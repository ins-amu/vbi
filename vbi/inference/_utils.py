"""
Simulation and training utilities - sbi-compatible helpers.

MI0-utils: simulate_for_sbi, process_prior
MI6:       simulate_for_vbi_sweep, simulate_for_vbi_sweep_cached, extract_from_cache
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def simulate_for_vbi_sweep(
    sim_spec,
    prior,
    pipeline,
    num_simulations: int,
    duration: float,
    sim_backend: str = "numba",
    seed: int | None = None,
    proposal=None,
    x_obs=None,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """
    Sample parameters, run a VBI sweep, extract features.

    Parameters
    ----------
    sim_spec        : SimulationSpec
    prior           : any prior with .sample() and ._resolved_param_names
    pipeline        : FeaturePipeline
    num_simulations : int
    duration        : float  simulation length in ms
    sim_backend     : str    backend for Sweeper ('numpy'|'numba'|'cuda'|'jax')
    seed            : int | None  RNG seed for prior sampling
    proposal        : Posterior | None
        When set, sample theta from posterior instead of prior.
        Requires x_obs to be supplied.
    x_obs           : ndarray | None  required when proposal is not None

    Returns
    -------
    theta          : (n, d_theta) float64  - parameter vectors
    x              : (n, d_x)     float64  - feature vectors
    param_names    : list[str]
    feature_labels : list[str]

    Notes
    -----
    The numpy/numba sweeper returns (labels, values) where
    values[:, :n_params] are the parameters and values[:, n_params:] are
    the features - confirmed from NumpySweeper and NumbaSweeperCPU source.
    Rows where any feature value is non-finite are dropped.
    """
    from vbi.simulator.api import Sweeper
    from vbi.simulator.spec.sweep import SweepSpec

    param_names = prior._resolved_param_names
    n_params = len(param_names)

    if proposal is None:
        theta = prior.sample((num_simulations,), seed=seed)
    else:
        if x_obs is None:
            raise ValueError("x_obs is required when proposal is provided.")
        rng = np.random.default_rng(seed)
        proposal_seed = int(rng.integers(0, 2**31))
        theta = proposal.sample((num_simulations,), x=np.asarray(x_obs),
                                seed=proposal_seed)

    theta = np.asarray(theta, dtype=np.float64)

    sweep_spec = SweepSpec(
        params=theta,
        param_names=tuple(param_names),
        pipeline=pipeline,
    )
    sweeper = Sweeper(sim_spec, sweep_spec, backend=sim_backend)
    labels, values = sweeper.run(duration)

    feature_labels = list(labels[n_params:])
    theta_out = values[:, :n_params].astype(np.float64)
    x_out     = values[:, n_params:].astype(np.float64)

    # Drop rows with any non-finite feature
    valid = np.all(np.isfinite(x_out), axis=1)
    n_dropped = int((~valid).sum())
    if n_dropped:
        log.warning(
            "simulate_for_vbi_sweep: dropped %d / %d rows with non-finite features.",
            n_dropped, num_simulations,
        )
    theta_out = theta_out[valid]
    x_out     = x_out[valid]

    return theta_out, x_out, param_names, feature_labels


# ---------------------------------------------------------------------------
# Simulation cache - Step 2 of MI6
# ---------------------------------------------------------------------------

def simulate_for_vbi_sweep_cached(
    sim_spec,
    prior,
    pipeline,
    num_simulations: int,
    duration: float,
    cache_dir,
    chunk_size: int = 500,
    sim_backend: str = "numba",
    seed: int | None = None,
    proposal=None,
    x_obs=None,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """
    Run a VBI sweep in chunks, write raw monitor recordings to disk, then
    extract features.  Use when simulations are expensive and you may want to
    re-extract features with different pipeline settings later.

    Cache layout::

        cache_dir/
          metadata.json       n_samples, chunk_size, param_names, signal, ...
          chunk_0000.npz      theta (chunk, d_θ), ts (chunk, n_record, ...), t (n_record,)
          chunk_0001.npz
          ...

    Parameters
    ----------
    sim_spec        : SimulationSpec
    prior           : prior object with ._resolved_param_names and .sample()
    pipeline        : FeaturePipeline  - determines which signal to cache
    num_simulations : int
    duration        : float  ms
    cache_dir       : str | Path  created if it doesn't exist
    chunk_size      : int  simulations per chunk; tune to fit GPU/RAM budget
    sim_backend     : str
    seed            : int | None
    proposal        : Posterior | None
    x_obs           : ndarray | None  required when proposal is set

    Returns
    -------
    theta          : (n, d_theta) float64
    x              : (n, d_x)     float64
    param_names    : list[str]
    feature_labels : list[str]

    Notes
    -----
    Raw time-series are stored as float32 to halve disk usage.  Feature
    extraction on reload converts back to float64 before calling
    pipeline.extract, matching the normal (non-cached) precision.

    To re-extract with a different pipeline later::

        theta, x = extract_from_cache(cache_dir, new_pipeline)
    """
    from vbi.simulator.api import Sweeper
    from vbi.simulator.spec.sweep import SweepSpec

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    param_names = prior._resolved_param_names
    signal = pipeline.signal

    if proposal is None:
        theta_all = prior.sample((num_simulations,), seed=seed)
    else:
        if x_obs is None:
            raise ValueError("x_obs is required when proposal is provided.")
        rng = np.random.default_rng(seed)
        proposal_seed = int(rng.integers(0, 2**31))
        theta_all = proposal.sample(
            (num_simulations,), x=np.asarray(x_obs), seed=proposal_seed
        )
    theta_all = np.asarray(theta_all, dtype=np.float64)

    n_chunks = (num_simulations + chunk_size - 1) // chunk_size

    for chunk_idx in range(n_chunks):
        start = chunk_idx * chunk_size
        end   = min(start + chunk_size, num_simulations)
        theta_chunk = theta_all[start:end]

        sweep_spec = SweepSpec(
            params=theta_chunk,
            param_names=tuple(param_names),
            pipeline=None,  # raw output - no feature extraction yet
        )
        sweeper = Sweeper(sim_spec, sweep_spec, backend=sim_backend)
        results = sweeper.run(duration)  # list[{monitor_kind: (t, data)}]

        ts_list = []
        t_vec   = None
        for r in results:
            t_i, data_i = r[signal]
            if t_vec is None:
                t_vec = np.asarray(t_i, dtype=np.float64)
            ts_list.append(np.asarray(data_i, dtype=np.float32))

        ts_chunk = np.stack(ts_list)  # (actual_chunk, n_record, ...)
        chunk_path = cache_dir / f"chunk_{chunk_idx:04d}.npz"
        np.savez_compressed(chunk_path,
                            theta=theta_chunk, ts=ts_chunk, t=t_vec)
        del ts_list, ts_chunk  # release before next chunk

        log.info(
            "simulate_for_vbi_sweep_cached: chunk %d/%d (%d sims) → %s",
            chunk_idx + 1, n_chunks, len(theta_chunk), chunk_path,
        )

    meta = {
        "n_samples":   num_simulations,
        "chunk_size":  chunk_size,
        "n_chunks":    n_chunks,
        "param_names": param_names,
        "signal":      signal,
        "duration":    duration,
        "sim_backend": sim_backend,
    }
    with open(cache_dir / "metadata.json", "w") as fh:
        json.dump(meta, fh, indent=2)

    theta_out, x_out, feat_labels = _extract_from_cache_impl(cache_dir, pipeline)
    return theta_out, x_out, param_names, feat_labels


def _extract_from_cache_impl(
    cache_dir,
    pipeline,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Internal: load chunks, extract features, return (theta, x, feat_labels)."""
    cache_dir = Path(cache_dir)
    with open(cache_dir / "metadata.json") as fh:
        meta = json.load(fh)

    cached_signal = meta["signal"]
    if pipeline.signal != cached_signal:
        raise ValueError(
            f"Pipeline signal {pipeline.signal!r} does not match the cached "
            f"signal {cached_signal!r}.  Either update pipeline.signal or "
            f"re-simulate with the new signal."
        )

    theta_rows: list[np.ndarray] = []
    x_rows:     list[np.ndarray] = []
    feat_labels: list[str] | None = None

    for chunk_idx in range(meta["n_chunks"]):
        chunk_path = cache_dir / f"chunk_{chunk_idx:04d}.npz"
        chunk  = np.load(chunk_path)
        theta_c = chunk["theta"]                    # (actual_chunk, d_theta)
        ts_c    = chunk["ts"]                       # (actual_chunk, n_record, ...)
        t_c     = chunk["t"]                        # (n_record,)

        for i in range(len(theta_c)):
            monitor_result = {cached_signal: (t_c, ts_c[i].astype(np.float64))}
            labels_i, vals_i = pipeline.extract(monitor_result)
            if feat_labels is None:
                feat_labels = labels_i
            elif labels_i != feat_labels:
                raise ValueError(
                    f"Feature labels changed at chunk {chunk_idx}, row {i}: "
                    f"expected {feat_labels}, got {labels_i}. "
                    "The pipeline may be data-dependent; ensure it produces "
                    "consistent labels across all cached samples."
                )
            theta_rows.append(theta_c[i])
            x_rows.append(vals_i)

        del chunk, ts_c  # free chunk memory

    theta_out = np.stack(theta_rows).astype(np.float64)
    x_out     = np.stack(x_rows).astype(np.float64)

    valid = np.all(np.isfinite(x_out), axis=1)
    n_dropped = int((~valid).sum())
    if n_dropped:
        log.warning(
            "extract_from_cache: dropped %d rows with non-finite features.",
            n_dropped,
        )

    return theta_out[valid], x_out[valid], feat_labels or []


def extract_from_cache(
    cache_dir,
    pipeline,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load cached raw recordings and extract features with a (possibly new) pipeline.

    Parameters
    ----------
    cache_dir : str | Path  directory written by simulate_for_vbi_sweep_cached
    pipeline  : FeaturePipeline

    Returns
    -------
    theta : (n, d_theta) float64
    x     : (n, d_x)     float64
    """
    theta, x, _ = _extract_from_cache_impl(cache_dir, pipeline)
    return theta, x


def simulate_for_vbi(
    simulator_fn,
    prior,
    num_simulations: int,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run a simulator for ``num_simulations`` parameter draws and collect
    ``(theta, x)`` pairs, mirroring ``sbi.utils.simulate_for_sbi``.

    Failed simulations (exceptions or non-finite x) are replaced with NaN
    rows so that ``SNPE.append_simulations(..., exclude_invalid_x=True)``
    silently filters them out.

    Parameters
    ----------
    simulator_fn   : callable  ``theta_1d -> x_1d``  (one simulation at a time)
    prior          : prior object with ``.sample((n,))``
    num_simulations : int
    seed           : int | None   RNG seed for prior sampling

    Returns
    -------
    theta : ndarray  (num_simulations, d_theta)
    x     : ndarray  (num_simulations, d_x)

    Examples
    --------
    >>> theta, x = simulate_for_sbi(my_sim, prior, num_simulations=1000)
    >>> inference.append_simulations(theta, x)
    """
    process_prior(prior)

    rng       = np.random.default_rng(seed)
    prior_seed = int(rng.integers(0, 2 ** 31))
    theta     = prior.sample((num_simulations,), seed=prior_seed)

    x_list: list[np.ndarray | None] = []
    x_dim:  int | None = None
    failed = 0

    for th in theta:
        try:
            x_i = np.asarray(simulator_fn(th), dtype=np.float32)
            if x_i.ndim == 0:
                x_i = x_i.reshape(1)
            if x_dim is None:
                x_dim = int(x_i.shape[0])
            x_list.append(x_i)
        except Exception:
            x_list.append(None)
            failed += 1

    # Replace None placeholders (failures before x_dim was known, or just None)
    fill = np.full(x_dim or 1, np.nan, dtype=np.float32)
    x_list = [xi if xi is not None else fill for xi in x_list]

    if failed:
        log.warning(
            "simulate_for_sbi: %d / %d simulations failed (NaN rows added; "
            "will be filtered by append_simulations).",
            failed, num_simulations,
        )

    x = np.stack(x_list)
    return theta, x


def process_prior(prior) -> object:
    """
    Validate that *prior* exposes the required ``.sample`` and ``.log_prob``
    interface, mirroring ``sbi.utils.process_prior``.

    Parameters
    ----------
    prior : object

    Returns
    -------
    prior  (returned unchanged for chaining)

    Raises
    ------
    ValueError  if the prior is missing required methods.
    """
    if not (hasattr(prior, "sample") and callable(prior.sample)):
        raise ValueError(
            "prior must have a callable .sample(sample_shape) method. "
            f"Got {type(prior).__name__!r}."
        )
    if not (hasattr(prior, "log_prob") and callable(prior.log_prob)):
        raise ValueError(
            "prior must have a callable .log_prob(theta) method. "
            f"Got {type(prior).__name__!r}."
        )
    return prior


# For backward compatibility
simulate_for_sbi = simulate_for_vbi
