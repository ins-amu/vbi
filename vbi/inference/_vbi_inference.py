"""
VBIInference — end-to-end SBI workflow object.  MI6.

Owns the full loop: prior sampling → sweep simulation → feature extraction
→ SNPE training → posterior.  Raw recordings can optionally be cached to
disk so features can be recomputed with different pipeline settings later.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ._api import SNPE
from ._utils import (
    simulate_for_vbi_sweep,
    simulate_for_vbi_sweep_cached,
    extract_from_cache as _extract_from_cache,
    _extract_from_cache_impl,
)

if TYPE_CHECKING:
    from ._posterior import Posterior
    from ._estimators import ConditionalDensityEstimator

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Estimator serialisation helpers
# ---------------------------------------------------------------------------

def _pack_estimator(est, data: dict) -> None:
    """
    Write all state needed to reconstruct ``est`` for inference (not training)
    into the flat numpy array dict ``data`` using an ``est__`` prefix.

    Saved beyond the basic ``weights / param_dim / feature_dim``:
      - z-score normalizers  (theta_mean/std, x_mean/std)
      - autoregressive masks (model_constants per flow)
      - ActNorm init flags
      - PCA components (if used)
      - n_flows (so load can set up the right list sizes)
    """
    for k, v in est.weights.items():
        data[f"est__w__{k}"] = np.asarray(v)

    data["est__param_dim"]   = np.array(est.param_dim)
    data["est__feature_dim"] = np.array(est.feature_dim)
    data["est__n_flows"]     = np.array(getattr(est, "n_flows", 0))

    for attr in ("theta_mean", "theta_std", "x_mean", "x_std"):
        val = getattr(est, attr, None)
        if val is not None:
            data[f"est__{attr}"] = np.asarray(val)

    mc = getattr(est, "model_constants", None)
    if mc is not None:
        for k_flow, layer in enumerate(mc["layers"]):
            for key in ("M1", "M2", "Mm", "perm", "inv_perm"):
                if key in layer:
                    data[f"est__mc__{k_flow}__{key}"] = np.asarray(layer[key])

    actnorm = getattr(est, "_actnorm_initialized", None)
    if actnorm is not None:
        data["est__actnorm_init"] = np.array(actnorm, dtype=bool)

    if getattr(est, "_use_pca", False) and getattr(est, "_pca_components", None) is not None:
        data["est__pca_components"] = np.asarray(est._pca_components)


def _unpack_estimator(d, EstCls):
    """
    Reconstruct a fully functional estimator from the flat array dict ``d``
    (keys produced by ``_pack_estimator``).

    Uses ``EstCls(n_flows=n)`` so ``__post_init__`` initialises the correct
    list sizes, then overrides all attributes with the restored values.
    """
    n_flows = int(d["est__n_flows"]) if "est__n_flows" in d else 4
    est = EstCls(n_flows=n_flows)

    est.weights        = {k[len("est__w__"):]: d[k] for k in d if k.startswith("est__w__")}
    est.param_dim      = int(d["est__param_dim"])
    est.feature_dim    = int(d["est__feature_dim"])
    est._dims_inferred = True
    est.loss_history   = []

    for attr in ("theta_mean", "theta_std", "x_mean", "x_std"):
        if f"est__{attr}" in d:
            setattr(est, attr, d[f"est__{attr}"])

    mc_keys = [k for k in d if k.startswith("est__mc__")]
    if mc_keys:
        n_mc_flows = max(int(k.split("__")[2]) for k in mc_keys) + 1
        layers = []
        for k_flow in range(n_mc_flows):
            layer = {
                key: d[f"est__mc__{k_flow}__{key}"]
                for key in ("M1", "M2", "Mm", "perm", "inv_perm")
                if f"est__mc__{k_flow}__{key}" in d
            }
            layers.append(layer)
        est.model_constants = {"layers": layers}

    if "est__actnorm_init" in d:
        est._actnorm_initialized = list(d["est__actnorm_init"])

    if "est__pca_components" in d:
        est._pca_components = d["est__pca_components"]
        est._use_pca = True

    return est


class VBIInference:
    """
    End-to-end VBI inference workflow.

    Owns the full loop: prior sampling → sweep simulation → feature
    extraction → SNPE training → posterior.

    Parameters
    ----------
    sim_spec : SimulationSpec
    prior : prior object
        Must expose ``._resolved_param_names``, ``.sample()``, ``.log_prob()``.
    pipeline : FeaturePipeline
    density_estimator : 'maf' | 'mdn' | 'nsf'
    sim_backend : str
        Backend for the Sweeper ('numpy' | 'numba' | 'cuda' | 'jax').
    backend : str
        Backend for the density estimator ('auto' | 'numpy' | 'jax').
    show_progress_bars : bool
    embedding_net : EmbeddingNet | None

    Examples
    --------
    >>> inf = VBIInference(sim_spec, prior, pipeline)
    >>> theta, x = inf.simulate(num_simulations=2000, duration=5000.0)
    >>> est = inf.train(training_batch_size=256, stop_after_epochs=30)
    >>> post = inf.build_posterior(est)
    >>> samples = post.sample((1000,), x=x_obs)
    >>> inf.save("run.npz")
    """

    def __init__(
        self,
        sim_spec,
        prior,
        pipeline,
        density_estimator: str = "maf",
        sim_backend: str = "numba",
        backend: str = "auto",
        show_progress_bars: bool = True,
        embedding_net=None,
    ):
        self._sim_spec          = sim_spec
        self._prior             = prior
        self._pipeline          = pipeline
        self._sim_backend       = sim_backend
        self._de_type           = density_estimator
        self._snpe              = SNPE(
            prior               = prior,
            density_estimator   = density_estimator,
            backend             = backend,
            show_progress_bars  = show_progress_bars,
            embedding_net       = embedding_net,
        )
        self._feature_labels: list[str] | None = None
        self._param_names:    list[str] | None = None
        self._default_train_kwargs: dict       = {}
        self._last_estimator                   = None

    # ------------------------------------------------------------------
    # Core workflow
    # ------------------------------------------------------------------

    def simulate(
        self,
        num_simulations: int,
        duration: float,
        seed: int | None = None,
        proposal=None,
        x_obs=None,
        cache_dir=None,
        chunk_size: int = 500,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Sample parameters and run the sweep simulator.

        Appends the resulting ``(theta, x)`` pairs to the internal SNPE
        object, ready for training.

        Parameters
        ----------
        num_simulations : int
        duration : float  simulation length in ms
        seed : int | None
        proposal : Posterior | None
            When set, sample theta from this posterior instead of the prior.
            Requires ``x_obs``.
        x_obs : ndarray | None  required when proposal is not None
        cache_dir : str | Path | None
            When set, raw monitor recordings are written to this directory
            in chunked ``.npz`` files.  Use ``extract_from_cache()`` later
            to re-extract features with a different pipeline.
        chunk_size : int
            Simulations per chunk (only used when cache_dir is set).
            Tune to stay within GPU/RAM budget.

        Returns
        -------
        theta : (n, d_theta) ndarray
        x     : (n, d_x)     ndarray
        """
        common = dict(
            sim_spec        = self._sim_spec,
            prior           = self._prior,
            pipeline        = self._pipeline,
            num_simulations = num_simulations,
            duration        = duration,
            sim_backend     = self._sim_backend,
            seed            = seed,
            proposal        = proposal,
            x_obs           = x_obs,
        )

        if cache_dir is None:
            theta, x, param_names, feat_labels = simulate_for_vbi_sweep(**common)
        else:
            theta, x, param_names, feat_labels = simulate_for_vbi_sweep_cached(
                **common, cache_dir=cache_dir, chunk_size=chunk_size
            )

        if self._param_names is None:
            self._param_names = param_names
        if self._feature_labels is None:
            self._feature_labels = feat_labels

        self._snpe.append_simulations(theta, x, proposal=proposal)
        return theta, x

    def train(self, **train_kwargs) -> "ConditionalDensityEstimator":
        """
        Train the density estimator on all accumulated simulations.

        Keyword arguments are forwarded to ``SNPE.train()``.  Any kwargs
        stored via ``from_config`` serve as defaults and can be overridden
        here.

        Returns
        -------
        estimator : ConditionalDensityEstimator
        """
        if self._snpe.n_simulations == 0:
            raise RuntimeError(
                "No simulations available.  Call simulate() before train()."
            )
        merged = {**self._default_train_kwargs, **train_kwargs}
        self._last_estimator = self._snpe.train(**merged)
        return self._last_estimator

    def build_posterior(
        self,
        estimator=None,
        sample_with: str = "direct",
        **kwargs,
    ) -> "Posterior":
        """
        Wrap the trained estimator in a Posterior object.

        Parameters
        ----------
        estimator : ConditionalDensityEstimator | None
            Uses the last one returned by ``train()`` if None.
        sample_with : 'direct' | 'mcmc' | 'rejection'

        Returns
        -------
        Posterior
        """
        est = estimator if estimator is not None else self._last_estimator
        return self._snpe.build_posterior(
            density_estimator=est, sample_with=sample_with, **kwargs
        )

    def get_simulations(self, starting_round: int = 0):
        """
        Return all accumulated (theta, x, proposals) from ``starting_round``.

        Returns
        -------
        theta     : (N, d_theta)
        x         : (N, d_x)
        proposals : list
        """
        return self._snpe.get_simulations(starting_round)

    # ------------------------------------------------------------------
    # Cache helper (static — usable without an instance)
    # ------------------------------------------------------------------

    @staticmethod
    def extract_from_cache(cache_dir, pipeline) -> tuple[np.ndarray, np.ndarray]:
        """
        Load cached raw recordings and extract features with a new pipeline.

        Parameters
        ----------
        cache_dir : str | Path
        pipeline  : FeaturePipeline

        Returns
        -------
        theta : (n, d_theta) float64
        x     : (n, d_x)     float64
        """
        return _extract_from_cache(cache_dir, pipeline)

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, path) -> None:
        """
        Persist inference state to a single ``.npz`` checkpoint.

        Saved: simulation data (theta, x across all rounds), full
        estimator state (weights + normalizers + masks), feature/param
        label lists, and constructor kwargs.

        NOT saved: ``sim_spec`` and ``pipeline`` — supply them on load.

        Parameters
        ----------
        path : str | Path  target file (``.npz`` extension added if absent)
        """
        path = Path(path)
        if not path.suffix:
            path = path.with_suffix(".npz")

        data: dict[str, np.ndarray] = {}

        # -- simulation data -------------------------------------------------
        if self._snpe.n_simulations > 0:
            theta_all, x_all, _ = self._snpe.get_simulations()
            data["sim__theta_all"] = theta_all
            data["sim__x_all"]     = x_all

        # -- estimator state -------------------------------------------------
        if self._last_estimator is not None:
            _pack_estimator(self._last_estimator, data)

        # -- metadata --------------------------------------------------------
        data["meta__feature_labels"]    = np.array("|".join(self._feature_labels or []))
        data["meta__param_names"]       = np.array("|".join(self._param_names or []))
        data["meta__density_estimator"] = np.array(self._de_type)
        data["meta__sim_backend"]       = np.array(self._sim_backend)
        data["meta__backend"]           = np.array(self._snpe._backend)
        data["meta__n_rounds"]          = np.array(self._snpe.n_rounds)

        np.savez(path, **data)
        log.info("VBIInference saved to %s", path)

    @classmethod
    def load(
        cls,
        path,
        sim_spec,
        pipeline,
        prior=None,
        show_progress_bars: bool = True,
        embedding_net=None,
    ) -> "VBIInference":
        """
        Restore from a checkpoint written by ``save()``.

        Parameters
        ----------
        path : str | Path
        sim_spec : SimulationSpec
        pipeline : FeaturePipeline  may differ from the saved one
        prior : prior object | None
            Required to call ``build_posterior()`` after load.

        Returns
        -------
        VBIInference  fully reconstructed — ``train()`` and
                      ``build_posterior()`` work immediately.
        """
        from ._backends import resolve_backend, get_estimator_map

        path = Path(path)
        d    = np.load(path, allow_pickle=False)

        de_type     = str(d["meta__density_estimator"])
        sim_backend = str(d["meta__sim_backend"])
        backend     = str(d["meta__backend"])

        inf = cls(
            sim_spec           = sim_spec,
            prior              = prior,
            pipeline           = pipeline,
            density_estimator  = de_type,
            sim_backend        = sim_backend,
            backend            = backend,
            show_progress_bars = show_progress_bars,
            embedding_net      = embedding_net,
        )

        fl_raw = str(d["meta__feature_labels"])
        pn_raw = str(d["meta__param_names"])
        inf._feature_labels = fl_raw.split("|") if fl_raw else []
        inf._param_names    = pn_raw.split("|") if pn_raw else []

        if "sim__theta_all" in d and "sim__x_all" in d:
            inf._snpe.append_simulations(
                d["sim__theta_all"], d["sim__x_all"],
                exclude_invalid_x=False,
            )

        if "est__param_dim" in d:
            backend_resolved = resolve_backend(backend)
            de_map           = get_estimator_map(backend_resolved)
            EstCls           = de_map[de_type]
            est = _unpack_estimator(d, EstCls)
            inf._last_estimator  = est
            inf._snpe._estimator = est

        log.info(
            "VBIInference loaded from %s  (n_sims=%d, estimator=%s)",
            path, inf._snpe.n_simulations,
            "restored" if inf._last_estimator is not None else "not found",
        )
        return inf

    # ------------------------------------------------------------------
    # Config loader  (Step 5)
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config) -> "VBIInference":
        """Build from a YAML / JSON config dict or file path.  Step 5."""
        raise NotImplementedError("from_config coming in Step 5.")

    # ------------------------------------------------------------------
    # Diagnostics  (Step 6)
    # ------------------------------------------------------------------

    def plot_loss(self):
        """Plot training / validation loss curves.  Step 6."""
        raise NotImplementedError("Diagnostics coming in Step 6.")

    def pairplot(self, x_obs, num_samples: int = 1000, **kwargs):
        """Pairplot of posterior samples at x_obs.  Step 6."""
        raise NotImplementedError("Diagnostics coming in Step 6.")

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"VBIInference("
            f"density_estimator={self._de_type!r}, "
            f"sim_backend={self._sim_backend!r}, "
            f"n_rounds={self._snpe.n_rounds}, "
            f"n_sims={self._snpe.n_simulations})"
        )
