from __future__ import annotations

import numpy as np


class FeaturePipeline:
    """
    Connects vbi.simulator monitor output to the existing cfg-dict feature
    extraction machinery.

    Per-feature parameters (window sizes, band ranges, etc.) belong in the
    cfg dict - built with get_features_by_domain / get_features_by_given_names /
    update_cfg as usual.  This class only adds the three sim-level concerns:
    which monitor signal to consume, how much burn-in to discard, and (in
    future milestones) which JIT backend to dispatch to.

    Parameters
    ----------
    cfg : dict
        Feature configuration from get_features_by_domain or similar.
    signal : str
        Monitor kind to consume: "tavg", "subsample", "raw", "bold".
    t_cut : float
        Burn-in to discard in ms before extracting features.
        When used with NumbaSweeperCPU the time array already starts at
        t_cut, so this has no additional effect (searchsorted returns 0).
    voi : int | tuple[int, int] | None
        Which VOI (state-variable index) to use from a 3-D monitor output
        ``(n_steps, n_voi, n_nodes)``.

        * ``0`` (default) — use only VOI 0 (backward-compatible; correct for
          most brain models where VOI 0 is the primary output variable).
        * ``(a, b)`` — use the derived channel ``VOI[a] - VOI[b]``. Useful for
          Jansen-Rit EEG/LFP proxy ``y1 - y2``.
        * ``None`` — use **all** VOIs; the resulting feature matrix has
          ``n_voi × n_nodes`` channels.  Useful for models with multiple
          meaningful state variables, e.g. DampedOscillator (x and y).
    pruner : FeaturePruner | None
        Optional :class:`~vbi.feature_extraction.pruner.FeaturePruner` to
        associate with this pipeline.  It is **not** applied automatically
        inside :meth:`extract` (pruning requires many simulation rows);
        attach it here so that downstream code (inference scripts, notebooks)
        can access it via ``pipeline.pruner`` without carrying a separate
        variable.

        Typical workflow::

            pruner = FeaturePruner(min_std=1e-4, max_corr=0.98)
            pipeline = FeaturePipeline(cfg, signal="raw", t_cut=500.0,
                                       voi=(1, 2), pruner=pruner)

            theta, x = inf.simulate(N_SIM, DURATION)
            x, labels = pipeline.pruner.fit_transform(x, feature_labels)
            x_obs_pruned = pipeline.pruner.transform(x_obs[None])[0]

    Examples
    --------
    cfg = get_features_by_domain("connectivity")
    cfg = get_features_by_given_names(cfg, ["calc_fc", "calc_fcd"])
    cfg = update_cfg(cfg, "calc_fcd", {"window_size": 30})

    pipeline = FeaturePipeline(cfg, signal="tavg", t_cut=500.0)

    # Single run
    result = Simulator(spec, backend="numpy").run(5000.0)
    labels, values = pipeline.extract(result)
    df = pipeline.extract_df(result)

    # Sweep
    sweep_spec = SweepSpec(params={"G": np.linspace(1, 4, 50)}, pipeline=pipeline)
    df = Sweeper(spec, sweep_spec, backend="numba").run_df(5000.0)
    """

    def __init__(
        self,
        cfg: dict,
        signal: str = "tavg",
        t_cut: float = 500.0,
        voi: int | tuple[int, int] | str | None = 0,
        pruner=None,
    ):
        self.cfg = cfg
        self.signal = signal
        self.t_cut = t_cut
        self.voi = voi
        self.pruner = pruner

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare(
        self, t: np.ndarray, ts: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """
        Apply burn-in cut and reshape to (n_nodes, n_steps) for feature fns.

        Returns
        -------
        ts_2d : (n_nodes, n_steps) float64
        fs    : sampling frequency in Hz, inferred from the time array (ms)
        """
        t_cut_idx = int(np.searchsorted(t, self.t_cut))
        ts_cut = ts[t_cut_idx:]

        # fs from the time array (t is in ms → Hz = 1000 / dt_ms)
        dt_ms = float(t[1] - t[0]) if len(t) > 1 else 1.0
        fs = 1000.0 / dt_ms

        # Normalise shape to (n_channels, n_steps) for feature functions.
        # n_channels = n_nodes         (when voi is an int or a difference tuple)
        # n_channels = n_voi * n_nodes (when voi is "all" -> all VOIs)
        if ts_cut.ndim == 3:
            # (n_steps, n_voi, n_nodes) - tavg / subsample / raw monitors
            if self.voi in ("all", None):
                # flatten all VOIs: (n_steps, n_voi, n_nodes) -> (n_voi*n_nodes, n_steps)
                n_steps, n_voi, n_nodes = ts_cut.shape
                ts_2d = ts_cut.reshape(n_steps, n_voi * n_nodes).T
            elif isinstance(self.voi, tuple):
                if len(self.voi) != 2:
                    raise ValueError(
                        "FeaturePipeline voi tuple must contain exactly two "
                        "indices: (positive_idx, negative_idx)."
                    )
                pos, neg = self.voi
                ts_2d = (ts_cut[:, pos, :] - ts_cut[:, neg, :]).T
            else:
                ts_2d = ts_cut[:, self.voi, :].T
        elif ts_cut.ndim == 2:
            # (n_steps, n_nodes) - bold monitor
            ts_2d = ts_cut.T
        else:
            raise ValueError(
                f"Unexpected time-series shape {ts_cut.shape}; "
                "expected (n_steps, n_voi, n_nodes) or (n_steps, n_nodes)."
            )

        return ts_2d.astype(np.float64, copy=False), fs

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self, monitor_result: dict
    ) -> tuple[list[str], np.ndarray]:
        """
        Extract features from a single monitor result dict.

        Parameters
        ----------
        monitor_result : dict
            Output of Simulator.run() - {monitor_kind: (times_ms, data)}.

        Returns
        -------
        labels : list[str]
        values : np.ndarray, shape (n_features,), dtype float64
        """
        from vbi.feature_extraction.calc_features import calc_features

        t, ts = monitor_result[self.signal]
        ts_2d, fs = self._prepare(np.asarray(t), np.asarray(ts))
        values, labels, _ = calc_features(ts_2d, fs, self.cfg)
        return labels, np.array(values, dtype=np.float64)

    def extract_df(self, monitor_result: dict):
        """Extract features and return a one-row pandas DataFrame."""
        import pandas as pd

        labels, values = self.extract(monitor_result)
        return pd.DataFrame([values], columns=labels)

    # ------------------------------------------------------------------
    # Tier-2 JIT descriptor
    # ------------------------------------------------------------------

    @property
    def nb_extractor(self):
        """
        Return an NbExtractorSpec if every feature in this pipeline's cfg
        is supported by the Numba JIT tier, otherwise return None.

        Currently supported cfg keys: calc_mean, calc_std.
        When None is returned the sweeper falls back to the Tier-1 Python path
        (simulation still runs in numba; only feature extraction is in Python).
        """
        from vbi.feature_extraction.features_utils_nb import (
            NbExtractorSpec,
            _PYTHON_TIER_COMPAT,
        )

        # Collect all feature names across every domain in the cfg
        all_names: set[str] = set()
        for domain_feats in self.cfg.values():
            all_names |= set(domain_feats.keys())

        # Any unsupported feature → fall back to Tier 1
        if not all_names.issubset(_PYTHON_TIER_COMPAT):
            return None

        spec = NbExtractorSpec()
        for name in all_names:
            flag = _PYTHON_TIER_COMPAT[name]
            setattr(spec, flag, True)

        if isinstance(self.voi, tuple):
            pos, neg = self.voi
            spec.n_voi_feat   = 1
            spec.voi_diff_pos = int(pos)
            spec.voi_diff_neg = int(neg)
        else:
            # voi="all" -> use all VOIs (-1 is a sentinel; sweeper resolves to n_sv)
            # voi=0     -> VOI 0 only (default, backward-compatible)
            spec.n_voi_feat = -1 if self.voi in ("all", None) else 1

        return spec
