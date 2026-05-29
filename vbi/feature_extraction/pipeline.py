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

    def __init__(self, cfg: dict, signal: str = "tavg", t_cut: float = 500.0):
        self.cfg = cfg
        self.signal = signal
        self.t_cut = t_cut

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

        # Normalise shape to (n_nodes, n_steps)
        if ts_cut.ndim == 3:
            # (n_steps, n_voi, n_nodes) - tavg / subsample / raw monitors
            ts_2d = ts_cut[:, 0, :].T
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
        When None is returned the sweeper falls back to the Tier-1 Python path.
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

        return spec
