"""
Tier-2 JIT feature kernels for the Numba CPU backend.

These @njit functions are called inside prange, so they must be pure numeric
kernels with no Python-level dispatch, dicts, or pandas.

Supported core set
------------------
calc_mean   per-node temporal mean           → (n_nodes,)
calc_std    per-node temporal std            → (n_nodes,)
fc_flat     upper-triangle Pearson FC        → (n_nodes*(n_nodes-1)//2,)
fcd_mean    mean of FCD upper triangle       → scalar

Input convention: ts_2d is always (n_steps, n_nodes) float64.
This matches nb_simulate_det output ts_i[:, voi, :].

NbExtractorSpec
---------------
A plain Python dataclass (not a numba jitclass) that the pipeline and sweeper
use to communicate which features to compute and how large the output will be.
The actual number-crunching is done by nb_extract(), which is @njit.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numba import njit


# ---------------------------------------------------------------------------
# Core kernels
# ---------------------------------------------------------------------------

@njit(cache=True)
def nb_mean(ts):
    """Per-node temporal mean.  ts: (n_steps, n_nodes) → (n_nodes,)"""
    n_steps, n_nodes = ts.shape
    out = np.zeros(n_nodes)
    for t in range(n_steps):
        for n in range(n_nodes):
            out[n] += ts[t, n]
    for n in range(n_nodes):
        out[n] /= n_steps
    return out


@njit(cache=True)
def nb_std(ts):
    """Per-node temporal std (ddof=1).  ts: (n_steps, n_nodes) → (n_nodes,)"""
    n_steps, n_nodes = ts.shape
    mean = nb_mean(ts)
    out = np.zeros(n_nodes)
    for t in range(n_steps):
        for n in range(n_nodes):
            d = ts[t, n] - mean[n]
            out[n] += d * d
    for n in range(n_nodes):
        out[n] = (out[n] / (n_steps - 1)) ** 0.5 if n_steps > 1 else 0.0
    return out


@njit(cache=True)
def nb_fc_flat(ts):
    """
    Upper triangle (k=1) of Pearson correlation matrix.
    ts: (n_steps, n_nodes) → (n_nodes*(n_nodes-1)//2,)
    """
    n_steps, n_nodes = ts.shape
    mean = nb_mean(ts)

    n_upper = n_nodes * (n_nodes - 1) // 2
    out = np.zeros(n_upper)
    idx = 0
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            cov = 0.0
            var_i = 0.0
            var_j = 0.0
            for t in range(n_steps):
                xi = ts[t, i] - mean[i]
                xj = ts[t, j] - mean[j]
                cov   += xi * xj
                var_i += xi * xi
                var_j += xj * xj
            denom = (var_i * var_j) ** 0.5
            out[idx] = cov / denom if denom > 0.0 else 0.0
            idx += 1
    return out


@njit(cache=True)
def nb_fcd_mean(ts, window_size):
    """
    Mean of the FCD (functional connectivity dynamics) upper triangle.

    Slides a window of `window_size` steps over the time series, computes the
    FC vector for each window, then returns the mean pairwise Pearson
    correlation of those FC vectors (= mean of FCD matrix upper triangle).

    ts: (n_steps, n_nodes), window_size: int → scalar float64
    """
    n_steps, n_nodes = ts.shape
    n_upper = n_nodes * (n_nodes - 1) // 2
    n_windows = n_steps - window_size + 1

    if n_windows < 2:
        return 0.0

    # Compute FC vector for each window
    fc_wins = np.empty((n_windows, n_upper))
    for w in range(n_windows):
        fc_wins[w] = nb_fc_flat(ts[w: w + window_size])

    # Mean of FCD upper triangle (pairwise correlation of FC vectors)
    total = 0.0
    count = 0
    for i in range(n_windows):
        for j in range(i + 1, n_windows):
            fi = fc_wins[i]
            fj = fc_wins[j]
            mi = 0.0
            mj = 0.0
            for k in range(n_upper):
                mi += fi[k]
                mj += fj[k]
            mi /= n_upper
            mj /= n_upper
            num = 0.0
            di  = 0.0
            dj  = 0.0
            for k in range(n_upper):
                xi = fi[k] - mi
                xj = fj[k] - mj
                num += xi * xj
                di  += xi * xi
                dj  += xj * xj
            denom = (di * dj) ** 0.5
            total += num / denom if denom > 0.0 else 0.0
            count += 1

    return total / count if count > 0 else 0.0


# ---------------------------------------------------------------------------
# Combiner - called inside prange, writes into a pre-allocated buffer
# ---------------------------------------------------------------------------

@njit(cache=True)
def nb_extract(ts_2d, do_mean, do_std, do_fc, do_fcd, fcd_window, out):
    """
    Fill pre-allocated `out` with the requested features.

    Feature order (whichever are enabled) - matches Tier-1 (JSON) order:
        std_0 … std_{n-1}  |  mean_0 … mean_{n-1}
        fc_0_1 … fc_{n-2}_{n-1}  |  fcd_mean

    Parameters
    ----------
    ts_2d      : (n_steps, n_nodes) float64
    do_mean    : bool
    do_std     : bool
    do_fc      : bool
    do_fcd     : bool
    fcd_window : int   sliding-window size for FCD (ignored if do_fcd=False)
    out        : (n_features,) float64  - pre-allocated output buffer
    """
    _, n_nodes = ts_2d.shape
    idx = 0

    if do_std:
        s = nb_std(ts_2d)
        for i in range(n_nodes):
            out[idx] = s[i]
            idx += 1

    if do_mean:
        m = nb_mean(ts_2d)
        for i in range(n_nodes):
            out[idx] = m[i]
            idx += 1

    if do_fc:
        fc = nb_fc_flat(ts_2d)
        n_fc = n_nodes * (n_nodes - 1) // 2
        for i in range(n_fc):
            out[idx] = fc[i]
            idx += 1

    if do_fcd:
        out[idx] = nb_fcd_mean(ts_2d, fcd_window)
        idx += 1


# ---------------------------------------------------------------------------
# NbExtractorSpec - Python-level descriptor used by pipeline + sweeper
# ---------------------------------------------------------------------------

# Mapping from cfg feature name → attribute on NbExtractorSpec
_CFG_TO_FLAG: dict[str, str] = {
    "calc_mean": "do_mean",
    "calc_std":  "do_std",
    "fc_flat":   "do_fc",    # Tier-2 only key (not in Python-tier cfg)
    "fcd_mean":  "do_fcd",   # Tier-2 only key
}

# Python-tier cfg keys that map to a Tier-2 equivalent (best-effort)
_PYTHON_TIER_COMPAT: dict[str, str] = {
    "calc_mean": "do_mean",
    "calc_std":  "do_std",
}


@dataclass
class NbExtractorSpec:
    """
    Descriptor for which JIT features to compute and how.

    Created by FeaturePipeline.nb_extractor; consumed by NumbaSweeperCPU
    to dispatch nb_sweep_det_feat / nb_sweep_stoch_feat instead of the
    Tier-1 Python-loop path.

    n_voi_feat
    ----------
    How many VOIs (state variables) to include as channels for feature
    extraction.
      1  (default)  — VOI 0 only (backward-compatible for brain models)
     -1             — all VOIs; the sweeper resolves this to model.n_sv at
                      run time and flattens (n_voi, n_nodes) into n_channels.
    """
    do_mean:      bool = False
    do_std:       bool = False
    do_fc:        bool = False
    do_fcd:       bool = False
    fcd_window:   int  = 30
    n_voi_feat:   int  = 1   # 1 = VOI 0 only; -1 = all VOIs (resolved by sweeper)
    voi_diff_pos: int  = -1  # >=0: compute VOI[pos]-VOI[neg] as single channel; -1 = disabled
    voi_diff_neg: int  = -1

    def n_features(self, n_channels: int) -> int:
        """n_channels = n_voi_feat * n_nodes (after VOI flattening)."""
        n = 0
        if self.do_mean: n += n_channels
        if self.do_std:  n += n_channels
        if self.do_fc:   n += n_channels * (n_channels - 1) // 2
        if self.do_fcd:  n += 1
        return n

    def labels(self, n_channels: int) -> list[str]:
        """n_channels = n_voi_feat * n_nodes (after VOI flattening)."""
        lbls: list[str] = []
        if self.do_std:
            lbls += [f"std_{i}" for i in range(n_channels)]
        if self.do_mean:
            lbls += [f"mean_{i}" for i in range(n_channels)]
        if self.do_fc:
            lbls += [f"fc_{i}_{j}"
                     for i in range(n_channels)
                     for j in range(i + 1, n_channels)]
        if self.do_fcd:
            lbls += ["fcd_mean"]
        return lbls
