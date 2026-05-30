"""
Shared helpers for VBIInference workflow scripts.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sp_signal

from vbi.feature_extraction import (
    FeaturePipeline,
    get_features_by_domain,
    get_features_by_given_names,
    update_cfg,
)


def build_jr_spectral_pipeline(
    fs_hz: float,
    t_cut: float = 500.0,
    voi: int | None = 1,
) -> FeaturePipeline:
    """
    Spectral feature pipeline for Jansen-Rit inference.

    Features: spectrum_stats, spectrum_auc, spectrum_moments (all averaged
    across nodes, Welch method).

    Parameters
    ----------
    fs_hz  : sampling frequency in Hz (= 1000 / tavg_period_ms)
    t_cut  : burn-in to discard in ms
    voi    : state-variable index to use; 1 = y1 (excitatory dendritic
             potential, the standard EEG proxy for JR)
    """
    cfg = get_features_by_domain("spectral")
    cfg = get_features_by_given_names(
        cfg, ["spectrum_stats", "spectrum_auc", "spectrum_moments"]
    )
    for key in ("spectrum_stats", "spectrum_auc", "spectrum_moments"):
        update_cfg(cfg, key, {"fs": fs_hz, "method": "welch", "average": True})

    return FeaturePipeline(cfg, signal="tavg", t_cut=t_cut, voi=voi)


def plot_jr_timeseries_psd(
    t: np.ndarray,
    ts: np.ndarray,
    fs_hz: float,
    title: str,
    out_path: Path,
    n_plot_nodes: int = 5,
    t_window_ms: tuple[float, float] | None = None,
) -> None:
    """
    Two-panel figure: time series (subset of nodes) + mean PSD across nodes.

    Parameters
    ----------
    t          : time array in ms, shape (T,)
    ts         : state array, shape (T, n_sv, N) — raw monitor output
    fs_hz      : sampling frequency in Hz
    title      : figure title
    out_path   : file path to save
    n_plot_nodes : how many nodes to overlay
    t_window_ms  : optional (t_start, t_end) zoom window in ms
    """
    # EEG proxy: y1 (index 1) for selected nodes
    y1 = ts[:, 1, :n_plot_nodes]   # (T, n_plot_nodes)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3))

    # Left: time series
    if t_window_ms is not None:
        mask = (t >= t_window_ms[0]) & (t <= t_window_ms[1])
        t_plot, y_plot = t[mask], y1[mask]
    else:
        t_plot, y_plot = t, y1
    axes[0].plot(t_plot, y_plot, lw=0.8, alpha=0.7)
    axes[0].set_xlabel("t (ms)")
    axes[0].set_ylabel("y1 (mV)")
    axes[0].set_title("Time series (y1, first nodes)")
    axes[0].margins(x=0)

    # Right: mean PSD across all nodes of y1
    freqs, psd = sp_signal.welch(ts[:, 1, :].T, fs=fs_hz, nperseg=min(512, ts.shape[0] // 4))
    axes[1].semilogy(freqs, psd.mean(axis=0), lw=1.2, color="steelblue")
    axes[1].set_xlim(0, 80)
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("PSD")
    axes[1].set_title("Mean PSD across nodes")

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
