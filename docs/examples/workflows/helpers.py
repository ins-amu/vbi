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
    FeaturePruner,
    get_features_by_domain,
    get_features_by_given_names,
    update_cfg,
)


def plot_feature_scatter(theta, x, x_obs, feature_labels, true_a2, out_path):
    theta = np.asarray(theta)
    x = np.asarray(x)
    x_obs = np.asarray(x_obs)

    if x.ndim != 2 or theta.ndim != 2:
        raise ValueError(f"Expected theta and x to be 2-D, got {theta.shape}, {x.shape}.")
    if x_obs.shape[0] != x.shape[1]:
        raise ValueError(
            f"x_obs has {x_obs.shape[0]} features but training x has {x.shape[1]}."
        )

    labels = list(feature_labels or [])
    if len(labels) != x.shape[1]:
        labels = [f"feature_{i}" for i in range(x.shape[1])]

    n_feat = x.shape[1]
    n_cols = min(4, n_feat)
    n_rows = int(np.ceil(n_feat / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.8 * n_cols, 2.8 * n_rows),
        squeeze=False,
    )

    a2 = theta[:, 0]
    for i, ax in enumerate(axes.flat):
        if i >= n_feat:
            ax.axis("off")
            continue
        ax.scatter(a2, x[:, i], s=18, alpha=0.7, color="steelblue", edgecolor="none")
        ax.scatter(
            [true_a2],
            [x_obs[i]],
            s=70,
            marker="*",
            color="crimson",
            edgecolor="black",
            linewidth=0.4,
            zorder=3,
            label="x_obs",
        )
        ax.axvline(true_a2, color="crimson", lw=0.9, alpha=0.55)
        ax.set_title(labels[i], fontsize=8)
        ax.set_xlabel(r"$a_2$")
        ax.set_ylabel("feature")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

    axes.flat[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Training features vs a_2 with observed feature point", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def build_jr_spectral_pipeline(
    fs_hz: float,
    t_cut: float = 500.0,
    voi: int | tuple[int, int] | None = (1, 2),
    signal: str = "tavg",
    pruner=None,
) -> FeaturePipeline:
    """
    Spectral feature pipeline for Jansen-Rit inference.

    Features: spectrum_stats, spectrum_auc, spectrum_moments (all averaged
    across nodes, Welch method).

    Parameters
    ----------
    fs_hz  : sampling frequency in Hz
    t_cut  : burn-in to discard in ms
    voi    : state-variable index or difference to use; (1, 2) = y1 - y2,
             the standard EEG/LFP proxy for JR.
    signal : monitor output key to extract features from.
    pruner : FeaturePruner | None
        Optional pruner to attach to the pipeline.  Not applied automatically;
        call ``pipeline.pruner.fit_transform(x, labels)`` after the sweep.
        Example: ``pruner=FeaturePruner(min_std=1e-4, max_corr=0.98)``.
    """
    cfg = get_features_by_domain("spectral")
    cfg = get_features_by_given_names(
        cfg, ["spectrum_stats", "spectrum_moments"]
    )
    for key in ("spectrum_stats", "spectrum_moments"):
        update_cfg(cfg, key, {"fs": fs_hz, "method": "welch", "average": True})

    return FeaturePipeline(cfg, signal=signal, t_cut=t_cut, voi=voi, pruner=pruner)


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
    Two-panel figure: JR observable time series + mean PSD across nodes.

    Parameters
    ----------
    t          : time array in ms, shape (T,)
    ts         : state array, shape (T, n_sv, N) - raw/tavg monitor output
    fs_hz      : sampling frequency in Hz
    title      : figure title
    out_path   : file path to save
    n_plot_nodes : how many nodes to overlay
    t_window_ms  : optional (t_start, t_end) zoom window in ms
    """
    # EEG/LFP proxy for JR: excitatory minus inhibitory PSP.
    y_eeg = ts[:, 1, :] - ts[:, 2, :]       # (T, N), y1 - y2
    y_plot_nodes = y_eeg[:, :n_plot_nodes]  # (T, n_plot_nodes)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3))

    # Left: time series
    if t_window_ms is not None:
        mask = (t >= t_window_ms[0]) & (t <= t_window_ms[1])
        t_plot, y_plot = t[mask], y_plot_nodes[mask]
    else:
        t_plot, y_plot = t, y_plot_nodes
    axes[0].plot(t_plot, y_plot, lw=0.8, alpha=0.7)
    axes[0].set_xlabel("t (ms)")
    axes[0].set_ylabel("y1 - y2 (mV)")
    axes[0].set_title("Time series (y1 - y2, first nodes)")
    axes[0].margins(x=0)

    # Right: mean PSD across all nodes of y1 - y2
    freqs, psd = sp_signal.welch(
        y_eeg.T, fs=fs_hz, nperseg=max(512, ts.shape[0] // 1)
    )
    axes[1].plot(freqs, psd.mean(axis=0), lw=1.2, color="steelblue")
    axes[1].set_xlim(0, 80)
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("PSD")
    axes[1].set_title("Mean PSD across nodes")

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
