"""
Shared visualisation helpers for vbi.inference demos.

All functions return a matplotlib Figure and optionally save to disk.
Designed to be lightweight: no sbi dependency, pure matplotlib/numpy.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# pairplot - posterior samples triangle plot
# ---------------------------------------------------------------------------

def pairplot(
    samples: np.ndarray,
    labels: list[str] | None = None,
    points: np.ndarray | None = None,
    points_color: str = "red",
    points_label: str = "true θ",
    title: str = "",
    figsize: tuple = None,
    out_path: str | Path | None = None,
) -> plt.Figure:
    """
    Triangle plot of posterior samples (marginals on diagonal, joints off-diagonal).

    Parameters
    ----------
    samples : (N, d)  posterior samples
    labels  : list[str]  parameter names, default ["θ0", "θ1", ...]
    points  : (d,) or (k, d)  reference points to overlay (e.g. true theta)
    title   : figure title
    out_path: save path (PNG)

    Returns
    -------
    matplotlib Figure
    """
    samples = np.asarray(samples)
    N, d    = samples.shape
    if labels is None:
        labels = [f"θ{i}" for i in range(d)]
    if figsize is None:
        figsize = (2.5 * d, 2.5 * d)

    fig, axes = plt.subplots(d, d, figsize=figsize)
    if d == 1:
        axes = np.array([[axes]])

    for row in range(d):
        for col in range(d):
            ax = axes[row, col]
            if col > row:
                ax.set_visible(False)
                continue

            if col == row:
                # Diagonal: 1-D marginal
                ax.hist(samples[:, row], bins=40, density=True,
                        color="steelblue", alpha=0.7, edgecolor="none")
                if points is not None:
                    pts = np.atleast_2d(points)
                    for pt in pts:
                        ax.axvline(pt[row], color=points_color, lw=1.5,
                                   label=points_label if (row == 0 and col == 0) else "")
                ax.set_xlabel(labels[row], fontsize=9)
                ax.set_yticks([])
            else:
                # Off-diagonal: 2-D joint density (scatter + contour)
                ax.scatter(samples[:, col], samples[:, row],
                           s=2, alpha=0.3, color="steelblue", rasterized=True)
                if points is not None:
                    pts = np.atleast_2d(points)
                    for pt in pts:
                        ax.scatter(pt[col], pt[row], color=points_color,
                                   s=60, zorder=5, marker="*",
                                   label=points_label if (row == 1 and col == 0) else "")
                ax.set_xlabel(labels[col], fontsize=9)
                ax.set_ylabel(labels[row], fontsize=9)

    if title:
        fig.suptitle(title, y=1.02, fontsize=11)
    # One legend entry for reference point
    if points is not None:
        handles, lbls = axes[min(1, d-1), 0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, lbls, loc="upper right", fontsize=8)

    fig.tight_layout()
    _maybe_save(fig, out_path)
    return fig


# ---------------------------------------------------------------------------
# posterior_1d - 1-D posterior with analytical overlay
# ---------------------------------------------------------------------------

def posterior_1d(
    samples: np.ndarray,
    true_mean: float | None = None,
    true_std:  float | None = None,
    x_obs_val: float | None = None,
    label_est: str = "estimated posterior",
    title: str = "",
    xlim: tuple[float, float] | None = None,
    out_path: str | Path | None = None,
) -> plt.Figure:
    """
    1-D histogram of posterior samples with optional analytical Gaussian overlay.
    """
    samples = np.asarray(samples).ravel()
    fig, ax = plt.subplots(figsize=(6, 3.5))

    ax.hist(samples, bins=50, density=True, alpha=0.6,
            color="steelblue", label=label_est, range=xlim)

    if true_mean is not None and true_std is not None:
        if xlim is None:
            xg = np.linspace(samples.min() - true_std, samples.max() + true_std, 300)
        else:
            xg = np.linspace(xlim[0], xlim[1], 300)
        from scipy.stats import norm
        ax.plot(xg, norm.pdf(xg, true_mean, true_std),
                "k--", lw=2, label="analytical posterior")
        ax.axvline(true_mean, color="k", lw=1, ls=":")

    if x_obs_val is not None:
        ax.axvline(x_obs_val, color="tomato", lw=1.5, ls="--", label=f"x_obs={x_obs_val:.2f}")

    ax.set_xlabel("θ");  ax.set_ylabel("density")
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.legend(fontsize=8)
    if title: ax.set_title(title, fontsize=10)
    fig.tight_layout()
    _maybe_save(fig, out_path)
    return fig


# ---------------------------------------------------------------------------
# loss_plot - training and validation loss curves
# ---------------------------------------------------------------------------

def loss_plot(
    loss_history: list[float],
    val_loss_history: list[float] | None = None,
    title: str = "Training loss",
    out_path: str | Path | None = None,
) -> plt.Figure:
    """Plot training (and optional validation) loss over epochs."""
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(loss_history, label="train loss", color="steelblue")
    if val_loss_history:
        ax.plot(val_loss_history, label="val loss", color="tomato", ls="--")
    ax.set_xlabel("epoch"); ax.set_ylabel("NLL loss")
    ax.legend(fontsize=8)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    _maybe_save(fig, out_path)
    return fig


# ---------------------------------------------------------------------------
# coverage_plot - 90% CI coverage across multiple x_obs values
# ---------------------------------------------------------------------------

def coverage_plot(
    x_test: np.ndarray,
    true_means: np.ndarray,
    est_means: np.ndarray,
    est_lo: np.ndarray,
    est_hi: np.ndarray,
    xlabel: str = "x_obs",
    title: str = "Posterior coverage",
    out_path: str | Path | None = None,
) -> plt.Figure:
    """
    Shows estimated posterior mean ± 90% CI vs true posterior mean.
    """
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(x_test, true_means, "k--", lw=1.5, label="analytical mean")
    ax.plot(x_test, est_means,  "o-",  color="steelblue", label="estimated mean")
    ax.fill_between(x_test, est_lo, est_hi, alpha=0.25,
                    color="steelblue", label="90% CI")
    ax.set_xlabel(xlabel); ax.set_ylabel("θ")
    ax.legend(fontsize=8); ax.set_title(title, fontsize=10)
    fig.tight_layout()
    _maybe_save(fig, out_path)
    return fig


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _maybe_save(fig: plt.Figure, out_path) -> None:
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        print(f"  saved: {out_path}")
    plt.close(fig)
