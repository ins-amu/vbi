"""Shared helpers for simulator model examples."""

from __future__ import annotations

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from io import StringIO
import logging
from pathlib import Path
import sys
from typing import Iterator

import numpy as np


def ensure_repo_on_path(file: str, parents_up: int = 3) -> Path:
    repo_root = Path(file).resolve().parents[parents_up]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


def complete_graph_weights(n_nodes: int) -> np.ndarray:
    weights = np.ones((n_nodes, n_nodes), dtype=np.float64)
    np.fill_diagonal(weights, 0.0)
    return weights


def constant_tract_lengths(weights: np.ndarray, length: float) -> np.ndarray:
    tract_lengths = np.full_like(weights, float(length), dtype=np.float64)
    tract_lengths[weights == 0.0] = 0.0
    np.fill_diagonal(tract_lengths, 0.0)
    return tract_lengths


def homogeneous_node_params(
    n_nodes: int,
    params: dict[str, float],
    scalar_names: tuple[str, ...] = ("G",),
) -> dict[str, float | np.ndarray]:
    return {
        name: value if name in scalar_names else np.full(n_nodes, value)
        for name, value in params.items()
    }


def make_tvb_connectivity(
    weights: np.ndarray,
    tract_lengths: np.ndarray | None = None,
    speed: float = 1.0,
):
    from tvb.datatypes.connectivity import Connectivity

    n_nodes = weights.shape[0]
    if tract_lengths is None:
        tract_lengths = np.zeros_like(weights)

    conn = Connectivity(
        weights=weights,
        tract_lengths=tract_lengths,
        region_labels=np.array([str(i) for i in range(n_nodes)]),
        centres=np.zeros((n_nodes, 3)),
        speed=np.array([speed]),
    )
    conn.configure()
    return conn


@contextmanager
def quiet_optional_imports() -> Iterator[None]:
    with redirect_stderr(StringIO()):
        yield


@contextmanager
def quiet_tvb() -> Iterator[None]:
    previous_logging_disable = logging.root.manager.disable
    try:
        logging.disable(logging.WARNING)
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            yield
    finally:
        logging.disable(previous_logging_disable)


def comparison_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    diff = candidate - reference
    return {
        "max_abs": float(np.max(np.abs(diff))),
        "rms": float(np.sqrt(np.mean(diff**2))),
    }


def save_state_comparison_plot(
    times: np.ndarray,
    left_data: np.ndarray,
    right_data: np.ndarray,
    out_path: Path,
    variable_names: tuple[str, ...],
    labels: tuple[str, str] = ("VBI simulator", "TVB"),
    title: str = "Simulator trajectory comparison",
    decimate: int = 1,
) -> None:
    import matplotlib.pyplot as plt

    decimate = max(1, decimate)
    t = times[::decimate]
    n_vars = len(variable_names)
    fig, axes = plt.subplots(n_vars, 1, figsize=(12, 3 * n_vars), sharex=True)
    axes_arr = np.atleast_1d(axes)

    for idx, (ax, name) in enumerate(zip(axes_arr, variable_names)):
        ax.plot(t, left_data[::decimate, idx, :], lw=1.0, alpha=0.85)
        ax.plot(t, right_data[::decimate, idx, :], "--", lw=0.9, alpha=0.75)
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.25)
        ax.margins(x=0.01)

    axes_arr[0].set_title(title)
    axes_arr[-1].set_xlabel("time [ms]")
    axes_arr[0].plot([], [], color="black", lw=1.0, label=labels[0])
    axes_arr[0].plot([], [], color="black", ls="--", lw=0.9, label=labels[1])
    axes_arr[0].legend(frameon=False, loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
