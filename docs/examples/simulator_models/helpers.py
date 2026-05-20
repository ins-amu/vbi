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
    """Add the repository root inferred from ``file`` to ``sys.path``."""
    repo_root = Path(file).resolve().parents[parents_up]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


def complete_graph_weights(n_nodes: int) -> np.ndarray:
    """Return a dense complete-graph weight matrix with zero diagonal."""
    weights = np.ones((n_nodes, n_nodes), dtype=np.float64)
    np.fill_diagonal(weights, 0.0)
    return weights


def constant_tract_lengths(weights: np.ndarray, length: float) -> np.ndarray:
    """Return tract lengths matching nonzero connections in ``weights``."""
    tract_lengths = np.full_like(weights, float(length), dtype=np.float64)
    tract_lengths[weights == 0.0] = 0.0
    np.fill_diagonal(tract_lengths, 0.0)
    return tract_lengths


def homogeneous_node_params(
    n_nodes: int,
    params: dict[str, float],
    scalar_names: tuple[str, ...] = ("G",),
) -> dict[str, float | np.ndarray]:
    """Expand scalar parameters to per-node arrays, except named globals."""
    return {
        name: value if name in scalar_names else np.full(n_nodes, value)
        for name, value in params.items()
    }


def make_tvb_connectivity(
    weights: np.ndarray,
    tract_lengths: np.ndarray | None = None,
    speed: float = 1.0,
):
    """Build and configure a minimal TVB connectivity object."""
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
    """Suppress stderr noise from optional dependency probes during imports."""
    with redirect_stderr(StringIO()):
        yield


@contextmanager
def quiet_tvb() -> Iterator[None]:
    """Suppress TVB logging and stream output inside the context."""
    previous_logging_disable = logging.root.manager.disable
    try:
        logging.disable(logging.WARNING)
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            yield
    finally:
        logging.disable(previous_logging_disable)


def comparison_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    """Return max absolute and RMS error between two trajectories."""
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
    """Save an overlay plot comparing state variables from two simulators."""
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
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


# ---------------------------------------------------------------------------
# Backend sweep benchmark plotting
# ---------------------------------------------------------------------------

def extrapolate_wall(ns_measured: list[int], times: list[float],
                     ns_extrap: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit  t = slope * n  on the last ≤3 measured CPU points and evaluate at
    ns_extrap.  CPU sweep time is linear in n_samples after JIT warmup.
    """
    ns    = np.array(ns_measured[-3:], dtype=float)
    ts    = np.array(times[-3:],       dtype=float)
    slope = float(np.mean(ts / ns))
    return np.array(ns_extrap, dtype=float), slope * np.array(ns_extrap, dtype=float)


def plot_sweep_benchmark(
    cpu_rows: list[dict],
    cuda_rows: list[dict],
    model_name: str,
    n_nodes: int,
    duration: float,
    dt: float,
    n_workers: int,
    repeats: int,
    out_dir: Path,
) -> None:
    """
    Three-panel benchmark figure.

    Panel 1 — Wall time (log-log):
        Measured CPU lines + dotted extrapolation to the CUDA x-range.
        CUDA measured points.

    Panel 2 — CPU speedup vs NumPy (linear scale).

    Panel 3 — Throughput in samples/s (log-log):
        CPU plateau extrapolated with dotted lines.
        CUDA measured; crossover vs C++ parallel annotated.

    Parameters
    ----------
    cpu_rows  : list of dicts with keys n, t_np, t_nb1, t_nbN, t_cs, t_cp
    cuda_rows : list of dicts with keys n, t_cuda, rate
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plot.")
        return

    cpu_max_n  = max(r["n"] for r in cpu_rows)  if cpu_rows  else 64
    cuda_max_n = max(r["n"] for r in cuda_rows) if cuda_rows else cpu_max_n
    extrap_ns  = sorted({n for n in [128, 256, 512, 1024, 2048, 4096, 8192]
                          if cpu_max_n < n <= cuda_max_n * 2})

    STYLES = [
        # (row_key, label, color, linestyle, marker, markerfacecolor)
        ("t_np",  "NumPy serial",          "tab:blue",   "-",  "o", None),
        ("t_nb1", "Numba 1T",              "tab:orange", "--", "s", None),
        ("t_nbN", f"Numba {n_workers}T",   "tab:orange", "-",  "s", "white"),
        ("t_cs",  "C++ serial",            "tab:green",  "--", "^", None),
        ("t_cp",  f"C++ par {n_workers}T", "tab:green",  "-",  "D", "white"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"Sweep benchmark — {model_name}  |  {n_nodes} nodes  |  "
        f"{duration} ms  |  dt={dt} ms  |  {n_workers} CPU threads  |  "
        f"best-of-{repeats}",
        fontsize=10,
    )

    # ---- Panel 1: wall time (log-log + CPU extrapolation) ----
    ax = axes[0]
    if cpu_rows:
        ns = [r["n"] for r in cpu_rows]
        for key, label, color, ls, marker, mfc in STYLES:
            ts = [r[key] for r in cpu_rows]
            ax.plot(ns, ts, marker=marker, linestyle=ls, label=label,
                    color=color, markerfacecolor=mfc or color, zorder=3)
            if extrap_ns:
                ex_ns, ex_ts = extrapolate_wall(ns, ts, extrap_ns)
                ax.plot(ex_ns, ex_ts, linestyle=":", color=color,
                        alpha=0.45, lw=1.4)
    if cuda_rows:
        gn = [r["n"] for r in cuda_rows]
        ax.plot(gn, [r["t_cuda"] for r in cuda_rows], "P-",
                label="CUDA (float32)", color="tab:red", lw=2, zorder=4)
    if extrap_ns:
        ax.text(0.97, 0.03, "··· = CPU extrapolation (linear fit)",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=7, color="gray")
    ax.set_xlabel("n_samples");  ax.set_ylabel("Wall time (s)")
    ax.set_title("Wall-clock time");  ax.set_xscale("log");  ax.set_yscale("log")
    ax.legend(fontsize=7, loc="best");  ax.grid(True, alpha=0.3, which="both")

    # ---- Panel 2: CPU speedup vs NumPy (linear) ----
    ax = axes[1]
    if cpu_rows:
        ns = [r["n"] for r in cpu_rows]
        for key, label, color, ls, marker, mfc in STYLES[1:]:  # skip NumPy
            su = [r["t_np"] / r[key] for r in cpu_rows]
            ax.plot(ns, su, marker=marker, linestyle=ls,
                    label=f"{label} / NumPy",
                    color=color, markerfacecolor=mfc or color)
    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    ax.set_xlabel("n_samples");  ax.set_ylabel("Speedup vs NumPy serial")
    ax.set_title("CPU speedup")
    ax.legend(fontsize=8, loc="best");  ax.grid(True, alpha=0.3)

    # ---- Panel 3: throughput (log-log + CPU plateau + CUDA crossover) ----
    ax = axes[2]
    if cpu_rows:
        ns = [r["n"] for r in cpu_rows]
        for key, label, color, ls, marker, mfc in STYLES:
            rates = [r["n"] / r[key] for r in cpu_rows]
            ax.plot(ns, rates, marker=marker, linestyle=ls, label=label,
                    color=color, markerfacecolor=mfc or color, zorder=3)
            if extrap_ns:
                plateau = float(np.mean(rates[-3:]))
                ax.plot(extrap_ns, [plateau] * len(extrap_ns),
                        linestyle=":", color=color, alpha=0.45, lw=1.4)
    if cuda_rows:
        gn    = [r["n"] for r in cuda_rows]
        rates = [r["rate"] for r in cuda_rows]
        ax.plot(gn, rates, "P-", label="CUDA (float32)",
                color="tab:red", lw=2, markersize=8, zorder=4)
        if cpu_rows:
            cp_plateau = float(np.mean(
                [r["n"] / r["t_cp"] for r in cpu_rows[-3:]]
            ))
            for r in sorted(cuda_rows, key=lambda x: x["n"]):
                if r["rate"] >= cp_plateau:
                    ax.axvline(r["n"], color="tab:red", ls=":", alpha=0.6, lw=1.2)
                    ax.text(r["n"] * 1.05, cp_plateau * 0.55,
                            f'CUDA > C++\n@ n≈{r["n"]}',
                            color="tab:red", fontsize=7)
                    break
    if extrap_ns:
        ax.text(0.97, 0.03, "··· = CPU plateau (avg last 3 pts)",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=7, color="gray")
    ax.set_xlabel("n_samples");  ax.set_ylabel("Samples / second")
    ax.set_title("Throughput");  ax.set_xscale("log");  ax.set_yscale("log")
    ax.legend(fontsize=7, loc="lower right", frameon=False);  ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"benchmark_all_backends_{model_name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {out}")
    # plt.show()
