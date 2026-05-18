"""
Benchmark: NumPy vs Numba vs C++ (serial and parallel) for parameter sweeps.

NumPy runs sequentially in Python.
Numba uses @njit(parallel=True) / prange — true thread parallelism within one process.
C++ serial   — Python loop calling pybind11 run_simulation() per point.
C++ parallel — ThreadPoolExecutor; each thread calls pybind11 (GIL released in C++).

Run from this directory:
    python benchmark_cpp_vs_numpy_numba.py
    python benchmark_cpp_vs_numpy_numba.py --n-nodes 80 --duration 2000
    python benchmark_cpp_vs_numpy_numba.py --no-plot
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
import sys

sys.dont_write_bytecode = True

import numpy as np

from helpers import ensure_repo_on_path, complete_graph_weights

ensure_repo_on_path(__file__)

from vbi.simulator.api import Sweeper
from vbi.simulator.models.mpr import mpr
from vbi.simulator.spec import (
    CouplingSpec,
    IntegratorSpec,
    MonitorSpec,
    SimulationSpec,
)
from vbi.simulator.spec.sweep import SweepSpec
from vbi.simulator.backend.cpp.sweeper import CppSweeper


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_N_NODES  = 40
DEFAULT_DURATION = 500.0   # ms  (short enough for NumPy to finish)
DEFAULT_DT       = 0.1
ETA_RANGE        = (-5.5, -3.5)
N_CPU            = os.cpu_count() // 2 or 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_spec(n_nodes: int, dt: float) -> SimulationSpec:
    W = complete_graph_weights(n_nodes)
    W /= W.sum(axis=1, keepdims=True).clip(1e-8)
    return SimulationSpec(
        model=mpr,
        integrator=IntegratorSpec(method="heun", dt=dt, stochastic=False),
        coupling=CouplingSpec(kind="linear", a=0.1, b=0.0),
        monitors=(MonitorSpec(kind="raw"),),
        weights=W,
    )


def make_sweep(n_samples: int) -> SweepSpec:
    return SweepSpec(params={"eta": np.linspace(*ETA_RANGE, n_samples)})


def timed(fn, *args, **kwargs) -> float:
    t0 = time.perf_counter()
    fn(*args, **kwargs)
    return time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Warmup (compile / JIT)
# ---------------------------------------------------------------------------

def warmup(spec: SimulationSpec) -> None:
    print("Warming up (Numba JIT + C++ compile)…", flush=True)
    tiny = SweepSpec(params={"eta": np.array([-4.6, -4.0])})
    Sweeper(spec, tiny, backend="numba").run(100.0)
    CppSweeper(spec, tiny).run_serial(100.0)
    print("  done.\n", flush=True)


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_benchmark(n_nodes: int, duration: float, dt: float,
                  sweep_sizes: list[int], plot: bool) -> None:
    spec = make_spec(n_nodes, dt)
    warmup(spec)

    cols = ["n_samples", "numpy(s)", "numba(s)", "cpp-ser(s)", f"cpp-par×{N_CPU}(s)",
            "nb/np", f"cpp-par/np"]
    col_w = [10, 10, 10, 12, 16, 8, 12]
    header = "  ".join(c.rjust(w) for c, w in zip(cols, col_w))
    sep    = "-" * len(header)
    print(header)
    print(sep)

    rows = []
    for n in sweep_sizes:
        sw = make_sweep(n)

        t_np  = timed(Sweeper(spec, sw, backend="numpy").run,  duration)
        t_nb  = timed(Sweeper(spec, sw, backend="numba").run,  duration)

        cpp = CppSweeper(spec, sw, n_workers=N_CPU)
        t_cs  = timed(cpp.run_serial,   duration)
        t_cp  = timed(cpp.run_parallel, duration)

        row = [n, t_np, t_nb, t_cs, t_cp, t_np / t_nb, t_np / t_cp]
        rows.append(row)

        vals = [f"{n:>10}",
                f"{t_np:>10.2f}", f"{t_nb:>10.2f}",
                f"{t_cs:>12.2f}", f"{t_cp:>16.2f}",
                f"{t_np/t_nb:>8.1f}x", f"{t_np/t_cp:>12.1f}x"]
        print("  ".join(vals))

    print(sep)
    print(f"\nNetwork : {n_nodes} nodes  |  duration={duration} ms  |  dt={dt} ms")
    print(f"Parallel C++ uses {N_CPU} threads (os.cpu_count()).")

    if plot:
        _plot(rows, sweep_sizes, n_nodes, duration)


def _plot(rows, sweep_sizes, n_nodes, duration):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plot.")
        return

    ns   = [r[0] for r in rows]
    t_np = [r[1] for r in rows]
    t_nb = [r[2] for r in rows]
    t_cs = [r[3] for r in rows]
    t_cp = [r[4] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Wall-clock time
    ax = axes[0]
    ax.plot(ns, t_np, "o-", label="NumPy (serial)")
    ax.plot(ns, t_nb, "s-", label="Numba (parallel)")
    ax.plot(ns, t_cs, "^-", label="C++ serial")
    ax.plot(ns, t_cp, "D-", label=f"C++ parallel ×{N_CPU}")
    ax.set_xlabel("n_samples")
    ax.set_ylabel("Wall time (s)")
    ax.set_title(f"Sweep time — {n_nodes} nodes, {duration} ms")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Speedup vs NumPy
    ax = axes[1]
    ax.plot(ns, [t_np[i] / t_nb[i] for i in range(len(ns))],
            "s-", label="Numba / NumPy")
    ax.plot(ns, [t_np[i] / t_cs[i] for i in range(len(ns))],
            "^-", label="C++ serial / NumPy")
    ax.plot(ns, [t_np[i] / t_cp[i] for i in range(len(ns))],
            "D-", label=f"C++ par×{N_CPU} / NumPy")
    ax.axhline(1.0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("n_samples")
    ax.set_ylabel("Speedup")
    ax.set_title("Speedup vs NumPy baseline")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = Path(__file__).parent / "outputs" / "benchmark_cpp.png"
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, dpi=150)
    print(f"Plot saved → {out}")
    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VBI C++ backend benchmark")
    parser.add_argument("--n-nodes",  type=int,   default=DEFAULT_N_NODES)
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION)
    parser.add_argument("--dt",       type=float, default=DEFAULT_DT)
    parser.add_argument("--sizes",    type=int,   nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--no-plot",  action="store_true")
    args = parser.parse_args()

    run_benchmark(
        n_nodes=args.n_nodes,
        duration=args.duration,
        dt=args.dt,
        sweep_sizes=args.sizes,
        plot=not args.no_plot,
    )
