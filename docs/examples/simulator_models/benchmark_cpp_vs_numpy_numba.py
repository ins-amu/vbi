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
import numba

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
DEFAULT_DURATION = 2000.0  # ms  (longer for stable timing across n_samples)
DEFAULT_DT       = 0.1
ETA_RANGE        = (-5.5, -3.5)

# Both parallel backends (Numba prange + C++ ThreadPoolExecutor) are pinned
# to the same thread count so the comparison is apples-to-apples.
N_WORKERS = int(os.environ.get("VBI_BENCH_THREADS", os.cpu_count() // 2 or 4))


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

def warmup(spec: SimulationSpec, n_workers: int) -> None:
    print(f"Warming up (Numba JIT + C++ compile, {n_workers} threads each)…", flush=True)
    # Pin Numba's thread pool before the first JIT compile so the compiled
    # code uses exactly n_workers threads for every subsequent prange call.
    numba.set_num_threads(n_workers)
    tiny = SweepSpec(params={"eta": np.array([-4.6, -4.0])})
    Sweeper(spec, tiny, backend="numba").run(500.0)
    cpp_tiny = CppSweeper(spec, tiny, n_workers=n_workers)
    cpp_tiny.run_serial(500.0)    # compiles .so and warms ThreadPoolExecutor
    cpp_tiny.run_parallel(500.0)
    print("  done.\n", flush=True)


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def timed_numba_serial(sweeper, duration: float, n_workers: int) -> float:
    """Run Numba sweep with 1 thread (forces prange to be sequential)."""
    numba.set_num_threads(1)
    try:
        return timed(sweeper.run, duration)
    finally:
        numba.set_num_threads(n_workers)


def run_benchmark(n_nodes: int, duration: float, dt: float,
                  sweep_sizes: list[int], n_workers: int, plot: bool) -> None:
    spec = make_spec(n_nodes, dt)
    warmup(spec, n_workers)

    T = n_workers
    cols = ["n_samples", "np-ser(s)", "nb-ser(s)", f"nb-par×{T}(s)",
            "cpp-ser(s)", f"cpp-par×{T}(s)",
            "nb-ser/np", f"nb-par/np", f"cpp-ser/np", f"cpp-par/np"]
    col_w = [10, 10, 10, 14, 11, 14, 11, 11, 12, 11]
    header = "  ".join(c.rjust(w) for c, w in zip(cols, col_w))
    sep    = "-" * len(header)
    print(header)
    print(sep)

    rows = []
    for n in sweep_sizes:
        sw  = make_sweep(n)
        nb_sweeper = Sweeper(spec, sw, backend="numba")
        cpp        = CppSweeper(spec, sw, n_workers=n_workers)

        t_np  = timed(Sweeper(spec, sw, backend="numpy").run, duration)
        t_nbs = timed_numba_serial(nb_sweeper,                duration, n_workers)
        t_nbp = timed(nb_sweeper.run,                         duration)
        t_cs  = timed(cpp.run_serial,                         duration)
        t_cp  = timed(cpp.run_parallel,                       duration)

        row = [n, t_np, t_nbs, t_nbp, t_cs, t_cp]
        rows.append(row)

        vals = [
            f"{n:>10}",
            f"{t_np:>10.2f}", f"{t_nbs:>10.2f}", f"{t_nbp:>14.2f}",
            f"{t_cs:>11.2f}", f"{t_cp:>14.2f}",
            f"{t_np/t_nbs:>11.1f}x", f"{t_np/t_nbp:>11.1f}x",
            f"{t_np/t_cs:>12.1f}x",  f"{t_np/t_cp:>11.1f}x",
        ]
        print("  ".join(vals))

    print(sep)
    print(f"\nNetwork : {n_nodes} nodes  |  duration={duration} ms  |  dt={dt} ms")
    print(f"Parallel backends pinned to {n_workers} threads "
          f"(override: VBI_BENCH_THREADS=N or --n-workers).")
    print("nb-ser  = Numba prange, 1 thread")
    print(f"nb-par  = Numba prange, {T} threads  (prange)")
    print("cpp-ser = C++ Python loop, 1 thread")
    print(f"cpp-par = C++ ThreadPoolExecutor, {T} threads  (GIL released per call)")

    if plot:
        _plot(rows, sweep_sizes, n_nodes, duration, n_workers)


def _plot(rows, sweep_sizes, n_nodes, duration, n_workers):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plot.")
        return

    ns,  t_np, t_nbs, t_nbp, t_cs, t_cp = zip(*rows)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Wall-clock time
    ax = axes[0]
    ax.plot(ns, t_np,  "o-", label="NumPy serial",            color="tab:blue")
    ax.plot(ns, t_nbs, "s--",label="Numba serial (1 thread)", color="tab:orange")
    ax.plot(ns, t_nbp, "s-", label=f"Numba parallel ×{n_workers}", color="tab:orange",
            markerfacecolor="white")
    ax.plot(ns, t_cs,  "^--",label="C++ serial",              color="tab:green")
    ax.plot(ns, t_cp,  "D-", label=f"C++ parallel ×{n_workers}",   color="tab:green",
            markerfacecolor="white")
    ax.set_xlabel("n_samples")
    ax.set_ylabel("Wall time (s)")
    ax.set_title(f"Sweep time — {n_nodes} nodes, {duration} ms")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Speedup vs NumPy serial
    ax = axes[1]
    ax.plot(ns, [t_np[i]/t_nbs[i] for i in range(len(ns))],
            "s--", label="Numba serial / NumPy",            color="tab:orange")
    ax.plot(ns, [t_np[i]/t_nbp[i] for i in range(len(ns))],
            "s-",  label=f"Numba par×{n_workers} / NumPy", color="tab:orange",
            markerfacecolor="white")
    ax.plot(ns, [t_np[i]/t_cs[i]  for i in range(len(ns))],
            "^--", label="C++ serial / NumPy",              color="tab:green")
    ax.plot(ns, [t_np[i]/t_cp[i]  for i in range(len(ns))],
            "D-",  label=f"C++ par×{n_workers} / NumPy",   color="tab:green",
            markerfacecolor="white")
    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    ax.set_xlabel("n_samples")
    ax.set_ylabel("Speedup vs NumPy serial")
    ax.set_title("Speedup vs NumPy baseline")
    ax.legend(fontsize=9)
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
    parser.add_argument("--sizes",     type=int,   nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--n-workers", type=int,   default=N_WORKERS)
    parser.add_argument("--no-plot",   action="store_true")
    args = parser.parse_args()

    run_benchmark(
        n_nodes=args.n_nodes,
        duration=args.duration,
        dt=args.dt,
        sweep_sizes=args.sizes,
        n_workers=args.n_workers,
        plot=not args.no_plot,
    )
