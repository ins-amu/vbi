"""
Backend sweep benchmark: NumPy · Numba (1T / NT) · C++ (serial / parallel) · CUDA.

Strategy
--------
* CPU backends (NumPy, Numba, C++) use small-to-medium sweep sizes because
  their time scales linearly with n_samples.  Running them at thousands of
  samples would make the benchmark very slow.

* CUDA is only launched for ≥ 64 samples where GPU scheduling overhead is
  amortised; it shines at ≥ 1024 samples.

* All CPU backends are timed at the same sweep sizes so speedups are
  directly comparable on the same x-axis.  The CUDA curve continues to
  larger n_samples on a separate panel (or same log-scale axis).

Model choice
------------
SupHopf (2 SVs, 2 params) — fast equations, good for benchmarking.
Can be overridden to Generic2dOscillator or MPR via CLI.

Usage
-----
    python benchmark_all_backends.py
    python benchmark_all_backends.py --model generic_2d_oscillator --n-nodes 40
    python benchmark_all_backends.py --no-cuda --no-plot
    python benchmark_all_backends.py --n-workers 8 --cpu-sizes 4 8 16 32 64 128
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import warnings
import numpy as np

# Suppress Numba CUDA low-occupancy warnings for small batches during warmup
warnings.filterwarnings(
    "ignore",
    message="Grid size.*will likely result in GPU under-utilization",
)

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from helpers import ensure_repo_on_path, complete_graph_weights
ensure_repo_on_path(__file__)

import numba

from vbi.simulator import Sweeper
from vbi.simulator.spec import (
    SimulationSpec, IntegratorSpec, CouplingSpec, MonitorSpec,
)
from vbi.simulator.spec.sweep import SweepSpec
from vbi.simulator.backend.cpp.sweeper import CppSweeper

from vbi.simulator.models.sup_hopf              import sup_hopf
from vbi.simulator.models.generic_2d_oscillator import generic_2d_oscillator
from vbi.simulator.models.kuramoto              import kuramoto
from vbi.simulator.models.mpr                   import mpr

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODELS = {
    "sup_hopf":              (sup_hopf,              "a",   (-0.5, 0.5), 0.01, 0.1),
    "generic_2d_oscillator": (generic_2d_oscillator, "I",   (-1.0, 1.0), 0.01, 0.05),
    "kuramoto":              (kuramoto,              "omega",(0.5,  2.0), 0.01, 0.05),
    "mpr":                   (mpr,                  "eta",  (-5.5,-4.0), 0.01, 0.1),
}

DEFAULT_MODEL    = "sup_hopf"
DEFAULT_N_NODES  = 20
DEFAULT_DURATION = 500.0   # ms  — kept short so CPU runs complete in seconds
DEFAULT_DT       = 0.01

# CPU sweep sizes — linear cost, so keep small; overlap with CUDA at 32–128
DEFAULT_CPU_SIZES  = [4, 8, 16, 32, 64, 128]
# CUDA sweep sizes — starts where CPU overlap begins; GPU shines at >= 512
DEFAULT_CUDA_SIZES = [32, 64, 128, 256, 512, 1024, 2048, 4096]

N_WORKERS = int(os.environ.get("VBI_BENCH_THREADS", str(max(2, (os.cpu_count() or 4) // 2))))
N_REPEATS = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _weights(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    W = complete_graph_weights(n).astype(np.float64)
    W /= W.sum(axis=1, keepdims=True).clip(1e-8)
    return W


def _make_spec(model_name: str, n_nodes: int, dt: float) -> SimulationSpec:
    model, _, _, _, coup_a = MODELS[model_name]
    return SimulationSpec(
        model=model,
        integrator=IntegratorSpec(method="heun", dt=dt, stochastic=False),
        coupling=CouplingSpec("linear", a=coup_a),
        monitors=(MonitorSpec("raw"),),
        weights=_weights(n_nodes),
    )


def _make_sweep(model_name: str, n_samples: int) -> SweepSpec:
    _, param, (lo, hi), _, _ = MODELS[model_name]
    return SweepSpec(params={param: np.linspace(lo, hi, n_samples)})


def _best_of(fn, repeats: int) -> float:
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


def _try(fn, label: str) -> float | None:
    try:
        return fn()
    except Exception as e:
        print(f"    [{label}] skipped: {type(e).__name__}: {e}")
        return None


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------

def warmup(spec: SimulationSpec, model_name: str, n_workers: int) -> None:
    print(f"Warming up (Numba JIT + C++ compile, {n_workers} threads)…", flush=True)
    numba.set_num_threads(n_workers)
    tiny = _make_sweep(model_name, 2)
    Sweeper(spec, tiny, backend="numba").run(100.0)
    cpp = CppSweeper(spec, tiny, n_workers=n_workers)
    cpp.run_serial(100.0)
    cpp.run_parallel(100.0, n_workers=n_workers)
    print("  Numba + C++ warmed up.")

    try:
        from vbi.simulator.backend.numba_cuda import CUDA_AVAILABLE
        if CUDA_AVAILABLE:
            print("  Warming up CUDA (JIT compile — may take 30-120 s)…",
                  end="", flush=True)
            Sweeper(spec, tiny, backend="cuda").run(100.0)
            print(" done.")
    except Exception as e:
        print(f"  CUDA warmup skipped: {e}")

    print()


# ---------------------------------------------------------------------------
# CPU benchmark
# ---------------------------------------------------------------------------

def benchmark_cpu(spec: SimulationSpec, model_name: str,
                  cpu_sizes: list[int], duration: float,
                  n_workers: int, repeats: int) -> list[dict]:
    print(f"{'n_samples':>10}  {'NumPy':>10}  {'Nb-1T':>10}  {'Nb-NT':>10}  "
          f"{'Cpp-ser':>10}  {'Cpp-par':>10}  "
          f"{'×Nb-1T':>8}  {'×Nb-NT':>8}  {'×Cpp-ser':>9}  {'×Cpp-par':>9}")
    print("-" * 105)

    rows = []
    for n in cpu_sizes:
        sw  = _make_sweep(model_name, n)
        cpp = CppSweeper(spec, sw, n_workers=n_workers)
        nb  = Sweeper(spec, sw, backend="numba")

        # NumPy (always serial)
        t_np = _best_of(lambda: Sweeper(spec, sw, backend="numpy").run(duration), repeats)

        # Numba single-thread
        numba.set_num_threads(1)
        t_nb1 = _best_of(lambda: nb.run(duration), repeats)

        # Numba n_workers threads
        numba.set_num_threads(n_workers)
        t_nbN = _best_of(lambda: nb.run(duration), repeats)

        # C++ serial
        t_cs = _best_of(lambda: cpp.run_serial(duration), repeats)

        # C++ parallel
        t_cp = _best_of(lambda: cpp.run_parallel(duration, n_workers=n_workers), repeats)

        def sp(ref, t): return ref / t if t and t > 0 else float("nan")

        row = dict(n=n, t_np=t_np, t_nb1=t_nb1, t_nbN=t_nbN, t_cs=t_cs, t_cp=t_cp)
        rows.append(row)

        print(
            f"{n:>10}  {t_np:>10.3f}  {t_nb1:>10.3f}  {t_nbN:>10.3f}  "
            f"{t_cs:>10.3f}  {t_cp:>10.3f}  "
            f"{sp(t_np,t_nb1):>7.1f}x  {sp(t_np,t_nbN):>7.1f}x  "
            f"{sp(t_np,t_cs):>8.1f}x  {sp(t_np,t_cp):>8.1f}x"
        )

    numba.set_num_threads(n_workers)  # restore
    print("-" * 105)
    print(f"  Times in seconds.  Nb-1T=Numba 1 thread, Nb-NT=Numba {n_workers} threads,")
    print(f"  Cpp-ser=C++ serial, Cpp-par=C++ ThreadPoolExecutor {n_workers} threads.")
    return rows


# ---------------------------------------------------------------------------
# CUDA benchmark
# ---------------------------------------------------------------------------

def benchmark_cuda(spec: SimulationSpec, model_name: str,
                   cuda_sizes: list[int], duration: float,
                   repeats: int) -> list[dict]:
    try:
        from vbi.simulator.backend.numba_cuda import CUDA_AVAILABLE
        if not CUDA_AVAILABLE:
            print("  CUDA not available — skipping GPU benchmark.")
            return []
    except ImportError:
        print("  numba[cuda] not installed — skipping GPU benchmark.")
        return []

    print(f"\n{'n_samples':>10}  {'CUDA(s)':>10}  {'sam/s':>10}")
    print("-" * 35)

    rows = []
    for n in cuda_sizes:
        sw = _make_sweep(model_name, n)

        def _run(): return Sweeper(spec, sw, backend="cuda").run(duration)

        t = _best_of(_run, repeats)
        rate = n / t
        row = dict(n=n, t_cuda=t, rate=rate)
        rows.append(row)
        print(f"{n:>10}  {t:>10.3f}  {rate:>10.0f}")

    print("-" * 35)
    print("  sam/s = parameter sets per second.")
    return rows


# ---------------------------------------------------------------------------
# Combined throughput table (CPU + CUDA at matching sizes)
# ---------------------------------------------------------------------------

def throughput_comparison(cpu_rows: list[dict], cuda_rows: list[dict],
                          n_workers: int) -> None:
    if not cuda_rows:
        return

    print("\n--- Throughput comparison at matched sweep sizes (samples/second) ---")
    cuda_map = {r["n"]: r["rate"] for r in cuda_rows}
    cpu_map  = {r["n"]: r for r in cpu_rows}

    common = sorted(set(cuda_map) & set(cpu_map))
    if not common:
        print("  (no overlapping sweep sizes between CPU and CUDA runs)")
        return

    print(f"{'n_samples':>10}  {'NumPy':>10}  {'Nb-NT':>10}  "
          f"{'Cpp-par':>10}  {'CUDA':>10}  {'CUDA/Nb':>10}  {'CUDA/Cpp':>10}")
    print("-" * 75)
    for n in common:
        r = cpu_map[n]
        def rate(t): return n / t if t and t > 0 else float("nan")
        rn  = rate(r["t_np"])
        rnb = rate(r["t_nbN"])
        rc  = rate(r["t_cp"])
        rg  = cuda_map[n]
        print(
            f"{n:>10}  {rn:>10.0f}  {rnb:>10.0f}  {rc:>10.0f}  "
            f"{rg:>10.0f}  {rg/rnb:>9.1f}x  {rg/rc:>9.1f}x"
        )
    print("-" * 75)
    print(f"  CUDA/Nb = CUDA speedup over Numba {n_workers}T  |  CUDA/Cpp = over C++ parallel")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(cpu_rows: list[dict], cuda_rows: list[dict],
                 model_name: str, n_nodes: int, duration: float,
                 n_workers: int, out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plot.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"Sweep benchmark — model: {model_name}  |  {n_nodes} nodes  |  "
        f"{duration} ms  |  CPU {n_workers} threads",
        fontsize=11,
    )

    # ---- Panel 1: wall-clock time (CPU backends) ----
    ax = axes[0]
    if cpu_rows:
        ns = [r["n"] for r in cpu_rows]
        ax.plot(ns, [r["t_np"]  for r in cpu_rows], "o-",  label="NumPy serial",
                color="tab:blue")
        ax.plot(ns, [r["t_nb1"] for r in cpu_rows], "s--", label="Numba 1 thread",
                color="tab:orange")
        ax.plot(ns, [r["t_nbN"] for r in cpu_rows], "s-",  label=f"Numba {n_workers}T",
                color="tab:orange", markerfacecolor="white")
        ax.plot(ns, [r["t_cs"]  for r in cpu_rows], "^--", label="C++ serial",
                color="tab:green")
        ax.plot(ns, [r["t_cp"]  for r in cpu_rows], "D-",  label=f"C++ par {n_workers}T",
                color="tab:green", markerfacecolor="white")
    if cuda_rows:
        gn = [r["n"] for r in cuda_rows]
        ax.plot(gn, [r["t_cuda"] for r in cuda_rows], "P-", label="CUDA",
                color="tab:red", lw=2)
    ax.set_xlabel("n_samples")
    ax.set_ylabel("Wall time (s)")
    ax.set_title("Wall-clock time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- Panel 2: speedup vs NumPy serial (CPU only) ----
    ax = axes[1]
    if cpu_rows:
        ns = [r["n"] for r in cpu_rows]
        def sp(key): return [r["t_np"] / r[key] for r in cpu_rows]
        ax.plot(ns, sp("t_nb1"), "s--", label="Numba 1T / NumPy", color="tab:orange")
        ax.plot(ns, sp("t_nbN"), "s-",  label=f"Numba {n_workers}T / NumPy",
                color="tab:orange", markerfacecolor="white")
        ax.plot(ns, sp("t_cs"),  "^--", label="C++ serial / NumPy", color="tab:green")
        ax.plot(ns, sp("t_cp"),  "D-",  label=f"C++ par {n_workers}T / NumPy",
                color="tab:green", markerfacecolor="white")
    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    ax.set_xlabel("n_samples")
    ax.set_ylabel("Speedup vs NumPy serial")
    ax.set_title("CPU speedup")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- Panel 3: throughput (samples/s) for all backends ----
    ax = axes[2]
    if cpu_rows:
        ns = [r["n"] for r in cpu_rows]
        ax.plot(ns, [r["n"]/r["t_np"]  for r in cpu_rows], "o-",
                label="NumPy serial", color="tab:blue")
        ax.plot(ns, [r["n"]/r["t_nb1"] for r in cpu_rows], "s--",
                label="Numba 1T", color="tab:orange")
        ax.plot(ns, [r["n"]/r["t_nbN"] for r in cpu_rows], "s-",
                label=f"Numba {n_workers}T", color="tab:orange", markerfacecolor="white")
        ax.plot(ns, [r["n"]/r["t_cs"]  for r in cpu_rows], "^--",
                label="C++ serial", color="tab:green")
        ax.plot(ns, [r["n"]/r["t_cp"]  for r in cpu_rows], "D-",
                label=f"C++ par {n_workers}T", color="tab:green", markerfacecolor="white")
    if cuda_rows:
        gn = [r["n"] for r in cuda_rows]
        ax.plot(gn, [r["rate"] for r in cuda_rows], "P-",
                label="CUDA", color="tab:red", lw=2)
    ax.set_xlabel("n_samples")
    ax.set_ylabel("Samples / second")
    ax.set_title("Throughput")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"benchmark_all_backends_{model_name}.png"
    fig.savefig(out, dpi=150)
    print(f"\nPlot saved → {out}")
    plt.show()


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------

def save_json(cpu_rows: list[dict], cuda_rows: list[dict],
              meta: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"meta": meta, "cpu": cpu_rows, "cuda": cuda_rows}
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"JSON saved   → {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VBI all-backend sweep benchmark")
    p.add_argument("--model",       default=DEFAULT_MODEL,
                   choices=list(MODELS.keys()))
    p.add_argument("--n-nodes",     type=int,   default=DEFAULT_N_NODES)
    p.add_argument("--duration",    type=float, default=DEFAULT_DURATION)
    p.add_argument("--dt",          type=float, default=DEFAULT_DT)
    p.add_argument("--n-workers",   type=int,   default=N_WORKERS,
                   help="threads for Numba prange and C++ ThreadPoolExecutor")
    p.add_argument("--repeats",     type=int,   default=N_REPEATS,
                   help="timing repetitions per cell (min is reported)")
    p.add_argument("--cpu-sizes",   type=int,   nargs="+", default=DEFAULT_CPU_SIZES)
    p.add_argument("--cuda-sizes",  type=int,   nargs="+", default=DEFAULT_CUDA_SIZES)
    p.add_argument("--no-cuda",     action="store_true")
    p.add_argument("--no-numpy",    action="store_true",
                   help="skip NumPy (very slow at large n_samples)")
    p.add_argument("--no-plot",     action="store_true")
    p.add_argument("--output-dir",  type=Path,
                   default=Path(__file__).parent / "outputs")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 70)
    print("  VBI all-backend parameter sweep benchmark")
    print("=" * 70)
    print(f"  Model     : {args.model}")
    print(f"  n_nodes   : {args.n_nodes}")
    print(f"  duration  : {args.duration} ms  |  dt={args.dt} ms")
    print(f"  CPU sizes : {args.cpu_sizes}")
    print(f"  CUDA sizes: {args.cuda_sizes if not args.no_cuda else '(disabled)'}")
    print(f"  n_workers : {args.n_workers}")
    print(f"  repeats   : {args.repeats}  (best-of)")
    print()

    spec = _make_spec(args.model, args.n_nodes, args.dt)
    warmup(spec, args.model, args.n_workers)

    # ---- CPU benchmark ----
    print(f"\n{'='*70}")
    print("  CPU backends")
    print(f"{'='*70}")
    cpu_rows = benchmark_cpu(
        spec, args.model, args.cpu_sizes,
        args.duration, args.n_workers, args.repeats,
    )

    # ---- CUDA benchmark ----
    cuda_rows: list[dict] = []
    if not args.no_cuda:
        print(f"\n{'='*70}")
        print("  CUDA backend")
        print(f"{'='*70}")
        cuda_rows = benchmark_cuda(
            spec, args.model, args.cuda_sizes,
            args.duration, args.repeats,
        )

    # ---- Summary ----
    throughput_comparison(cpu_rows, cuda_rows, args.n_workers)

    # ---- Save ----
    meta = dict(
        model=args.model, n_nodes=args.n_nodes,
        duration=args.duration, dt=args.dt,
        n_workers=args.n_workers, repeats=args.repeats,
    )
    save_json(
        cpu_rows, cuda_rows, meta,
        args.output_dir / f"benchmark_all_backends_{args.model}.json",
    )

    if not args.no_plot:
        plot_results(
            cpu_rows, cuda_rows,
            args.model, args.n_nodes, args.duration,
            args.n_workers, args.output_dir,
        )


if __name__ == "__main__":
    main()
