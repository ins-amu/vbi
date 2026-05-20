"""
Backend sweep benchmark: NumPy · Numba (1T / NT) · C++ (serial / parallel) · CUDA · JAX.

Strategy
--------
* CPU backends (NumPy, Numba, C++) use small-to-medium sweep sizes because
  their time scales linearly with n_samples.  Running them at thousands of
  samples would make the benchmark very slow.

* CUDA and JAX are only launched for ≥ 32 samples where GPU scheduling
  overhead is amortised; they shine at ≥ 256 samples.

* JAX uses jax.vmap to batch all samples into a single JIT-compiled XLA
  program.  Performance improves with batch size (better XLA utilisation).
  On GPU, JAX competes with CUDA; on CPU, it is comparable to Numba N-thread.

* All CPU backends are timed at the same sweep sizes so speedups are
  directly comparable on the same x-axis.  The GPU curves continue to
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
    python benchmark_all_backends.py --no-jax           # skip JAX
    python benchmark_all_backends.py --jax-sizes 16 32 64 128 256 512 1024
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
from helpers import (
    ensure_repo_on_path,
    complete_graph_weights,
    plot_sweep_benchmark,
)
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

# ---- Last validated configuration ----
# SupHopf, 20 nodes, dt=0.1, duration=500ms  →  runs in ~3 min (repeats=2)
# For brain-scale: --model mpr --n-nodes 80 --cuda-sizes 64 128 512 2048 4096 --repeats 1
DEFAULT_MODEL    = "sup_hopf"
DEFAULT_N_NODES  = 20
DEFAULT_DURATION = 500.0   # ms  — 5000 steps per simulation (dt=0.1)
DEFAULT_DT       = 0.1     # dt=0.1 ms → fast on all backends

# CPU: kept ≤ 64 so NumPy finishes in seconds (cost is linear in n_samples)
DEFAULT_CPU_SIZES  = [4, 8, 16, 32, 64]
# CUDA: starts at 32 (overlaps CPU), extends to 2048 where GPU starts winning
DEFAULT_CUDA_SIZES = [32, 64, 128, 256, 512, 1024, 2048]
# JAX: vmap batches compile per n_samples; warmup is included in first call
DEFAULT_JAX_SIZES  = [16, 32, 64, 128, 256, 512, 1024]

N_WORKERS = int(os.environ.get("VBI_BENCH_THREADS", str(max(2, (os.cpu_count() or 4) // 2))))
N_REPEATS = 2


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
    # Use tavg with period = 10*dt so n_record = duration/(10*dt) — manageable on GPU.
    # For CPU backends this makes no practical difference (they store on host RAM).
    tavg_period = round(10 * dt, 6)
    return SimulationSpec(
        model=model,
        integrator=IntegratorSpec(method="heun", dt=dt, stochastic=False),
        coupling=CouplingSpec("linear", a=coup_a),
        monitors=(MonitorSpec("tavg", period=tavg_period),),
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
# JAX benchmark
# ---------------------------------------------------------------------------

def benchmark_jax(spec: SimulationSpec, model_name: str,
                  jax_sizes: list[int], duration: float,
                  repeats: int) -> list[dict]:
    """Benchmark JAX vmap sweep. Each n_samples triggers a fresh JIT compile."""
    try:
        import jax
        platform = jax.default_backend()
    except ImportError:
        print("  JAX not installed — skipping JAX benchmark.")
        return []

    print(f"\n{'n_samples':>10}  {'JAX(s)':>10}  {'sam/s':>10}  {'platform':>10}")
    print("-" * 45)

    # Use subsample monitor for JAX (avoids large raw output transfers)
    dt = spec.integrator.dt
    jax_spec = SimulationSpec(
        model=spec.model,
        integrator=spec.integrator,
        coupling=spec.coupling,
        monitors=(MonitorSpec("subsample", period=max(dt * 10, 1.0)),),
        weights=spec.weights,
    )

    rows = []
    for n in jax_sizes:
        sw = _make_sweep(model_name, n)

        # Warmup this batch size (JAX recompiles per unique n_samples)
        _try(lambda: Sweeper(jax_spec, sw, backend="jax").run(duration),
             f"JAX warmup n={n}")

        def _run():
            return Sweeper(jax_spec, sw, backend="jax").run(duration)

        t = _best_of(_run, repeats)
        rate = n / t
        row = dict(n=n, t_jax=t, rate_jax=rate, platform=platform)
        rows.append(row)
        print(f"{n:>10}  {t:>10.3f}  {rate:>10.0f}  {platform:>10}")

    print("-" * 45)
    print(f"  sam/s = parameter sets per second  |  platform: {platform}")
    return rows


# ---------------------------------------------------------------------------
# Combined throughput table (CPU + CUDA + JAX at matching sizes)
# ---------------------------------------------------------------------------

def throughput_comparison(cpu_rows: list[dict], cuda_rows: list[dict],
                          jax_rows: list[dict], n_workers: int) -> None:
    gpu_rows = cuda_rows or jax_rows
    if not gpu_rows:
        return

    print("\n--- Throughput comparison at matched sweep sizes (samples/second) ---")
    cuda_map = {r["n"]: r["rate"] for r in cuda_rows}
    jax_map  = {r["n"]: r["rate_jax"] for r in jax_rows}
    cpu_map  = {r["n"]: r for r in cpu_rows}

    all_gpu = set(cuda_map) | set(jax_map)
    common  = sorted(all_gpu & set(cpu_map))
    if not common:
        print("  (no overlapping sweep sizes between CPU and GPU runs)")
        return

    print(f"{'n_samples':>10}  {'NumPy':>10}  {'Nb-NT':>10}  "
          f"{'Cpp-par':>10}  {'CUDA':>10}  {'JAX':>10}  {'JAX/Nb':>8}  {'JAX/Cpp':>8}")
    print("-" * 90)
    for n in common:
        def rate(t): return n / t if t and t > 0 else float("nan")
        r   = cpu_map[n]
        rn  = rate(r["t_np"])
        rnb = rate(r["t_nbN"])
        rc  = rate(r["t_cp"])
        rg  = cuda_map.get(n, float("nan"))
        rj  = jax_map.get(n, float("nan"))
        def _sp(num, den): return f"{num/den:>7.1f}x" if den > 0 and den == den else "    N/A"
        print(
            f"{n:>10}  {rn:>10.0f}  {rnb:>10.0f}  {rc:>10.0f}  "
            f"{rg:>10.0f}  {rj:>10.0f}  {_sp(rj,rnb):>8}  {_sp(rj,rc):>8}"
        )
    print("-" * 90)
    print(f"  JAX/Nb = JAX speedup over Numba {n_workers}T  |  JAX/Cpp = over C++ parallel")


# ---------------------------------------------------------------------------
# Plotting  — delegated to helpers.plot_sweep_benchmark
# ---------------------------------------------------------------------------

def plot_results(cpu_rows: list[dict], cuda_rows: list[dict],
                 model_name: str, n_nodes: int, duration: float,
                 n_workers: int, repeats: int, dt: float,
                 out_dir: Path) -> None:
    """Delegate to helpers.plot_sweep_benchmark."""
    plot_sweep_benchmark(
        cpu_rows=cpu_rows,
        cuda_rows=cuda_rows,
        model_name=model_name,
        n_nodes=n_nodes,
        duration=duration,
        dt=dt,
        n_workers=n_workers,
        repeats=repeats,
        out_dir=out_dir,
    )


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------

def save_json(cpu_rows: list[dict], cuda_rows: list[dict],
              jax_rows: list[dict], meta: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"meta": meta, "cpu": cpu_rows, "cuda": cuda_rows, "jax": jax_rows}
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
    p.add_argument("--jax-sizes",   type=int,   nargs="+", default=DEFAULT_JAX_SIZES)
    p.add_argument("--no-cuda",     action="store_true")
    p.add_argument("--no-jax",      action="store_true")
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
    print(f"  JAX sizes : {args.jax_sizes if not args.no_jax else '(disabled)'}")
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

    # ---- JAX benchmark ----
    jax_rows: list[dict] = []
    if not args.no_jax:
        print(f"\n{'='*70}")
        print("  JAX backend  (jax.vmap + jax.jit)")
        print(f"{'='*70}")
        jax_rows = benchmark_jax(
            spec, args.model, args.jax_sizes,
            args.duration, args.repeats,
        )

    # ---- Summary ----
    throughput_comparison(cpu_rows, cuda_rows, jax_rows, args.n_workers)

    # ---- Save ----
    meta = dict(
        model=args.model, n_nodes=args.n_nodes,
        duration=args.duration, dt=args.dt,
        n_workers=args.n_workers, repeats=args.repeats,
    )
    save_json(
        cpu_rows, cuda_rows, jax_rows, meta,
        args.output_dir / f"benchmark_all_backends_{args.model}.json",
    )

    if not args.no_plot:
        plot_results(
            cpu_rows, cuda_rows,
            args.model, args.n_nodes, args.duration,
            args.n_workers, args.repeats, args.dt,
            args.output_dir,
        )


if __name__ == "__main__":
    main()
