"""
Run the sweep benchmark for every model defined in MODELS and save one
figure per model.

Each model is benchmarked with the same network size and sweep ranges but
its own coupling strength (as configured in MODELS).  The Numba JIT and C++
compilation happen once per model; CUDA JIT is cached to disk after the first
model.  JAX recompiles for each (model, n_samples) pair; the first call per
model includes XLA compilation overhead.

Usage
-----
    python benchmark_all_models.py                      # all defaults
    python benchmark_all_models.py --n-nodes 40         # larger network
    python benchmark_all_models.py --no-cuda            # skip GPU
    python benchmark_all_models.py --no-jax             # skip JAX
    python benchmark_all_models.py --models sup_hopf mpr  # subset

Outputs
-------
    outputs/benchmark_all_backends_{model_name}.json
    outputs/benchmark_all_backends_{model_name}.png
    outputs/benchmark_all_models_summary.json           # cross-model table
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# ---- repo path ----
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from helpers import ensure_repo_on_path, plot_sweep_benchmark
ensure_repo_on_path(__file__)

# Import benchmark primitives from the sibling script
from benchmark_all_backends import (
    MODELS,
    DEFAULT_N_NODES,
    DEFAULT_DURATION,
    DEFAULT_DT,
    DEFAULT_CPU_SIZES,
    DEFAULT_CUDA_SIZES,
    DEFAULT_JAX_SIZES,
    N_WORKERS,
    _make_spec,
    warmup,
    benchmark_cpu,
    benchmark_cuda,
    benchmark_jax,
    throughput_comparison,
    save_json,
)

OUTPUT_DIR = Path(__file__).parent / "outputs"


# ---------------------------------------------------------------------------
# Per-model run
# ---------------------------------------------------------------------------

def run_one_model(
    model_name: str,
    n_nodes: int,
    duration: float,
    dt: float,
    cpu_sizes: list[int],
    cuda_sizes: list[int],
    jax_sizes: list[int],
    n_workers: int,
    repeats: int,
    no_cuda: bool,
    no_jax: bool,
    out_dir: Path,
) -> dict:
    """Benchmark a single model, save JSON + figure, return summary dict."""
    print()
    print("=" * 70)
    print(f"  Model: {model_name}")
    print("=" * 70)

    spec = _make_spec(model_name, n_nodes, dt)
    warmup(spec, model_name, n_workers)

    # CPU
    print(f"\n{'='*70}")
    print("  CPU backends")
    print(f"{'='*70}")
    cpu_rows = benchmark_cpu(spec, model_name, cpu_sizes, duration,
                             n_workers, repeats)

    # CUDA
    cuda_rows: list[dict] = []
    if not no_cuda:
        print(f"\n{'='*70}")
        print("  CUDA backend")
        print(f"{'='*70}")
        cuda_rows = benchmark_cuda(spec, model_name, cuda_sizes,
                                   duration, repeats)

    # JAX
    jax_rows: list[dict] = []
    if not no_jax:
        print(f"\n{'='*70}")
        print("  JAX backend  (jax.vmap + jax.jit)")
        print(f"{'='*70}")
        jax_rows = benchmark_jax(spec, model_name, jax_sizes,
                                 duration, repeats)

    # Summary table
    throughput_comparison(cpu_rows, cuda_rows, jax_rows, n_workers)

    # Save JSON
    meta = dict(
        model=model_name, n_nodes=n_nodes,
        duration=duration, dt=dt,
        n_workers=n_workers, repeats=repeats,
    )
    save_json(cpu_rows, cuda_rows, jax_rows, meta,
              out_dir / f"benchmark_all_backends_{model_name}.json")

    # Save figure
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
        jax_rows=jax_rows,
    )

    # Collect peak throughputs for the summary
    cp_rates   = [r["n"] / r["t_cp"]    for r in cpu_rows]  if cpu_rows  else []
    nb_rates   = [r["n"] / r["t_nbN"]   for r in cpu_rows]  if cpu_rows  else []
    cuda_rates = [r["rate"]              for r in cuda_rows] if cuda_rows else []
    jax_rates  = [r["rate_jax"]         for r in jax_rows]  if jax_rows  else []

    return {
        "model":       model_name,
        "n_nodes":     n_nodes,
        "duration_ms": duration,
        "dt":          dt,
        "peak_cpp_par_sam_s":  float(np.mean(cp_rates[-3:]))  if cp_rates   else None,
        "peak_numba_NT_sam_s": float(np.mean(nb_rates[-3:]))  if nb_rates   else None,
        "peak_cuda_sam_s":     float(max(cuda_rates))          if cuda_rates else None,
        "peak_jax_sam_s":      float(max(jax_rates))           if jax_rates  else None,
        "cuda_max_n":          max(r["n"] for r in cuda_rows)  if cuda_rows  else None,
        "jax_max_n":           max(r["n"] for r in jax_rows)   if jax_rows   else None,
        "jax_platform":        jax_rows[0]["platform"]          if jax_rows   else None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run sweep benchmark for all (or selected) models."
    )
    p.add_argument("--models",     nargs="+", default=list(MODELS.keys()),
                   choices=list(MODELS.keys()),
                   help="Models to benchmark (default: all)")
    p.add_argument("--n-nodes",    type=int,   default=DEFAULT_N_NODES)
    p.add_argument("--duration",   type=float, default=DEFAULT_DURATION)
    p.add_argument("--dt",         type=float, default=DEFAULT_DT)
    p.add_argument("--repeats",    type=int,   default=3)
    p.add_argument("--n-workers",  type=int,   default=N_WORKERS)
    p.add_argument("--cpu-sizes",  type=int,   nargs="+", default=DEFAULT_CPU_SIZES)
    p.add_argument("--cuda-sizes", type=int,   nargs="+", default=DEFAULT_CUDA_SIZES)
    p.add_argument("--jax-sizes",  type=int,   nargs="+", default=DEFAULT_JAX_SIZES)
    p.add_argument("--no-cuda",    action="store_true")
    p.add_argument("--no-jax",     action="store_true")
    p.add_argument("--output-dir", type=Path,  default=OUTPUT_DIR)
    return p.parse_args()


def main() -> None:
    args  = parse_args()
    t_all = time.perf_counter()

    print("=" * 70)
    print("  VBI all-models parameter sweep benchmark")
    print("=" * 70)
    print(f"  Models    : {', '.join(args.models)}")
    print(f"  n_nodes   : {args.n_nodes}")
    print(f"  duration  : {args.duration} ms  |  dt={args.dt} ms")
    print(f"  CPU sizes : {args.cpu_sizes}")
    print(f"  CUDA sizes: {args.cuda_sizes if not args.no_cuda else '(disabled)'}")
    print(f"  JAX sizes : {args.jax_sizes  if not args.no_jax  else '(disabled)'}")
    print(f"  n_workers : {args.n_workers}")
    print(f"  repeats   : {args.repeats}  (best-of)")
    print()

    summaries = []
    for model_name in args.models:
        t0 = time.perf_counter()
        summary = run_one_model(
            model_name  = model_name,
            n_nodes     = args.n_nodes,
            duration    = args.duration,
            dt          = args.dt,
            cpu_sizes   = args.cpu_sizes,
            cuda_sizes  = args.cuda_sizes,
            jax_sizes   = args.jax_sizes,
            n_workers   = args.n_workers,
            repeats     = args.repeats,
            no_cuda     = args.no_cuda,
            no_jax      = args.no_jax,
            out_dir     = args.output_dir,
        )
        summary["wall_s"] = round(time.perf_counter() - t0, 1)
        summaries.append(summary)

    # ---- Cross-model summary ----
    print()
    print("=" * 70)
    print("  Cross-model summary")
    print("=" * 70)
    print(f"{'Model':<26}  {'Nb-NT':>10}  {'C++par':>10}  {'CUDA':>10}  "
          f"{'JAX':>10}  {'platform':>8}  {'time(s)':>8}")
    print("-" * 92)
    for s in summaries:
        nb  = f"{s['peak_numba_NT_sam_s']:.0f}" if s["peak_numba_NT_sam_s"] else "—"
        cp  = f"{s['peak_cpp_par_sam_s']:.0f}"  if s["peak_cpp_par_sam_s"]  else "—"
        cu  = f"{s['peak_cuda_sam_s']:.0f}"      if s["peak_cuda_sam_s"]     else "—"
        jx  = f"{s['peak_jax_sam_s']:.0f}"       if s["peak_jax_sam_s"]      else "—"
        plt = s["jax_platform"] or "—"
        print(f"{s['model']:<26}  {nb:>10}  {cp:>10}  {cu:>10}  "
              f"{jx:>10}  {plt:>8}  {s['wall_s']:>8.1f}")
    print("-" * 92)
    print("  Throughput = samples/second (peak over measured range)")
    print("  JAX platform: cpu or gpu (depends on installed jaxlib)")
    print(f"\n  Total wall time: {time.perf_counter() - t_all:.0f} s")

    # Save summary JSON
    out = args.output_dir / "benchmark_all_models_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"  Summary JSON → {out}")

    print()
    print("  Figures:")
    for s in summaries:
        fig = args.output_dir / f"benchmark_all_backends_{s['model']}.png"
        print(f"    {fig}")


if __name__ == "__main__":
    main()
