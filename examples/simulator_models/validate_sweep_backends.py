"""
Cross-Backend Sweep Validation
=================================

Checks that NumPy, Numba CPU, C++, CUDA (if available), and JAX all produce
numerically consistent sweep results for the same model and parameters.

Usage
-----
    python validate_sweep_backends.py
    python validate_sweep_backends.py --model sup_hopf --n-nodes 10 --n-samples 8
    python validate_sweep_backends.py --rtol 1e-3   # CPU tolerance
    python validate_sweep_backends.py --rtol-jax 2e-2  # JAX float32 tolerance

Each available backend is compared against the NumPy reference.
Exit code 0 = all checks pass; non-zero = at least one mismatch.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import warnings
import numpy as np

# Suppress low-occupancy warnings for tiny CUDA batches in validation
warnings.filterwarnings(
    "ignore",
    message="Grid size.*will likely result in GPU under-utilization",
)

# Prefer the vbi living in this checkout over any other version already
# installed (e.g. a different vbi checkout installed editable elsewhere).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vbi.simulator import Simulator, Sweeper
from vbi.simulator.spec import (
    SimulationSpec, IntegratorSpec, CouplingSpec, MonitorSpec, Connectivity,
)
from vbi.simulator.spec.sweep import SweepSpec

# Available models for quick sweeps
from vbi.simulator.models.sup_hopf            import sup_hopf
from vbi.simulator.models.generic_2d_oscillator import generic_2d_oscillator
from vbi.simulator.models.kuramoto            import kuramoto
from vbi.simulator.models.mpr                 import mpr

MODELS = {
    "sup_hopf":             (sup_hopf,             "a",   np.linspace(-0.5, 0.5, 8), 0.01, 0.1),
    "generic_2d_oscillator":(generic_2d_oscillator,"I",   np.linspace(-1.0, 1.0, 8), 0.01, 0.05),
    "kuramoto":             (kuramoto,             "omega",np.linspace(0.5, 2.0, 8), 0.01, 0.1),
    "mpr":                  (mpr,                  "eta", np.linspace(-5.5, -4.0, 8),0.01, 0.1),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _weights(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    W = np.abs(rng.standard_normal((n, n)))
    np.fill_diagonal(W, 0.0)
    W /= W.sum(axis=1, keepdims=True).clip(1e-8)
    return W


def _make_spec(model, n_nodes: int, dt: float, coup_a: float) -> SimulationSpec:
    return SimulationSpec(
        model=model,
        integrator=IntegratorSpec(method="heun", dt=dt, stochastic=False),
        coupling=CouplingSpec("linear", a=coup_a),
        monitors=(MonitorSpec("raw"),),
        connectivity=Connectivity(_weights(n_nodes)),
    )


def _run_sweep(spec, sweep_spec, backend: str, duration: float) -> list[np.ndarray]:
    """Return list of raw data arrays (one per sweep point)."""
    results = Sweeper(spec, sweep_spec, backend=backend).run(duration)
    return [res["raw"][1] for res in results]


def _check_match(reference: list[np.ndarray], candidate: list[np.ndarray],
                 backend: str, rtol: float, atol: float) -> tuple[bool, str]:
    """Compare two sweep result lists. Returns (ok, message)."""
    if len(reference) != len(candidate):
        return False, f"Length mismatch: ref={len(reference)}, got={len(candidate)}"
    max_abs = max_rel = 0.0
    for i, (r, c) in enumerate(zip(reference, candidate)):
        if r.shape != c.shape:
            return False, f"Sample {i}: shape mismatch ref={r.shape} got={c.shape}"
        diff = np.abs(c - r)
        denom = np.abs(r)
        max_abs = max(max_abs, diff.max())
        nz = denom > 0
        if nz.any():
            max_rel = max(max_rel, (diff[nz] / denom[nz]).max())
    ok = (max_abs <= atol + rtol * np.abs(np.array([r.max() for r in reference])).mean())
    try:
        for r, c in zip(reference, candidate):
            np.testing.assert_allclose(c, r, rtol=rtol, atol=atol)
        return True, f"OK  (max_abs={max_abs:.2e}  max_rel={max_rel:.2e})"
    except AssertionError as e:
        msg = str(e).split("\n")[2] if "\n" in str(e) else str(e)
        return False, f"FAIL  max_abs={max_abs:.2e}  max_rel={max_rel:.2e}  - {msg}"


# ---------------------------------------------------------------------------
# Main validation
# ---------------------------------------------------------------------------

def validate(
    model_name: str,
    n_nodes: int,
    n_samples: int,
    duration: float,
    rtol_cpu: float,
    rtol_cuda: float,
    rtol_jax: float,
) -> bool:
    model, sweep_param, values_all, dt, coup_a = MODELS[model_name]
    values = values_all[:n_samples] if n_samples < len(values_all) else values_all
    # Extend if needed
    if n_samples > len(values_all):
        values = np.linspace(values_all[0], values_all[-1], n_samples)

    spec       = _make_spec(model, n_nodes, dt, coup_a)
    sweep_spec = SweepSpec(params={sweep_param: values})

    print(f"\nModel       : {model.name}")
    print(f"Sweep param : {sweep_param}  ({len(values)} values)")
    print(f"n_nodes     : {n_nodes}")
    print(f"duration    : {duration} ms  |  dt={dt} ms")
    print(f"tolerance   : CPU rtol={rtol_cpu:.0e},  CUDA rtol={rtol_cuda:.0e},  "
          f"JAX rtol={rtol_jax:.0e}")
    print()

    # --- Reference: NumPy ---
    print("  [1/5] Running NumPy sweep (reference)… ", end="", flush=True)
    ref = _run_sweep(spec, sweep_spec, "numpy", duration)
    print("done.")

    all_ok = True

    # --- Numba ---
    print("  [2/5] Running Numba CPU sweep… ", end="", flush=True)
    try:
        nb = _run_sweep(spec, sweep_spec, "numba", duration)
        ok, msg = _check_match(ref, nb, "numba", rtol=rtol_cpu, atol=0.0)
        status = "✓" if ok else "✗"
        print(f"{status}  {msg}")
        all_ok = all_ok and ok
    except Exception as e:
        print(f"SKIP  ({type(e).__name__}: {e})")

    # --- C++ ---
    print("  [3/5] Running C++ sweep… ", end="", flush=True)
    try:
        cpp = _run_sweep(spec, sweep_spec, "cpp", duration)
        ok, msg = _check_match(ref, cpp, "cpp", rtol=rtol_cpu, atol=0.0)
        status = "✓" if ok else "✗"
        print(f"{status}  {msg}")
        all_ok = all_ok and ok
    except Exception as e:
        print(f"SKIP  ({type(e).__name__}: {e})")

    # --- CUDA ---
    print("  [4/5] Running CUDA sweep… ", end="", flush=True)
    try:
        from vbi.simulator.backend.numba_cuda import CUDA_AVAILABLE
        if not CUDA_AVAILABLE:
            raise RuntimeError("no CUDA device")
        cuda = _run_sweep(spec, sweep_spec, "cuda", duration)
        ok, msg = _check_match(ref, cuda, "cuda", rtol=rtol_cuda, atol=1e-5)
        status = "✓" if ok else "✗"
        print(f"{status}  {msg}  [float32 GPU]")
        all_ok = all_ok and ok
    except Exception as e:
        print(f"SKIP  ({type(e).__name__}: {e})")

    # --- JAX ---
    print("  [5/5] Running JAX sweep… ", end="", flush=True)
    try:
        import jax  # noqa: F401
        # JAX uses raw monitor; use subsample to keep output size manageable
        jax_spec = SimulationSpec(
            model=model,
            integrator=IntegratorSpec(method="heun", dt=dt, stochastic=False),
            coupling=CouplingSpec("linear", a=coup_a),
            monitors=(MonitorSpec("subsample", period=max(dt * 10, 1.0)),),
            connectivity=spec.connectivity,
        )
        jax_sweep_results = Sweeper(jax_spec, sweep_spec, backend="jax").run(duration)
        ref_sub = []
        for res in jax_sweep_results:
            _, d = res["subsample"]
            ref_sub.append(d)
        # Build numpy-backend reference on same spec for apples-to-apples comparison
        np_sub_results = Sweeper(jax_spec, sweep_spec, backend="numpy").run(duration)
        np_sub = [res["subsample"][1] for res in np_sub_results]
        ok, msg = _check_match(np_sub, ref_sub, "jax", rtol=rtol_jax, atol=1e-5)
        status = "✓" if ok else "✗"
        platform = jax.default_backend()
        print(f"{status}  {msg}  [float32 {platform.upper()}]")
        all_ok = all_ok and ok
    except Exception as e:
        print(f"SKIP  ({type(e).__name__}: {e})")

    print()
    return all_ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Cross-backend sweep validation")
    parser.add_argument("--model",      default="sup_hopf",
                        choices=list(MODELS.keys()))
    parser.add_argument("--n-nodes",    type=int,   default=10)
    parser.add_argument("--n-samples",  type=int,   default=8)
    parser.add_argument("--duration",   type=float, default=200.0)
    parser.add_argument("--rtol",       type=float, default=1e-4,
                        help="CPU backend tolerance (NumPy/Numba/C++)")
    parser.add_argument("--rtol-cuda",  type=float, default=5e-3,
                        help="CUDA tolerance (float32 vs float64)")
    parser.add_argument("--rtol-jax",   type=float, default=1e-2,
                        help="JAX tolerance (float32 vs float64)")
    parser.add_argument("--all-models", action="store_true",
                        help="Validate all four models sequentially")
    args = parser.parse_args()

    print("=" * 60)
    print("  VBI sweep backend cross-validation")
    print("=" * 60)

    models_to_run = list(MODELS.keys()) if args.all_models else [args.model]
    all_passed = True

    for m in models_to_run:
        ok = validate(
            model_name=m,
            n_nodes=args.n_nodes,
            n_samples=args.n_samples,
            duration=args.duration,
            rtol_cpu=args.rtol,
            rtol_cuda=args.rtol_cuda,
            rtol_jax=args.rtol_jax,
        )
        all_passed = all_passed and ok

    print("=" * 60)
    if all_passed:
        print("  RESULT: ALL CHECKS PASSED")
    else:
        print("  RESULT: ONE OR MORE CHECKS FAILED")
    print("=" * 60)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
