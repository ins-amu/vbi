"""
JAX backend demo - differentiable brain-network simulator.

Demonstrates four unique capabilities of the JAX backend:

1. Single-run simulation (Heun, with delays, subsample monitor)
2. BOLD monitor (Balloon-Windkessel via lax.scan)
3. Parameter sweep via jax.vmap - all samples in one JIT call
4. Gradient of a loss through the full simulation (jax.grad)

The JAX backend runs on CPU or GPU transparently.  Activate the vbienv
environment (which has CUDA-enabled JAX) to run on GPU.

Usage
-----
    python jax_demo.py
    python jax_demo.py --n-nodes 20 --n-samples 32 --duration 500
    python jax_demo.py --model mpr --no-grad
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from helpers import ensure_repo_on_path
ensure_repo_on_path(__file__)

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

from vbi.simulator import Simulator, Sweeper
from vbi.simulator.spec import (
    SimulationSpec, IntegratorSpec, CouplingSpec, MonitorSpec, Connectivity,
)
from vbi.simulator.spec.sweep import SweepSpec
from vbi.simulator.models.mpr        import mpr
from vbi.simulator.models.sup_hopf   import sup_hopf
from vbi.simulator.models.generic_2d_oscillator import generic_2d_oscillator

MODELS = {
    "mpr":                  (mpr,                  "eta", (-5.5, -4.0), 0.1, 0.1),
    "sup_hopf":             (sup_hopf,             "a",   (-0.5,  0.5), 0.1, 0.1),
    "generic_2d_oscillator":(generic_2d_oscillator,"I",   (-1.0,  1.0), 0.1, 0.05),
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


def _tract_lengths(n: int, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    D = np.abs(rng.standard_normal((n, n))) * 20.0
    np.fill_diagonal(D, 0.0)
    return D


def _make_spec(model_name: str, n_nodes: int, dt: float,
               monitor: MonitorSpec) -> SimulationSpec:
    model, _, _, _, coup_a = MODELS[model_name]
    return SimulationSpec(
        model=model,
        integrator=IntegratorSpec(method="heun", dt=dt),
        coupling=CouplingSpec("linear", a=coup_a),
        monitors=(monitor,),
        connectivity=Connectivity(_weights(n_nodes), _tract_lengths(n_nodes)),
    )


# ---------------------------------------------------------------------------
# Demo 1 - single run
# ---------------------------------------------------------------------------

def demo_single_run(model_name: str, n_nodes: int, duration: float,
                    dt: float) -> None:
    print("\n" + "─" * 60)
    print("  Demo 1 - Single run (subsample monitor)")
    print("─" * 60)

    spec = _make_spec(model_name, n_nodes, dt,
                      MonitorSpec("subsample", period=max(dt * 10, 1.0)))

    # First call: JIT compile
    t0 = time.perf_counter()
    sim = Simulator(spec, backend="jax")
    result = sim.run(duration)
    t_first = time.perf_counter() - t0

    # Second call: compiled - much faster
    t0 = time.perf_counter()
    result = sim.run(duration)
    t_cached = time.perf_counter() - t0

    t, data = result["subsample"]
    print(f"  model      : {model_name}  |  {n_nodes} nodes  |  {duration} ms")
    print(f"  output     : t.shape={t.shape}  data.shape={data.shape}")
    print(f"  1st call   : {t_first:.3f} s  (includes JIT compilation)")
    print(f"  2nd call   : {t_cached:.3f} s  (compiled, cache hit)")
    print(f"  JIT ratio  : {t_first/t_cached:.1f}×")
    print(f"  data range : [{data.min():.4f}, {data.max():.4f}]")


# ---------------------------------------------------------------------------
# Demo 2 - BOLD monitor
# ---------------------------------------------------------------------------

def demo_bold(model_name: str, n_nodes: int, duration: float,
              dt: float) -> None:
    print("\n" + "─" * 60)
    print("  Demo 2 - BOLD monitor (Balloon-Windkessel via lax.scan)")
    print("─" * 60)

    tr = 2000.0   # ms
    spec = _make_spec(model_name, n_nodes, dt, MonitorSpec("bold", tr=tr))

    sim = Simulator(spec, backend="jax")
    sim.run(duration)  # warmup

    t0 = time.perf_counter()
    result = sim.run(duration)
    elapsed = time.perf_counter() - t0

    t_bold, bold = result["bold"]
    n_tr = int(duration / tr)
    print(f"  duration   : {duration} ms  |  TR={tr} ms  →  {n_tr} expected samples")
    print(f"  BOLD shape : {bold.shape}  (n_TR × n_nodes)")
    print(f"  run time   : {elapsed:.3f} s  (compiled)")
    print(f"  BOLD range : [{bold.min():.4f}, {bold.max():.4f}]")


# ---------------------------------------------------------------------------
# Demo 3 - parameter sweep via vmap
# ---------------------------------------------------------------------------

def demo_sweep(model_name: str, n_nodes: int, n_samples: int,
               duration: float, dt: float) -> None:
    print("\n" + "─" * 60)
    print("  Demo 3 - Parameter sweep (jax.vmap, one JIT call)")
    print("─" * 60)

    model, sweep_param, (lo, hi), _, coup_a = MODELS[model_name]
    spec = SimulationSpec(
        model=model,
        integrator=IntegratorSpec(method="heun", dt=dt),
        coupling=CouplingSpec("linear", a=coup_a),
        monitors=(MonitorSpec("subsample", period=max(dt * 10, 1.0)),),
        connectivity=Connectivity(_weights(n_nodes), _tract_lengths(n_nodes)),
    )
    sweep_spec = SweepSpec(params={sweep_param: np.linspace(lo, hi, n_samples)})
    sweeper = Sweeper(spec, sweep_spec, backend="jax")

    # Warmup - first call compiles for (n_samples, n_nodes) batch
    sweeper.run(duration)

    t0 = time.perf_counter()
    results = sweeper.run(duration)
    elapsed = time.perf_counter() - t0

    _, data_0 = results[0]["subsample"]
    print(f"  model      : {model_name}  |  {n_nodes} nodes  |  {duration} ms")
    print(f"  sweep      : {sweep_param}  ∈  [{lo}, {hi}]  ×  {n_samples} samples")
    print(f"  per-sample : t={data_0.shape[0]} steps × {data_0.shape[1]} SVs × {n_nodes} nodes")
    print(f"  total time : {elapsed:.3f} s")
    print(f"  throughput : {n_samples / elapsed:.0f} samples/s")


# ---------------------------------------------------------------------------
# Demo 4 - gradient through simulation
# ---------------------------------------------------------------------------

def demo_gradient(model_name: str, n_nodes: int, duration: float,
                  dt: float) -> None:
    print("\n" + "─" * 60)
    print("  Demo 4 - jax.grad through the full simulation")
    print("─" * 60)

    if not JAX_AVAILABLE:
        print("  JAX not installed - skipping gradient demo.")
        return

    from vbi.simulator.backend.jax_.simulator import JaxSimulator

    model, sweep_param, (lo, hi), _, coup_a = MODELS[model_name]
    spec = SimulationSpec(
        model=model,
        integrator=IntegratorSpec(method="heun", dt=dt),
        coupling=CouplingSpec("linear", a=coup_a),
        monitors=(MonitorSpec("subsample", period=max(dt * 10, 1.0)),),
        connectivity=Connectivity(_weights(n_nodes), _tract_lengths(n_nodes)),
    )

    sim = JaxSimulator()
    sim.build(spec)
    n_steps = round(duration / dt)

    # Pick which parameter to differentiate w.r.t.
    grad_param = sweep_param   # same as sweep param for consistency

    def loss(param_val):
        params = {**sim._params, grad_param: param_val}
        result = sim._run_core(params, n_steps)
        _, data = result["subsample"]
        return jnp.mean(data ** 2)

    # Compile the gradient
    grad_fn = jax.jit(jax.grad(loss))
    param0 = jnp.float32((lo + hi) / 2)

    # Warmup
    grad_fn(param0)

    t0 = time.perf_counter()
    g = grad_fn(param0)
    elapsed = time.perf_counter() - t0

    print(f"  model      : {model_name}  |  {n_nodes} nodes  |  {duration} ms")
    print(f"  ∂loss/∂{grad_param:<6} = {float(g):.6f}  at  {grad_param}={float(param0):.3f}")
    print(f"  grad time  : {elapsed:.4f} s  (compiled)")
    print(f"  gradient finite: {bool(jnp.isfinite(g))}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model",     default="mpr",   choices=list(MODELS.keys()))
    p.add_argument("--n-nodes",   type=int,   default=10)
    p.add_argument("--n-samples", type=int,   default=16)
    p.add_argument("--duration",  type=float, default=500.0)
    p.add_argument("--dt",        type=float, default=0.1)
    p.add_argument("--no-bold",   action="store_true")
    p.add_argument("--no-sweep",  action="store_true")
    p.add_argument("--no-grad",   action="store_true")
    return p.parse_args()


def main() -> None:
    if not JAX_AVAILABLE:
        print("JAX is not installed.  Install with:  pip install jax[cuda12]")
        sys.exit(1)

    args = parse_args()

    print("=" * 60)
    print("  VBI JAX backend demo")
    print(f"  JAX version: {jax.__version__}  |  "
          f"platform: {jax.default_backend()}")
    print(f"  devices: {jax.devices()}")
    print("=" * 60)

    demo_single_run(args.model, args.n_nodes, args.duration, args.dt)

    if not args.no_bold:
        bold_dur = max(args.duration, 6000.0)  # need at least a few TR periods
        demo_bold(args.model, args.n_nodes, bold_dur, args.dt)

    if not args.no_sweep:
        demo_sweep(args.model, args.n_nodes, args.n_samples, args.duration, args.dt)

    if not args.no_grad:
        demo_gradient(args.model, args.n_nodes, args.duration, args.dt)

    print("\n" + "=" * 60)
    print("  Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
