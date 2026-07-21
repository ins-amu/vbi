"""
Kuramoto Model Demo
====================

Verifies that the VBI KuramotoCoupling produces trajectories identical to a
single self-contained NumPy implementation using the same Heun integrator.

Both implementations solve the (frustrated) Kuramoto model:

    dθ_i/dt = ω_i + (G/N) Σ_j W_{ij} sin(θ_j − θ_i + α)

where α (--alpha) is the frustration angle. α=0 gives the standard model.

Run
---
::

    python kuramoto_demo.py                        # 5-node complete graph
    python kuramoto_demo.py --n 10 --duration 1000
    python kuramoto_demo.py --delayed              # add 5 ms axonal delay
    python kuramoto_demo.py --alpha 0.5236         # frustrated  (α = π/6)
"""

# %%
# Setup
# -----

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.dont_write_bytecode = True

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    _SCRIPT_PATH = Path(__file__)
except NameError:
    # sphinx-gallery execs this file without setting __file__; it already
    # chdirs into the script's own directory first, so cwd is equivalent.
    _SCRIPT_PATH = Path.cwd() / "kuramoto_demo.py"

# Prefer the vbi living in this checkout over any other version already
# installed (e.g. a different vbi checkout installed editable elsewhere).
# Downloaded standalone copies of this script should `pip install vbi`
# instead - see the first notebook cell.
_repo_root = _SCRIPT_PATH.resolve().parents[2]
sys.path.insert(0, str(_repo_root))

from vbi.simulator import Simulator
from vbi.simulator.models.kuramoto import kuramoto
from vbi.simulator.spec.coupling import CouplingSpec
from vbi.simulator.spec.integrator import IntegratorSpec
from vbi.simulator.spec.monitor import MonitorSpec
from vbi.simulator.spec.simulation import SimulationSpec
from vbi.simulator.spec.connectivity import Connectivity


def complete_graph_weights(n_nodes: int) -> np.ndarray:
    """Return a dense complete-graph weight matrix with zero diagonal."""
    weights = np.ones((n_nodes, n_nodes), dtype=np.float64)
    np.fill_diagonal(weights, 0.0)
    return weights


# %%
# Reference implementation (pure NumPy)
# --------------------------------------
# A single, dependency-free function that integrates the (frustrated)
# Kuramoto model with Heun's method, including optional axonal delays.

def kuramoto_heun(
    theta0: np.ndarray,
    omega: np.ndarray,
    weights: np.ndarray,
    G: float,
    dt: float,
    n_steps: int,
    delays: np.ndarray | None = None,
    alpha: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Integrate the (frustrated) Kuramoto model with Heun's method.

    Uses the frozen-coupling convention (VBI / TVB): coupling is computed once
    per step and held fixed for both Heun stages.  For Kuramoto (dfun = ω + c)
    this reduces to Euler.

    Parameters
    ----------
    theta0  : (N,)            initial phases [rad]
    omega   : (N,)            natural frequencies [rad/ms]
    weights : (N, N)          weight matrix, weights[tgt, src]
    G       : float           global coupling strength
    dt      : float           time step [ms]
    n_steps : int             number of integration steps
    delays  : (N, N) | None   axonal delays [ms]; None → zero delay
    alpha   : float           frustration angle [rad]  (default 0)

    Returns
    -------
    t     : (n_steps,)    time vector
    theta : (n_steps, N)  phase trajectories
    """
    N = len(theta0)
    theta = np.empty((n_steps, N))
    theta[0] = theta0.copy()

    if delays is not None:
        delay_steps = np.round(delays / dt).astype(int)
        horizon = int(delay_steps.max()) + 2
        # Fill all horizon slots with theta0 - matches VBI History.initialize()
        buf = np.tile(theta0, (horizon, 1))  # (horizon, N)
    else:
        delay_steps = None
        horizon = 1
        buf = None

    def compute_coupling(step: int, current_theta: np.ndarray) -> np.ndarray:
        """c[tgt] = (G/N) Σ_src W[tgt,src] sin(θ_src − θ_tgt + alpha)"""
        if buf is None:
            diff = current_theta[:, np.newaxis] - current_theta[np.newaxis, :]
        else:
            diff = np.empty((N, N))   # diff[src, tgt]
            t_last = step - 1
            for src in range(N):
                for tgt in range(N):
                    d = delay_steps[src, tgt]
                    idx = (t_last - d + horizon) % horizon
                    diff[src, tgt] = buf[idx, src] - current_theta[tgt]
        return (G / N) * (weights * np.sin(diff.T + alpha)).sum(axis=1)

    for step in range(1, n_steps):
        th = theta[step - 1]
        # Frozen-coupling Heun (VBI / TVB convention):
        #   coupling is computed once from the current (or delayed) phases and
        #   held fixed for both predictor and corrector stages.
        # For Kuramoto, dfun = ω + c has no further θ-dependence once c is frozen,
        # so k1 == k2 and Heun reduces to Euler.  The step below is equivalent to
        #   k1 = omega + c;  k2 = omega + c;  th_new = th + dt/2*(k1+k2) = th + dt*k1
        c  = compute_coupling(step, th)
        k1 = omega + c
        k2 = omega + c          # same c → same as k1 for Kuramoto
        theta[step] = th + 0.5 * dt * (k1 + k2)

        if buf is not None:
            buf[step % horizon] = theta[step]

    t = np.arange(n_steps) * dt
    return t, theta


# %%
# VBI simulator
# -------------
# Builds a :class:`~vbi.simulator.Simulator` with a Kuramoto coupling and
# Heun integrator, mirroring the reference implementation's setup.

def run_vbi(
    theta0: np.ndarray,
    omega: np.ndarray,
    weights: np.ndarray,
    G: float,
    dt: float,
    duration: float,
    tract_lengths: np.ndarray | None,
    speed: float,
    alpha: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    spec = SimulationSpec(
        model=kuramoto.with_init(theta=theta0),
        integrator=IntegratorSpec(method="heun", dt=dt),
        coupling=CouplingSpec(kind="kuramoto", alpha=alpha),
        monitors=(MonitorSpec(kind="raw"),),
        connectivity=Connectivity(weights, tract_lengths, speed=speed),
        node_params={"omega": omega, "G": G},
    )

    sim = Simulator(spec, backend="numpy")
    t, d = sim.run(duration)["raw"]
    theta = d[:, 0, :]  # (time, N)
    # Prepend t=0 / initial state so the time axis matches the reference
    t    = np.concatenate([[0.0], t])
    theta = np.vstack([theta0[np.newaxis, :], theta])
    return t, theta


# %%
# Comparison plot
# ---------------
# Plots per-node ``sin(θ)``, unwrapped phase, the Kuramoto order parameter
# R(t), and the absolute error between the VBI and reference trajectories.

def comparison_plot(
    t: np.ndarray,
    vbi_theta: np.ndarray,
    ref_theta: np.ndarray,
    out_path: Path,
    delayed: bool,
    alpha: float = 0.0,
) -> None:
    N = vbi_theta.shape[1]
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), tight_layout=True)
    colors = plt.cm.tab10(np.linspace(0, 0.9, N))
    tags = []
    if delayed:
        tags.append("delayed")
    if alpha != 0.0:
        tags.append(f"α={alpha:.4g} rad")
    subtitle = ("  (" + ", ".join(tags) + ")") if tags else ""

    # --- panel 1: sin(θ) per node ---
    ax = axes[0]
    for i in range(N):
        ax.plot(t, np.sin(ref_theta[:, i]),  color=colors[i], lw=1.5,
                label=f"node {i}")
        ax.plot(t, np.sin(vbi_theta[:, i]),  color=colors[i], lw=0.8,
                ls="--", alpha=0.7)
    ax.set_ylabel("sin(θ)")
    ax.set_ylim(-1.15, 1.15)
    ax.set_title(f"sin(θ) per node - solid: reference, dashed: VBI{subtitle}")
    ax.legend(fontsize=7, ncol=N, loc="upper right")

    # --- panel 2: raw unwrapped phases ---
    ax = axes[1]
    for i in range(N):
        ax.plot(t, ref_theta[:, i],  color=colors[i], lw=1.5)
        ax.plot(t, vbi_theta[:, i],  color=colors[i], lw=0.8, ls="--", alpha=0.7)
    ax.set_ylabel("θ  [rad]")
    ax.set_title("Unwrapped phase θ(t)")

    # --- panel 3: Kuramoto order parameter R(t) ---
    ax = axes[2]
    r_ref = np.abs(np.exp(1j * ref_theta).mean(axis=1))
    r_vbi = np.abs(np.exp(1j * vbi_theta).mean(axis=1))
    ax.plot(t, r_ref, color="steelblue", lw=1.5, label="reference R(t)")
    ax.plot(t, r_vbi, color="tomato",    lw=0.8, ls="--", label="VBI R(t)")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("order parameter R(t)")
    ax.legend(fontsize=8)

    # --- panel 4: absolute error (handles zero error gracefully) ---
    ax = axes[3]
    err = np.abs(vbi_theta - ref_theta)
    max_err_per_t = err.max(axis=1)
    max_err = max_err_per_t.max()
    if max_err > 0:
        ax.semilogy(t, max_err_per_t, color="k", lw=1.0)
        ax.set_ylabel("max |θ_VBI − θ_ref|  (log)")
    else:
        ax.plot(t, max_err_per_t, color="k", lw=1.0)
        ax.set_ylabel("max |θ_VBI − θ_ref|")
        ax.set_ylim(-0.1, 1.0)
        ax.text(0.5, 0.5, "error = 0  (machine precision)",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=11, color="green")
    ax.set_xlabel("time  [ms]")
    ax.set_title(f"max abs error: {max_err:.2e}   rms: {err.mean():.2e}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    print(f"saved figure: {out_path}")
    # Left open (not plt.close'd) so sphinx-gallery can capture it inline.


# %%
# Run the comparison
# -------------------
# Runs both implementations with default parameters (5-node complete graph)
# and reports the maximum/RMS phase error alongside the comparison plot.
# Pass ``--n``, ``--delayed``, ``--alpha``, etc. on the command line to
# change the setup.

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n",        type=int,   default=5,      help="number of nodes")
    p.add_argument("--duration", type=float, default=100.0,  help="simulation time [ms]")
    p.add_argument("--dt",       type=float, default=0.01,   help="time step [ms]")
    p.add_argument("--G",        type=float, default=0.5,    help="global coupling strength")
    p.add_argument("--alpha",    type=float, default=0.0,    help="frustration angle [rad]")
    p.add_argument("--delayed",  action="store_true",        help="add 5 ms axonal delay")
    p.add_argument("--speed",    type=float, default=1.0,    help="axonal speed [mm/ms]")
    p.add_argument("--tract",    type=float, default=5.0,    help="tract length [mm]")
    p.add_argument("--seed",     type=int,   default=42)
    p.add_argument("--output", type=Path,
                   default=_SCRIPT_PATH.with_name("outputs") / "kuramoto_comparison.png")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    N = args.n

    # shared setup
    weights = complete_graph_weights(N)
    omega   = rng.uniform(0.9, 0.95, N)
    theta0  = rng.uniform(-np.pi, np.pi, N)
    n_steps = round(args.duration / args.dt)

    tract_lengths = None
    delays_ms     = None
    speed         = args.speed

    if args.delayed:
        tract_lengths = np.full((N, N), args.tract, dtype=float)
        np.fill_diagonal(tract_lengths, 0.0)
        delays_ms = tract_lengths / speed  # (N, N) [ms]

    # --- reference (pure NumPy) ---
    t_ref, theta_ref = kuramoto_heun(
        theta0=theta0,
        omega=omega,
        weights=weights,
        G=args.G,
        dt=args.dt,
        n_steps=n_steps,
        delays=delays_ms,
        alpha=args.alpha,
    )

    # --- VBI simulator ---
    t_vbi, theta_vbi = run_vbi(
        theta0=theta0,
        omega=omega,
        weights=weights,
        G=args.G,
        dt=args.dt,
        duration=args.duration,
        tract_lengths=tract_lengths,
        speed=speed,
        alpha=args.alpha,
    )

    # align lengths (VBI may produce one fewer sample depending on rounding)
    n = min(len(t_ref), len(t_vbi))
    t_ref, theta_ref = t_ref[:n], theta_ref[:n]
    t_vbi, theta_vbi = t_vbi[:n], theta_vbi[:n]

    max_err = np.abs(theta_vbi - theta_ref).max()
    rms_err = np.sqrt(np.mean((theta_vbi - theta_ref) ** 2))

    extras = []
    if args.delayed:
        extras.append(f"delayed={args.tract} mm")
    if args.alpha != 0.0:
        extras.append(f"alpha={args.alpha:.4g} rad")
    print(f"Kuramoto  N={N}  G={args.G}  dt={args.dt} ms"
          + (("  " + "  ".join(extras)) if extras else ""))
    print(f"omega: {np.round(omega, 3)}")
    print(f"max |err|: {max_err:.3e}   rms |err|: {rms_err:.3e}")

    comparison_plot(t_ref, theta_vbi, theta_ref, args.output,
                         args.delayed, alpha=args.alpha)


if __name__ == "__main__":
    main()
