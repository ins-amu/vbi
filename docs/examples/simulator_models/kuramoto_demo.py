"""
Kuramoto model: VBI simulator vs pure-NumPy reference.

Verifies that the VBI KuramotoCoupling produces trajectories identical to a
single self-contained NumPy implementation using the same Heun integrator.

Both implementations solve:

    dθ_i/dt = ω_i + (G/N) Σ_j W_{ij} sin(θ_j − θ_i)

Run
---
    python kuramoto_demo.py                  # 5-node complete graph, 500 ms
    python kuramoto_demo.py --n 10 --duration 1000
    python kuramoto_demo.py --delayed        # add 5 ms axonal delay
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.dont_write_bytecode = True

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from helpers import ensure_repo_on_path, complete_graph_weights

ensure_repo_on_path(__file__)

from vbi.simulator import Simulator
from vbi.simulator.models.kuramoto import kuramoto
from vbi.simulator.spec.coupling import CouplingSpec
from vbi.simulator.spec.integrator import IntegratorSpec
from vbi.simulator.spec.monitor import MonitorSpec
from vbi.simulator.spec.simulation import SimulationSpec


# ---------------------------------------------------------------------------
# Pure-NumPy reference (single function, no VBI dependency)
# ---------------------------------------------------------------------------

def kuramoto_heun(
    theta0: np.ndarray,
    omega: np.ndarray,
    weights: np.ndarray,
    G: float,
    dt: float,
    n_steps: int,
    delays: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Integrate the Kuramoto model with Heun's method (frozen-coupling convention).

    This matches the VBI / TVB simulator convention: the coupling term
    c_i = (G/N) Σ_j W_{ij} sin(θ_j − θ_i)  is computed once at the start
    of each step from the current (or delayed) phases and held fixed for
    both stages of Heun.  Because the Kuramoto dfun is entirely the coupling
    term (dθ/dt = ω + c, no additional state-dependent nonlinearity), the
    frozen-coupling Heun reduces to a first-order Euler step.

    Parameters
    ----------
    theta0  : (N,)            initial phases [rad]
    omega   : (N,)            natural frequencies [rad/ms]
    weights : (N, N)          weight matrix, weights[tgt, src]
    G       : float           global coupling strength
    dt      : float           time step [ms]
    n_steps : int             number of integration steps
    delays  : (N, N) | None   axonal delays [ms]; None → zero delay

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
        # Fill all horizon slots with theta0 — matches VBI History.initialize()
        buf = np.tile(theta0, (horizon, 1))  # (horizon, N)
    else:
        delay_steps = None
        horizon = 1
        buf = None

    def compute_coupling(step: int, current_theta: np.ndarray) -> np.ndarray:
        """c[tgt] = (G/N) Σ_src W[tgt,src] sin(θ_src − θ_tgt)"""
        if buf is None:
            # zero delay: diff[src, tgt] = theta[src] - theta[tgt]
            diff = current_theta[:, np.newaxis] - current_theta[np.newaxis, :]
        else:
            # delayed: θ_src(t−τ_{src→tgt}) − θ_tgt(t)
            diff = np.empty((N, N))   # diff[src, tgt]
            t_last = step - 1
            for src in range(N):
                for tgt in range(N):
                    d = delay_steps[src, tgt]
                    idx = (t_last - d + horizon) % horizon
                    diff[src, tgt] = buf[idx, src] - current_theta[tgt]
        # c[tgt] = (G/N) * Σ_src W[tgt,src] * sin(diff[src,tgt])
        return (G / N) * (weights * np.sin(diff.T)).sum(axis=1)

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


# ---------------------------------------------------------------------------
# VBI runner
# ---------------------------------------------------------------------------

def run_vbi(
    theta0: np.ndarray,
    omega: np.ndarray,
    weights: np.ndarray,
    G: float,
    dt: float,
    duration: float,
    tract_lengths: np.ndarray | None,
    speed: float,
) -> tuple[np.ndarray, np.ndarray]:
    import dataclasses

    # Set per-node initial phases via StateVar.default_init (broadcast by _build_initial_state)
    sv_with_init = (dataclasses.replace(kuramoto.state_variables[0], default_init=theta0),)
    model_with_init = dataclasses.replace(kuramoto, state_variables=sv_with_init)

    spec = SimulationSpec(
        model=model_with_init,
        integrator=IntegratorSpec(method="heun", dt=dt),
        coupling=CouplingSpec(kind="kuramoto"),
        monitors=(MonitorSpec(kind="raw"),),
        weights=weights,
        tract_lengths=tract_lengths,
        speed=speed,
        node_params={"omega": omega, "G": G},
    )

    sim = Simulator(spec, backend="numpy")
    t, d = sim.run(duration)["raw"]
    theta = d[:, 0, :]  # (time, N)
    # Prepend t=0 / initial state so the time axis matches the reference
    t    = np.concatenate([[0.0], t])
    theta = np.vstack([theta0[np.newaxis, :], theta])
    return t, theta


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def save_comparison_plot(
    t: np.ndarray,
    vbi_theta: np.ndarray,
    ref_theta: np.ndarray,
    out_path: Path,
    delayed: bool,
) -> None:
    N = vbi_theta.shape[1]
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), tight_layout=True)
    colors = plt.cm.tab10(np.linspace(0, 0.9, N))
    subtitle = "  (delayed)" if delayed else ""

    # --- panel 1: sin(θ) per node ---
    ax = axes[0]
    for i in range(N):
        ax.plot(t, np.sin(ref_theta[:, i]),  color=colors[i], lw=1.5,
                label=f"node {i}")
        ax.plot(t, np.sin(vbi_theta[:, i]),  color=colors[i], lw=0.8,
                ls="--", alpha=0.7)
    ax.set_ylabel("sin(θ)")
    ax.set_ylim(-1.15, 1.15)
    ax.set_title(f"sin(θ) per node — solid: reference, dashed: VBI{subtitle}")
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
    plt.close(fig)
    print(f"saved figure: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n",        type=int,   default=5,      help="number of nodes")
    p.add_argument("--duration", type=float, default=100.0,  help="simulation time [ms]")
    p.add_argument("--dt",       type=float, default=0.01,   help="time step [ms]")
    p.add_argument("--G",        type=float, default=0.5,    help="global coupling strength")
    p.add_argument("--delayed",  action="store_true",        help="add 5 ms axonal delay")
    p.add_argument("--speed",    type=float, default=1.0,    help="axonal speed [mm/ms]")
    p.add_argument("--tract",    type=float, default=5.0,    help="tract length [mm]")
    p.add_argument("--seed",     type=int,   default=42)
    p.add_argument("--output", type=Path,
                   default=Path(__file__).with_name("outputs") / "kuramoto_comparison.png")
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
    )

    # align lengths (VBI may produce one fewer sample depending on rounding)
    n = min(len(t_ref), len(t_vbi))
    t_ref, theta_ref = t_ref[:n], theta_ref[:n]
    t_vbi, theta_vbi = t_vbi[:n], theta_vbi[:n]

    max_err = np.abs(theta_vbi - theta_ref).max()
    rms_err = np.sqrt(np.mean((theta_vbi - theta_ref) ** 2))

    print(f"Kuramoto  N={N}  G={args.G}  dt={args.dt} ms"
          f"{'  delayed=' + str(args.tract) + ' mm' if args.delayed else ''}")
    print(f"omega: {np.round(omega, 3)}")
    print(f"max |err|: {max_err:.3e}   rms |err|: {rms_err:.3e}")

    save_comparison_plot(t_ref, theta_vbi, theta_ref, args.output, args.delayed)


if __name__ == "__main__":
    main()
