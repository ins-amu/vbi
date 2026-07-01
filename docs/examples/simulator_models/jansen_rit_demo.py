"""
Jansen-Rit Model Demo
=======================

Compares the VBI simulator's Jansen-Rit cortical-column model against The
Virtual Brain (TVB)'s own Jansen-Rit implementation, using the same Heun
integrator and connectivity. Requires the optional ``tvb-library`` package.

Run
---
::

    python jansen_rit_demo.py                  # 6-node normalized complete graph
    python jansen_rit_demo.py --duration 100
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
    _SCRIPT_PATH = Path.cwd() / "jansen_rit_demo.py"

# Prefer the vbi living in this checkout over any other version already
# installed (e.g. a different vbi checkout installed editable elsewhere).
# Downloaded standalone copies of this script should `pip install vbi`
# instead - see the first notebook cell.
_repo_root = _SCRIPT_PATH.resolve().parents[3]
sys.path.insert(0, str(_repo_root))

from vbi.simulator import Simulator
from vbi.simulator.models.jansen_rit import jansen_rit
from vbi.simulator.spec.coupling import CouplingSpec
from vbi.simulator.spec.integrator import IntegratorSpec
from vbi.simulator.spec.monitor import MonitorSpec
from vbi.simulator.spec.simulation import SimulationSpec
from vbi.simulator.spec.connectivity import Connectivity

N_NODES = 6
COUPLING_STRENGTH = 0.01
DT = 0.05
# Model parameters that differ from jansen_rit's own defaults.
PARAM_OVERRIDES = {"a_2": 1.0, "mu": 0.04}


def normalized_complete_graph(n_nodes: int) -> np.ndarray:
    """Complete graph with each row normalized to sum to 1."""
    weights = np.ones((n_nodes, n_nodes), dtype=np.float64)
    np.fill_diagonal(weights, 0.0)
    return weights / weights.sum(axis=1, keepdims=True)


# %%
# VBI simulator
# -------------

def run_vbi(duration: float) -> tuple[np.ndarray, np.ndarray]:
    weights = normalized_complete_graph(N_NODES)

    spec = SimulationSpec(
        model=jansen_rit,
        integrator=IntegratorSpec(method="heun", dt=DT),
        coupling=CouplingSpec(kind="linear", a=COUPLING_STRENGTH),
        monitors=(MonitorSpec(kind="raw"),),
        connectivity=Connectivity(weights, speed=1.0),
        node_params=PARAM_OVERRIDES,
    )
    return Simulator(spec, backend="numpy").run(duration)["raw"]


# %%
# TVB reference
# -------------

def run_tvb(duration: float) -> np.ndarray:
    try:
        from tvb.datatypes.connectivity import Connectivity as TVBConnectivity
        from tvb.simulator.coupling import Linear
        from tvb.simulator.integrators import HeunDeterministic
        from tvb.simulator.models.jansen_rit import JansenRit
        from tvb.simulator.monitors import Raw
        from tvb.simulator.simulator import Simulator as TVBSimulator
    except ImportError as exc:
        raise RuntimeError("TVB comparison requires the 'tvb-library' package") from exc

    weights = normalized_complete_graph(N_NODES)

    conn = TVBConnectivity(
        weights=weights,
        tract_lengths=np.zeros_like(weights),
        region_labels=np.array([str(i) for i in range(N_NODES)]),
        centres=np.zeros((N_NODES, 3)),
        speed=np.array([1.0]),
    )
    conn.configure()

    model = JansenRit(
        A=np.array([3.25]), B=np.array([22.0]), a=np.array([0.1]), b=np.array([0.05]),
        v0=np.array([5.52]), nu_max=np.array([0.0025]), r=np.array([0.56]), J=np.array([135.0]),
        a_1=np.array([1.0]), a_2=np.array([PARAM_OVERRIDES["a_2"]]),
        a_3=np.array([0.25]), a_4=np.array([0.25]), mu=np.array([PARAM_OVERRIDES["mu"]]),
        variables_of_interest=("y0", "y1", "y2", "y3", "y4", "y5"),
    )
    sim = TVBSimulator(
        connectivity=conn,
        model=model,
        coupling=Linear(a=np.array([COUPLING_STRENGTH])),
        integrator=HeunDeterministic(dt=DT),
        monitors=[Raw()],
        simulation_length=duration,
    ).configure()

    initial_state = np.zeros((model.nvar, N_NODES, 1))
    sim.current_state[:] = initial_state
    sim.history.buffer[:] = initial_state[model.cvar][np.newaxis, ...]

    (_times, data), = sim.run()
    return data[:, :, :, 0]  # (time, n_sv, n_nodes)


# %%
# Comparison plot
# ---------------
# ``y0`` and the "EEG-like" signal ``y1 - y2``, overlaid for every node.

def comparison_plot(t: np.ndarray, vbi_data: np.ndarray, tvb_data: np.ndarray,
                    out_path: Path) -> None:
    vbi_eeg = vbi_data[:, 1, :] - vbi_data[:, 2, :]
    tvb_eeg = tvb_data[:, 1, :] - tvb_data[:, 2, :]

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    colors = plt.cm.tab10(np.linspace(0, 0.9, N_NODES))
    for i in range(N_NODES):
        axes[0].plot(t, tvb_data[:, 0, i], color=colors[i], lw=1.5, label=f"node {i}")
        axes[0].plot(t, vbi_data[:, 0, i], color=colors[i], lw=0.8, ls="--")
        axes[1].plot(t, tvb_eeg[:, i], color=colors[i], lw=1.5)
        axes[1].plot(t, vbi_eeg[:, i], color=colors[i], lw=0.8, ls="--")
    axes[0].set_ylabel("y0")
    axes[0].set_title("Jansen-Rit - solid: TVB, dashed: VBI")
    axes[0].legend(fontsize=8, ncol=N_NODES, loc="upper right")
    axes[1].set_ylabel("y1 - y2 (EEG-like)")
    axes[1].set_xlabel("time [ms]")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    print(f"saved figure: {out_path}")


# %%
# Run the comparison
# -------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--duration", type=float, default=25.0, help="simulation time [ms]")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t, vbi_data = run_vbi(args.duration)
    tvb_data = run_tvb(args.duration)

    n = min(len(t), tvb_data.shape[0])
    t, vbi_data, tvb_data = t[:n], vbi_data[:n], tvb_data[:n]

    err = np.abs(vbi_data - tvb_data)
    print(f"Jansen-Rit  nodes={N_NODES}  coupling={COUPLING_STRENGTH}  dt={DT} ms  duration={args.duration} ms")
    print(f"max |err|: {err.max():.3e}   rms |err|: {np.sqrt((err ** 2).mean()):.3e}")

    comparison_plot(t, vbi_data, tvb_data,
                    _SCRIPT_PATH.with_name("outputs") / "jansen_rit_comparison.png")


if __name__ == "__main__":
    main()
