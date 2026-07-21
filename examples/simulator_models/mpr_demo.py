"""
MPR Model Demo
==============

Compares the VBI simulator's Montbrio-Pazo-Roxin (MPR) mean-field model
against The Virtual Brain (TVB)'s own MPR implementation, using the same
Heun integrator and connectivity. Requires the optional ``tvb-library``
package.

Run
---
::

    python mpr_demo.py                  # 6-node complete graph, 40 ms
    python mpr_demo.py --duration 100
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
    _SCRIPT_PATH = Path.cwd() / "mpr_demo.py"

# Prefer the vbi living in this checkout over any other version already
# installed (e.g. a different vbi checkout installed editable elsewhere).
# Downloaded standalone copies of this script should `pip install vbi`
# instead - see the first notebook cell.
_repo_root = _SCRIPT_PATH.resolve().parents[2]
sys.path.insert(0, str(_repo_root))

from vbi.simulator import Simulator
from vbi.simulator.models.mpr import mpr
from vbi.simulator.spec.coupling import CouplingSpec
from vbi.simulator.spec.integrator import IntegratorSpec
from vbi.simulator.spec.monitor import MonitorSpec
from vbi.simulator.spec.simulation import SimulationSpec
from vbi.simulator.spec.connectivity import Connectivity

N_NODES = 6
G = 0.33
DT = 0.01
TRACT_LENGTH = 4.0
SPEED = 4.0


def complete_graph_weights(n_nodes: int) -> np.ndarray:
    """Return a dense complete-graph weight matrix with zero diagonal."""
    weights = np.ones((n_nodes, n_nodes), dtype=np.float64)
    np.fill_diagonal(weights, 0.0)
    return weights


def complete_graph_tract_lengths(weights: np.ndarray) -> np.ndarray:
    """Return tract lengths matching the nonzero entries of ``weights``."""
    tract_lengths = np.full_like(weights, TRACT_LENGTH)
    np.fill_diagonal(tract_lengths, 0.0)
    return tract_lengths


# %%
# VBI simulator
# -------------
# ``I`` is bumped above the model's default (0) to push MPR into an
# oscillatory regime; all other parameters use the model's own defaults.

def run_vbi(duration: float) -> tuple[np.ndarray, np.ndarray]:
    weights = complete_graph_weights(N_NODES)
    tract_lengths = complete_graph_tract_lengths(weights)

    spec = SimulationSpec(
        model=mpr,
        integrator=IntegratorSpec(method="heun", dt=DT),
        coupling=CouplingSpec(kind="linear", a=1.0, b=0.0),
        monitors=(MonitorSpec(kind="raw"),),
        connectivity=Connectivity(weights, tract_lengths, speed=SPEED),
        node_params={"I": 2.0, "G": G},
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
        from tvb.simulator.models.infinite_theta import MontbrioPazoRoxin
        from tvb.simulator.monitors import Raw
        from tvb.simulator.simulator import Simulator as TVBSimulator
    except ImportError as exc:
        raise RuntimeError("TVB comparison requires the 'tvb-library' package") from exc

    weights = complete_graph_weights(N_NODES)
    tract_lengths = complete_graph_tract_lengths(weights)

    conn = TVBConnectivity(
        weights=weights,
        tract_lengths=tract_lengths,
        region_labels=np.array([str(i) for i in range(N_NODES)]),
        centres=np.zeros((N_NODES, 3)),
        speed=np.array([SPEED]),
    )
    conn.configure()

    model = MontbrioPazoRoxin(
        tau=np.array([1.0]), I=np.array([2.0]), Delta=np.array([0.7]),
        J=np.array([14.5]), eta=np.array([-4.6]), Gamma=np.array([0.0]),
        cr=np.array([1.0]), cv=np.array([0.0]),
    )
    sim = TVBSimulator(
        connectivity=conn,
        model=model,
        coupling=Linear(a=np.array([G])),
        integrator=HeunDeterministic(dt=DT),
        monitors=[Raw()],
        simulation_length=duration,
    ).configure()

    # Match VBI's default initial state (r=0, V=-2.0; see mpr.state_variables).
    initial_state = np.zeros((model.nvar, N_NODES, 1))
    initial_state[1, :, 0] = -2.0
    sim.current_state[:] = initial_state
    sim.history.buffer[:] = initial_state[model.cvar][np.newaxis, ...]

    (_times, data), = sim.run()
    return data[:, :, :, 0]  # (time, n_sv, n_nodes)


# %%
# Comparison plot
# ---------------
# Overlays the firing-rate variable ``r`` for every node.

def comparison_plot(t: np.ndarray, vbi_data: np.ndarray, tvb_data: np.ndarray,
                    out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.tab10(np.linspace(0, 0.9, vbi_data.shape[-1]))
    for i in range(vbi_data.shape[-1]):
        ax.plot(t, tvb_data[:, 0, i], color=colors[i], lw=1.5, label=f"node {i}")
        ax.plot(t, vbi_data[:, 0, i], color=colors[i], lw=0.8, ls="--")
    ax.set_xlabel("time [ms]")
    ax.set_ylabel("r (firing rate)")
    ax.set_title("MPR firing rate r(t) - solid: TVB, dashed: VBI")
    ax.legend(fontsize=8, ncol=vbi_data.shape[-1], loc="upper right")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    print(f"saved figure: {out_path}")


# %%
# Run the comparison
# -------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--duration", type=float, default=40.0, help="simulation time [ms]")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t, vbi_data = run_vbi(args.duration)
    tvb_data = run_tvb(args.duration)

    n = min(len(t), tvb_data.shape[0])
    t, vbi_data, tvb_data = t[:n], vbi_data[:n], tvb_data[:n]

    err = np.abs(vbi_data - tvb_data)
    print(f"MPR  nodes={N_NODES}  G={G}  dt={DT} ms  duration={args.duration} ms")
    print(f"max |err|: {err.max():.3e}   rms |err|: {np.sqrt((err ** 2).mean()):.3e}")

    comparison_plot(t, vbi_data, tvb_data,
                    _SCRIPT_PATH.with_name("outputs") / "mpr_comparison.png")


if __name__ == "__main__":
    main()
