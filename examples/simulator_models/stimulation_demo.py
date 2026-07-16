"""
Stimulation Demo
==================

External stimulation with the VBI NumPy backend: a 10 ms pulse train
targets a single node in an otherwise unconnected 5-node network, and the
stimulated trajectory is compared against an unstimulated control run.

Run
---
::

    python stimulation_demo.py
"""

# %%
# Setup
# -----

from __future__ import annotations

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
    _SCRIPT_PATH = Path.cwd() / "stimulation_demo.py"

# Prefer the vbi living in this checkout over any other version already
# installed (e.g. a different vbi checkout installed editable elsewhere).
# Downloaded standalone copies of this script should `pip install vbi`
# instead - see the first notebook cell.
_repo_root = _SCRIPT_PATH.resolve().parents[2]
sys.path.insert(0, str(_repo_root))

from vbi.simulator import Simulator
from vbi.simulator.models.generic_2d_oscillator import generic_2d_oscillator
from vbi.simulator.spec.coupling import CouplingSpec
from vbi.simulator.spec.integrator import IntegratorSpec
from vbi.simulator.spec.monitor import MonitorSpec
from vbi.simulator.spec.simulation import SimulationSpec
from vbi.simulator.spec.connectivity import Connectivity
from vbi.simulator.spec.stimulus import StimSpec

N_NODES = 5
DT = 0.1
DURATION = 200.0


def pulse_train(t_ms: float) -> float:
    """A 10 ms pulse every 40 ms, starting at 20 ms."""
    if t_ms < 20.0:
        return 0.0
    return 1.0 if (t_ms - 20.0) % 40.0 < 10.0 else 0.0


# %%
# VBI simulator
# -------------
# Stimulates only node 0; passing no stimulus gives the unstimulated control.

def build_spec(stimulus: StimSpec | None = None) -> SimulationSpec:
    weights = np.zeros((N_NODES, N_NODES), dtype=np.float64)
    return SimulationSpec(
        model=generic_2d_oscillator,
        integrator=IntegratorSpec(method="euler", dt=DT, stochastic=False),
        coupling=CouplingSpec(kind="linear", a=0.0),
        monitors=(MonitorSpec(kind="raw", variables=("V",)),),
        connectivity=Connectivity(weights),
        stimuli=() if stimulus is None else (stimulus,),
    )


# %%
# Run the demo
# -------------

def main() -> None:
    spatial_weights = np.zeros(N_NODES)
    spatial_weights[0] = 2.0
    stimulus = StimSpec(sv_name="V", amplitude=spatial_weights, waveform=pulse_train)

    t_stim, y_stim = Simulator(build_spec(stimulus), backend="numpy").run(DURATION)["raw"]
    t_ctrl, y_ctrl = Simulator(build_spec(), backend="numpy").run(DURATION)["raw"]

    stim_waveform = np.array([pulse_train(t) for t in t_stim])

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(t_stim, 0.0, 0.25 * stim_waveform, color="tab:orange", alpha=0.25, label="stimulus")
    ax.plot(t_stim, y_stim[:, 0, 0], color="tab:red", label="node 0, stimulated", alpha=0.5)
    ax.plot(t_ctrl, y_ctrl[:, 0, 0], color="0.4", ls="--", label="node 0, control", alpha=0.5)
    ax.plot(t_stim, y_stim[:, 0, 4], color="tab:blue", label="node 4, unstimulated", alpha=0.5)
    ax.set_xlabel("time [ms]")
    ax.set_ylabel("V")
    ax.set_title("Pulse-train stimulation with VBI numpy backend")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    out_path = _SCRIPT_PATH.with_name("outputs") / "stimulation_comparison.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)

    print("VBI numpy stimulation demo")
    print(f"raw shape: {y_stim.shape}  # (time, variable, node)")
    print(f"saved figure: {out_path}")


if __name__ == "__main__":
    main()
