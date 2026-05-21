"""Simple external-stimulation demo for the VBI NumPy backend.

Run from the repository root:

    python docs/examples/simulator_models/stimulation_demo.py
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True

from helpers import ensure_repo_on_path, quiet_optional_imports

ensure_repo_on_path(__file__)
os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "vbi_matplotlib"))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

with quiet_optional_imports():
    from vbi.simulator import Simulator
    from vbi.simulator.models.generic_2d_oscillator import generic_2d_oscillator
    from vbi.simulator.spec.coupling import CouplingSpec
    from vbi.simulator.spec.integrator import IntegratorSpec
    from vbi.simulator.spec.monitor import MonitorSpec
    from vbi.simulator.spec.simulation import SimulationSpec
    from vbi.simulator.spec.stimulus import StimSpec


N_NODES = 5
DT = 0.1
DURATION = 200.0


def build_spec(stimulus: StimSpec | None = None) -> SimulationSpec:
    weights = np.zeros((N_NODES, N_NODES), dtype=np.float64)
    return SimulationSpec(
        model=generic_2d_oscillator,
        integrator=IntegratorSpec(method="euler", dt=DT, stochastic=False),
        coupling=CouplingSpec(kind="linear", a=0.0),
        monitors=(MonitorSpec(kind="raw", variables=("V",)),),
        weights=weights,
        tract_lengths=np.zeros_like(weights),
        stimuli=() if stimulus is None else (stimulus,),
    )


def pulse_train(t_ms: float) -> float:
    """A 10 ms pulse every 40 ms, starting at 20 ms."""
    if t_ms < 20.0:
        return 0.0
    return 1.0 if (t_ms - 20.0) % 40.0 < 10.0 else 0.0


def main() -> None:
    # Spatial pattern: stimulate only node 0.
    spatial_weights = np.zeros(N_NODES)
    spatial_weights[0] = 2.0

    stimulus = StimSpec(
        sv_name="V",                # must be in models.cvar
        amplitude=spatial_weights,
        waveform=pulse_train,
    )

    t_stim, y_stim = Simulator(build_spec(stimulus), backend="numpy").run(DURATION)["raw"]
    t_ctrl, y_ctrl = Simulator(build_spec(), backend="numpy").run(DURATION)["raw"]

    out_path = Path(__file__).with_name("outputs") / "stimulation_demo.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stim_waveform = np.array([pulse_train(t) for t in t_stim])

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(t_stim, 0.0, 0.25 * stim_waveform, color="tab:orange", alpha=0.25, label="stimulus")
    ax.plot(t_stim, y_stim[:, 0, 0], color="tab:red", label="node 0, stimulated", alpha=0.5)
    ax.plot(t_ctrl, y_ctrl[:, 0, 0], color="0.4", ls="--", label="node 0, control", alpha=0.5)
    ax.plot(t_stim, y_stim[:, 0, 4], color="tab:blue", label="node 4, unstimulated", alpha=0.5)
    ax.set_xlabel("time [ms]")
    ax.set_ylabel("V")
    ax.set_title("Pulse-train stimulation with VBI NumPy backend")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

    print("VBI NumPy stimulation demo")
    print(f"raw shape: {y_stim.shape}  # (time, variable, node)")
    print(f"saved figure: {out_path}")


if __name__ == "__main__":
    main()
