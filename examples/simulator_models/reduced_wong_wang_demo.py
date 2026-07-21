"""
Reduced Wong-Wang BOLD + FCD Demo
===================================

Reproduces the BOLD + FCD (functional connectivity dynamics) analysis from
``rww_sde_numba.ipynb`` using the current :mod:`vbi.simulator` API on an
84-node structural connectome.

Run
---
::

    python reduced_wong_wang_demo.py
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
    _SCRIPT_PATH = Path.cwd() / "reduced_wong_wang_demo.py"

# Prefer the vbi living in this checkout over any other version already
# installed (e.g. a different vbi checkout installed editable elsewhere).
# Downloaded standalone copies of this script should `pip install vbi`
# instead - see the first notebook cell.
_repo_root = _SCRIPT_PATH.resolve().parents[2]
sys.path.insert(0, str(_repo_root))

import vbi
from vbi.simulator import Simulator
from vbi.simulator.models.reduced_wong_wang import reduced_wong_wang
from vbi.simulator.spec.coupling import CouplingSpec
from vbi.simulator.spec.integrator import IntegratorSpec
from vbi.simulator.spec.monitor import MonitorSpec
from vbi.simulator.spec.simulation import SimulationSpec
from vbi.simulator.spec.connectivity import Connectivity
from vbi.feature_extraction.features import get_fcd

SEED = 42
G = 1.2                     # global coupling strength
I_O = 0.05                  # external input
W_REC = 1.0                 # local recurrence (vbi.simulator default is 0.6)
SIGMA = 0.05                # noise amplitude
DT = 2.5                    # ms
DURATION = 5 * 60 * 1000.0  # ms (5 min)
T_CUT = 1 * 60 * 1000.0     # ms (1 min burn-in)
TR = 300.0                  # ms BOLD repetition time


# %%
# VBI simulator
# -------------

def run_vbi(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nn = weights.shape[0]
    spec = SimulationSpec(
        model=reduced_wong_wang,
        integrator=IntegratorSpec(
            method="heun", dt=DT, stochastic=True,
            noise_nsig=np.array([SIGMA]),
            noise_style="amplitude", noise_seed=SEED,
        ),
        coupling=CouplingSpec(kind="linear", a=G),
        monitors=(MonitorSpec(kind="bold", tr=TR),),
        connectivity=Connectivity(weights),
        node_params={"I_o": np.full(nn, I_O), "w": np.full(nn, W_REC)},
    )
    bold_t, bold_d = Simulator(spec, backend="numpy").run(duration=DURATION)["bold"]
    return bold_t, bold_d[:, 0, :]  # squeeze the single state-variable axis


# %%
# Run the demo
# -------------
# Simulates BOLD on an 84-node connectome, trims the burn-in period, and
# computes the FCD (functional connectivity dynamics) matrix.

def main() -> None:
    weights = vbi.LoadSample(nn=84).get_weights()
    nn = weights.shape[0]

    print(f"Simulating {nn} nodes, T={DURATION / 1000:.0f}s, dt={DT} ms ...")
    bold_t, bold_d = run_vbi(weights)

    cut_idx = int(T_CUT / TR)
    bold_t, bold_d = bold_t[cut_idx:], bold_d[cut_idx:]
    print(f"BOLD shape after trim: {bold_d.shape}  ({bold_d.shape[0]} time points x {nn} nodes)")

    fcd = get_fcd(bold_d.T, win_len=30, tr=TR / 1000.0)["full"]

    fig = plt.figure(figsize=(10, 3.5))
    ax1 = plt.subplot(121)
    ax1.plot(bold_t / 1000, bold_d, lw=1, alpha=0.2, color="C1")
    ax1.set_xlabel("Time (s)")
    ax1.margins(x=0, y=0.01)

    ax2 = plt.subplot(122)
    im = ax2.imshow(fcd, cmap="viridis", aspect="equal")
    plt.colorbar(im, ax=ax2)
    ax2.set_xlabel("Time shift")
    ax2.set_ylabel("Time shift")

    out_path = _SCRIPT_PATH.with_name("outputs") / "reduced_wong_wang_comparison.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"saved figure: {out_path}")


if __name__ == "__main__":
    main()
