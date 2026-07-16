"""
Wilson-Cowan Model Demo
=========================

Reproduces the first Wilson-Cowan SDE notebook visualization: a two-node
excitatory-inhibitory oscillator driven into a limit cycle by external
drive ``P``, integrated with additive noise. Plots the stochastic
trajectory, its power spectrum, and the E-I phase plane.

Run
---
::

    python wilson_cowan_demo.py
    python wilson_cowan_demo.py --duration 4000
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
    _SCRIPT_PATH = Path.cwd() / "wilson_cowan_demo.py"

# Prefer the vbi living in this checkout over any other version already
# installed (e.g. a different vbi checkout installed editable elsewhere).
# Downloaded standalone copies of this script should `pip install vbi`
# instead - see the first notebook cell.
_repo_root = _SCRIPT_PATH.resolve().parents[2]
sys.path.insert(0, str(_repo_root))

from vbi.simulator import Simulator
from vbi.simulator.models.wilson_cowan import wilson_cowan
from vbi.simulator.spec.coupling import CouplingSpec
from vbi.simulator.spec.integrator import IntegratorSpec
from vbi.simulator.spec.monitor import MonitorSpec
from vbi.simulator.spec.simulation import SimulationSpec
from vbi.simulator.spec.connectivity import Connectivity

DT = 0.05
NOISE_AMP = 0.0005
NOISE_SEED = 42
T_CUT = 101.0  # drop this much initial transient before analysis/plotting

# Working point that pushes the model into a stable limit cycle; every
# parameter not listed here uses wilson_cowan's own default.
PARAM_OVERRIDES = {
    "c_ee": 16.0, "c_ei": 12.0, "c_ie": 15.0, "c_ii": 3.0,
    "tau_e": 8.0, "tau_i": 8.0,
    "a_e": 1.3, "b_e": 4.0, "a_i": 2.0, "b_i": 3.7,
    "k_e": 0.994, "k_i": 0.999,
    "P": 1.025, "shift_sigmoid": 0.0,
}


# %%
# VBI simulator
# -------------
# Two nodes, coupling strength zero, so each evolves independently under
# its own noise realization.

def run_vbi(duration: float) -> tuple[np.ndarray, np.ndarray]:
    weights = np.array([[0.0, 1.0], [1.0, 0.0]])

    spec = SimulationSpec(
        model=wilson_cowan,
        integrator=IntegratorSpec(
            method="heun", dt=DT, stochastic=True,
            noise_nsig=np.array([NOISE_AMP, NOISE_AMP]),
            noise_style="amplitude", noise_seed=NOISE_SEED,
        ),
        coupling=CouplingSpec(kind="linear", a=0.0),
        monitors=(MonitorSpec(kind="raw"),),
        connectivity=Connectivity(weights, speed=1.0),
        node_params=PARAM_OVERRIDES,
    )
    return Simulator(spec, backend="numpy").run(duration)["raw"]


# %%
# Analysis helpers
# ----------------

def spectrum(signal: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the Welch power spectrum of a one-dimensional signal."""
    signal = signal - np.mean(signal)
    from scipy.signal import welch
    return welch(signal, fs=1000.0 / dt, nperseg=min(4096, signal.size))


def after_transient(times: np.ndarray, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop the initial transient (t < T_CUT) before analysis/plotting."""
    keep = times >= T_CUT
    return times[keep], data[keep]


# %%
# Comparison plot
# ---------------
# Time series (E/I), Welch power spectrum, and the E-I phase plane.

def comparison_plot(times: np.ndarray, data: np.ndarray, out_path: Path) -> None:
    post_times, post_data = after_transient(times, data)
    post_e, post_i = post_data[:, 0, 0], post_data[:, 1, 0]
    freqs_e, power_e = spectrum(post_e, DT)
    freqs_i, power_i = spectrum(post_i, DT)

    fig = plt.figure(constrained_layout=True, figsize=(11, 6))
    axes = fig.subplot_mosaic("AA\nBC")

    axes["A"].plot(post_times, post_e, color="tab:red", lw=0.8, label="E")
    axes["A"].plot(post_times, post_i, color="tab:blue", lw=0.8, label="I")
    axes["A"].set_ylabel("activity")
    axes["A"].set_title("Wilson-Cowan stochastic trajectory")
    axes["A"].legend(frameon=False)
    axes["A"].grid(True, alpha=0.25)

    axes["B"].plot(freqs_e, power_e, color="tab:red", lw=1.0, label="E")
    axes["B"].plot(freqs_i, power_i, color="tab:blue", lw=1.0, label="I")
    axes["B"].set_xlim(0, 100)
    axes["B"].set_xlabel("frequency [Hz]")
    axes["B"].set_ylabel("power")
    axes["B"].legend(frameon=False)
    axes["B"].grid(True, alpha=0.25)

    axes["C"].plot(post_e, post_i, color="black", lw=0.6, alpha=0.75)
    axes["C"].set_xlabel("E")
    axes["C"].set_ylabel("I")
    axes["C"].grid(True, alpha=0.25)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    print(f"saved figure: {out_path}")


# %%
# Run the demo
# -------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--duration", type=float, default=2000.0, help="simulation time [ms]")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    times, data = run_vbi(args.duration)

    _post_times, post_data = after_transient(times, data)
    freqs, power = spectrum(post_data[:, 0, 0], DT)
    nonzero = freqs > 0.0
    peak_freq = float(freqs[nonzero][np.argmax(power[nonzero])]) if np.any(nonzero) else 0.0

    print(f"Wilson-Cowan  nodes=2  P={PARAM_OVERRIDES['P']}  dt={DT} ms  duration={args.duration} ms")
    print(f"trajectory shape: {data.shape}  # (time, variable, node)")
    print(f"peak E frequency: {peak_freq:.3f} Hz")

    comparison_plot(times, data,
                    _SCRIPT_PATH.with_name("outputs") / "wilson_cowan_comparison.png")


if __name__ == "__main__":
    main()
