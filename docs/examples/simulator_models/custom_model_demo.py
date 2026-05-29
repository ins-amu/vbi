"""
Custom model demo — adding a user-defined model to the VBI simulator.

This script shows the full workflow:

  1. Define a custom ModelSpec (FitzHugh-Nagumo as a concrete example).
  2. Register it so SimulationSpec.from_dict() / from_config() can find it.
  3. Run it directly with Simulator and with a parameter sweep.
  4. Use it via a config dict (as would come from a YAML file).

Run
---
    python custom_model_demo.py
    python custom_model_demo.py --duration 200 --backend numba
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True

# Make sure the package root is importable when running from the examples dir.
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

# ---------------------------------------------------------------------------
# Step 1 — Define the custom model
# ---------------------------------------------------------------------------
#
# ModelSpec is the backend-agnostic description understood by every VBI
# simulator backend (numpy, numba, CUDA, JAX, C++).
#
# dfun_str uses plain math with these symbols:
#   - state variable names (defined in state_variables)
#   - parameter names      (defined in parameters)
#   - 'c'                  the net coupling input at this node
#   - exp, log, sin, cos, tanh, sqrt, abs, pi
#
# No 'np.' prefix — the code generator handles the namespace.

from vbi.simulator.spec.model import ModelSpec, StateVar, Parameter

fitzhugh_nagumo = ModelSpec(
    name="FitzHughNagumo",
    state_variables=(
        # v: fast membrane variable; bounded to avoid runaway
        StateVar("v", default_init=0.0, noise=True,
                 lower_bound=-2.5, upper_bound=2.5),
        # w: slow recovery variable
        StateVar("w", default_init=0.0, noise=False),
    ),
    parameters=(
        Parameter("a",   0.7,   "threshold parameter"),
        Parameter("b",   0.8,   "recovery rate"),
        Parameter("tau", 12.5,  "time-scale separation (1/tau multiplies dw/dt)"),
        Parameter("I",   0.5,   "external current"),
    ),
    # v is the coupling variable — it enters other nodes' equations via c.
    cvar=("v",),
    dfun_str={
        "v": "v - (v**3) / 3.0 - w + I + c",
        "w": "(v + a - b * w) / tau",
    },
    noise_variables=("v",),
    reference="FitzHugh (1961) / Nagumo et al. (1962)",
    dfun_latex={
        "v": r"\dot{v} = v - \frac{v^3}{3} - w + I + c^{\rm net}",
        "w": r"\dot{w} = \frac{v + a - b\,w}{\tau}",
    },
    latex_notes=(
        r"$c^{\rm net} = G\sum_j W_{ij}\,v_j$."
        r" Oscillatory regime: $|I| > (a - 1)$ approximately."
    ),
)

# ---------------------------------------------------------------------------
# Step 2 — Register the model
# ---------------------------------------------------------------------------
#
# register_model() adds the model to the global lookup table used by
# SimulationSpec.from_dict() and VBIInference.from_config().
# Call it once, before any from_dict()/from_config() that references this model.
#
# The first alias you supply becomes the string to use in the config file:
#
#   sim:
#     model: fitzhugh_nagumo
#
# The model's own .name attribute ("FitzHughNagumo") is also registered
# automatically.

from vbi.simulator.spec.simulation import register_model

register_model(fitzhugh_nagumo, "fitzhugh_nagumo", "fhn")

print("Registered model:", fitzhugh_nagumo.name)
print("  state variables:", fitzhugh_nagumo.sv_names)
print("  parameters     :", fitzhugh_nagumo.param_names)
print("  coupling vars  :", fitzhugh_nagumo.cvar)
print()


# ---------------------------------------------------------------------------
# Step 3 — Run directly with Simulator
# ---------------------------------------------------------------------------

import argparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--duration", type=float, default=500.0,
                    help="simulation length in ms")
parser.add_argument("--backend", default="numpy",
                    choices=("numpy", "numba"),
                    help="simulator backend")
parser.add_argument("--n-nodes", type=int, default=6)
args = parser.parse_args()

from vbi.simulator import Simulator
from vbi.simulator.spec import (
    SimulationSpec, IntegratorSpec, CouplingSpec, MonitorSpec,
)

rng = np.random.default_rng(0)
W = np.abs(rng.standard_normal((args.n_nodes, args.n_nodes)))
np.fill_diagonal(W, 0.0)
W /= W.sum(axis=1, keepdims=True).clip(1e-8)

spec = SimulationSpec(
    model       = fitzhugh_nagumo,
    integrator  = IntegratorSpec(method="heun", dt=0.1,
                                 stochastic=True,
                                 noise_nsig=np.array([0.01])),
    coupling    = CouplingSpec("linear", a=0.05),
    monitors    = (MonitorSpec("tavg", period=1.0),),
    weights     = W,
)

t, data = Simulator(spec, backend=args.backend).run(args.duration)["tavg"]
print(f"Direct run   backend={args.backend!r}  duration={args.duration} ms")
print(f"  output shape : {data.shape}  # (time, sv, nodes)")
print(f"  v mean ± std : {data[:, 0, :].mean():.3f} ± {data[:, 0, :].std():.3f}")
print()


# ---------------------------------------------------------------------------
# Step 4 — Parameter sweep
# ---------------------------------------------------------------------------

from vbi.simulator import Sweeper
from vbi.simulator.spec.sweep import SweepSpec

sweep_spec = SweepSpec(params={"I": np.linspace(0.0, 1.5, 6)})
results = Sweeper(spec, sweep_spec, backend=args.backend).run(args.duration)

print(f"Sweep over I  ({len(results)} samples)")
for i, res in enumerate(results):
    _, d = res["tavg"]
    print(f"  I={sweep_spec.param_sets[i, 0]:.2f}  v_mean={d[:, 0, :].mean():.3f}")
print()


# ---------------------------------------------------------------------------
# Step 5 — Use via config dict  (equivalent to loading a YAML file)
# ---------------------------------------------------------------------------
#
# After register_model(), any config-dict or YAML file can reference the
# model by the registered alias.  This is the path used by
# VBIInference.from_config() as well.

config = {
    "model": "fitzhugh_nagumo",    # registered alias
    "connectivity": {              # inline weights
        "weights": W.tolist(),
    },
    "dt": 0.1,
    "method": "heun",
    "stochastic": True,
    "noise_nsig": [0.01],
    "coupling": {"kind": "linear", "a": 0.05},
    "monitors": [{"kind": "tavg", "period": 1.0}],
    "node_params": {"I": 0.5},
}

spec_from_cfg = SimulationSpec.from_dict(config)
t2, data2 = Simulator(spec_from_cfg, backend=args.backend).run(args.duration)["tavg"]
print(f"Config-driven run  model={spec_from_cfg.model.name!r}")
print(f"  output shape : {data2.shape}")
print(f"  v mean ± std : {data2[:, 0, :].mean():.3f} ± {data2[:, 0, :].std():.3f}")
print()

print("Done.  To use the custom model in a YAML config, add:\n")
print("  # register_model() must be called once before from_config()")
print("  from vbi.simulator.spec.simulation import register_model")
print("  register_model(fitzhugh_nagumo, 'fitzhugh_nagumo', 'fhn')")
print()
print("  sim:")
print("    model: fitzhugh_nagumo")
print("    ...")
