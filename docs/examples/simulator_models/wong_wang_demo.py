"""Simulate full excitatory-inhibitory Wong-Wang BOLD and FCD.

This mirrors rww_bold_fcd.py, but uses the full Wong-Wang excitatory-
inhibitory model. The BOLD monitor is driven by the excitatory gating
variable S_e, which is the full model's long-range coupled variable.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.dont_write_bytecode = True
warnings.filterwarnings("ignore")

from helpers import ensure_repo_on_path

ensure_repo_on_path(__file__)

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import vbi
from vbi.feature_extraction.features import get_fcd
from vbi.simulator import Simulator
from vbi.simulator.models.wong_wang_exc_inh import wong_wang_exc_inh
from vbi.simulator.spec.coupling import CouplingSpec
from vbi.simulator.spec.integrator import IntegratorSpec
from vbi.simulator.spec.monitor import MonitorSpec
from vbi.simulator.spec.simulation import SimulationSpec


D = vbi.LoadSample(nn=84)
weights = D.get_weights()
nn = weights.shape[0]

seed = 42
dt = 1.0                  # ms
T = 5 * 60 * 1000.0       # ms  (5 min)
t_cut = 1 * 60 * 1000.0   # ms  (1 min burn-in)
tr = 300.0                # ms  BOLD repetition time
sigma = 0.005             # noise amplitude for S_e and S_i

# Full Wong-Wang parameters from the simulator demo.
params = {
    "G": 1.91,
    "a_e": 310.0,
    "b_e": 125.0,
    "d_e": 0.160,
    "gamma_e": 0.641 / 1000,
    "tau_e": 100.0,
    "w_p": 1.4,
    "W_e": 1.0,
    "J_N": 0.15,
    "a_i": 615.0,
    "b_i": 177.0,
    "d_i": 0.087,
    "gamma_i": 1.0 / 1000,
    "tau_i": 10.0,
    "J_i": 1.0,
    "W_i": 0.7,
    "I_o": 0.382,
    "I_ext": 0.0,
    "lamda": 0.0,
}

node_params = {
    name: (value if name == "G" else np.full(nn, value))
    for name, value in params.items()
}

spec = SimulationSpec(
    model=wong_wang_exc_inh,
    integrator=IntegratorSpec(
        method="heun",
        dt=dt,
        stochastic=True,
        noise_nsig=np.array([sigma, sigma]),
        noise_style="amplitude",
        noise_seed=seed,
    ),
    coupling=CouplingSpec(kind="linear", a=1.0),
    monitors=(MonitorSpec(kind="bold", variables=("S_e",), tr=tr),),
    weights=weights,
    tract_lengths=np.zeros_like(weights),
    speed=1.0,
    node_params=node_params,
)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
print(f"Simulating full Wong-Wang, {nn} nodes, T={T/1000:.0f}s, dt={dt}ms ...")
result = Simulator(spec, backend="numpy").run(duration=T)
bold_t, bold_d = result["bold"]  # (n_steps,), (n_steps, nn)

# Trim burn-in
cut_idx = int(t_cut / tr)
bold_t = bold_t[cut_idx:]
bold_d = bold_d[cut_idx:]

print(f"BOLD shape after trim: {bold_d.shape}  ({bold_d.shape[0]} time points x {nn} nodes)")

fcd = get_fcd(bold_d.T, win_len=30, tr=tr / 1000.0)["full"]

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
out_path = Path(__file__).with_name("outputs") / "wong_wang_demo.png"
out_path.parent.mkdir(exist_ok=True)

fig = plt.figure(figsize=(10, 3.5))
ax1 = plt.subplot(121)
ax1.plot(bold_t / 1000, bold_d, lw=1, alpha=0.2, color="C0")
ax1.set_xlabel("Time (s)")
ax1.margins(x=0, y=0.01)

ax2 = plt.subplot(122)
im = ax2.imshow(fcd, cmap="viridis", aspect="equal")
plt.colorbar(im, ax=ax2)
ax2.set_xlabel("Time shift")
ax2.set_ylabel("Time shift")

plt.tight_layout()
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")
