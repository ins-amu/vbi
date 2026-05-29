"""Reproduce the BOLD + FCD plot from rww_sde_numba.ipynb using vbi.simulator.

Matches notebook parameters exactly:
  G=1.2, I_ext=0.05, w=1.0, sigma=0.05, dt=2.5ms, T=5min, t_cut=1min, tr=300ms
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.dont_write_bytecode = True
warnings.filterwarnings("ignore")

from helpers import ensure_repo_on_path
ensure_repo_on_path(__file__)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import vbi
from vbi.simulator import Simulator
from vbi.simulator.models.reduced_wong_wang import reduced_wong_wang
from vbi.simulator.spec.model import StateVar
from vbi.simulator.spec.coupling import CouplingSpec
from vbi.simulator.spec.integrator import IntegratorSpec
from vbi.simulator.spec.monitor import MonitorSpec
from vbi.simulator.spec.simulation import SimulationSpec


D = vbi.LoadSample(nn=84)
weights = D.get_weights()
nn = weights.shape[0]

seed   = 42
G      = 1.2          # global coupling strength
I_o    = 0.05         # external input (notebook: I_ext)
w_rec  = 1.0          # local recurrence (notebook RWW default, vbi.simulator default is 0.6)
sigma  = 0.05         # noise amplitude
dt     = 2.5          # ms
T      = 5 * 60 * 1000.0   # ms  (5 min)
t_cut  = 1 * 60 * 1000.0   # ms  (1 min burn-in)
tr     = 300.0        # ms  BOLD repetition time


np.random.seed(seed)
x0 = np.random.rand(nn) * 0.1

spec = SimulationSpec(
    model=reduced_wong_wang,
    integrator=IntegratorSpec(
        method="heun",
        dt=dt,
        stochastic=True,
        noise_nsig=np.array([sigma]),
        noise_style="amplitude",   # dW = sigma * sqrt(dt) * N(0,1)
        noise_seed=seed,
    ),
    coupling=CouplingSpec(kind="linear", a=G),
    monitors=(MonitorSpec(kind="bold", tr=tr),),
    weights=weights,
    node_params={
        "I_o": np.full(nn, I_o),
        "w":   np.full(nn, w_rec),
    },
)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
print(f"Simulating {nn} nodes, T={T/1000:.0f}s, dt={dt}ms ...")
result = Simulator(spec, backend="numpy").run(duration=T)
bold_t, bold_d = result["bold"]   # (n_steps,), (n_steps, nn)

# Trim burn-in
cut_idx = int(t_cut / tr)
bold_t  = bold_t[cut_idx:]
bold_d  = bold_d[cut_idx:]

print(f"BOLD shape after trim: {bold_d.shape}  ({bold_d.shape[0]} time points × {nn} nodes)")

from vbi.feature_extraction.features import get_fcd
fcd = get_fcd(bold_d.T, win_len=30, tr=tr / 1000.0)['full']

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
out_path = Path(__file__).with_name("outputs") / "rww.png"
out_path.parent.mkdir(exist_ok=True)

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

plt.tight_layout()
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")
