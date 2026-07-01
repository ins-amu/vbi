"""
CUDA Sweep Demo
=================

G sweep and same_noise control.

Demonstrates two features added to the CUDA backend:

1. Sweeping the global coupling strength G.
   G scales the coupling term for every sample without re-compiling the kernel.

2. same_noise semantics.
   same_noise=True  - all sweep samples share the same noise realisation.
                      Differences in output reflect only the swept parameter.
   same_noise=False - each sample gets an independent noise seed.
                      More realistic for SBI training data.

Run:
    python cuda_sweep_demo.py
    python cuda_sweep_demo.py --no-plot
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

try:
    from numba import cuda as _cuda

    if not _cuda.is_available():
        print("No CUDA device found. Exiting.")
        sys.exit(0)
except Exception:
    print("numba[cuda] not installed. Exiting.")
    sys.exit(0)

from vbi.simulator import Sweeper
from vbi.simulator.models.mpr import mpr
from vbi.simulator.spec import (
    Connectivity,
    CouplingSpec,
    IntegratorSpec,
    MonitorSpec,
    SimulationSpec,
)
from vbi.simulator.spec.sweep import SweepSpec

# ---------------------------------------------------------------------------
# Shared setup
# ---------------------------------------------------------------------------

N_NODES = 8
DT = 0.1  # ms
DURATION = 500.0  # ms

rng = np.random.default_rng(0)
W = np.abs(rng.standard_normal((N_NODES, N_NODES))).astype(np.float64)
np.fill_diagonal(W, 0.0)
W /= W.sum(axis=1, keepdims=True).clip(1e-8)

N_NOISE = len(mpr.noise_indices)
NSIG = np.full(N_NOISE, 0.1)

base_spec = SimulationSpec(
    model=mpr,
    integrator=IntegratorSpec(
        method="heun",
        dt=DT,
        stochastic=True,
        noise_nsig=NSIG,
        noise_seed=0,
    ),
    coupling=CouplingSpec("linear", a=1.0),  # a=1 so that G is the only scale
    monitors=(MonitorSpec("tavg", period=1.0),),
    connectivity=Connectivity(W),
)


# ---------------------------------------------------------------------------
# 1. G sweep
# ---------------------------------------------------------------------------

print("Running G sweep …")

G_values = np.linspace(0.0, 0.5, 8)
sweep_G = SweepSpec(params={"G": G_values}, same_noise=True)
results_G = Sweeper(base_spec, sweep_G, backend="cuda").run(DURATION)

# Collect mean activity (averaged across nodes and time) for each G
mean_activity = []
for res in results_G:
    t, d = res["tavg"]  # d: (n_time, n_sv, n_nodes)
    mean_activity.append(float(d[:, 0, :].mean()))

print(f"  G values  : {np.round(G_values, 3)}")
print(f"  mean |r|  : {np.round(mean_activity, 4)}")


# ---------------------------------------------------------------------------
# 2. same_noise comparison - eta sweep
# ---------------------------------------------------------------------------

print("\nRunning eta sweep with same_noise=True  …")
ETA_VALUES = np.linspace(-5.5, -4.0, 8)

sweep_same = SweepSpec(params={"eta": ETA_VALUES}, same_noise=True)
results_same = Sweeper(base_spec, sweep_same, backend="cuda").run(DURATION)

print("Running eta sweep with same_noise=False …")
sweep_indep = SweepSpec(params={"eta": ETA_VALUES}, same_noise=False)
results_indep = Sweeper(base_spec, sweep_indep, backend="cuda").run(DURATION)

# Variance of the mean activity across sweep samples (lower with same_noise=True
# when parameters are close together - noise no longer adds sample-to-sample spread)
means_same = [float(res["tavg"][1][:, 0, :].mean()) for res in results_same]
means_indep = [float(res["tavg"][1][:, 0, :].mean()) for res in results_indep]
mean_delta = np.asarray(means_indep) - np.asarray(means_same)
traj_delta = []
for res_a, res_b in zip(results_same, results_indep):
    traj_delta.append(float(np.max(np.abs(res_a["tavg"][1] - res_b["tavg"][1]))))

print(f"\nsame_noise=True , mean r per eta: {np.round(means_same,  6)}")
print(f"  same_noise=False  , mean r per eta: {np.round(means_indep, 6)}")
print(f"  independent-minus-shared mean delta: {np.round(mean_delta, 8)}")
print(f"  max trajectory delta per eta: {np.round(traj_delta, 8)}")

# Duplicate-parameter sanity check: two identical eta values with same_noise=True
# must produce identical trajectories.
sweep_dup = SweepSpec(params={"eta": np.array([-4.6, -4.6])}, same_noise=True)
res_dup = Sweeper(base_spec, sweep_dup, backend="cuda").run(DURATION)
_, d0 = res_dup[0]["tavg"]
_, d1 = res_dup[1]["tavg"]
identical = np.array_equal(d0, d1)
print(f"\n  Duplicate-param same_noise=True identical: {identical}  (expected True)")

sweep_dup2 = SweepSpec(params={"eta": np.array([-4.6, -4.6])}, same_noise=False)
res_dup2 = Sweeper(base_spec, sweep_dup2, backend="cuda").run(DURATION)
_, e0 = res_dup2[0]["tavg"]
_, e1 = res_dup2[1]["tavg"]
different = not np.array_equal(e0, e1)
print(f"  Duplicate-param same_noise=False different: {different}  (expected True)")


# ---------------------------------------------------------------------------
# 3. Optional plots
# ---------------------------------------------------------------------------


def _plot(G_values, mean_activity, ETA_VALUES, means_same, means_indep):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    ax.plot(G_values, mean_activity, "o-", color="steelblue")
    ax.set_xlabel("Global coupling G")
    ax.set_ylabel("Mean firing rate r (a.u.)")
    ax.set_title("CUDA: G sweep (MPR, tavg)")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(ETA_VALUES, means_same, "s-", label="same_noise=True", color="darkorange")
    ax.plot(ETA_VALUES, means_indep, "^--", label="same_noise=False", color="seagreen")
    ax.set_xlabel("η (excitability)")
    ax.set_ylabel("Mean firing rate r (a.u.)")
    ax.set_title("CUDA: eta sweep - noise sharing")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = Path(__file__).parent / "outputs" / "cuda_sweep_demo.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"\nFigure saved to {out}")
    plt.show()


parser = argparse.ArgumentParser()
parser.add_argument("--no-plot", action="store_true")
args = parser.parse_args()

if not args.no_plot:
    try:
        _plot(G_values, mean_activity, ETA_VALUES, means_same, means_indep)
    except ImportError:
        print("matplotlib not available - skipping plot.")
