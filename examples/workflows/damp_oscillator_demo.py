"""
Damped Oscillator — end-to-end SBI with InferencePipeline
===========================================================

Uses the ``damped_oscillator`` ModelSpec registered in the new VBI simulator
pipeline, so the full ``InferencePipeline`` workflow applies — the same object
that drives MPR, JR, WC, etc.

The model
---------
A nonlinear damped oscillator (Lotka-Volterra type)::

    dx/dt = x - xy - a*x²
    dy/dt = xy - y  - b*y²

Parameters to infer: ``a`` (x-damping) and ``b`` (y-damping).

Single-node setup (N=1, no network coupling).

Monitor
-------
``subsample`` with period=0.1 ms records every 10 integration steps.
Use ``MonitorSpec('raw')`` for the full dt-resolution signal (larger memory).

Features
--------
Pipeline uses ``voi="all"`` to expose **both** state variables (x and y) as
separate channels.  Select whichever statistical / spectral features suit
the problem — the user should customise the feature list in section 3.

Workflow
--------
1. Build SimulationSpec (single node, no coupling).
2. Define prior and feature pipeline.
3. Build InferencePipeline.
4. Simulate + train MAF + build posterior (single round).
5. Plot loss and pairplot.

Expected runtime: ~2–4 min  (numba backend, 2000 simulations × 250 ms).
"""

from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vbi.simulator.models.damped_oscillator import damped_oscillator
from vbi.simulator.spec import (
    SimulationSpec,
    IntegratorSpec,
    CouplingSpec,
    MonitorSpec,
    Connectivity,
)
from vbi.simulator import Simulator
from vbi.feature_extraction import (
    FeaturePipeline,
    get_features_by_domain,
    get_features_by_given_names,
)
from vbi.inference import (
    InferencePipeline,
    SNPE,
    MAF,
    TrainingOptions,
    BoxUniform,
    pairplot
)

try:
    _SCRIPT_DIR = Path(__file__).parent
except NameError:
    _SCRIPT_DIR = Path.cwd()

OUT = _SCRIPT_DIR / "outputs/damped_oscillator"
OUT.mkdir(exist_ok=True)

print("=" * 62)
print("Damped Oscillator  —  InferencePipeline end-to-end pipeline")
print("=" * 62)

# ── 1 - SimulationSpec (single node, no coupling) ─────────────────────────────

N = 1
dt = 0.1
monitor = MonitorSpec("raw")
coupling = CouplingSpec("linear", a=0.0)  # no coupling (a=0.0)
integrator_backend = "numba"

conn = Connectivity(weights=np.zeros((N, N)))

sim_spec = SimulationSpec(
    model=damped_oscillator,
    integrator=IntegratorSpec(method="heun", dt=dt),
    coupling=coupling,
    monitors=(monitor,),
    connectivity=conn,
)

# ── 2 - True parameters and observed data ────────────────────────────────────

THETA_TRUE = np.array([0.1, 0.05])  # a=0.1, b=0.05
PRIOR_LOW = np.array([0.0, 0.0])
PRIOR_HIGH = np.array([1.0, 1.0])
DURATION = 250.0  # ms  (system reaches steady state by ~100 ms)
T_CUT = 20.0  # ms  (discard initial transient)

# Run a single simulation at the true parameters for visual check
sim_spec_true = SimulationSpec(
    model=damped_oscillator,
    integrator=IntegratorSpec(method="heun", dt=dt),
    coupling=coupling,
    monitors=(monitor,),
    connectivity=conn,
    node_params={"a": np.array([THETA_TRUE[0]]), "b": np.array([THETA_TRUE[1]])},
)
result_true = Simulator(sim_spec_true, backend=integrator_backend).run(DURATION)
t_obs, ts_obs = result_true["raw"]  # ts: (T, 2, 1)

fig, axes = plt.subplots(2, 1, figsize=(6, 4), sharex=True)
for i, (sv, color) in enumerate([("x", "steelblue"), ("y", "tomato")]):
    axes[i].plot(t_obs, ts_obs[:, i, 0], color=color, lw=1.2, label=sv)
    axes[i].set_ylabel(sv)
    axes[i].spines["right"].set_visible(False)
    axes[i].spines["top"].set_visible(False)
axes[-1].set_xlabel("t (ms)")
axes[0].set_title(f"True dynamics  a={THETA_TRUE[0]}, b={THETA_TRUE[1]}", fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "do_vbi_timeseries.png", dpi=120)
plt.close(fig)
print(f"\n  True θ     : a={THETA_TRUE[0]}, b={THETA_TRUE[1]}")
print(f"  Time-series → {OUT/'do_vbi_timeseries.png'}")

# ── 3 - Feature pipeline ──────────────────────────────────────────────────────
#
# voi="all" exposes BOTH state variables (x = channel 0, y = channel 1).
# Select the features you want by editing get_features_by_given_names below.
# Available statistical features include: calc_mean, calc_std, calc_skewness,
#   calc_kurtosis, calc_energy, calc_rms, calc_max, calc_min, ...
# Run:  vbi.feature_extraction.report_cfg(get_features_by_domain("statistical"))
# to see the full list.

cfg = get_features_by_domain("statistical")
cfg = get_features_by_given_names(cfg, ["calc_mean", "calc_std"])

pipeline = FeaturePipeline(
    cfg,
    signal="raw",
    t_cut=T_CUT,
    voi="all",  # use ALL state variables (x and y)
)

# Confirm feature labels on the true-parameter run
labels, values = pipeline.extract(result_true)
print(f"\n  Feature labels  : {labels}")
print(f"  True obs values : {np.round(values, 4)}")
print(f"  (4 features: std_x, std_y, mean_x, mean_y)")

# ── 4 - Prior ─────────────────────────────────────────────────────────────────

prior = BoxUniform(
    low=PRIOR_LOW,
    high=PRIOR_HIGH,
    param_names=["a", "b"],
)

# ── 5 - Build InferencePipeline ────────────────────────────────────────────────

inf = InferencePipeline(
    sim_spec=sim_spec,
    prior=prior,
    feature_pipeline=pipeline,
    integrator_backend=integrator_backend,
    engine=SNPE(prior, density_estimator=MAF(backend="auto")),
    training=TrainingOptions(
        batch_size=256,
        learning_rate=5e-4,
        stop_after_epochs=20,
        max_epochs=500,
    ),
    show_progress_bars=True,
)
print(f"\n  {inf}")

# ── 6 - Simulate + train + posterior  (single round) ─────────────────────────

N_SIM = 2000

print(f"\n  Simulating {N_SIM} × {DURATION} ms …", flush=True)
theta, x = inf.simulate(N_SIM, DURATION, seed=0)
print(f"  theta {theta.shape}   x {x.shape}")

print("  Training MAF …", flush=True)
estimator = inf.train()   # uses the TrainingOptions passed above
print(
    f"  Best val loss : {estimator.best_val_loss:.4f}  (epoch {estimator.best_epoch})"
)

fig_loss = inf.plot_loss()
fig_loss.savefig(OUT / "do_vbi_loss.png", dpi=120)
plt.close(fig_loss)
print(f"  Loss curve    → {OUT/'do_vbi_loss.png'}")

posterior = inf.build_posterior(estimator)
x_obs = values  # feature vector at THETA_TRUE

samples = posterior.sample((5000,), x=x_obs[None], seed=0)
print(f"\n  Posterior samples : {samples.shape}")
print(f"  True θ            : {THETA_TRUE}")
print(f"  Posterior mean    : {np.round(samples.mean(0), 4)}")
print(f"  Posterior std     : {np.round(samples.std(0),  4)}")

fig_pair = pairplot(samples, labels=["a", "b"], points=THETA_TRUE[None])
fig_pair.savefig(OUT / "do_vbi_pairplot.png", dpi=120)
plt.close(fig_pair)
print(f"  Pairplot          → {OUT/'do_vbi_pairplot.png'}")

# ── Summary ───────────────────────────────────────────────────────────────────

print("\n" + "=" * 62)
print("Summary")
print("=" * 62)
print(f"  Model    : DampedOscillator  (N=1, no coupling)")
print(f"  Backend  : {integrator_backend} (sim) + auto (inference)")
print(f"  Monitor  : raw")
print(f"  Features : {labels}  (voi='all' → both SVs)")
print(f"  N sims   : {N_SIM}  ×  {DURATION} ms")
print(f"  True θ   : a={THETA_TRUE[0]}, b={THETA_TRUE[1]}")
print(f"  Post mean: {np.round(samples.mean(0), 4)}")
print(f"  Outputs  : {OUT}/")
