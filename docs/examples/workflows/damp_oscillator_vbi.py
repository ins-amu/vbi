"""
Damped Oscillator — end-to-end SBI with VBIInference
=====================================================

Uses the ``damped_oscillator`` ModelSpec registered in the new VBI simulator
pipeline, so the full ``VBIInference`` workflow applies — the same object
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
Pipeline uses ``voi=None`` to expose **both** state variables (x and y) as
separate channels.  Select whichever statistical / spectral features suit
the problem — the user should customise the feature list in section 3.

Workflow
--------
1. Build SimulationSpec (single node, no coupling).
2. Define prior and feature pipeline.
3. Build VBIInference.
4. Round 1: simulate + train MAF + build posterior.
5. Plot loss and pairplot.
6. Save / load checkpoint.
7. Round 2: posterior-focused simulation (sequential SNPE).

Expected runtime: ~2–4 min  (numba backend, 2000 simulations × 200 ms).
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vbi.simulator.models.damped_oscillator import damped_oscillator
from vbi.simulator.spec import (
    SimulationSpec, IntegratorSpec, CouplingSpec, MonitorSpec,
)
from vbi.simulator import Simulator
from vbi.feature_extraction import (
    FeaturePipeline, get_features_by_domain, get_features_by_given_names,
)
from vbi.inference import (
    VBIInference, BoxUniform,
    pairplot, plot_loss,
)

OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)

print("=" * 62)
print("Damped Oscillator  —  VBIInference end-to-end pipeline")
print("=" * 62)

# ── 1 - SimulationSpec (single node, no coupling) ─────────────────────────────

N = 1
W = np.zeros((N, N))
D = np.zeros((N, N))

sim_spec = SimulationSpec(
    model         = damped_oscillator,
    integrator    = IntegratorSpec(method="heun", dt=0.01),
    coupling      = CouplingSpec("linear", a=0.0),
    monitors      = (MonitorSpec("subsample", period=0.1),),
    # Use MonitorSpec("raw") for full dt-resolution signal instead.
    weights       = W,
    tract_lengths = D,
    speed         = 4.0,
)

# ── 2 - True parameters and observed data ────────────────────────────────────

THETA_TRUE = np.array([0.1, 0.05])   # a=0.1, b=0.05
PRIOR_LOW  = np.array([0.0, 0.0])
PRIOR_HIGH = np.array([1.0, 1.0])
DURATION   = 200.0   # ms  (system reaches steady state by ~100 ms)
T_CUT      = 20.0    # ms  (discard initial transient)

# Run a single simulation at the true parameters for visual check
sim_spec_true = SimulationSpec(
    model         = damped_oscillator,
    integrator    = IntegratorSpec(method="heun", dt=0.01),
    coupling      = CouplingSpec("linear", a=0.0),
    monitors      = (MonitorSpec("subsample", period=0.1),),
    weights       = W,
    tract_lengths = D,
    speed         = 4.0,
    node_params   = {"a": np.array([THETA_TRUE[0]]),
                     "b": np.array([THETA_TRUE[1]])},
)
result_true = Simulator(sim_spec_true, backend="numba").run(DURATION)
t_obs, ts_obs = result_true["subsample"]       # ts: (T, 2, 1)

fig, axes = plt.subplots(2, 1, figsize=(6, 4), sharex=True)
for i, (sv, color) in enumerate([("x", "steelblue"), ("y", "tomato")]):
    axes[i].plot(t_obs, ts_obs[:, i, 0], color=color, lw=1.2, label=sv)
    axes[i].set_ylabel(sv)
    axes[i].spines["right"].set_visible(False)
    axes[i].spines["top"].set_visible(False)
axes[-1].set_xlabel("t (ms)")
axes[0].set_title(
    f"True dynamics  a={THETA_TRUE[0]}, b={THETA_TRUE[1]}", fontsize=10
)
fig.tight_layout()
fig.savefig(OUT / "do_vbi_timeseries.png", dpi=120)
plt.close(fig)
print(f"\n  True θ     : a={THETA_TRUE[0]}, b={THETA_TRUE[1]}")
print(f"  Time-series → {OUT/'do_vbi_timeseries.png'}")

# ── 3 - Feature pipeline ──────────────────────────────────────────────────────
#
# voi=None exposes BOTH state variables (x = channel 0, y = channel 1).
# Select the features you want by editing get_features_by_given_names below.
# Available statistical features include: calc_mean, calc_std, calc_skewness,
#   calc_kurtosis, calc_energy, calc_rms, calc_max, calc_min, ...
# Run:  vbi.feature_extraction.report_cfg(get_features_by_domain("statistical"))
# to see the full list.

cfg = get_features_by_domain("statistical")
cfg = get_features_by_given_names(cfg, ["calc_mean", "calc_std"])

pipeline = FeaturePipeline(
    cfg,
    signal = "subsample",
    t_cut  = T_CUT,
    voi    = None,      # use ALL state variables (x and y)
)

# Confirm feature labels on the true-parameter run
labels, values = pipeline.extract(result_true)
print(f"\n  Feature labels  : {labels}")
print(f"  True obs values : {np.round(values, 4)}")
print(f"  (4 features: std_x, std_y, mean_x, mean_y)")

# ── 4 - Prior ─────────────────────────────────────────────────────────────────

prior = BoxUniform(
    low         = PRIOR_LOW,
    high        = PRIOR_HIGH,
    param_names = ["a", "b"],
)

# ── 5 - Build VBIInference ────────────────────────────────────────────────────

inf = VBIInference(
    sim_spec          = sim_spec,
    prior             = prior,
    pipeline          = pipeline,
    density_estimator = "maf",
    sim_backend       = "numba",
    backend           = "auto",
    show_progress_bars = True,
)
print(f"\n  {inf}")

# ── 6 - Round 1: simulate + train + posterior ─────────────────────────────────

N_R1 = 200


print(f"\n  Round 1: {N_R1} simulations × {DURATION} ms …", flush=True)
theta_r1, x_r1 = inf.simulate(N_R1, DURATION, seed=0)
print(f"  theta {theta_r1.shape}   x {x_r1.shape}")

print("  Training MAF …", flush=True)
est_r1 = inf.train(
    training_batch_size = 256,
    learning_rate       = 5e-4,
    stop_after_epochs   = 20,
    max_num_epochs      = 500,
)
print(f"  Best val loss : {est_r1.best_val_loss:.4f}  (epoch {est_r1.best_epoch})")
exit(0)

fig_loss = inf.plot_loss()
fig_loss.savefig(OUT / "do_vbi_r1_loss.png", dpi=120)
plt.close(fig_loss)
print(f"  Loss curve    → {OUT/'do_vbi_r1_loss.png'}")

# Observed feature vector from true parameters
post_r1 = inf.build_posterior(est_r1)
x_obs   = values   # features at THETA_TRUE

samples_r1 = post_r1.sample((5000,), x=x_obs[None], seed=0)
print(f"\n  Posterior samples : {samples_r1.shape}")

post_mean = samples_r1.mean(axis=0)
post_std  = samples_r1.std(axis=0)
print(f"  True θ            : {THETA_TRUE}")
print(f"  Posterior mean    : {np.round(post_mean, 4)}")
print(f"  Posterior std     : {np.round(post_std,  4)}")

fig_pair = pairplot(samples_r1, labels=["a", "b"], points=THETA_TRUE[None])
fig_pair.savefig(OUT / "do_vbi_r1_pairplot.png", dpi=120)
plt.close(fig_pair)
print(f"  Pairplot          → {OUT/'do_vbi_r1_pairplot.png'}")

# ── 7 - Save / load checkpoint ────────────────────────────────────────────────

ckpt = OUT / "do_vbi_r1.npz"
inf.save(ckpt)
print(f"\n  Checkpoint saved  → {ckpt}")

inf_loaded = VBIInference.load(ckpt, sim_spec=sim_spec, pipeline=pipeline, prior=prior)
post_ld    = inf_loaded.build_posterior()
s_ld       = post_ld.sample((100,), x=x_obs[None], seed=1)
assert s_ld.shape == (100, 2)
print("  ✓ Load + sample OK")

# ── 8 - Round 2: posterior-focused simulation ─────────────────────────────────

N_R2 = 500
post_r1.set_default_x(x_obs[None])

print(f"\n  Round 2: {N_R2} sims from round-1 posterior …", flush=True)
theta_r2, x_r2 = inf.simulate(
    N_R2, DURATION,
    proposal = post_r1,
    x_obs    = x_obs[None],
    seed     = 1,
)
print(f"  theta {theta_r2.shape}   x {x_r2.shape}")
print(f"  Total sims: {inf._snpe.n_simulations}")

est_r2  = inf.train(
    training_batch_size = 256,
    learning_rate       = 5e-4,
    stop_after_epochs   = 20,
    max_num_epochs      = 500,
)
post_r2    = inf.build_posterior(est_r2)
samples_r2 = post_r2.sample((5000,), x=x_obs[None], seed=0)
print(f"  Round-2 mean: {np.round(samples_r2.mean(axis=0), 4)}")
print(f"  Round-2 std : {np.round(samples_r2.std(axis=0),  4)}")

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, samp, label in zip(
    axes,
    [samples_r1, samples_r2],
    [f"Round 1  (n={N_R1})",
     f"Round 2  (+{N_R2} posterior sims)"],
):
    ax.scatter(samp[:, 0], samp[:, 1], s=3, alpha=0.3, color="steelblue")
    ax.scatter(*THETA_TRUE, color="red", s=80, marker="*", zorder=5, label="true θ")
    ax.set_xlabel("a"); ax.set_ylabel("b")
    ax.set_title(label, fontsize=10)
    ax.legend(fontsize=8)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
fig.suptitle("Round-1 vs Round-2 posterior", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "do_vbi_r1_vs_r2.png", dpi=120)
plt.close(fig)
print(f"\n  Comparison plot   → {OUT/'do_vbi_r1_vs_r2.png'}")

# ── Summary ───────────────────────────────────────────────────────────────────

print("\n" + "=" * 62)
print("Summary")
print("=" * 62)
print(f"  Model    : DampedOscillator  (N=1, no coupling)")
print(f"  Monitor  : subsample  period=0.1 ms")
print(f"  Features : {labels}  (voi=None → both SVs)")
print(f"  R1 sims  : {N_R1}  |  R2 sims: {N_R2}")
print(f"  True θ   : a={THETA_TRUE[0]}, b={THETA_TRUE[1]}")
print(f"  R1 mean  : {np.round(samples_r1.mean(0), 4)}")
print(f"  R2 mean  : {np.round(samples_r2.mean(0), 4)}")
print(f"  Outputs  : {OUT}/")
