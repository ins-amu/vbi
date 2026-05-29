"""
Damped Oscillator - SBI with vbi.inference.SNPE
================================================

Reproduce the ``damp_oscillator_cde.ipynb`` workflow using the new
``vbi.inference`` API.  No torch, no sbi dependency.

The model
---------
A nonlinear damped oscillator::

    dx/dt = x - xy - a*x²
    dy/dt = xy - y  - b*y²

Parameters to infer: ``a`` (x-damping) and ``b`` (y-damping).

Workflow
--------
1. Run the model at the true parameters to get the "observed" data.
2. Define a simulator callable ``theta_1d → x_features``.
3. Use ``simulate_for_sbi`` to batch-generate 2000 training samples.
4. Train a MAF density estimator with ``SNPE``.
5. Draw posterior samples and print shrinkage / z-score diagnostics.
6. Plot: time-series, training loss, posterior pairplot.
7. (Optional) Run SBC + PP plot for calibration check.

Note
----
``VBIInference`` is designed for large brain models (MPR, JR, …) that use the
``SimulationSpec`` / ``Sweeper`` pipeline.  The DampedOscillator is a
standalone ODE without a connectome, so we use ``SNPE`` directly with a
custom simulator function instead.

Expected runtime: ~2–3 min on a modern CPU (numba backend, 2000 simulations).
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vbi.models.numba.damp_oscillator import DO
from vbi.inference import (
    SNPE, BoxUniform,
    simulate_for_sbi,
    pairplot,
    plot_loss,
    run_sbc, pp_plot,
)
from vbi.utils import posterior_shrinkage_numpy, posterior_zscore_numpy

OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)

SEED = 2
rng  = np.random.default_rng(SEED)

# ── Model parameters (fixed, not inferred) ────────────────────────────────────

BASE_PAR = {
    "dt":           0.01,
    "t_start":      0.0,
    "method":       "rk4",
    "t_end":        100.0,
    "t_cut":        20.0,
    "output":       "output",
    "initial_state": [0.5, 1.0],
}

THETA_TRUE = np.array([0.1, 0.05])   # a=0.1, b=0.05
PRIOR_LOW  = np.array([0.0, 0.0])
PRIOR_HIGH = np.array([1.0, 1.0])

# ── 1 - Run the model at the true parameters ──────────────────────────────────

print("=" * 60)
print("Damped Oscillator  —  SBI with vbi.inference.SNPE")
print("=" * 60)

ode_true = DO({**BASE_PAR, "a": THETA_TRUE[0], "b": THETA_TRUE[1]})
sol      = ode_true.run()
t_obs    = sol["t"]
x_obs_ts = sol["x"]   # (T, 2)

fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(t_obs, x_obs_ts[:, 0], label="x  (state 0)", lw=1.2)
ax.plot(t_obs, x_obs_ts[:, 1], label="y  (state 1)", lw=1.2)
ax.set_xlabel("t")
ax.set_ylabel("state")
ax.set_title(f"True dynamics  a={THETA_TRUE[0]}, b={THETA_TRUE[1]}")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "do_timeseries.png", dpi=120)
plt.close(fig)
print(f"  Time-series saved → {OUT/'do_timeseries.png'}")

# ── 2 - Feature extractor: std and mean of each state variable ────────────────

def _features(x: np.ndarray) -> np.ndarray:
    """4 features: [std(x0), std(x1), mean(x0), mean(x1)]."""
    return np.array([x[:, 0].std(), x[:, 1].std(),
                     x[:, 0].mean(), x[:, 1].mean()], dtype=np.float64)


def simulator(theta_1d: np.ndarray) -> np.ndarray:
    """Run one simulation and return the feature vector."""
    a, b = float(theta_1d[0]), float(theta_1d[1])
    try:
        ode = DO({**BASE_PAR, "a": a, "b": b})
        sol = ode.run()
        return _features(sol["x"])
    except Exception:
        return np.full(4, np.nan)


# Observed feature vector
x_obs = _features(x_obs_ts)
print(f"\n  True θ          : a={THETA_TRUE[0]}, b={THETA_TRUE[1]}")
print(f"  Observed features: {np.round(x_obs, 4)}")

# ── 3 - Generate training data ────────────────────────────────────────────────

N_SIM = 2000
prior = BoxUniform(low=PRIOR_LOW, high=PRIOR_HIGH, param_names=["a", "b"])

print(f"\n  Generating {N_SIM} simulations … ", end="", flush=True)
theta_train, x_train = simulate_for_sbi(simulator, prior, N_SIM, seed=SEED)
valid = np.all(np.isfinite(x_train), axis=1)
print(f"done  ({valid.sum()} / {N_SIM} valid)")

theta_train = theta_train[valid]
x_train     = x_train[valid]
print(f"  theta: {theta_train.shape}   x: {x_train.shape}")

# ── 4 - Train SNPE with MAF ───────────────────────────────────────────────────

inference = SNPE(prior=prior, density_estimator="maf", backend="auto")
inference.append_simulations(theta_train, x_train)

print("\n  Training MAF … ", flush=True)
estimator = inference.train(
    training_batch_size = 256,
    learning_rate       = 5e-4,
    stop_after_epochs   = 20,
    max_num_epochs      = 500,
    verbose             = True,
)
print(f"  Best val loss: {estimator.best_val_loss:.4f}  "
      f"(epoch {estimator.best_epoch})")

# Plot training loss
fig_loss = plot_loss(estimator.loss_history, estimator.val_loss_history)
fig_loss.savefig(OUT / "do_loss.png", dpi=120)
plt.close(fig_loss)
print(f"  Loss curve saved → {OUT/'do_loss.png'}")

# ── 5 - Build posterior and draw samples ──────────────────────────────────────

posterior = inference.build_posterior(estimator)
posterior.set_default_x(x_obs[None])   # shape (1, 4)

N_POST  = 5000
samples = posterior.sample((N_POST,), x=x_obs[None], seed=0)
print(f"\n  Posterior samples: {samples.shape}")

# ── 6 - Diagnostics ───────────────────────────────────────────────────────────

shrinkage = posterior_shrinkage_numpy(theta_train, samples)
zscore    = posterior_zscore_numpy(THETA_TRUE, samples)
post_mean = samples.mean(axis=0)
post_std  = samples.std(axis=0)

print("\n  ── Posterior diagnostics ─────────────────────────────")
print(f"  True parameters  : {THETA_TRUE}")
print(f"  Posterior mean   : {np.round(post_mean, 4)}")
print(f"  Posterior std    : {np.round(post_std,  4)}")
print(f"  Shrinkage        : {np.round(shrinkage, 3)}")
print(f"  Z-score          : {np.round(zscore,    3)}")

# Sanity checks
assert np.all(shrinkage > 0.5), "Posterior not more concentrated than prior!"
assert np.all(np.abs(zscore) < 3.0), "Z-score > 3: posterior may be miscalibrated!"
print("  ✓ Shrinkage > 0.5 and |z-score| < 3")

# ── 7 - Posterior pairplot ────────────────────────────────────────────────────

fig_pp = pairplot(
    samples,
    labels = ["a", "b"],
    points = THETA_TRUE[None],
)
fig_pp.savefig(OUT / "do_posterior_pairplot.png", dpi=120)
plt.close(fig_pp)
print(f"\n  Pairplot saved → {OUT/'do_posterior_pairplot.png'}")

# ── 8 - SBC calibration check (optional, ~30 s) ───────────────────────────────

print("\n  Running SBC (50 runs) … ", end="", flush=True)
sbc_result = run_sbc(
    posterior,
    simulator         = simulator,
    prior             = prior,
    num_sbc_runs      = 50,
    num_posterior_samples = 100,
    seed              = 1,
)
print("done")

fig_sbc = pp_plot(sbc_result["ranks"], labels=["a", "b"])
fig_sbc.savefig(OUT / "do_sbc_ppplot.png", dpi=120)
plt.close(fig_sbc)
print(f"  PP plot saved → {OUT/'do_sbc_ppplot.png'}")

# ── Summary ───────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print(f"  Training samples : {len(theta_train)}")
print(f"  Posterior samples: {N_POST}")
print(f"  True θ = [a={THETA_TRUE[0]}, b={THETA_TRUE[1]}]")
print(f"  MAP estimate ≈   [a={post_mean[0]:.4f}, b={post_mean[1]:.4f}]")
print(f"  Outputs in       : {OUT}/")
