"""
Demo 1 - 1-D Gaussian with known analytical posterior.

Simulator:   x | theta ~ N(theta, sigma_lik)
Prior:       theta ~ N(0, sigma_prior)
Posterior:   theta | x  ~ N(mu_post, sigma_post)   (exact formula below)

We train vbi.inference.SNPE and compare:
  - posterior mean error
  - posterior std error
  - coverage: does the 90% credible interval contain the true theta?

Expected runtime: < 10 seconds.
"""

import numpy as np
from vbi.inference import SNPE, Gaussian

# ── Problem definition ────────────────────────────────────────────────────────
SIGMA_LIK   = 0.3     # likelihood noise
SIGMA_PRIOR = 2.0     # prior std
N_TRAIN     = 2000    # number of simulated training pairs
N_POST      = 2000    # posterior samples per test point
SEED        = 42

rng = np.random.default_rng(SEED)

# ── Analytical posterior ──────────────────────────────────────────────────────
def analytical_posterior(x_obs: float):
    """theta | x ~ N(mu_post, sigma_post²)."""
    prec_prior = 1.0 / SIGMA_PRIOR ** 2
    prec_lik   = 1.0 / SIGMA_LIK   ** 2
    sigma_post = np.sqrt(1.0 / (prec_prior + prec_lik))
    mu_post    = sigma_post ** 2 * prec_lik * x_obs
    return mu_post, sigma_post

# ── Simulate training data ────────────────────────────────────────────────────
prior = Gaussian(mean=np.array([0.0]), std=np.array([SIGMA_PRIOR]))
theta_train = prior.sample((N_TRAIN,), seed=SEED)            # (N, 1)
x_train     = theta_train + rng.normal(0, SIGMA_LIK, theta_train.shape)

# ── Train ─────────────────────────────────────────────────────────────────────
inference = SNPE(prior=prior, density_estimator="maf")
inference.append_simulations(theta_train, x_train)
estimator = inference.train(
    training_batch_size=256,
    learning_rate=5e-4,
    stop_after_epochs=20,
    max_num_epochs=300,
    verbose=False,
)
posterior = inference.build_posterior(estimator)

# ── Evaluate on test observations ─────────────────────────────────────────────
x_test_vals = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
rng_eval = np.random.RandomState(0)

mean_errors, std_errors, in_90ci = [], [], []

for x_obs_val in x_test_vals:
    x_obs     = np.array([[x_obs_val]])
    mu_true, sigma_true = analytical_posterior(x_obs_val)

    samples = posterior.sample((N_POST,), x=x_obs, seed=0)  # (N_POST, 1)
    mu_est  = samples[:, 0].mean()
    std_est = samples[:, 0].std()

    lo, hi = np.percentile(samples[:, 0], [5, 95])
    inside = (lo <= mu_true <= hi)

    mean_errors.append(abs(mu_est - mu_true))
    std_errors.append(abs(std_est - sigma_true))
    in_90ci.append(inside)

mean_err_avg = np.mean(mean_errors)
std_err_avg  = np.mean(std_errors)
coverage_90  = np.mean(in_90ci)

# ── Report ─────────────────────────────────────────────────────────────────────
print("=" * 55)
print("Demo 1 - 1-D Gaussian (known posterior)")
print("=" * 55)
print(f"  sigma_lik={SIGMA_LIK}  sigma_prior={SIGMA_PRIOR}  N_train={N_TRAIN}")
print()
print(f"  {'x_obs':>6}  {'true_mean':>10}  {'est_mean':>10}  "
      f"{'true_std':>9}  {'est_std':>8}  {'in 90% CI':>10}")
print("  " + "-" * 60)

rng_eval2 = np.random.RandomState(0)
for x_val in x_test_vals:
    x_obs     = np.array([[x_val]])
    mu_t, s_t = analytical_posterior(x_val)
    samp      = posterior.sample((N_POST,), x=x_obs, seed=0)
    mu_e      = samp[:, 0].mean()
    std_e     = samp[:, 0].std()
    lo, hi    = np.percentile(samp[:, 0], [5, 95])
    inside    = "✓" if lo <= mu_t <= hi else "✗"
    print(f"  {x_val:>6.1f}  {mu_t:>10.4f}  {mu_e:>10.4f}  "
          f"{s_t:>9.4f}  {std_e:>8.4f}  {inside:>10}")

print()
print(f"  Mean absolute posterior mean error : {mean_err_avg:.4f}")
print(f"  Mean absolute posterior std  error : {std_err_avg:.4f}")
print(f"  90% CI coverage (expected 0.90)    : {coverage_90:.2f}")
print()

# Pass/fail thresholds - loose (fast training; tighten for quality benchmarks)
assert mean_err_avg  < 0.20, f"Posterior mean error too large: {mean_err_avg:.4f}"
assert std_err_avg   < 0.15, f"Posterior std  error too large: {std_err_avg:.4f}"
assert coverage_90   >= 0.6,  f"90% CI coverage too low: {coverage_90:.2f}"
print("  ✓ All accuracy checks passed.")

# ── Plots ──────────────────────────────────────────────────────────────────────
try:
    import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from helpers import posterior_1d, coverage_plot
    import numpy as _np
    from pathlib import Path
    out = Path(__file__).parent / "outputs"

    # Fig 1: posterior at x_obs=1.5
    x_s = _np.array([[1.5]])
    s   = posterior.sample((2000,), x=x_s, seed=0)
    mu_t, s_t = analytical_posterior(1.5)
    posterior_1d(s[:, 0], true_mean=mu_t, true_std=s_t, x_obs_val=1.5,
                 title="Demo 1 - posterior at x_obs=1.5",
                 out_path=out / "01_posterior_1d.png")

    # Fig 2: coverage across x range
    x_range = _np.linspace(-3, 3, 15)
    t_means, e_means, e_lo, e_hi = [], [], [], []
    for xv in x_range:
        mu_t, _ = analytical_posterior(xv)
        t_means.append(mu_t)
        s2 = posterior.sample((500,), x=_np.array([[xv]]), seed=0)[:, 0]
        e_means.append(s2.mean())
        e_lo.append(_np.percentile(s2, 5))
        e_hi.append(_np.percentile(s2, 95))
    coverage_plot(x_range, _np.array(t_means), _np.array(e_means),
                  _np.array(e_lo), _np.array(e_hi),
                  title="Demo 1 - 90% CI coverage",
                  out_path=out / "01_coverage.png")
except Exception as e:
    print(f"  (plots skipped: {e})")
