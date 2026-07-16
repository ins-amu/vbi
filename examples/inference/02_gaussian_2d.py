"""
Demo 2 - 2-D correlated Gaussian: checks that covariance structure is captured.

Simulator:   x = A @ theta + noise,   noise ~ N(0, I * sigma²)
Prior:       theta ~ Uniform([-3,-3], [3,3])
Posterior:   theta | x  - analytically derived (linear-Gaussian)

With A = [[1,0],[0,1]] (identity) and BoxUniform prior, the posterior is
approximately Gaussian near the observation, which lets us check that the
estimator recovers the right mean AND that the two parameters are inferred
independently (zero covariance expected).

We also add a cross-correlation case: x[0] = theta[0] + 0.5*theta[1] + noise,
x[1] = theta[1] + noise  - here the joint posterior shows correlation.

Expected runtime: < 15 seconds.
"""

import numpy as np
from vbi.inference import SNPE, BoxUniform

SIGMA = 0.2
N_TRAIN = 3000
SEED = 7

rng = np.random.default_rng(SEED)

# ── Case A: identity simulator (independent parameters) ───────────────────────
print("=" * 60)
print("Demo 2 - 2-D Gaussian (covariance structure)")
print("=" * 60)

prior = BoxUniform(low=np.array([-3., -3.]), high=np.array([3., 3.]))
theta_tr = prior.sample((N_TRAIN,), seed=SEED)
x_tr     = theta_tr + rng.normal(0, SIGMA, theta_tr.shape)

inf = SNPE(prior=prior, density_estimator="maf")
inf.append_simulations(theta_tr, x_tr)
est = inf.train(
    training_batch_size=256, learning_rate=5e-4,
    stop_after_epochs=20, max_num_epochs=300, verbose=False,
)
post = inf.build_posterior(est)

x_obs = np.array([[1.0, -1.0]])
samp  = post.sample((3000,), x=x_obs, seed=0)   # (3000, 2)

mu_est  = samp.mean(axis=0)
std_est = samp.std(axis=0)
corr_est = np.corrcoef(samp[:, 0], samp[:, 1])[0, 1]

print("\n  Case A - identity: x = theta + noise (params should be independent)")
print(f"  x_obs = {x_obs[0]}")
print(f"  Posterior mean : {mu_est}  (expected ≈ {x_obs[0]})")
print(f"  Posterior std  : {std_est}  (expected ≈ [{SIGMA:.2f}, {SIGMA:.2f}])")
print(f"  Correlation    : {corr_est:.3f}  (expected ≈ 0.0)")

assert np.all(np.abs(mu_est - x_obs[0]) < 0.15), \
    f"Posterior mean too far from truth: {mu_est}"
assert abs(corr_est) < 0.25, \
    f"Unexpected correlation in independent case: {corr_est:.3f}"
print("  ✓ Independent case: OK")

# ── Case B: correlated simulator ──────────────────────────────────────────────
# x0 = t0 + 0.5*t1 + noise,  x1 = t1 + noise
# The coupling of x0 to t1 creates posterior correlation between t0 and t1
A = np.array([[1.0, 0.5], [0.0, 1.0]])

theta_tr2 = prior.sample((N_TRAIN,), seed=SEED + 1)
x_tr2     = (A @ theta_tr2.T).T + rng.normal(0, SIGMA, theta_tr2.shape)

inf2 = SNPE(prior=prior, density_estimator="maf")
inf2.append_simulations(theta_tr2, x_tr2)
est2 = inf2.train(
    training_batch_size=256, learning_rate=5e-4,
    stop_after_epochs=20, max_num_epochs=300, verbose=False,
)
post2 = inf2.build_posterior(est2)

x_obs2   = np.array([[1.5, -0.5]])
samp2    = post2.sample((3000,), x=x_obs2, seed=0)
corr2    = np.corrcoef(samp2[:, 0], samp2[:, 1])[0, 1]
mu_est2  = samp2.mean(axis=0)

# Analytical posterior mean via least squares (for the Gaussian case)
# Ignoring prior (it's wide), posterior mean ≈ A^-1 @ x_obs
A_inv    = np.linalg.inv(A)
mu_true2 = (A_inv @ x_obs2.T).T[0]

print("\n  Case B - coupled: x0 = t0 + 0.5*t1, x1 = t1 (params should correlate)")
print(f"  x_obs = {x_obs2[0]}")
print(f"  Posterior mean : {mu_est2}  (expected ≈ {mu_true2})")
print(f"  Correlation    : {corr2:.3f}  (expected < 0, coupling x0↔t1)")

assert np.all(np.abs(mu_est2 - mu_true2) < 0.3), \
    f"Posterior mean too far: est={mu_est2}  true={mu_true2}"
assert corr2 < -0.1, \
    f"Expected negative correlation in coupled case, got {corr2:.3f}"
print("  ✓ Correlated case: OK")

print()
print("  ✓ All 2-D covariance checks passed.")

# ── Plots ──────────────────────────────────────────────────────────────────────
try:
    import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from helpers import pairplot
    from pathlib import Path
    out = Path(__file__).parent / "outputs"

    # Case A - independent: should show no correlation
    pairplot(samp, labels=["θ0", "θ1"],
             points=x_obs[0],   points_label="x_obs",
             title="Demo 2A - independent params (x=θ+noise)",
             out_path=out / "02a_pairplot_independent.png")

    # Case B - coupled: should show negative correlation
    pairplot(samp2, labels=["θ0", "θ1"],
             points=mu_true2,   points_label="analytical mean",
             title="Demo 2B - coupled params (x0=θ0+0.5θ1)",
             out_path=out / "02b_pairplot_coupled.png")
except Exception as e:
    print(f"  (plots skipped: {e})")
