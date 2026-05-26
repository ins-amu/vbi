"""
Demo 5 — Sequential rounds: 1 round vs 2 rounds.

Uses the 1-D Gaussian problem to show that a second round of simulation
focused near the posterior concentrates samples closer to the true parameter.

Round 1: simulate from prior theta ~ N(0, 2)  → broad coverage
Round 2: simulate from the round-1 posterior  → focused near true theta

We compare posterior mean error and std between the two strategies.

Expected runtime: < 20 seconds.
"""

import numpy as np
from vbi.inference import SNPE, Gaussian

SIGMA_LIK   = 0.3
SIGMA_PRIOR = 2.0
TRUE_THETA  = 1.5     # the "true" parameter we're trying to infer
N_ROUND1    = 1500
N_ROUND2    = 500
N_POST      = 1000
SEED        = 0
# monitor_collapse=True (default) stops training automatically when the
# posterior std drops below collapse_threshold * data_std.
# This means max_num_epochs is just a safety cap — training stops
# long before it from either early stopping OR collapse prevention.
max_num_epochs = 2000

rng = np.random.default_rng(SEED)
prior  = Gaussian(mean=np.array([0.0]), std=np.array([SIGMA_PRIOR]))
x_obs  = np.array([[TRUE_THETA + rng.normal(0, SIGMA_LIK)]])

# Analytical posterior
prec_p   = 1 / SIGMA_PRIOR ** 2
prec_l   = 1 / SIGMA_LIK   ** 2
s_true   = np.sqrt(1 / (prec_p + prec_l))
mu_true  = s_true ** 2 * prec_l * x_obs[0, 0]

print("=" * 55)
print("Demo 5 — Sequential rounds")
print("=" * 55)
print(f"  True theta = {TRUE_THETA},  x_obs = {x_obs[0,0]:.3f}")
print(f"  Analytical posterior: mean={mu_true:.3f}  std={s_true:.3f}")


def simulate_from(sampler, n, rng_seed):
    """Sample theta from sampler, simulate x = theta + noise."""
    rng_s  = np.random.default_rng(rng_seed)
    theta  = sampler(n, rng_seed)
    x      = theta + rng_s.normal(0, SIGMA_LIK, theta.shape)
    return theta, x


# ── 1-round strategy ───────────────────────────────────────────────────────────
theta1, x1 = simulate_from(lambda n, s: prior.sample((n,), seed=s), N_ROUND1, SEED)

inf1 = SNPE(prior=prior, density_estimator="maf")
inf1.append_simulations(theta1, x1)
est1 = inf1.train(
    training_batch_size=256, learning_rate=5e-4,
    stop_after_epochs=20, max_num_epochs=max_num_epochs, verbose=False,
)
post1   = inf1.build_posterior(est1)
samp1   = post1.sample((N_POST,), x=x_obs, seed=0)
me1     = abs(samp1[:, 0].mean() - mu_true)
se1     = abs(samp1[:, 0].std()  - s_true)

print(f"\n  Round 1 only  ({N_ROUND1} sims from prior):")
print(f"    mean = {samp1[:,0].mean():.4f}  err = {me1:.4f}")
print(f"    std  = {samp1[:,0].std():.4f}  err = {se1:.4f}")

# ── 2-round strategy ───────────────────────────────────────────────────────────
# Round 2: sample theta from the round-1 posterior, then simulate.
#
# WHY we use round-2 data ONLY (not round1 + round2):
#   Round 1 was sampled from the prior; round 2 from the posterior.
#   Mixing both proposals without importance-weighting (APT / num_atoms)
#   biases the estimator — the prior-sampled round 1 pulls the posterior
#   back toward the prior.
#   → Correct fix: SNPE-C APT weights (MI3), which corrects for proposal mismatch.
#   → Safe current fix: train on the focused round-2 data only (no bias,
#     less total data, no broad coverage from round 1).
post1.set_default_x(x_obs)
theta2_rng = lambda n, s: post1.sample((n,), seed=s)

theta2, x2 = simulate_from(theta2_rng, N_ROUND2, SEED + 1)

inf2 = SNPE(prior=prior, density_estimator="maf")
# NOTE: do NOT add theta1/x1 here without APT importance weights — it would
# bias the posterior.  Once MI3 (num_atoms) is implemented, uncomment below:
# inf2.append_simulations(theta1, x1)  # round 1 (add when APT is available)
inf2.append_simulations(theta2, x2)    # round 2 — focused near posterior
est2 = inf2.train(
    training_batch_size=256, learning_rate=5e-4,
    stop_after_epochs=20, max_num_epochs=max_num_epochs, verbose=False,
)
post2   = inf2.build_posterior(est2)
samp2   = post2.sample((N_POST,), x=x_obs, seed=0)
me2     = abs(samp2[:, 0].mean() - mu_true)
se2     = abs(samp2[:, 0].std()  - s_true)

print(f"\n  Round 2 only  ({N_ROUND2} focused sims, no APT weights yet):")
print(f"    mean = {samp2[:,0].mean():.4f}  err = {me2:.4f}")
print(f"    std  = {samp2[:,0].std():.4f}  err = {se2:.4f}")

print(f"\n  {'Strategy':>20}  {'Mean err':>10}  {'Std err':>10}")
print("  " + "-" * 44)
print(f"  {'Round 1 (prior)':>20}  {me1:>10.4f}  {se1:>10.4f}")
print(f"  {'Round 2 (focused)':>20}  {me2:>10.4f}  {se2:>10.4f}")
print()
print("  Note: with APT weights (MI3), combining both rounds")
print("  will give lower error than either alone.")

# Both should be accurate; round 2 should use fewer total sims for same quality
assert me1 < 0.2, f"Round-1 mean error too large: {me1:.4f}"
assert me2 < 0.2, f"Round-2 mean error too large: {me2:.4f}"
print()
print("  ✓ Both strategies pass accuracy checks.")
print(f"  ✓ Round 2 used {N_ROUND2} focused sims vs {N_ROUND1} broad sims in round 1.")
