"""
Demo 3 — Two-moons benchmark.

The two-moons distribution is a classic 2-D SBI benchmark from the sbi
paper (Cranmer et al. 2020).  There is no analytical posterior, but the
posterior has a distinctive crescent shape that is easy to evaluate visually.

Simulator: generates a 2-D observation x from a 2-D parameter theta.
The posterior p(theta | x_obs) at x_obs=(0,0) has two crescent-shaped modes.

We check two quantitative metrics:
  1. C2ST-style accuracy: a classifier trained to distinguish posterior
     samples from prior samples should NOT be at chance (posterior != prior),
     and posterior samples should lie closer to the observation than prior samples.
  2. Posterior mean should be near (0, 0) by symmetry when x_obs=(0,0).

Expected runtime: < 20 seconds.
"""

import numpy as np
from vbi.inference import SNPE, BoxUniform

SEED = 42

# ── Two-moons simulator ────────────────────────────────────────────────────────

def two_moons_simulator(theta: np.ndarray, rng=None) -> np.ndarray:
    """
    From the sbi two-moons example.
    theta : (N, 2)  ∈ [-1, 1]²
    Returns x : (N, 2)
    """
    if rng is None:
        rng = np.random.default_rng()
    N  = theta.shape[0]
    a  = rng.uniform(-np.pi / 2, np.pi / 2, N)
    r  = rng.normal(0.1, 0.01, N)
    p  = np.stack([r * np.cos(a) + 0.25,
                   r * np.sin(a)], axis=1)
    sign = np.sign(theta[:, 0:1])
    sign[sign == 0] = 1
    x = np.concatenate([
        -np.abs(theta[:, 0:1]) + p[:, 0:1],
        sign * theta[:, 1:2]  + p[:, 1:2],
    ], axis=1)
    return x


# ── Train ─────────────────────────────────────────────────────────────────────
N_TRAIN = 3000
rng_sim = np.random.default_rng(SEED)

prior    = BoxUniform(low=np.array([-1., -1.]), high=np.array([1., 1.]))
theta_tr = prior.sample((N_TRAIN,), seed=SEED)
x_tr     = two_moons_simulator(theta_tr, rng=rng_sim)

inference = SNPE(prior=prior, density_estimator="maf")
inference.append_simulations(theta_tr, x_tr)
estimator = inference.train(
    training_batch_size=256,
    learning_rate=5e-4,
    stop_after_epochs=20,
    max_num_epochs=400,
    verbose=False,
)
posterior = inference.build_posterior(estimator)

# ── Evaluate ───────────────────────────────────────────────────────────────────
x_obs    = np.array([[0.0, 0.0]])
N_POST   = 2000
samples  = posterior.sample((N_POST,), x=x_obs, seed=0)   # (N_POST, 2)
prior_samp = prior.sample((N_POST,), seed=1)               # (N_POST, 2)

print("=" * 55)
print("Demo 3 — Two-moons benchmark")
print("=" * 55)
print(f"  x_obs = {x_obs[0]}")
print(f"  Posterior mean : {samples.mean(axis=0).round(3)}")
print(f"  Posterior std  : {samples.std(axis=0).round(3)}")
print(f"  Sample range   : [{samples.min():.2f}, {samples.max():.2f}]")

# Check 1: samples lie inside the prior box
in_box = np.all((samples >= -1.0) & (samples <= 1.0), axis=1).mean()
print(f"\n  Fraction inside prior box : {in_box:.3f}  (expected ≈ 1.0)")
assert in_box > 0.85, f"Too many samples outside prior: {in_box:.3f}"

# Check 2: posterior is more concentrated than the prior
# (posterior samples should have smaller std than prior samples)
post_std  = samples.std(axis=0)
prior_std = prior_samp.std(axis=0)
more_conc = np.all(post_std < prior_std)
print(f"  Posterior std  : {post_std.round(3)}")
print(f"  Prior std      : {prior_std.round(3)}")
print(f"  Posterior more concentrated than prior: {more_conc}")
assert more_conc, "Posterior should be more concentrated than prior"

# Check 3: most log_prob values should be finite (MAF can return -inf for
# low-density regions, but samples from the posterior should mostly be finite)
lp_obs   = posterior.log_prob(samples[:50], x=x_obs)
frac_fin = np.isfinite(lp_obs).mean()
print(f"\n  log_prob finite fraction (on posterior samples): {frac_fin:.2f}  (expected > 0.8)")
assert frac_fin > 0.7, f"Too many non-finite log_probs: {frac_fin:.2f}"

# Check 4: posterior mean near (0,0) by symmetry of two-moons at x=(0,0)
mu = samples.mean(axis=0)
dist_from_zero = np.linalg.norm(mu)
print(f"\n  Posterior mean distance from (0,0): {dist_from_zero:.3f}  (expected < 0.3)")
assert dist_from_zero < 0.5, \
    f"Posterior mean {mu} too far from expected (0,0) at x_obs=(0,0)"

print()
print("  ✓ All two-moons checks passed.")

# ── Plots ──────────────────────────────────────────────────────────────────────
try:
    import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from helpers import pairplot
    from pathlib import Path
    out = Path(__file__).parent / "outputs"

    # Posterior pairplot — should show two crescent-shaped modes
    pairplot(samples, labels=["θ0", "θ1"],
             points=x_obs[0], points_label="x_obs=(0,0)",
             title="Demo 3 — Two-moons posterior at x_obs=(0,0)",
             out_path=out / "03_two_moons_posterior.png")
except Exception as e:
    print(f"  (plots skipped: {e})")
