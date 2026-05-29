"""
MI4 tests - MCMC posterior sampling.

Covers:
  1. MetropolisHastings: samples shape, finiteness, stays near true posterior.
  2. HMC: same checks with JAX backend.
  3. r_hat: converges to ~1 on well-mixed chains; flags non-convergence.
  4. effective_sample_size: positive and <= chain length.
  5. SNPE.build_posterior(sample_with='mcmc') end-to-end.
  6. Posterior.sample() dispatches to MCMC correctly.
  7. Negative-path: invalid mcmc_method, batched x, out-of-support init.
"""
import numpy as np
import pytest

pytest.importorskip("autograd", reason="autograd not installed")

from vbi.inference import (
    SNPE, BoxUniform,
    MetropolisHastings, HMC, NUTS, r_hat, effective_sample_size,
)
from vbi.inference._estimators.maf import MAFEstimator


# ---------------------------------------------------------------------------
# Shared fixture: 1-D Gaussian, analytical posterior known
# ---------------------------------------------------------------------------

SIGMA_LIK   = 0.3
SIGMA_PRIOR = 2.0
TRUE_THETA  = 0.8


@pytest.fixture(scope="module")
def trained_1d():
    """
    Trains MAF on 1-D Gaussian simulator and returns
    (estimator, prior, x_obs, mu_true, sigma_true).
    """
    rng   = np.random.default_rng(0)
    prior = BoxUniform(low=np.array([-3.0]), high=np.array([3.0]))

    theta = prior.sample((600,), seed=0)
    x     = theta + rng.normal(0, SIGMA_LIK, theta.shape)
    x_obs = np.array([[TRUE_THETA + rng.normal(0, SIGMA_LIK)]])

    # Analytical posterior for comparison
    prec_p  = 1 / SIGMA_PRIOR ** 2
    prec_l  = 1 / SIGMA_LIK   ** 2
    s_true  = np.sqrt(1 / (prec_p + prec_l))
    mu_true = s_true ** 2 * prec_l * x_obs[0, 0]

    inf = SNPE(prior=prior, density_estimator="maf", backend="numpy")
    inf.append_simulations(theta, x)
    est = inf.train(
        training_batch_size=128,
        max_num_epochs=100,
        stop_after_epochs=30,
        learning_rate=5e-3,
        verbose=False,
    )
    return est, prior, x_obs, mu_true, s_true


# ---------------------------------------------------------------------------
# MetropolisHastings
# ---------------------------------------------------------------------------

class TestMetropolisHastings:

    def test_samples_shape(self, trained_1d):
        est, prior, x_obs, mu_true, s_true = trained_1d
        mh = MetropolisHastings(est, prior=prior, step_size=0.1)
        samples = mh.run(x_obs, n_samples=200, n_warmup=100, seed=0)
        assert samples.shape == (200, 1)

    def test_samples_finite(self, trained_1d):
        est, prior, x_obs, mu_true, s_true = trained_1d
        mh = MetropolisHastings(est, prior=prior, step_size=0.1)
        samples = mh.run(x_obs, n_samples=200, n_warmup=100, seed=1)
        assert np.all(np.isfinite(samples))

    def test_mean_near_true(self, trained_1d):
        est, prior, x_obs, mu_true, s_true = trained_1d
        mh = MetropolisHastings(est, prior=prior, step_size=0.15)
        samples = mh.run(x_obs, n_samples=500, n_warmup=300, seed=2)
        mean_err = abs(samples[:, 0].mean() - mu_true)
        assert mean_err < 0.4, f"MH mean error too large: {mean_err:.3f}"

    def test_no_prior_still_runs(self, trained_1d):
        est, prior, x_obs, _, _ = trained_1d
        mh = MetropolisHastings(est, prior=None, step_size=0.1)
        s  = mh.run(x_obs, n_samples=100, n_warmup=50, seed=3)
        assert s.shape == (100, 1)


# ---------------------------------------------------------------------------
# SNPE.build_posterior(sample_with='mcmc') end-to-end
# ---------------------------------------------------------------------------

class TestSNPEMCMCPosterior:

    def test_mh_via_build_posterior(self, trained_1d):
        est, prior, x_obs, mu_true, s_true = trained_1d
        inf = SNPE(prior=prior, density_estimator="maf", backend="numpy")
        # inject trained estimator manually
        inf._estimator = est
        inf._rounds    = [
            (np.zeros((1, 1), "f"), np.zeros((1, 1), "f"), None)
        ]

        post = inf.build_posterior(
            est,
            sample_with="mcmc",
            mcmc_method="mh",
            mcmc_step_size=0.15,
            mcmc_num_warmup=100,
        )
        samples = post.sample((100,), x=x_obs, seed=10)
        assert samples.shape == (100, 1)
        assert np.all(np.isfinite(samples))

    def test_posterior_sample_with_mcmc_attr(self, trained_1d):
        """Posterior constructed with sample_with='mcmc' uses MCMC in .sample()."""
        from vbi.inference import Posterior
        est, prior, x_obs, _, _ = trained_1d
        post = Posterior(est, prior=prior, sample_with="mcmc",
                         mcmc_step_size=0.15, mcmc_num_warmup=50)
        s = post.sample((50,), x=x_obs, seed=5)
        assert s.shape == (50, 1)


# ---------------------------------------------------------------------------
# HMC (JAX only)
# ---------------------------------------------------------------------------

class TestHMC:

    def test_hmc_requires_jax_estimator(self, trained_1d):
        est, prior, x_obs, _, _ = trained_1d
        # autograd MAFEstimator - should raise TypeError
        with pytest.raises(TypeError, match="JAX-backend"):
            HMC(est, prior=prior)

    def test_hmc_jax_samples_shape(self, trained_1d):
        pytest.importorskip("jax")
        from vbi.inference._backends.jax_.maf_jax import JaxMAFEstimator

        est_base, prior, x_obs, mu_true, s_true = trained_1d
        rng   = np.random.default_rng(20)
        theta = prior.sample((400,), seed=20)
        x     = theta + rng.normal(0, SIGMA_LIK, theta.shape)

        jax_est = JaxMAFEstimator(n_flows=2, hidden_units=16, num_blocks=1)
        jax_est.train(theta, x, n_iter=80, learning_rate=5e-3, seed=0,
                      verbose=False, validation_fraction=0.0, stop_after_epochs=80,
                      lr_schedule=None)

        hmc = HMC(jax_est, prior=prior, step_size=0.1, max_tree_depth=4)
        samples = hmc.run(x_obs, n_samples=100, n_warmup=100, seed=21)
        assert samples.shape == (100, 1)
        assert np.all(np.isfinite(samples))

    def test_nuts_alias(self, trained_1d):
        """NUTS is a backward-compatible alias for HMC."""
        assert NUTS is HMC


# ---------------------------------------------------------------------------
# Negative-path tests (findings 4, 5, 6)
# ---------------------------------------------------------------------------

class TestNegativePaths:

    def test_invalid_mcmc_method_raises(self, trained_1d):
        """Unknown mcmc_method should raise ValueError at construction time."""
        from vbi.inference import Posterior
        est, prior, x_obs, _, _ = trained_1d
        with pytest.raises(ValueError, match="mcmc_method"):
            Posterior(est, prior=prior, sample_with="mcmc", mcmc_method="nust")

    def test_batched_x_raises_for_mcmc(self, trained_1d):
        """Batched observations must be rejected with a clear error."""
        from vbi.inference import Posterior
        est, prior, x_obs, _, _ = trained_1d
        post = Posterior(est, prior=prior, sample_with="mcmc",
                         mcmc_step_size=0.15, mcmc_num_warmup=10)
        x_batched = np.tile(x_obs, (3, 1))   # shape (3, feature_dim)
        with pytest.raises(ValueError, match="single observation"):
            post.sample((10,), x=x_batched, seed=0)

    def test_mh_samples_within_prior_support(self, trained_1d):
        """MH with a bounded prior must stay within bounds (support masking)."""
        est, prior, x_obs, _, _ = trained_1d
        # prior is BoxUniform([-3], [3])
        mh = MetropolisHastings(est, prior=prior, step_size=0.5)
        samples = mh.run(x_obs, n_samples=200, n_warmup=100, seed=42)
        assert np.all(samples >= -3.0) and np.all(samples <= 3.0), \
            "MH samples escaped BoxUniform support"


# ---------------------------------------------------------------------------
# Diagnostics: r_hat and effective_sample_size
# ---------------------------------------------------------------------------

class TestDiagnostics:

    def test_r_hat_converged(self):
        """Well-mixed Gaussian chains should give R-hat near 1."""
        rng    = np.random.default_rng(99)
        chains = rng.normal(0, 1, (4, 1000, 2))   # (chains, samples, params)
        rh     = r_hat(chains)
        assert rh.shape == (2,)
        assert np.all(rh < 1.1), f"R-hat too large for iid chains: {rh}"

    def test_r_hat_flags_diverged(self):
        """Chains with very different means should give R-hat > 1.2."""
        rng = np.random.default_rng(100)
        chain1 = rng.normal( 5.0, 0.1, (500, 1))
        chain2 = rng.normal(-5.0, 0.1, (500, 1))
        chains = np.stack([chain1, chain2], axis=0)   # (2, 500, 1)
        rh     = r_hat(chains)
        assert rh[0] > 1.2, f"R-hat should flag diverged chains, got {rh}"

    def test_r_hat_single_chain(self):
        """Single chain: R-hat returns nan (undefined with M=1)."""
        chain = np.random.default_rng(0).normal(0, 1, (1, 100, 2))
        rh    = r_hat(chain)
        assert np.all(np.isnan(rh))

    def test_ess_positive(self):
        rng    = np.random.default_rng(50)
        chain  = rng.normal(0, 1, (500, 3))
        ess    = effective_sample_size(chain)
        assert ess.shape == (3,)
        assert np.all(ess > 0)
        assert np.all(ess <= 500)

    def test_ess_autocorrelated_lower(self):
        """Highly autocorrelated chain has lower ESS than iid."""
        rng = np.random.default_rng(60)
        # AR(1) with phi=0.95
        n, phi = 2000, 0.95
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = phi * x[i - 1] + rng.normal(0, 1) * (1 - phi ** 2) ** 0.5
        ess_ar  = float(effective_sample_size(x[:, None])[0])
        ess_iid = float(effective_sample_size(rng.normal(0, 1, (n, 1)))[0])
        assert ess_ar < ess_iid / 2, \
            f"AR ESS ({ess_ar:.0f}) should be much less than iid ({ess_iid:.0f})"
