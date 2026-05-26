"""
Tests for vbi.inference — sbi-compatible API.

Covers:
  - Prior objects (BoxUniform, Gaussian, CustomPrior)
  - SNPE workflow: append_simulations → train → build_posterior
  - Posterior: sample, log_prob, set_default_x  (sbi-compatible signatures)
  - MDN and MAF density estimators
  - vbi.cde shim emits DeprecationWarning
  - Mini-batch training matches full-batch (loose tolerance)
"""
import warnings

import numpy as np
import pytest

try:
    import autograd  # noqa: F401
    AUTOGRAD_AVAILABLE = True
except ImportError:
    AUTOGRAD_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not AUTOGRAD_AVAILABLE, reason="autograd not installed"
)

from vbi.inference import (
    SNPE, BoxUniform, Gaussian, CustomPrior,
    Posterior, MDNEstimator, MAFEstimator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _linear_gaussian(n: int = 300, d: int = 2, seed: int = 0):
    """Simple linear-Gaussian simulator: x = theta + N(0, 0.1)."""
    rng   = np.random.default_rng(seed)
    prior = BoxUniform(low=np.zeros(d), high=np.ones(d))
    theta = prior.sample((n,), seed=seed)
    x     = theta + rng.normal(0, 0.1, theta.shape)
    return prior, theta, x


# ---------------------------------------------------------------------------
# Prior tests
# ---------------------------------------------------------------------------

class TestBoxUniform:

    def test_sample_shape(self):
        p = BoxUniform(low=np.zeros(3), high=np.ones(3))
        s = p.sample((100,))
        assert s.shape == (100, 3)

    def test_log_prob_inside_zero(self):
        p  = BoxUniform(low=np.zeros(2), high=np.ones(2))
        lp = p.log_prob(np.array([[0.5, 0.5]]))
        assert np.isfinite(lp[0])

    def test_log_prob_outside_neginf(self):
        p  = BoxUniform(low=np.zeros(2), high=np.ones(2))
        lp = p.log_prob(np.array([[1.5, 0.5]]))
        assert lp[0] == -np.inf

    def test_invalid_bounds(self):
        with pytest.raises(ValueError):
            BoxUniform(low=np.ones(2), high=np.zeros(2))


class TestGaussian:

    def test_sample_shape(self):
        p = Gaussian(mean=np.zeros(3), std=np.ones(3))
        s = p.sample((50,))
        assert s.shape == (50, 3)

    def test_log_prob_shape(self):
        p  = Gaussian(mean=np.zeros(2), std=np.ones(2))
        lp = p.log_prob(np.random.randn(20, 2))
        assert lp.shape == (20,)

    def test_log_prob_finite(self):
        p  = Gaussian(mean=np.zeros(2), std=np.ones(2))
        lp = p.log_prob(np.array([[0.0, 0.0]]))
        assert np.isfinite(lp[0])


class TestCustomPrior:

    def test_sample_and_log_prob(self):
        p = CustomPrior(
            sample_fn   = lambda shape: np.random.uniform(0, 1, (*shape, 2)),
            log_prob_fn = lambda t: np.zeros(len(t)),
            dim         = 2,
        )
        s  = p.sample((10,))
        lp = p.log_prob(s)
        assert s.shape == (10, 2)
        assert lp.shape == (10,)


# ---------------------------------------------------------------------------
# SNPE workflow tests
# ---------------------------------------------------------------------------

class TestSNPEWorkflow:

    @pytest.fixture
    def data(self):
        return _linear_gaussian(n=200, d=2)

    def _train(self, de, data, n_epochs=20):
        prior, theta, x = data
        inf = SNPE(prior=prior, density_estimator=de)
        inf.append_simulations(theta, x)
        est = inf.train(
            training_batch_size=64,
            learning_rate=1e-3,
            stop_after_epochs=n_epochs,
            max_num_epochs=n_epochs,
            verbose=False,
        )
        return inf, est

    @pytest.mark.parametrize("de", ["maf", "mdn"])
    def test_full_workflow(self, de, data):
        prior, theta, x = data
        inf, est = self._train(de, data)
        post = inf.build_posterior(est)

        x_obs   = np.array([[0.5, 0.5]])
        samples = post.sample((200,), x=x_obs)
        assert samples.shape == (200, 2), f"{de}: sample shape wrong"
        assert np.all(np.isfinite(samples)),    f"{de}: non-finite samples"

    def test_append_returns_self(self, data):
        prior, theta, x = data
        inf  = SNPE(prior=prior)
        ret  = inf.append_simulations(theta, x)
        assert ret is inf

    def test_n_simulations_counter(self, data):
        prior, theta, x = data
        inf = SNPE(prior=prior)
        inf.append_simulations(theta[:100], x[:100])
        inf.append_simulations(theta[100:], x[100:])
        assert inf.n_simulations == len(theta)
        assert inf.n_rounds == 2

    def test_train_without_data_raises(self):
        with pytest.raises(RuntimeError, match="No simulations"):
            SNPE(prior=None).train()

    def test_build_posterior_without_train_raises(self):
        with pytest.raises(RuntimeError, match="No trained estimator"):
            SNPE(prior=None).build_posterior()


# ---------------------------------------------------------------------------
# Posterior sbi-compatible API tests
# ---------------------------------------------------------------------------

class TestPosterior:

    @pytest.fixture
    def posterior(self):
        prior, theta, x = _linear_gaussian(n=200, d=2)
        inf = SNPE(prior=prior, density_estimator='maf')
        inf.append_simulations(theta, x)
        est = inf.train(
            training_batch_size=64,
            max_num_epochs=15,
            verbose=False,
        )
        return inf.build_posterior(est)

    def test_sample_shape(self, posterior):
        s = posterior.sample((300,), x=np.array([[0.5, 0.5]]))
        assert s.shape == (300, 2)

    def test_sample_tuple_shape(self, posterior):
        """sbi uses tuple shape like (1000,)."""
        s = posterior.sample((100,), x=np.array([[0.3, 0.7]]))
        assert s.shape == (100, 2)

    def test_log_prob_theta_first(self, posterior):
        """sbi signature: log_prob(theta, x=x_obs) — theta FIRST."""
        # Use in-distribution theta (within the prior [0,1]²)
        rng   = np.random.default_rng(0)
        theta = rng.uniform(0, 1, (20, 2)).astype("f")
        lp    = posterior.log_prob(theta, x=np.array([[0.5, 0.5]]))
        assert lp.shape == (20,)
        assert np.all(np.isfinite(lp))

    def test_set_default_x(self, posterior):
        posterior.set_default_x(np.array([[0.5, 0.5]]))
        s = posterior.sample((50,))   # no x= argument
        assert s.shape == (50, 2)

    def test_no_x_raises(self, posterior):
        with pytest.raises(ValueError, match="No observation provided"):
            posterior.sample((10,))

    def test_prior_log_prob_added(self):
        """Posterior with Gaussian prior adds prior.log_prob to estimator output."""
        prior, theta, x = _linear_gaussian(n=100, d=1)
        inf  = SNPE(prior=prior, density_estimator='mdn')
        inf.append_simulations(theta, x)
        est  = inf.train(max_num_epochs=5, verbose=False)

        # Use Gaussian prior so log_prob is non-zero (BoxUniform is 0 inside)
        gaussian_prior = Gaussian(mean=np.array([0.5]), std=np.array([0.2]))
        post_with    = Posterior(estimator=est, prior=gaussian_prior)
        post_without = Posterior(estimator=est, prior=None)
        x_obs = np.array([[0.5]])
        t     = theta[:5].reshape(-1, 1)
        lp_w  = post_with.log_prob(t,    x=x_obs)
        lp_wo = post_without.log_prob(t, x=x_obs)
        # Gaussian prior adds a non-zero term; values should differ
        assert not np.allclose(lp_w, lp_wo)


# ---------------------------------------------------------------------------
# Mini-batch test
# ---------------------------------------------------------------------------

class TestMiniBatch:

    def test_minibatch_loss_converges(self):
        """Mini-batch training should reduce loss compared to no training."""
        prior, theta, x = _linear_gaussian(n=300, d=2)
        est = MAFEstimator()
        est.train(
            params=theta, features=x,
            n_iter=30, learning_rate=1e-3,
            batch_size=64, verbose=False,
        )
        assert len(est.loss_history) > 0
        # Loss should decrease overall
        assert est.loss_history[-1] < est.loss_history[0]


# ---------------------------------------------------------------------------
# Rejection sampling tests
# ---------------------------------------------------------------------------

class TestRejectionSampling:
    """MI0-rejection: samples stay within prior support; leakage_correction works."""

    @pytest.fixture
    def posterior_with_prior(self):
        """Train a small MAF posterior on a BoxUniform [0,1]^2 prior."""
        prior, theta, x = _linear_gaussian(n=300, d=2)
        inf = SNPE(prior=prior, density_estimator='maf')
        inf.append_simulations(theta, x)
        est = inf.train(
            training_batch_size=64,
            max_num_epochs=15,
            verbose=False,
        )
        return inf.build_posterior(est), prior

    def test_direct_samples_may_leak(self, posterior_with_prior):
        """Baseline: direct sampling does NOT enforce prior bounds (expected)."""
        posterior, prior = posterior_with_prior
        # Just verify the call works and returns the right shape.
        x_obs   = np.array([[0.5, 0.5]])
        samples = posterior.sample((200,), x=x_obs)
        assert samples.shape == (200, 2)

    def test_reject_outside_prior_samples_in_bounds(self, posterior_with_prior):
        """With reject_outside_prior=True, all samples are within [0,1]^2."""
        posterior, prior = posterior_with_prior
        x_obs   = np.array([[0.5, 0.5]])
        samples = posterior.sample((200,), x=x_obs, reject_outside_prior=True)
        assert samples.shape == (200, 2)
        assert np.all(samples >= 0.0), "samples below prior lower bound"
        assert np.all(samples <= 1.0), "samples above prior upper bound"

    def test_build_posterior_rejection_mode(self):
        """build_posterior(sample_with='rejection') auto-applies rejection."""
        prior, theta, x = _linear_gaussian(n=300, d=2)
        inf = SNPE(prior=prior, density_estimator='maf')
        inf.append_simulations(theta, x)
        est  = inf.train(max_num_epochs=10, verbose=False)
        post = inf.build_posterior(est, sample_with='rejection')
        x_obs   = np.array([[0.5, 0.5]])
        samples = post.sample((150,), x=x_obs)
        assert samples.shape == (150, 2)
        assert np.all(samples >= 0.0)
        assert np.all(samples <= 1.0)

    def test_leakage_correction_returns_float_in_range(self, posterior_with_prior):
        """leakage_correction returns a float in (0, 1]."""
        posterior, prior = posterior_with_prior
        x_obs = np.array([[0.5, 0.5]])
        frac  = posterior.leakage_correction(x_obs, num_rejection_samples=500)
        assert isinstance(frac, float)
        assert 0.0 < frac <= 1.0

    def test_reject_without_prior_raises(self):
        """Rejection sampling without a prior raises a clear error."""
        prior, theta, x = _linear_gaussian(n=100, d=1)
        inf  = SNPE(prior=None, density_estimator='mdn')
        inf.append_simulations(theta, x)
        est  = inf.train(max_num_epochs=5, verbose=False)
        post = inf.build_posterior(est, prior=None)
        with pytest.raises(ValueError, match="prior"):
            post.sample((10,), x=np.array([[0.5]]), reject_outside_prior=True)

    def test_leakage_correction_without_prior_raises(self, posterior_with_prior):
        """leakage_correction without a prior raises ValueError."""
        posterior, _ = posterior_with_prior
        post_no_prior = Posterior(estimator=posterior._estimator, prior=None)
        with pytest.raises(ValueError, match="prior"):
            post_no_prior.leakage_correction(x=np.array([[0.5, 0.5]]))

    def test_build_posterior_mcmc_still_raises(self):
        """sample_with='mcmc' still raises NotImplementedError (MI4)."""
        prior, theta, x = _linear_gaussian(n=100, d=1)
        inf = SNPE(prior=prior)
        inf.append_simulations(theta, x)
        inf.train(max_num_epochs=3, verbose=False)
        with pytest.raises(NotImplementedError, match="mcmc"):
            inf.build_posterior(sample_with='mcmc')


# ---------------------------------------------------------------------------
# Deprecated shim test
# ---------------------------------------------------------------------------

class TestDeprecatedShim:

    def test_cde_import_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import vbi.cde  # noqa: F401
            assert any(issubclass(warning.category, DeprecationWarning)
                       for warning in w), "No DeprecationWarning from vbi.cde"

    def test_cde_classes_still_importable(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from vbi.cde import MDNEstimator, MAFEstimator, ConditionalDensityEstimator
            assert MDNEstimator is not None
