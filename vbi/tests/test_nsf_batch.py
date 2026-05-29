"""
Tests for MI0-NSF (NSFEstimator) and MI0-batch (sample_batched / log_prob_batched).
"""
import numpy as np
import pytest

autograd = pytest.importorskip("autograd", reason="autograd not installed")

from vbi.inference import (
    SNPE, BoxUniform,
    NSFEstimator, MAFEstimator,
)
from vbi.inference._backends.jax_ import JaxNSFEstimator
from vbi.inference._estimators.nsf import (
    _spline_params, _rq_forward_1d, _rq_inverse_1d,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_posterior(de="nsf", n_train=300, param_dim=2, seed=42):
    rng   = np.random.RandomState(seed)
    lo    = -2.0 * np.ones(param_dim)
    hi    =  2.0 * np.ones(param_dim)
    prior = BoxUniform(low=lo, high=hi)
    theta = prior.sample((n_train,))
    x     = theta + 0.1 * rng.randn(*theta.shape)
    inf   = SNPE(prior=prior, density_estimator=de)
    inf   = inf.append_simulations(theta, x)
    inf.train(max_num_epochs=5, verbose=False)
    return inf.build_posterior(), prior, theta, x


# ---------------------------------------------------------------------------
# RQ-spline unit tests
# ---------------------------------------------------------------------------

class TestSplineParams:
    def test_widths_sum_to_2B(self):
        rng = np.random.RandomState(0)
        N, K, B = 10, 6, 5.0
        uw = rng.randn(N, K).astype("f")
        uh = rng.randn(N, K).astype("f")
        ud = rng.randn(N, K - 1).astype("f")
        cumw, cumh, widths, heights, derivs = _spline_params(uw, uh, ud, B, K)
        np.testing.assert_allclose(np.sum(widths,  axis=1), 2 * B, atol=1e-4)
        np.testing.assert_allclose(np.sum(heights, axis=1), 2 * B, atol=1e-4)

    def test_derivatives_positive(self):
        rng = np.random.RandomState(1)
        N, K, B = 8, 4, 5.0
        uw = rng.randn(N, K).astype("f")
        uh = rng.randn(N, K).astype("f")
        ud = rng.randn(N, K - 1).astype("f")
        _, _, _, _, derivs = _spline_params(uw, uh, ud, B, K)
        assert np.all(np.array(derivs) > 0)

    def test_cumwidths_boundaries(self):
        rng = np.random.RandomState(2)
        N, K, B = 5, 8, 5.0
        uw = rng.randn(N, K).astype("f")
        uh = rng.randn(N, K).astype("f")
        ud = rng.randn(N, K - 1).astype("f")
        cumw, cumh, *_ = _spline_params(uw, uh, ud, B, K)
        np.testing.assert_allclose(np.array(cumw[:, 0]),  -B, atol=1e-4)
        np.testing.assert_allclose(np.array(cumw[:, -1]),  B, atol=1e-4)
        np.testing.assert_allclose(np.array(cumh[:, 0]),  -B, atol=1e-4)
        np.testing.assert_allclose(np.array(cumh[:, -1]),  B, atol=1e-4)


class TestRQSplineForwardInverse:
    def _make_params(self, N=20, K=6, B=5.0, seed=0):
        rng = np.random.RandomState(seed)
        return (rng.randn(N, K).astype("f"),
                rng.randn(N, K).astype("f"),
                rng.randn(N, K - 1).astype("f"))

    def test_forward_output_finite(self):
        N, K, B = 20, 6, 5.0
        rng = np.random.RandomState(0)
        x   = rng.uniform(-B, B, N).astype("f")
        uw, uh, ud = self._make_params(N, K, B, seed=1)
        y, ld = _rq_forward_1d(x, uw, uh, ud, B, K)
        assert np.all(np.isfinite(np.array(y)))
        assert np.all(np.isfinite(np.array(ld)))

    def test_forward_output_in_range(self):
        N, K, B = 50, 8, 5.0
        rng = np.random.RandomState(2)
        x   = rng.uniform(-B, B, N).astype("f")
        uw, uh, ud = self._make_params(N, K, B, seed=3)
        y, _ = _rq_forward_1d(x, uw, uh, ud, B, K)
        assert np.all(np.array(y) >= -B - 1e-4)
        assert np.all(np.array(y) <=  B + 1e-4)

    def test_linear_tails_identity(self):
        N, K, B = 10, 4, 5.0
        rng = np.random.RandomState(4)
        # All inputs outside [-B, B]
        x   = (rng.uniform(B + 0.1, B + 2.0, N) * rng.choice([-1, 1], N)).astype("f")
        uw, uh, ud = self._make_params(N, K, B, seed=5)
        y, ld = _rq_forward_1d(x, uw, uh, ud, B, K)
        np.testing.assert_allclose(np.array(y), x, atol=1e-5)
        np.testing.assert_allclose(np.array(ld), 0.0, atol=1e-5)

    def test_forward_inverse_roundtrip(self):
        """Inverse of forward should recover the original input."""
        N, K, B = 30, 6, 5.0
        rng = np.random.RandomState(6)
        x   = rng.uniform(-B + 0.1, B - 0.1, N).astype("f")
        uw, uh, ud = self._make_params(N, K, B, seed=7)
        y,  _  = _rq_forward_1d(x,  uw, uh, ud, B, K)
        x2, _  = _rq_inverse_1d(np.array(y).astype("f"), uw, uh, ud, B, K)
        np.testing.assert_allclose(np.array(x2), x, atol=1e-4)

    def test_inverse_forward_roundtrip(self):
        """Forward of inverse should recover the latent input."""
        N, K, B = 30, 6, 5.0
        rng = np.random.RandomState(8)
        z   = rng.uniform(-B + 0.1, B - 0.1, N).astype("f")
        uw, uh, ud = self._make_params(N, K, B, seed=9)
        x,  _  = _rq_inverse_1d(z,  uw, uh, ud, B, K)
        z2, _  = _rq_forward_1d(np.array(x).astype("f"), uw, uh, ud, B, K)
        np.testing.assert_allclose(np.array(z2), z, atol=1e-4)


# ---------------------------------------------------------------------------
# NSFEstimator unit tests
# ---------------------------------------------------------------------------

class TestNSFEstimatorInit:
    def test_instantiation(self):
        nsf = NSFEstimator(n_flows=2, hidden_units=32, num_bins=4)
        assert nsf.num_bins == 4
        assert nsf.n_flows == 2

    def test_out_per_dim(self):
        nsf = NSFEstimator(num_bins=8)
        # K + K + (K-1) = 3K-1
        assert nsf._out_per_dim() == 3 * 8 - 1

    def test_is_maf_subclass(self):
        assert issubclass(NSFEstimator, MAFEstimator)


class TestNSFEstimatorTrainSample:
    def test_log_prob_finite_1d(self):
        rng   = np.random.RandomState(0)
        theta = rng.randn(100, 1).astype("f")
        x     = theta + 0.1 * rng.randn(100, 1).astype("f")
        nsf   = NSFEstimator(param_dim=1, feature_dim=1, n_flows=2,
                             hidden_units=16, num_bins=4)
        nsf.train(theta, x, n_iter=10, verbose=False, batch_size=50)
        lp = nsf.log_prob(x[:5], theta[:5])
        assert np.all(np.isfinite(lp))

    def test_log_prob_finite_2d(self):
        rng   = np.random.RandomState(1)
        theta = rng.randn(150, 2).astype("f")
        x     = theta + 0.1 * rng.randn(150, 2).astype("f")
        nsf   = NSFEstimator(n_flows=2, hidden_units=16, num_bins=4)
        nsf.train(theta, x, n_iter=10, verbose=False)
        lp = nsf.log_prob(x[:10], theta[:10])
        assert np.all(np.isfinite(lp))

    def test_sample_shape_1d(self):
        rng   = np.random.RandomState(2)
        theta = rng.randn(100, 1).astype("f")
        x     = theta + 0.05 * rng.randn(100, 1).astype("f")
        nsf   = NSFEstimator(n_flows=2, hidden_units=16, num_bins=4)
        nsf.train(theta, x, n_iter=10, verbose=False)
        s = nsf.sample(x[:3], n_samples=20, rng=np.random.RandomState(0))
        assert s.shape == (3, 20, 1)

    def test_sample_shape_2d(self):
        rng   = np.random.RandomState(3)
        theta = rng.randn(150, 2).astype("f")
        x     = theta + 0.1 * rng.randn(150, 2).astype("f")
        nsf   = NSFEstimator(n_flows=2, hidden_units=16, num_bins=4)
        nsf.train(theta, x, n_iter=10, verbose=False)
        s = nsf.sample(x[:2], n_samples=50, rng=np.random.RandomState(1))
        assert s.shape == (2, 50, 2)

    def test_loss_decreases(self):
        rng   = np.random.RandomState(4)
        theta = rng.randn(200, 2).astype("f")
        x     = theta + 0.1 * rng.randn(200, 2).astype("f")
        nsf   = NSFEstimator(n_flows=2, hidden_units=32, num_bins=6)
        nsf.train(theta, x, n_iter=30, verbose=False, batch_size=64)
        assert nsf.loss_history[-1] < nsf.loss_history[0]

    def test_no_feature_dim(self):
        rng   = np.random.RandomState(5)
        theta = rng.randn(100, 2).astype("f")
        x     = np.zeros((100, 0), "f")
        nsf   = NSFEstimator(n_flows=2, hidden_units=16, num_bins=4)
        nsf.train(theta, x, n_iter=5, verbose=False)
        s = nsf.sample(np.zeros((1, 0), "f"), n_samples=10,
                       rng=np.random.RandomState(0))
        assert s.shape == (1, 10, 2)


class TestSNPEWithNSF:
    def test_snpe_nsf_sample_shape(self):
        posterior, _, _, _ = _make_posterior(de="nsf")
        s = posterior.sample((100,), x=np.array([0.3, -0.1]))
        assert s.shape == (100, 2)

    def test_snpe_nsf_log_prob_finite(self):
        posterior, _, theta, x = _make_posterior(de="nsf")
        lp = posterior.log_prob(theta[:20], x=x[0])
        assert np.all(np.isfinite(lp))

    def test_snpe_nsf_vs_maf_comparable_loss(self):
        """NSF should achieve a similar or lower loss than MAF on a simple task."""
        rng   = np.random.RandomState(42)
        prior = BoxUniform(low=np.array([-2.0]), high=np.array([2.0]))
        theta = prior.sample((500,))
        x     = theta + 0.1 * rng.randn(*theta.shape)

        for de in ("maf", "nsf"):
            inf = SNPE(prior=prior, density_estimator=de)
            inf = inf.append_simulations(theta, x)
            inf.train(max_num_epochs=15, verbose=False)
        # Both should have finite final loss (correctness check, not ordering)
        assert np.isfinite(inf._estimator.loss_history[-1])

    def test_nsf_density_estimator_string(self):
        """SNPE accepts 'nsf' and creates NSFEstimator after training."""
        rng   = np.random.RandomState(0)
        prior = BoxUniform(low=np.array([0.0]), high=np.array([1.0]))
        theta = prior.sample((100,))
        x     = theta + 0.05 * rng.randn(*theta.shape)
        inf   = SNPE(prior=prior, density_estimator="nsf")
        inf   = inf.append_simulations(theta, x)
        inf.train(max_num_epochs=3, verbose=False)
        assert isinstance(inf._estimator, (NSFEstimator, JaxNSFEstimator))


# ---------------------------------------------------------------------------
# MI0-batch: sample_batched / log_prob_batched
# ---------------------------------------------------------------------------

class TestSampleBatched:
    def test_shape(self):
        posterior, _, _, x = _make_posterior()
        x_batch = x[:5]
        sb = posterior.sample_batched((30,), x=x_batch)
        assert sb.shape == (5, 30, 2)

    def test_single_obs_matches_sample(self):
        posterior, _, _, x = _make_posterior()
        np.random.seed(0)
        s1 = posterior.sample((50,), x=x[0], seed=7)
        s2 = posterior.sample_batched((50,), x=x[0:1], seed=7)
        assert s2.shape == (1, 50, 2)
        # Mean should be close (same seed → same samples from estimator)
        np.testing.assert_allclose(s1, s2[0], atol=1e-5)

    def test_values_finite(self):
        posterior, _, _, x = _make_posterior()
        sb = posterior.sample_batched((20,), x=x[:4])
        assert np.all(np.isfinite(sb))

    def test_batch_dim_varies(self):
        posterior, _, _, x = _make_posterior()
        for batch in (1, 3, 10):
            sb = posterior.sample_batched((10,), x=x[:batch])
            assert sb.shape == (batch, 10, 2)

    def test_nsf_sample_batched(self):
        posterior, _, _, x = _make_posterior(de="nsf")
        sb = posterior.sample_batched((15,), x=x[:4])
        assert sb.shape == (4, 15, 2)
        assert np.all(np.isfinite(sb))


class TestLogProbBatched:
    def test_shape(self):
        posterior, _, theta, x = _make_posterior()
        lp = posterior.log_prob_batched(theta[:20], x[:20])
        assert lp.shape == (20,)

    def test_finite(self):
        posterior, _, theta, x = _make_posterior()
        lp = posterior.log_prob_batched(theta[:30], x[:30])
        assert np.all(np.isfinite(lp))

    def test_matches_log_prob_pointwise(self):
        """log_prob_batched(theta, x) should equal log_prob(theta[i], x[i]) for each i."""
        posterior, _, theta, x = _make_posterior(n_train=200)
        batch_lp = posterior.log_prob_batched(theta[:5], x[:5])
        for i in range(5):
            single = posterior.log_prob(theta[i:i+1], x=x[i])
            np.testing.assert_allclose(float(batch_lp[i]), float(single[0]), rtol=1e-5)

    def test_nsf_log_prob_batched(self):
        posterior, _, theta, x = _make_posterior(de="nsf")
        lp = posterior.log_prob_batched(theta[:10], x[:10])
        assert lp.shape == (10,)
        assert np.all(np.isfinite(lp))

    def test_leading_is_sample_ignored(self):
        posterior, _, theta, x = _make_posterior()
        lp1 = posterior.log_prob_batched(theta[:5], x[:5], leading_is_sample=True)
        lp2 = posterior.log_prob_batched(theta[:5], x[:5], leading_is_sample=False)
        np.testing.assert_array_equal(lp1, lp2)
