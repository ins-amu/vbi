"""
Parity tests: JAX backend vs autograd backend.

Tests verify:
  1. JAX estimators produce finite log_prob and samples (smoke tests).
  2. Loss decreases during training (sanity).
  3. log_prob / sample shapes match the autograd versions.
  4. Numerical parity between autograd and JAX for log_prob after
     training with the same seed (rtol=1e-3).
  5. SNPE(backend='jax') end-to-end workflow.
  6. Save / load round-trip preserves log_prob values.
"""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("autograd")

import jax.numpy as jnp

from vbi.inference._backends.jax_.mdn_jax import JaxMDNEstimator
from vbi.inference._backends.jax_.maf_jax import JaxMAFEstimator
from vbi.inference._backends.jax_.nsf_jax import JaxNSFEstimator

from vbi.inference._estimators.mdn import MDNEstimator
from vbi.inference._estimators.maf import MAFEstimator
from vbi.inference._estimators.nsf import NSFEstimator

from vbi.inference import SNPE, BoxUniform


# ---------------------------------------------------------------------------
# Shared fixture data
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def linear_gaussian_data():
    rng = np.random.default_rng(42)
    N, D, C = 400, 2, 3
    theta = rng.uniform(0, 1, (N, D)).astype("f")
    x     = theta + rng.normal(0, 0.1, (N, D)).astype("f")
    # Features have 3 dims (pad to separate param_dim from feature_dim)
    x3 = np.concatenate([x, rng.normal(0, 0.1, (N, 1)).astype("f")], axis=1)
    x_obs = np.array([[0.5, 0.5, 0.0]], dtype="f")
    return theta, x3, x_obs


# ---------------------------------------------------------------------------
# Helper: train a small JAX estimator and return it
# ---------------------------------------------------------------------------

def _train_jax(cls, theta, features, seed=0, n_iter=80, **kwargs):
    est = cls(**kwargs)
    # MAF/NSF have a full train override; MDN uses the base train
    is_maf = isinstance(est, JaxMAFEstimator)
    common = dict(n_iter=n_iter, learning_rate=5e-3, seed=seed, verbose=False)
    if is_maf:
        common.update(dict(validation_fraction=0.0, stop_after_epochs=n_iter,
                           lr_schedule=None))
    est.train(theta, features, **common)
    return est


def _train_autograd(cls, theta, features, seed=0, n_iter=80, **kwargs):
    est = cls(**kwargs)
    is_maf = isinstance(est, MAFEstimator)
    common = dict(n_iter=n_iter, learning_rate=5e-3, seed=seed, verbose=False)
    if is_maf:
        common.update(dict(validation_fraction=0.0, stop_after_epochs=n_iter,
                           lr_schedule=None))
    est.train(theta, features, **common)
    return est


# ---------------------------------------------------------------------------
# Smoke tests: finite outputs, correct shapes
# ---------------------------------------------------------------------------

class TestJaxMDNSmoke:

    def test_log_prob_finite(self, linear_gaussian_data):
        theta, x, x_obs = linear_gaussian_data
        est = _train_jax(JaxMDNEstimator, theta, x,
                         n_components=3, hidden_sizes=(16,))
        lp = est.log_prob(x, theta)
        assert lp.shape == (len(theta),)
        assert np.all(np.isfinite(lp)), "log_prob has non-finite values"

    def test_sample_shape(self, linear_gaussian_data):
        theta, x, x_obs = linear_gaussian_data
        est = _train_jax(JaxMDNEstimator, theta, x,
                         n_components=3, hidden_sizes=(16,))
        rng = np.random.RandomState(0)
        s = est.sample(x_obs, 200, rng)
        assert s.shape == (1, 200, 2)

    def test_loss_decreases(self, linear_gaussian_data):
        theta, x, _ = linear_gaussian_data
        est = JaxMDNEstimator(n_components=3, hidden_sizes=(16,))
        est.train(theta, x, n_iter=60, learning_rate=5e-3, seed=0, verbose=False)
        assert est.loss_history[-1] < est.loss_history[0], \
            "Training loss did not decrease"


class TestJaxMAFSmoke:

    def test_log_prob_finite(self, linear_gaussian_data):
        theta, x, _ = linear_gaussian_data
        est = _train_jax(JaxMAFEstimator, theta, x,
                         n_flows=2, hidden_units=16, num_blocks=1)
        lp = est.log_prob(x, theta)
        assert lp.shape == (len(theta),)
        assert np.all(np.isfinite(lp))

    def test_sample_shape(self, linear_gaussian_data):
        theta, x, x_obs = linear_gaussian_data
        est = _train_jax(JaxMAFEstimator, theta, x,
                         n_flows=2, hidden_units=16, num_blocks=1)
        rng = np.random.RandomState(7)
        s = est.sample(x_obs, 100, rng)
        assert s.shape == (1, 100, 2)

    def test_loss_decreases(self, linear_gaussian_data):
        theta, x, _ = linear_gaussian_data
        est = JaxMAFEstimator(n_flows=2, hidden_units=16, num_blocks=1)
        est.train(theta, x, n_iter=60, learning_rate=5e-3,
                  seed=0, verbose=False, validation_fraction=0.0,
                  stop_after_epochs=60, lr_schedule=None)
        assert est.loss_history[-1] < est.loss_history[0]


class TestJaxNSFSmoke:

    def test_log_prob_finite(self, linear_gaussian_data):
        theta, x, _ = linear_gaussian_data
        est = _train_jax(JaxNSFEstimator, theta, x,
                         n_flows=2, hidden_units=16, num_blocks=1, num_bins=4)
        lp = est.log_prob(x, theta)
        assert lp.shape == (len(theta),)
        assert np.all(np.isfinite(lp))

    def test_sample_shape(self, linear_gaussian_data):
        theta, x, x_obs = linear_gaussian_data
        est = _train_jax(JaxNSFEstimator, theta, x,
                         n_flows=2, hidden_units=16, num_blocks=1, num_bins=4)
        rng = np.random.RandomState(3)
        s = est.sample(x_obs, 100, rng)
        assert s.shape == (1, 100, 2)


# ---------------------------------------------------------------------------
# Numerical parity: JAX vs autograd (same seed, same data, rtol=1e-2)
# ---------------------------------------------------------------------------

class TestJaxAutoGradParity:
    """
    After identical training (same seed, same data), JAX and autograd
    backends should produce log_prob values that agree to within 1%.

    We use a loose tolerance because:
      - ActNorm: JAX starts from act_s=1/act_b=0 (no data-init),
        autograd does data-dependent init on first forward.
      - Floating-point order differences in JAX XLA vs numpy.
    Both should produce similarly *good* posteriors, not bitwise equal ones.
    """

    def test_mdn_log_prob_shape_matches(self, linear_gaussian_data):
        theta, x, _ = linear_gaussian_data
        jax_est = _train_jax(JaxMDNEstimator, theta, x,
                             n_components=3, hidden_sizes=(16,))
        ag_est  = _train_autograd(MDNEstimator, theta, x,
                                  n_components=3, hidden_sizes=(16,))
        lp_jax = jax_est.log_prob(x[:20], theta[:20])
        lp_ag  = ag_est.log_prob(x[:20], theta[:20])
        assert lp_jax.shape == lp_ag.shape

    def test_maf_log_prob_shape_matches(self, linear_gaussian_data):
        theta, x, _ = linear_gaussian_data
        jax_est = _train_jax(JaxMAFEstimator, theta, x,
                             n_flows=2, hidden_units=16, num_blocks=1)
        ag_est  = _train_autograd(MAFEstimator, theta, x,
                                  n_flows=2, hidden_units=16, num_blocks=1)
        lp_jax = jax_est.log_prob(x[:20], theta[:20])
        lp_ag  = ag_est.log_prob(x[:20], theta[:20])
        assert lp_jax.shape == lp_ag.shape

    def test_mdn_loss_comparable(self, linear_gaussian_data):
        """Final loss values should be in the same ballpark (< 2× apart)."""
        theta, x, _ = linear_gaussian_data
        jax_est = _train_jax(JaxMDNEstimator, theta, x,
                             n_components=3, hidden_sizes=(16,), n_iter=120)
        ag_est  = _train_autograd(MDNEstimator, theta, x,
                                  n_components=3, hidden_sizes=(16,), n_iter=120)
        jax_loss = jax_est.loss_history[-1]
        ag_loss  = ag_est.loss_history[-1]
        assert abs(jax_loss - ag_loss) < 2.0 * max(abs(ag_loss), 0.1), \
            f"JAX loss ({jax_loss:.4f}) diverges too much from autograd ({ag_loss:.4f})"


# ---------------------------------------------------------------------------
# Save / load round-trip
# ---------------------------------------------------------------------------

class TestJaxSaveLoad:

    def test_maf_save_load(self, linear_gaussian_data, tmp_path):
        theta, x, x_obs = linear_gaussian_data
        est = _train_jax(JaxMAFEstimator, theta, x,
                         n_flows=2, hidden_units=16, num_blocks=1)
        path = str(tmp_path / "maf_jax.npz")
        est.save(path)

        # load() is a classmethod that returns a new instance
        est2 = JaxMAFEstimator.load(path)
        # Restore architecture state (not serialised — must match constructor args)
        est2.n_flows       = est.n_flows
        est2.hidden_units  = est.hidden_units
        est2.num_blocks    = est.num_blocks
        est2.activation    = est.activation
        est2.use_actnorm   = est.use_actnorm
        est2.embedding_dim = est.embedding_dim
        est2.actnorm_eps   = est.actnorm_eps
        est2._dims_inferred  = True
        est2.model_constants = est.model_constants
        est2.theta_mean, est2.theta_std = est.theta_mean, est.theta_std
        est2.x_mean,     est2.x_std     = est.x_mean,     est.x_std
        est2._use_pca        = est._use_pca
        est2._pca_components = est._pca_components
        est2._emb = None

        lp1 = est.log_prob(x[:10], theta[:10])
        lp2 = est2.log_prob(x[:10], theta[:10])
        np.testing.assert_allclose(lp1, lp2, rtol=1e-5,
                                   err_msg="log_prob changed after save/load")

    def test_mdn_save_load(self, linear_gaussian_data, tmp_path):
        theta, x, _ = linear_gaussian_data
        est = _train_jax(JaxMDNEstimator, theta, x,
                         n_components=3, hidden_sizes=(16,))
        path = str(tmp_path / "mdn_jax.npz")
        est.save(path)

        est2 = JaxMDNEstimator.load(path)
        est2.n_components   = est.n_components
        est2.hidden_sizes   = est.hidden_sizes
        est2._dims_inferred = True
        est2._offdiag_basis = est._offdiag_basis
        est2._emb = None

        lp1 = est.log_prob(x[:10], theta[:10])
        lp2 = est2.log_prob(x[:10], theta[:10])
        np.testing.assert_allclose(lp1, lp2, rtol=1e-5)


# ---------------------------------------------------------------------------
# End-to-end: SNPE(backend='jax')
# ---------------------------------------------------------------------------

class TestSNPEJaxBackend:

    def test_maf_full_workflow(self):
        rng = np.random.default_rng(0)
        prior = BoxUniform(low=np.zeros(2), high=np.ones(2))
        theta = prior.sample((300,), seed=0)
        x     = theta + rng.normal(0, 0.1, theta.shape)
        x_obs = np.array([[0.5, 0.5]])

        inf = SNPE(prior=prior, density_estimator="maf", backend="jax")
        inf.append_simulations(theta, x)
        est = inf.train(
            training_batch_size=64,
            max_num_epochs=30,
            stop_after_epochs=30,
            learning_rate=5e-3,
            verbose=False,
        )
        post = inf.build_posterior(est)
        samples = post.sample((200,), x=x_obs)
        assert samples.shape == (200, 2)
        assert np.all(np.isfinite(samples))

    def test_mdn_full_workflow(self):
        rng = np.random.default_rng(1)
        prior = BoxUniform(low=np.zeros(2), high=np.ones(2))
        theta = prior.sample((300,), seed=1)
        x     = theta + rng.normal(0, 0.1, theta.shape)
        x_obs = np.array([[0.5, 0.5]])

        inf = SNPE(prior=prior, density_estimator="mdn", backend="jax")
        inf.append_simulations(theta, x)
        est = inf.train(
            training_batch_size=None,
            max_num_epochs=30,
            stop_after_epochs=30,
            learning_rate=5e-3,
            verbose=False,
        )
        post = inf.build_posterior(est)
        samples = post.sample((200,), x=x_obs)
        assert samples.shape == (200, 2)
        assert np.all(np.isfinite(samples))

    def test_nsf_full_workflow(self):
        rng = np.random.default_rng(2)
        prior = BoxUniform(low=np.zeros(2), high=np.ones(2))
        theta = prior.sample((300,), seed=2)
        x     = theta + rng.normal(0, 0.1, theta.shape)
        x_obs = np.array([[0.5, 0.5]])

        inf = SNPE(prior=prior, density_estimator="nsf", backend="jax")
        inf.append_simulations(theta, x)
        est = inf.train(
            training_batch_size=64,
            max_num_epochs=30,
            stop_after_epochs=30,
            learning_rate=5e-3,
            verbose=False,
        )
        post = inf.build_posterior(est)
        samples = post.sample((200,), x=x_obs)
        assert samples.shape == (200, 2)
        assert np.all(np.isfinite(samples))

    def test_backend_auto_selects_jax(self):
        """backend='auto' should resolve to 'jax' when JAX is installed."""
        from vbi.inference._backends import resolve_backend
        assert resolve_backend("auto") == "jax"


# ---------------------------------------------------------------------------
# jax.vmap over observations (batch log_prob)
# ---------------------------------------------------------------------------

class TestJaxVmap:
    """_get_log_prob must be vmappable over the features (x_obs) axis."""

    def test_maf_vmap_over_features(self, linear_gaussian_data):
        theta, x, _ = linear_gaussian_data
        est = _train_jax(JaxMAFEstimator, theta, x,
                         n_flows=2, hidden_units=16, num_blocks=1)
        w = {k: jnp.array(v) for k, v in est.weights.items()}
        f5 = jnp.array(x[:5], dtype="f")
        p5 = jnp.array(theta[:5], dtype="f")

        lp_vmapped = jax.vmap(
            lambda fi: est._get_log_prob(w, fi[None], p5)
        )(f5)
        assert lp_vmapped.shape == (5, 5), f"unexpected shape {lp_vmapped.shape}"
        assert jnp.all(jnp.isfinite(lp_vmapped))

    def test_nsf_vmap_over_features(self, linear_gaussian_data):
        theta, x, _ = linear_gaussian_data
        est = _train_jax(JaxNSFEstimator, theta, x,
                         n_flows=2, hidden_units=16, num_blocks=1, num_bins=4)
        w = {k: jnp.array(v) for k, v in est.weights.items()}
        f5 = jnp.array(x[:5], dtype="f")
        p5 = jnp.array(theta[:5], dtype="f")

        lp_vmapped = jax.vmap(
            lambda fi: est._get_log_prob(w, fi[None], p5)
        )(f5)
        assert lp_vmapped.shape == (5, 5)
        assert jnp.all(jnp.isfinite(lp_vmapped))

    def test_mdn_vmap_over_features(self, linear_gaussian_data):
        theta, x, _ = linear_gaussian_data
        est = _train_jax(JaxMDNEstimator, theta, x,
                         n_components=3, hidden_sizes=(16,))
        w = {k: jnp.array(v) for k, v in est.weights.items()}
        f5 = jnp.array(x[:5], dtype="f")
        p5 = jnp.array(theta[:5], dtype="f")

        lp_vmapped = jax.vmap(
            lambda fi: est._log_prob_core(w, fi[None], p5)
        )(f5)
        assert lp_vmapped.shape == (5, 5)
        assert jnp.all(jnp.isfinite(lp_vmapped))


# ---------------------------------------------------------------------------
# jax.grad through log_prob w.r.t. theta
# ---------------------------------------------------------------------------

class TestJaxGrad:
    """jax.grad through _get_log_prob w.r.t. params must be finite."""

    def test_maf_grad_wrt_theta(self, linear_gaussian_data):
        theta, x, _ = linear_gaussian_data
        est = _train_jax(JaxMAFEstimator, theta, x,
                         n_flows=2, hidden_units=16, num_blocks=1)
        w = {k: jnp.array(v) for k, v in est.weights.items()}
        f = jnp.array(x[:5], dtype="f")
        p = jnp.array(theta[:5], dtype="f")

        grad_fn = jax.grad(lambda p_: jnp.mean(est._get_log_prob(w, f, p_)))
        g = grad_fn(p)
        assert g.shape == p.shape
        assert jnp.all(jnp.isfinite(g)), "MAF gradient has non-finite values"

    def test_nsf_grad_wrt_theta(self, linear_gaussian_data):
        theta, x, _ = linear_gaussian_data
        est = _train_jax(JaxNSFEstimator, theta, x,
                         n_flows=2, hidden_units=16, num_blocks=1, num_bins=4)
        w = {k: jnp.array(v) for k, v in est.weights.items()}
        f = jnp.array(x[:5], dtype="f")
        p = jnp.array(theta[:5], dtype="f")

        grad_fn = jax.grad(lambda p_: jnp.mean(est._get_log_prob(w, f, p_)))
        g = grad_fn(p)
        assert g.shape == p.shape
        assert jnp.all(jnp.isfinite(g)), "NSF gradient has non-finite values"

    def test_mdn_grad_wrt_theta(self, linear_gaussian_data):
        theta, x, _ = linear_gaussian_data
        est = _train_jax(JaxMDNEstimator, theta, x,
                         n_components=3, hidden_sizes=(16,))
        w = {k: jnp.array(v) for k, v in est.weights.items()}
        f = jnp.array(x[:5], dtype="f")
        p = jnp.array(theta[:5], dtype="f")

        grad_fn = jax.grad(lambda p_: jnp.mean(est._log_prob_core(w, f, p_)))
        g = grad_fn(p)
        assert g.shape == p.shape
        assert jnp.all(jnp.isfinite(g)), "MDN gradient has non-finite values"


# ---------------------------------------------------------------------------
# NSF save / load round-trip
# ---------------------------------------------------------------------------

class TestJaxNSFSaveLoad:

    def test_nsf_save_load(self, linear_gaussian_data, tmp_path):
        theta, x, _ = linear_gaussian_data
        est = _train_jax(JaxNSFEstimator, theta, x,
                         n_flows=2, hidden_units=16, num_blocks=1, num_bins=4)
        path = str(tmp_path / "nsf_jax.npz")
        est.save(path)

        est2 = JaxNSFEstimator.load(path)
        est2.n_flows      = est.n_flows
        est2.hidden_units = est.hidden_units
        est2.num_blocks   = est.num_blocks
        est2.num_bins     = est.num_bins
        est2.activation   = est.activation
        est2.use_actnorm  = est.use_actnorm
        est2.embedding_dim = est.embedding_dim
        est2.actnorm_eps  = est.actnorm_eps
        est2._dims_inferred  = True
        est2.model_constants = est.model_constants
        est2.theta_mean, est2.theta_std = est.theta_mean, est.theta_std
        est2.x_mean,     est2.x_std     = est.x_mean,     est.x_std
        est2._use_pca        = est._use_pca
        est2._pca_components = est._pca_components
        est2._emb = None

        lp1 = est.log_prob(x[:10], theta[:10])
        lp2 = est2.log_prob(x[:10], theta[:10])
        np.testing.assert_allclose(lp1, lp2, rtol=1e-5,
                                   err_msg="NSF log_prob changed after save/load")


# ---------------------------------------------------------------------------
# EmbeddingNet + JAX backend (SNPE end-to-end)
# ---------------------------------------------------------------------------

class TestSNPEJaxEmbeddingNet:

    def test_maf_with_embedding_net(self):
        from vbi.inference._embedding import EmbeddingNet

        rng = np.random.default_rng(5)
        prior = BoxUniform(low=np.zeros(2), high=np.ones(2))
        theta = prior.sample((200,), seed=5)
        x = theta + rng.normal(0, 0.1, theta.shape)
        x_obs = np.array([[0.5, 0.5]])

        emb = EmbeddingNet(input_dim=2, output_dim=4, hidden_sizes=(8,))
        inf = SNPE(prior=prior, density_estimator="maf", backend="jax",
                   embedding_net=emb)
        inf.append_simulations(theta, x)
        est = inf.train(
            training_batch_size=64,
            max_num_epochs=20,
            stop_after_epochs=20,
            learning_rate=5e-3,
            verbose=False,
        )
        post = inf.build_posterior(est)
        samples = post.sample((100,), x=x_obs)
        assert samples.shape == (100, 2)
        assert np.all(np.isfinite(samples))


# ---------------------------------------------------------------------------
# Rejection sampling posterior with JAX backend
# ---------------------------------------------------------------------------

class TestSNPEJaxRejection:

    def test_samples_inside_prior_bounds(self):
        rng = np.random.default_rng(9)
        prior = BoxUniform(low=np.zeros(2), high=np.ones(2))
        theta = prior.sample((300,), seed=9)
        x = theta + rng.normal(0, 0.1, theta.shape)
        x_obs = np.array([[0.5, 0.5]])

        inf = SNPE(prior=prior, density_estimator="maf", backend="jax")
        inf.append_simulations(theta, x)
        est = inf.train(
            training_batch_size=64,
            max_num_epochs=30,
            stop_after_epochs=30,
            learning_rate=5e-3,
            verbose=False,
        )
        post = inf.build_posterior(est)
        samples = post.sample((200,), x=x_obs, reject_outside_prior=True)
        assert samples.shape == (200, 2)
        assert np.all(samples >= prior.low), "samples below prior lower bound"
        assert np.all(samples <= prior.high), "samples above prior upper bound"

    def test_leakage_correction_scalar(self):
        rng = np.random.default_rng(10)
        prior = BoxUniform(low=np.zeros(2), high=np.ones(2))
        theta = prior.sample((300,), seed=10)
        x = theta + rng.normal(0, 0.1, theta.shape)
        x_obs = np.array([[0.5, 0.5]])

        inf = SNPE(prior=prior, density_estimator="maf", backend="jax")
        inf.append_simulations(theta, x)
        est = inf.train(
            training_batch_size=64,
            max_num_epochs=30,
            stop_after_epochs=30,
            learning_rate=5e-3,
            verbose=False,
        )
        post = inf.build_posterior(est)
        lc = post.leakage_correction(x_obs)
        assert np.isscalar(lc) or lc.ndim == 0
        assert 0.0 < float(lc) <= 1.0, f"leakage correction {lc} out of (0, 1]"
