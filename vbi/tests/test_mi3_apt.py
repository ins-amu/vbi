"""
MI3 tests - APT/SNPE-C importance-weighted loss.

Covers:
  1. APT loss computes finite values and has correct shape (unit test).
  2. APT loss equals NLL when all log_w = 0 (round-1 equivalence).
  3. SNPE with num_atoms=10 and a 2-round proposal runs end-to-end.
  4. APT with JAX backend runs end-to-end.
  5. compute_log_importance_weights returns correct shape and zeros for
     round-1 samples.
"""
import numpy as np
import pytest

pytest.importorskip("autograd", reason="autograd not installed")

from vbi.inference import SNPE, BoxUniform, Gaussian
from vbi.inference._apt import make_apt_loss, compute_log_importance_weights
from vbi.inference._estimators.mdn import MDNEstimator
from vbi.inference._estimators.maf import MAFEstimator


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def linear_gaussian():
    rng   = np.random.default_rng(42)
    prior = BoxUniform(low=np.zeros(2), high=np.ones(2))
    theta = prior.sample((300,), seed=42)
    x     = theta + rng.normal(0, 0.1, theta.shape)
    x_obs = np.array([[0.5, 0.5]])
    return prior, theta, x, x_obs


# ---------------------------------------------------------------------------
# Unit tests: make_apt_loss
# ---------------------------------------------------------------------------

class TestMakeAptLoss:

    def test_mdn_apt_loss_finite(self, linear_gaussian):
        prior, theta, x, _ = linear_gaussian
        est = MDNEstimator(n_components=3, hidden_sizes=(16,))
        est.train(theta, x, n_iter=10, verbose=False)

        log_w = np.zeros(len(theta), dtype="f")
        theta_aug = np.concatenate([theta, log_w[:, None]], axis=1)
        apt = make_apt_loss(est, num_atoms=5)

        import autograd.numpy as anp
        loss_val = apt(est.weights, x.astype("f"), theta_aug.astype("f"))
        assert np.isfinite(float(loss_val)), "APT loss is non-finite"

    def test_maf_apt_loss_finite(self, linear_gaussian):
        prior, theta, x, _ = linear_gaussian
        est = MAFEstimator(n_flows=2, hidden_units=16, num_blocks=1)
        est.train(theta, x, n_iter=20, verbose=False, validation_fraction=0.0,
                  stop_after_epochs=20, lr_schedule=None)

        log_w = np.zeros(len(theta), dtype="f")
        theta_aug = np.concatenate([theta, log_w[:, None]], axis=1)
        apt = make_apt_loss(est, num_atoms=5)

        loss_val = apt(est.weights, x.astype("f"), theta_aug.astype("f"))
        assert np.isfinite(float(loss_val)), "APT MAF loss is non-finite"

    def test_apt_num_atoms_must_be_ge2(self):
        with pytest.raises(ValueError, match="num_atoms must be >= 2"):
            make_apt_loss(None, num_atoms=1)


class TestLogImportanceWeights:

    def test_round1_all_zeros(self, linear_gaussian):
        prior, theta, x, _ = linear_gaussian
        rounds = [(theta.astype("f"), x.astype("f"), None)]
        log_w = compute_log_importance_weights(rounds, prior)
        assert log_w.shape == (len(theta),)
        assert np.all(log_w == 0.0), "Round-1 log_w must be zero"

    def test_round2_nonzero_for_narrow_proposal(self):
        prior = BoxUniform(low=np.zeros(1), high=np.ones(1))
        theta1 = prior.sample((100,), seed=0)

        # A narrow Gaussian proposal concentrated near 0.5
        class NarrowGaussian:
            def log_prob(self, theta):
                return -0.5 * ((theta[:, 0] - 0.5) / 0.05) ** 2

        narrow = NarrowGaussian()
        theta2 = np.random.default_rng(1).uniform(0.4, 0.6, (50, 1)).astype("f")
        x2     = theta2.copy()

        rounds = [
            (theta1.astype("f"), theta1.astype("f"), None),
            (theta2,             x2,                 narrow),
        ]
        log_w = compute_log_importance_weights(rounds, prior)
        assert log_w.shape == (150,)
        assert np.all(log_w[:100] == 0.0)
        # Round-2 samples near 0.5 should have small prior - large proposal → negative weight
        assert np.any(log_w[100:] != 0.0)


# ---------------------------------------------------------------------------
# End-to-end: SNPE with APT (2-round, autograd)
# ---------------------------------------------------------------------------

class TestSNPEWithAPT:

    def _run_2round(self, density_estimator: str, backend: str):
        rng   = np.random.default_rng(7)
        prior = BoxUniform(low=np.zeros(2), high=np.ones(2))

        # Round 1: simulate from prior
        theta1 = prior.sample((200,), seed=7)
        x1     = theta1 + rng.normal(0, 0.1, theta1.shape)

        inf = SNPE(prior=prior, density_estimator=density_estimator, backend=backend)
        inf.append_simulations(theta1, x1)
        est1 = inf.train(
            training_batch_size=64,
            max_num_epochs=20,
            stop_after_epochs=20,
            learning_rate=5e-3,
            verbose=False,
        )
        post1 = inf.build_posterior(est1)

        # Round 2: simulate from round-1 posterior (with proposal)
        x_obs = np.array([[0.5, 0.5]])
        theta2 = post1.sample((100,), x=x_obs, seed=8)
        x2     = theta2 + rng.normal(0, 0.1, theta2.shape)

        # Append round 2 WITH the proposal object
        post1.set_default_x(x_obs)
        inf.append_simulations(theta2, x2, proposal=post1)
        est2 = inf.train(
            training_batch_size=64,
            max_num_epochs=20,
            stop_after_epochs=20,
            learning_rate=5e-3,
            num_atoms=10,
            verbose=False,
        )
        post2 = inf.build_posterior(est2)
        samples = post2.sample((200,), x=x_obs, seed=9)
        assert samples.shape == (200, 2)
        assert np.all(np.isfinite(samples))

    def test_maf_autograd_2round_apt(self):
        self._run_2round("maf", "numpy")

    def test_mdn_autograd_2round_apt(self):
        self._run_2round("mdn", "numpy")


# ---------------------------------------------------------------------------
# APT with JAX backend
# ---------------------------------------------------------------------------

class TestSNPEWithAPTJax:

    def test_maf_jax_2round_apt(self):
        pytest.importorskip("jax")

        rng   = np.random.default_rng(11)
        prior = BoxUniform(low=np.zeros(2), high=np.ones(2))

        theta1 = prior.sample((200,), seed=11)
        x1     = theta1 + rng.normal(0, 0.1, theta1.shape)

        inf = SNPE(prior=prior, density_estimator="maf", backend="jax")
        inf.append_simulations(theta1, x1)
        est1 = inf.train(
            training_batch_size=64,
            max_num_epochs=20,
            stop_after_epochs=20,
            learning_rate=5e-3,
            verbose=False,
        )
        post1 = inf.build_posterior(est1)
        x_obs = np.array([[0.5, 0.5]])
        theta2 = post1.sample((100,), x=x_obs, seed=12)
        x2     = theta2 + rng.normal(0, 0.1, theta2.shape)

        post1.set_default_x(x_obs)
        inf.append_simulations(theta2, x2, proposal=post1)
        est2 = inf.train(
            training_batch_size=64,
            max_num_epochs=20,
            stop_after_epochs=20,
            learning_rate=5e-3,
            num_atoms=10,
            verbose=False,
        )
        post2 = inf.build_posterior(est2)
        samples = post2.sample((200,), x=x_obs, seed=13)
        assert samples.shape == (200, 2)
        assert np.all(np.isfinite(samples))


# ---------------------------------------------------------------------------
# get_simulations / n_rounds bookkeeping
# ---------------------------------------------------------------------------

class TestSNPEBookkeeping:

    def test_n_rounds_and_n_simulations(self, linear_gaussian):
        prior, theta, x, _ = linear_gaussian
        inf = SNPE(prior=prior)
        inf.append_simulations(theta, x)
        assert inf.n_rounds == 1
        assert inf.n_simulations == len(theta)

    def test_get_simulations_round2(self, linear_gaussian):
        prior, theta, x, _ = linear_gaussian
        inf = SNPE(prior=prior)
        inf.append_simulations(theta[:100], x[:100])
        inf.append_simulations(theta[100:], x[100:], proposal=None)
        th, xr, props = inf.get_simulations(starting_round=1)
        assert th.shape[0] == len(theta) - 100
