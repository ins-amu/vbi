"""
Tests for vbi.inference diagnostic tools.

Covers: run_sbc, check_sbc, sbc_rank_plot, run_tarp, check_tarp, plot_tarp,
        c2st, pairplot, conditional_pairplot, plot_loss
"""
import numpy as np
import pytest

autograd = pytest.importorskip("autograd", reason="autograd not installed")
matplotlib = pytest.importorskip("matplotlib", reason="matplotlib not installed")
sklearn = pytest.importorskip("sklearn", reason="scikit-learn not installed")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vbi.inference import (
    SNPE, BoxUniform,
    run_sbc, check_sbc, sbc_rank_plot,
    run_tarp, check_tarp, plot_tarp,
    c2st, pairplot, conditional_pairplot, plot_loss,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_1d_posterior(n_train=200, seed=42):
    """Train a quick 1-D posterior on a Gaussian toy model."""
    rng = np.random.RandomState(seed)
    prior = BoxUniform(low=np.array([-3.0]), high=np.array([3.0]))
    theta = prior.sample((n_train,))
    x     = theta + 0.1 * rng.randn(*theta.shape)

    inf = SNPE(prior=prior, density_estimator="mdn")
    inf = inf.append_simulations(theta, x)
    inf.train(max_num_epochs=20, verbose=False)
    posterior = inf.build_posterior()
    posterior.set_default_x(x[0])
    return posterior, prior


def _make_2d_posterior(n_train=300, seed=7):
    rng = np.random.RandomState(seed)
    prior = BoxUniform(low=np.array([-2.0, -2.0]), high=np.array([2.0, 2.0]))
    theta = prior.sample((n_train,))
    x     = theta + 0.1 * rng.randn(*theta.shape)

    inf = SNPE(prior=prior, density_estimator="mdn")
    inf = inf.append_simulations(theta, x)
    inf.train(max_num_epochs=20, verbose=False)
    posterior = inf.build_posterior()
    return posterior, prior


@pytest.fixture(scope="module")
def posterior_1d():
    return _make_1d_posterior()


@pytest.fixture(scope="module")
def posterior_2d():
    return _make_2d_posterior()


# ---------------------------------------------------------------------------
# SBC
# ---------------------------------------------------------------------------

class TestRunSBC:
    def test_returns_ranks_and_thetas(self, posterior_1d):
        posterior, prior = posterior_1d
        def sim(theta): return theta + 0.1 * np.random.randn(*theta.shape)

        result = run_sbc(posterior, sim, prior,
                         num_sbc_runs=20, num_posterior_samples=50, seed=0)
        assert "ranks" in result
        assert "thetas" in result

    def test_ranks_shape(self, posterior_1d):
        posterior, prior = posterior_1d
        def sim(theta): return theta + 0.0

        result = run_sbc(posterior, sim, prior,
                         num_sbc_runs=15, num_posterior_samples=40, seed=1)
        assert result["ranks"].shape == (15, 1)
        assert result["thetas"].shape == (15, 1)

    def test_ranks_in_valid_range(self, posterior_1d):
        posterior, prior = posterior_1d
        def sim(theta): return theta + 0.0

        result = run_sbc(posterior, sim, prior,
                         num_sbc_runs=20, num_posterior_samples=50, seed=2)
        ranks = result["ranks"]
        valid = ranks[ranks >= 0]
        assert np.all(valid >= 0)
        assert np.all(valid <= 50)

    def test_2d_ranks_shape(self, posterior_2d):
        posterior, prior = posterior_2d
        def sim(theta): return theta + 0.05 * np.random.randn(*theta.shape)

        result = run_sbc(posterior, sim, prior,
                         num_sbc_runs=10, num_posterior_samples=30, seed=3)
        assert result["ranks"].shape == (10, 2)

    def test_simulator_failure_handled(self, posterior_1d):
        posterior, prior = posterior_1d
        call_count = {"n": 0}

        def bad_sim(theta):
            call_count["n"] += 1
            if call_count["n"] <= 3:
                raise RuntimeError("boom")
            return theta

        result = run_sbc(posterior, bad_sim, prior,
                         num_sbc_runs=10, num_posterior_samples=20, seed=4)
        assert result["ranks"].shape[0] == 10  # all runs stored, failed as -1


class TestCheckSBC:
    def test_returns_pvalues(self, posterior_1d):
        posterior, prior = posterior_1d
        def sim(theta): return theta + 0.0

        result = run_sbc(posterior, sim, prior,
                         num_sbc_runs=50, num_posterior_samples=100, seed=5)
        check  = check_sbc(result["ranks"], num_posterior_samples=100)
        assert "uniformity_pvalues" in check
        assert check["uniformity_pvalues"].shape == (1,)

    def test_pvalues_in_01(self, posterior_1d):
        posterior, prior = posterior_1d
        def sim(theta): return theta + 0.0

        result = run_sbc(posterior, sim, prior,
                         num_sbc_runs=50, num_posterior_samples=100, seed=6)
        check  = check_sbc(result["ranks"], num_posterior_samples=100)
        pvals  = check["uniformity_pvalues"]
        assert np.all(pvals >= 0.0)
        assert np.all(pvals <= 1.0)

    def test_perfect_uniform_high_pvalue(self):
        """Synthetic perfectly-uniform ranks → high KS p-value."""
        rng   = np.random.RandomState(0)
        ranks = rng.randint(0, 100, size=(500, 1))
        check = check_sbc(ranks, num_posterior_samples=100)
        assert check["uniformity_pvalues"][0] > 0.05

    def test_degenerate_ranks_low_pvalue(self):
        """All ranks equal → should fail uniformity test."""
        ranks = np.zeros((200, 1), dtype=int)
        check = check_sbc(ranks, num_posterior_samples=100)
        assert check["uniformity_pvalues"][0] < 0.05

    def test_2d(self):
        rng   = np.random.RandomState(1)
        ranks = rng.randint(0, 50, size=(300, 2))
        check = check_sbc(ranks, num_posterior_samples=50)
        assert check["uniformity_pvalues"].shape == (2,)


class TestSBCRankPlot:
    def test_returns_figure(self):
        rng   = np.random.RandomState(0)
        ranks = rng.randint(0, 100, size=(100, 2))
        fig   = sbc_rank_plot(ranks, num_posterior_samples=100)
        assert hasattr(fig, "savefig")
        plt.close("all")

    def test_correct_number_of_axes(self):
        rng   = np.random.RandomState(0)
        ranks = rng.randint(0, 50, size=(80, 3))
        fig   = sbc_rank_plot(ranks, num_posterior_samples=50,
                              labels=["a", "b", "c"])
        assert len(fig.axes) == 3
        plt.close("all")

    def test_1d_ranks(self):
        rng   = np.random.RandomState(0)
        ranks = rng.randint(0, 50, size=(60,))
        fig   = sbc_rank_plot(ranks, num_posterior_samples=50)
        assert hasattr(fig, "savefig")
        plt.close("all")


# ---------------------------------------------------------------------------
# TARP
# ---------------------------------------------------------------------------

class TestRunTARP:
    def test_returns_dict_keys(self, posterior_1d):
        posterior, prior = posterior_1d
        def sim(theta): return theta + 0.0

        result = run_tarp(posterior, sim, prior,
                          num_runs=15, num_posterior_samples=30, seed=0)
        assert "alphas" in result
        assert "ecp"    in result
        assert "ranks"  in result

    def test_alphas_in_01(self, posterior_1d):
        posterior, prior = posterior_1d
        def sim(theta): return theta + 0.0

        result = run_tarp(posterior, sim, prior,
                          num_runs=20, num_posterior_samples=30, seed=1)
        assert np.all(result["alphas"] >= 0)
        assert np.all(result["alphas"] <= 1)

    def test_ecp_in_01(self, posterior_1d):
        posterior, prior = posterior_1d
        def sim(theta): return theta + 0.0

        result = run_tarp(posterior, sim, prior,
                          num_runs=20, num_posterior_samples=30, seed=2)
        assert np.all(result["ecp"] >= 0)
        assert np.all(result["ecp"] <= 1)

    def test_lengths_match(self, posterior_1d):
        posterior, prior = posterior_1d
        def sim(theta): return theta + 0.0

        result = run_tarp(posterior, sim, prior,
                          num_runs=10, num_posterior_samples=20, seed=3)
        assert len(result["alphas"]) == len(result["ecp"])


class TestCheckTARP:
    def test_ece_zero_for_perfect_coverage(self):
        alphas = np.linspace(0, 1, 100)
        ecp    = alphas.copy()  # perfect coverage
        result = check_tarp(alphas, ecp)
        assert "ece" in result
        assert result["ece"] == pytest.approx(0.0, abs=1e-10)

    def test_ece_positive_for_biased_coverage(self):
        alphas = np.linspace(0, 1, 100)
        ecp    = alphas + 0.1  # overconfident
        ecp    = np.clip(ecp, 0, 1)
        result = check_tarp(alphas, ecp)
        assert result["ece"] > 0.0

    def test_ece_is_float(self, posterior_1d):
        posterior, prior = posterior_1d
        def sim(theta): return theta + 0.0

        tarp_result  = run_tarp(posterior, sim, prior,
                                num_runs=15, num_posterior_samples=20, seed=0)
        check_result = check_tarp(tarp_result["alphas"], tarp_result["ecp"])
        assert isinstance(check_result["ece"], float)


class TestPlotTARP:
    def test_returns_figure(self):
        alphas = np.linspace(0, 1, 50)
        ecp    = alphas + 0.05 * np.sin(np.pi * alphas)
        fig    = plot_tarp(alphas, ecp)
        assert hasattr(fig, "savefig")
        plt.close("all")

    def test_axes_limits(self):
        alphas = np.linspace(0, 1, 30)
        ecp    = alphas
        fig    = plot_tarp(alphas, ecp)
        ax     = fig.axes[0]
        assert ax.get_xlim()[0] == pytest.approx(0.0)
        assert ax.get_xlim()[1] == pytest.approx(1.0)
        plt.close("all")


# ---------------------------------------------------------------------------
# C2ST
# ---------------------------------------------------------------------------

class TestC2ST:
    def test_identical_distributions_near_half(self):
        rng = np.random.RandomState(0)
        p   = rng.randn(500, 2)
        q   = rng.randn(500, 2)
        acc = c2st(p, q, seed=0)
        assert 0.4 <= acc <= 0.7, f"Expected ~0.5, got {acc}"

    def test_distinct_distributions_high_accuracy(self):
        rng = np.random.RandomState(1)
        p   = rng.randn(500, 2)
        q   = rng.randn(500, 2) + 5.0  # well separated
        acc = c2st(p, q, seed=1)
        assert acc > 0.8, f"Expected >0.8, got {acc}"

    def test_return_is_float(self):
        rng = np.random.RandomState(2)
        acc = c2st(rng.randn(100, 1), rng.randn(100, 1), seed=2)
        assert isinstance(acc, float)

    def test_range(self):
        rng = np.random.RandomState(3)
        acc = c2st(rng.randn(200, 3), rng.randn(200, 3), seed=3)
        assert 0.0 <= acc <= 1.0

    def test_1d_inputs(self):
        rng = np.random.RandomState(4)
        p   = rng.randn(200)
        q   = rng.randn(200) + 3.0
        acc = c2st(p, q, seed=4)
        assert acc > 0.7

    def test_unequal_sample_sizes(self):
        rng = np.random.RandomState(5)
        p   = rng.randn(300, 2)
        q   = rng.randn(100, 2)
        acc = c2st(p, q, seed=5)
        assert 0.0 <= acc <= 1.0


# ---------------------------------------------------------------------------
# Pairplot
# ---------------------------------------------------------------------------

class TestPairplot:
    def test_1d_returns_figure(self):
        rng     = np.random.RandomState(0)
        samples = rng.randn(200, 1)
        fig     = pairplot(samples)
        assert hasattr(fig, "savefig")
        plt.close("all")

    def test_2d_returns_figure(self):
        rng     = np.random.RandomState(1)
        samples = rng.randn(300, 2)
        fig     = pairplot(samples)
        assert hasattr(fig, "savefig")
        plt.close("all")

    def test_3d_returns_figure(self):
        rng     = np.random.RandomState(2)
        samples = rng.randn(200, 3)
        fig     = pairplot(samples)
        assert hasattr(fig, "savefig")
        plt.close("all")

    def test_with_points(self):
        rng     = np.random.RandomState(3)
        samples = rng.randn(200, 2)
        fig     = pairplot(samples, points=np.array([[0.0, 0.0]]))
        assert hasattr(fig, "savefig")
        plt.close("all")

    def test_with_limits(self):
        rng     = np.random.RandomState(4)
        samples = rng.randn(200, 2)
        fig     = pairplot(samples, limits=[(-3, 3), (-3, 3)])
        assert hasattr(fig, "savefig")
        plt.close("all")

    def test_with_labels(self):
        rng     = np.random.RandomState(5)
        samples = rng.randn(200, 2)
        fig     = pairplot(samples, labels=["α", "β"])
        assert hasattr(fig, "savefig")
        plt.close("all")

    def test_1d_array_input(self):
        rng     = np.random.RandomState(7)
        samples = rng.randn(100, 1)
        fig     = pairplot(samples)
        assert hasattr(fig, "savefig")
        plt.close("all")


class TestConditionalPairplot:
    def test_returns_figure(self, posterior_1d):
        posterior, _ = posterior_1d
        x_obs = np.array([0.5])
        fig   = conditional_pairplot(posterior, x_obs, n_samples=50,
                                     points=np.array([[0.5]]))
        assert hasattr(fig, "savefig")
        plt.close("all")

    def test_2d(self, posterior_2d):
        posterior, _ = posterior_2d
        x_obs = np.array([0.2, -0.3])
        fig   = conditional_pairplot(posterior, x_obs, n_samples=50)
        assert hasattr(fig, "savefig")
        plt.close("all")


# ---------------------------------------------------------------------------
# plot_loss
# ---------------------------------------------------------------------------

class TestPlotLoss:
    def test_returns_figure(self):
        losses = np.exp(-np.linspace(0, 3, 100))
        fig    = plot_loss(losses)
        assert hasattr(fig, "savefig")
        plt.close("all")

    def test_with_val_loss(self):
        losses     = np.exp(-np.linspace(0, 3, 100))
        val_losses = np.exp(-np.linspace(0, 2.5, 50))
        fig        = plot_loss(losses, val_losses)
        assert hasattr(fig, "savefig")
        plt.close("all")

    def test_log_scale(self):
        losses = np.exp(-np.linspace(0, 3, 100))
        fig    = plot_loss(losses, log_scale=True)
        ax     = fig.axes[0]
        assert ax.get_yscale() == "log"
        plt.close("all")

    def test_existing_axes(self):
        fig_in, ax_in = plt.subplots()
        losses        = list(range(50, 0, -1))
        fig_out       = plot_loss(losses, ax=ax_in)
        assert fig_out is fig_in
        plt.close("all")

    def test_uses_estimator_loss_history(self, posterior_1d):
        posterior, _ = posterior_1d
        # SNPE._estimator.loss_history should be non-empty
        from vbi.inference import SNPE, BoxUniform
        prior = BoxUniform(low=np.array([-1.0]), high=np.array([1.0]))
        rng   = np.random.RandomState(99)
        theta = prior.sample((100,))
        x     = theta + 0.1 * rng.randn(*theta.shape)
        inf   = SNPE(prior=prior, density_estimator="mdn")
        inf   = inf.append_simulations(theta, x)
        inf.train(max_num_epochs=5, verbose=False)
        losses = inf._estimator.loss_history
        fig    = plot_loss(losses)
        assert hasattr(fig, "savefig")
        plt.close("all")
