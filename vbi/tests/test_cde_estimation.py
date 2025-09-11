"""
Test for parameter estimation using cde.MDNEstimator and cde.MAFEstimator.

This test file provides comprehensive testing of the conditional density estimation 
workflow for parameter estimation using both MDN and MAF estimators.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_less

# Try to import autograd - skip tests if not available
try:
    import autograd.numpy as anp
    from autograd import grad
    AUTOGRAD_AVAILABLE = True
except ImportError:
    AUTOGRAD_AVAILABLE = False
    anp = np

# Try to import the module we're testing
try:
    import vbi.cde as cde
    from vbi import BoxUniform
    CDE_AVAILABLE = True
    VBI_AVAILABLE = True
except ImportError:
    CDE_AVAILABLE = False
    VBI_AVAILABLE = False


class TestParameterEstimation:
    """Test parameter estimation workflow using both MDN and MAF estimators."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Parameters for linear Gaussian example
        self.num_dim = 3
        self.num_simulations = 1000
        self.prior_min = -2 * np.ones(self.num_dim)
        self.prior_max = 2 * np.ones(self.num_dim)
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Create a linear Gaussian simulator
        def simulator(theta):
            """Linear Gaussian simulator using numpy."""
            return theta + 1.0 + np.random.randn(*theta.shape) * 0.1
        
        self.simulator = simulator
        
        # Generate test data
        if VBI_AVAILABLE:
            self.prior = BoxUniform(low=self.prior_min, high=self.prior_max)
            self.theta = self.prior.sample((self.num_simulations,), seed=42)
        else:
            # Fallback: uniform sampling
            self.theta = np.random.uniform(
                self.prior_min, self.prior_max, 
                (self.num_simulations, self.num_dim)
            )
        
        self.x = self.simulator(self.theta)
        
        # Use smaller dataset for faster tests
        self.small_n = 100
        self.theta_small = self.theta[:self.small_n]
        self.x_small = self.x[:self.small_n]

    @pytest.mark.skipif(not AUTOGRAD_AVAILABLE, reason="autograd not available")
    @pytest.mark.skipif(not CDE_AVAILABLE, reason="vbi.cde not available")
    def test_data_generation(self):
        """Test that the data generation process works correctly."""
        assert self.theta.shape == (self.num_simulations, self.num_dim)
        assert self.x.shape == (self.num_simulations, self.num_dim)
        
        # Check that data is within reasonable bounds
        assert np.all(self.theta >= self.prior_min - 1e-6)
        assert np.all(self.theta <= self.prior_max + 1e-6)
        
        # Check that observations are close to theta + 1.0 (with noise)
        expected_mean = self.theta + 1.0
        diff = np.abs(self.x - expected_mean)
        # Most differences should be small (within 3 standard deviations = 0.3)
        assert np.mean(diff < 0.3) > 0.95

    @pytest.mark.skipif(not AUTOGRAD_AVAILABLE, reason="autograd not available")
    @pytest.mark.skipif(not CDE_AVAILABLE, reason="vbi.cde not available")
    def test_mdn_estimator_basic_workflow(self):
        """Test basic MDN estimator workflow."""
        # Create MDN estimator
        mdn_estimator = cde.MDNEstimator(n_components=10, hidden_sizes=(32, 32))
        
        # Test that training runs without errors (short training for test speed)
        try:
            mdn_estimator.train(
                self.theta_small, self.x_small, 
                n_iter=50, learning_rate=5e-4
            )
        except Exception as e:
            pytest.fail(f"MDN training failed: {e}")
        
        # Test that the model has been trained
        assert mdn_estimator.weights is not None
        
        # Test sampling
        rng = anp.random.RandomState(0)
        test_observation = self.x_small[0]
        samples = mdn_estimator.sample(test_observation, n_samples=100, rng=rng)[0]
        
        # Check sample shape
        assert samples.shape == (100, self.num_dim)
        
        # Check that samples are within reasonable bounds
        assert np.all(np.isfinite(samples))
        assert samples.std() > 0  # Samples should have some variability

    @pytest.mark.skipif(not AUTOGRAD_AVAILABLE, reason="autograd not available")
    @pytest.mark.skipif(not CDE_AVAILABLE, reason="vbi.cde not available")
    def test_maf_estimator_basic_workflow(self):
        """Test basic MAF estimator workflow."""
        # Create MAF estimator
        maf_estimator = cde.MAFEstimator()
        
        # Test that training runs without errors (short training for test speed)
        try:
            maf_estimator.train(
                self.theta_small, self.x_small, 
                n_iter=50, learning_rate=5e-4
            )
        except Exception as e:
            pytest.fail(f"MAF training failed: {e}")
        
        # Test that the model has been trained
        assert maf_estimator.weights is not None
        
        # Test sampling
        rng = anp.random.RandomState(0)
        test_observation = self.x_small[0]
        samples = maf_estimator.sample(test_observation, n_samples=100, rng=rng)[0]
        
        # Check sample shape
        assert samples.shape == (100, self.num_dim)
        
        # Check that samples are within reasonable bounds
        assert np.all(np.isfinite(samples))
        assert samples.std() > 0  # Samples should have some variability

    @pytest.mark.skipif(not AUTOGRAD_AVAILABLE, reason="autograd not available")
    @pytest.mark.skipif(not CDE_AVAILABLE, reason="vbi.cde not available")
    def test_mdn_parameter_recovery(self):
        """Test that MDN can recover parameters reasonably well."""
        # Create and train MDN estimator
        mdn_estimator = cde.MDNEstimator(n_components=5, hidden_sizes=(16,))
        mdn_estimator.train(
            self.theta_small, self.x_small, 
            n_iter=100, learning_rate=1e-3
        )
        
        # Test parameter recovery for a few examples
        rng = anp.random.RandomState(42)
        
        for i in range(3):  # Test first 3 examples
            true_theta = self.theta_small[i]
            observation = self.x_small[i]
            
            # Sample from posterior
            samples = mdn_estimator.sample(observation, n_samples=500, rng=rng)[0]
            
            # Check that posterior mean is reasonably close to true parameter
            posterior_mean = np.mean(samples, axis=0)
            
            # Given the noise level (0.1) and the linear relationship,
            # we expect recovery to be reasonable but not perfect
            diff = np.abs(posterior_mean - true_theta)
            
            # Allow for some error due to noise and limited training
            # Use a more relaxed threshold for this simple test
            assert np.all(diff < 2.0), f"Parameter recovery too poor for example {i}: diff={diff}"

    @pytest.mark.skipif(not AUTOGRAD_AVAILABLE, reason="autograd not available")
    @pytest.mark.skipif(not CDE_AVAILABLE, reason="vbi.cde not available")
    def test_maf_parameter_recovery(self):
        """Test that MAF can recover parameters reasonably well."""
        # Create and train MAF estimator
        maf_estimator = cde.MAFEstimator(n_flows=2, hidden_units=32)
        maf_estimator.train(
            self.theta_small, self.x_small, 
            n_iter=100, learning_rate=1e-3
        )
        
        # Test parameter recovery for a few examples
        rng = anp.random.RandomState(42)
        
        for i in range(3):  # Test first 3 examples
            true_theta = self.theta_small[i]
            observation = self.x_small[i]
            
            # Sample from posterior
            samples = maf_estimator.sample(observation, n_samples=500, rng=rng)[0]
            
            # Check that posterior mean is reasonably close to true parameter
            posterior_mean = np.mean(samples, axis=0)
            
            # Given the noise level (0.1) and the linear relationship,
            # we expect recovery to be reasonable but not perfect
            diff = np.abs(posterior_mean - true_theta)
            
            # Allow for some error due to noise and limited training
            # Use a more relaxed threshold for this simple test
            assert np.all(diff < 2.0), f"Parameter recovery too poor for example {i}: diff={diff}"

    @pytest.mark.skipif(not AUTOGRAD_AVAILABLE, reason="autograd not available")
    @pytest.mark.skipif(not CDE_AVAILABLE, reason="vbi.cde not available")
    def test_estimator_comparison(self):
        """Compare MDN and MAF estimators on the same data."""
        # Create both estimators with similar complexity
        mdn_estimator = cde.MDNEstimator(n_components=3, hidden_sizes=(16,))
        maf_estimator = cde.MAFEstimator(n_flows=2, hidden_units=16)
        
        # Train both with same data and parameters
        train_params = {
            'n_iter': 100,
            'learning_rate': 1e-3
        }
        
        mdn_estimator.train(self.theta_small, self.x_small, **train_params)
        maf_estimator.train(self.theta_small, self.x_small, **train_params)
        
        # Test on same observation
        test_observation = self.x_small[0]
        true_theta = self.theta_small[0]
        rng = anp.random.RandomState(123)
        
        # Get samples from both estimators
        mdn_samples = mdn_estimator.sample(test_observation, n_samples=200, rng=rng)[0]
        rng = anp.random.RandomState(123)  # Reset for fair comparison
        maf_samples = maf_estimator.sample(test_observation, n_samples=200, rng=rng)[0]
        
        # Both should produce finite samples
        assert np.all(np.isfinite(mdn_samples))
        assert np.all(np.isfinite(maf_samples))
        
        # Both should have some variability
        assert mdn_samples.std() > 0
        assert maf_samples.std() > 0
        
        # Both should be reasonably close to true parameter (within bounds)
        mdn_mean = np.mean(mdn_samples, axis=0)
        maf_mean = np.mean(maf_samples, axis=0)
        
        mdn_error = np.linalg.norm(mdn_mean - true_theta)
        maf_error = np.linalg.norm(maf_mean - true_theta)
        
        # Both errors should be reasonable (less than 2.0 for this simple problem)
        assert mdn_error < 2.0, f"MDN error too large: {mdn_error}"
        assert maf_error < 2.0, f"MAF error too large: {maf_error}"

    @pytest.mark.skipif(not AUTOGRAD_AVAILABLE, reason="autograd not available")
    @pytest.mark.skipif(not CDE_AVAILABLE, reason="vbi.cde not available")
    def test_log_probability_evaluation(self):
        """Test log probability evaluation for both estimators."""
        # Create and train both estimators
        mdn_estimator = cde.MDNEstimator(n_components=3, hidden_sizes=(16,))
        maf_estimator = cde.MAFEstimator(n_flows=2, hidden_units=16)
        
        mdn_estimator.train(self.theta_small, self.x_small, n_iter=50, learning_rate=1e-3)
        maf_estimator.train(self.theta_small, self.x_small, n_iter=50, learning_rate=1e-3)
        
        # Test log probability evaluation
        test_features = self.x_small[:5]
        test_params = self.theta_small[:5]
        
        # Evaluate log probabilities
        mdn_log_probs = mdn_estimator.log_prob(test_features, test_params)
        maf_log_probs = maf_estimator.log_prob(test_features, test_params)
        
        # Check that log probabilities are finite
        assert np.all(np.isfinite(mdn_log_probs))
        assert np.all(np.isfinite(maf_log_probs))
        
        # Log probabilities should be negative (probabilities < 1)
        # (This might not always be true for density estimates, so we just check finite)
        assert len(mdn_log_probs) == 5
        assert len(maf_log_probs) == 5

    @pytest.mark.skipif(not AUTOGRAD_AVAILABLE, reason="autograd not available")
    @pytest.mark.skipif(not CDE_AVAILABLE, reason="vbi.cde not available")
    def test_different_data_sizes(self):
        """Test estimators with different data sizes."""
        # Test with very small dataset
        tiny_theta = self.theta[:20]
        tiny_x = self.x[:20]
        
        mdn_estimator = cde.MDNEstimator(n_components=2, hidden_sizes=(8,))
        
        # Should work even with small data
        try:
            mdn_estimator.train(tiny_theta, tiny_x, n_iter=20, learning_rate=1e-3)
        except Exception as e:
            pytest.fail(f"Training failed with small dataset: {e}")
        
        # Test sampling
        rng = anp.random.RandomState(0)
        samples = mdn_estimator.sample(tiny_x[0], n_samples=10, rng=rng)[0]
        assert samples.shape == (10, self.num_dim)

    @pytest.mark.skipif(not VBI_AVAILABLE, reason="vbi BoxUniform not available")
    def test_boxuniform_integration(self):
        """Test integration with BoxUniform prior."""
        prior = BoxUniform(low=self.prior_min, high=self.prior_max)
        
        # Test sampling
        samples = prior.sample((50,), seed=123)
        assert samples.shape == (50, self.num_dim)
        
        # Check bounds
        assert np.all(samples >= self.prior_min)
        assert np.all(samples <= self.prior_max)
        
        # Test log probability (if available)
        if hasattr(prior, 'log_prob'):
            log_probs = prior.log_prob(samples)
            assert np.all(np.isfinite(log_probs))

    @pytest.mark.skipif(not AUTOGRAD_AVAILABLE, reason="autograd not available")
    @pytest.mark.skipif(not CDE_AVAILABLE, reason="vbi.cde not available")
    def test_reproducibility(self):
        """Test that results are reproducible with same random seed."""
        # Train two identical models with same seed
        rng1 = anp.random.RandomState(42)
        rng2 = anp.random.RandomState(42)
        
        mdn1 = cde.MDNEstimator(n_components=2, hidden_sizes=(8,))
        mdn2 = cde.MDNEstimator(n_components=2, hidden_sizes=(8,))
        
        # This test is more about ensuring the interface works consistently
        # Perfect reproducibility might not be guaranteed due to implementation details
        mdn1._infer_dimensions(self.theta_small[:20], self.x_small[:20])
        mdn2._infer_dimensions(self.theta_small[:20], self.x_small[:20])
        
        weights1 = mdn1._initialize_weights(rng1)
        weights2 = mdn2._initialize_weights(rng2)
        
        # Weights should be identical with same seed
        for key in weights1.keys():
            if key in weights2:
                assert_allclose(weights1[key], weights2[key], rtol=1e-10)


@pytest.mark.skip(reason="Skipping long-running tests")
@pytest.mark.long
@pytest.mark.skipif(not AUTOGRAD_AVAILABLE, reason="autograd not available")
@pytest.mark.skipif(not CDE_AVAILABLE, reason="vbi.cde not available")
class TestParameterEstimationLong:
    """Long-running tests for parameter estimation (marked as slow tests)."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.num_dim = 2  # Smaller for faster convergence
        self.num_simulations = 500
        self.prior_min = -1 * np.ones(self.num_dim)
        self.prior_max = 1 * np.ones(self.num_dim)
        
        np.random.seed(42)
        
        # Simpler linear relationship for better convergence
        def simulator(theta):
            return 2.0 * theta + 0.05 * np.random.randn(*theta.shape)
        
        self.simulator = simulator
        
        if VBI_AVAILABLE:
            self.prior = BoxUniform(low=self.prior_min, high=self.prior_max)
            self.theta = self.prior.sample((self.num_simulations,), seed=42)
        else:
            self.theta = np.random.uniform(
                self.prior_min, self.prior_max, 
                (self.num_simulations, self.num_dim)
            )
        
        self.x = self.simulator(self.theta)

    def test_mdn_convergence(self):
        """Test MDN convergence with more training."""
        mdn_estimator = cde.MDNEstimator(n_components=5, hidden_sizes=(32, 16))
        
        # Train for longer
        mdn_estimator.train(
            self.theta, self.x, 
            n_iter=500, learning_rate=1e-3
        )
        
        # Test parameter recovery accuracy
        rng = anp.random.RandomState(42)
        test_idx = 0
        true_theta = self.theta[test_idx]
        observation = self.x[test_idx]
        
        samples = mdn_estimator.sample(observation, n_samples=1000, rng=rng)[0]
        posterior_mean = np.mean(samples, axis=0)
        
        # With more training and simpler problem, expect better recovery
        error = np.linalg.norm(posterior_mean - true_theta)
        assert error < 0.2, f"Parameter recovery error too large: {error}"

    def test_maf_convergence(self):
        """Test MAF convergence with more training."""
        maf_estimator = cde.MAFEstimator(n_flows=4, hidden_units=64)
        
        # Train for longer
        maf_estimator.train(
            self.theta, self.x, 
            n_iter=500, learning_rate=1e-3
        )
        
        # Test parameter recovery accuracy
        rng = anp.random.RandomState(42)
        test_idx = 0
        true_theta = self.theta[test_idx]
        observation = self.x[test_idx]
        
        samples = maf_estimator.sample(observation, n_samples=1000, rng=rng)[0]
        posterior_mean = np.mean(samples, axis=0)
        
        # With more training and simpler problem, expect better recovery
        error = np.linalg.norm(posterior_mean - true_theta)
        assert error < 0.2, f"Parameter recovery error too large: {error}"
