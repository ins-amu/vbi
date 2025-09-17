import unittest
import numpy as np
import pytest
from parameterized import parameterized
import sys
import os

# Add the parent directory to the path to import vbi
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from vbi.utils import BoxUniform


@pytest.mark.short
@pytest.mark.fast
class TestBoxUniform(unittest.TestCase):
    """Test suite for BoxUniform distribution class."""

    def setUp(self):
        """Set up test fixtures."""
        self.seed = 42
        np.random.seed(self.seed)

    @parameterized.expand([
        ("1d_case", 0.0, 1.0, np.float64),
        ("1d_case_float32", 0.0, 1.0, np.float32),
        ("2d_case", [0.0, -1.0], [1.0, 1.0], np.float64),
        ("3d_case", [-1.0, -2.0, 0.0], [2.0, 1.0, 3.0], np.float64),
    ])
    def test_initialization(self, name, low, high, dtype):
        """Test BoxUniform initialization with different configurations."""
        dist = BoxUniform(low=low, high=high, dtype=dtype)
        
        # Check dimensions
        expected_ndims = 1 if np.isscalar(low) else len(low)
        self.assertEqual(dist.ndims, expected_ndims)
        
        # Check dtype
        self.assertEqual(dist.dtype, dtype)
        self.assertEqual(dist.low.dtype, dtype)
        self.assertEqual(dist.high.dtype, dtype)
        
        # Check bounds
        if np.isscalar(low):
            self.assertEqual(dist.low[0], low)
            self.assertEqual(dist.high[0], high)
        else:
            np.testing.assert_array_equal(dist.low, low)
            np.testing.assert_array_equal(dist.high, high)

    def test_initialization_default_dtype(self):
        """Test that default dtype is np.float64."""
        dist = BoxUniform(low=0.0, high=1.0)
        self.assertEqual(dist.dtype, np.float64)
        self.assertEqual(dist.low.dtype, np.float64)
        self.assertEqual(dist.high.dtype, np.float64)

    def test_initialization_errors(self):
        """Test that initialization raises appropriate errors."""
        # Test low >= high
        with self.assertRaises(ValueError):
            BoxUniform(low=1.0, high=0.0)
        
        with self.assertRaises(ValueError):
            BoxUniform(low=[0.0, 1.0], high=[1.0, 0.5])
        
        # Test mismatched shapes
        with self.assertRaises(ValueError):
            BoxUniform(low=[0.0], high=[1.0, 2.0])

    def test_initialization_with_seed(self):
        """Test BoxUniform initialization with seed parameter."""
        seed_value = 12345
        dist = BoxUniform(low=0.0, high=1.0, seed=seed_value)
        
        # Test that the seed affects sampling
        samples1 = dist.sample(5)
        
        # Create another instance with same seed
        dist2 = BoxUniform(low=0.0, high=1.0, seed=seed_value)
        samples2 = dist2.sample(5)
        
        # Should be identical
        np.testing.assert_array_equal(samples1, samples2)

    def test_seed_in_sample_method(self):
        """Test seed parameter in sample method."""
        dist = BoxUniform(low=[0.0, -1.0], high=[2.0, 1.0])
        
        # Sample with specific seed
        seed_value = 54321
        samples1 = dist.sample(10, seed=seed_value)
        samples2 = dist.sample(10, seed=seed_value)
        
        # Should be identical
        np.testing.assert_array_equal(samples1, samples2)
        
        # Sample without seed should be different
        samples3 = dist.sample(10)
        self.assertFalse(np.allclose(samples1, samples3))

    def test_set_seed_method(self):
        """Test set_seed method functionality."""
        dist = BoxUniform(low=0.0, high=10.0)
        
        # Set seed and sample
        dist.set_seed(98765)
        samples1 = dist.sample(8)
        
        # Reset to same seed and sample again
        dist.set_seed(98765)
        samples2 = dist.sample(8)
        
        # Should be identical
        np.testing.assert_array_equal(samples1, samples2)

    def test_seed_none_behavior(self):
        """Test behavior when seed is None."""
        # Test in constructor
        dist1 = BoxUniform(low=0.0, high=1.0, seed=None)
        samples1 = dist1.sample(5)
        
        dist2 = BoxUniform(low=0.0, high=1.0, seed=None)
        samples2 = dist2.sample(5)
        
        # Should be different (very high probability)
        self.assertFalse(np.allclose(samples1, samples2))
        
        # Test set_seed with None
        dist1.set_seed(None)  # Should not raise error
        
        # Test sample with None seed
        samples3 = dist1.sample(5, seed=None)
        samples4 = dist1.sample(5, seed=None)
        self.assertFalse(np.allclose(samples3, samples4))

    def test_seed_dtype_compatibility(self):
        """Test that seed works with different dtypes."""
        seed_value = 11111
        
        for dtype in [np.float32, np.float64]:
            dist = BoxUniform(low=0.0, high=1.0, dtype=dtype, seed=seed_value)
            samples = dist.sample(5)
            
            self.assertEqual(samples.dtype, dtype)
            
            # Test reproducibility with dtype
            dist2 = BoxUniform(low=0.0, high=1.0, dtype=dtype, seed=seed_value)
            samples2 = dist2.sample(5)
            
            np.testing.assert_array_equal(samples, samples2)

    def test_seed_multidimensional(self):
        """Test seed functionality with multidimensional distributions."""
        seed_value = 77777
        low = [-2.0, 0.0, 1.0]
        high = [0.0, 3.0, 5.0]
        
        dist1 = BoxUniform(low=low, high=high, seed=seed_value)
        samples1 = dist1.sample(20)
        
        dist2 = BoxUniform(low=low, high=high, seed=seed_value)
        samples2 = dist2.sample(20)
        
        np.testing.assert_array_equal(samples1, samples2)
        
        # Check that all dimensions are within bounds
        for i in range(3):
            self.assertTrue(np.all(samples1[:, i] >= low[i]))
            self.assertTrue(np.all(samples1[:, i] <= high[i]))

    @parameterized.expand([
        ("1d_single", 0.0, 1.0, 1, (1, 1), np.float64),
        ("1d_multiple", 0.0, 1.0, 100, (100, 1), np.float64),
        ("2d_single", [0.0, -1.0], [1.0, 1.0], 1, (1, 2), np.float64),
        ("2d_multiple", [0.0, -1.0], [1.0, 1.0], 50, (50, 2), np.float64),
        ("3d_batch", [-1.0, 0.0, 2.0], [1.0, 2.0, 4.0], (10, 5), (10, 5, 3), np.float64),
        ("1d_float32", 0.0, 1.0, 10, (10, 1), np.float32),
        ("2d_float32", [0.0, -1.0], [1.0, 1.0], 20, (20, 2), np.float32),
    ])
    def test_sampling(self, name, low, high, sample_shape, expected_shape, dtype):
        """Test sampling with different shapes, dimensions, and dtypes."""
        dist = BoxUniform(low=low, high=high, dtype=dtype)
        samples = dist.sample(sample_shape)
        
        # Check shape
        self.assertEqual(samples.shape, expected_shape)
        
        # Check dtype
        self.assertEqual(samples.dtype, dtype)
        
        # Check bounds
        low_arr = np.atleast_1d(low)
        high_arr = np.atleast_1d(high)
        
        self.assertTrue(np.all(samples >= low_arr))
        self.assertTrue(np.all(samples <= high_arr))

    def test_sampling_dtype_consistency(self):
        """Test that samples have the correct dtype."""
        for dtype in [np.float32, np.float64]:
            dist = BoxUniform(low=0.0, high=1.0, dtype=dtype)
            samples = dist.sample(10)
            self.assertEqual(samples.dtype, dtype)

    @parameterized.expand([
        ("1d_inside", 0.0, 1.0, 0.5, 0.0),
        ("1d_outside_low", 0.0, 1.0, -0.5, -np.inf),
        ("1d_outside_high", 0.0, 1.0, 1.5, -np.inf),
        ("2d_inside", [0.0, 0.0], [1.0, 2.0], [0.5, 1.0], -np.log(2.0)),
        ("2d_outside", [0.0, 0.0], [1.0, 2.0], [1.5, 1.0], -np.inf),
    ])
    def test_log_prob(self, name, low, high, test_point, expected_log_prob):
        """Test log probability calculations."""
        dist = BoxUniform(low=low, high=high)
        log_prob = dist.log_prob(test_point)
        
        if np.isfinite(expected_log_prob):
            np.testing.assert_allclose(log_prob, expected_log_prob, rtol=1e-10)
        else:
            self.assertEqual(log_prob, expected_log_prob)

    def test_log_prob_batch(self):
        """Test log probability with batch inputs."""
        dist = BoxUniform(low=[0.0, 0.0], high=[1.0, 1.0])
        test_points = np.array([
            [0.5, 0.5],  # inside
            [1.5, 0.5],  # outside
            [0.5, 1.5],  # outside
            [0.2, 0.8],  # inside
        ])
        
        log_probs = dist.log_prob(test_points)
        expected = np.array([0.0, -np.inf, -np.inf, 0.0])
        
        np.testing.assert_array_equal(log_probs, expected)

    def test_prob(self):
        """Test probability density calculations."""
        dist = BoxUniform(low=[0.0, 0.0], high=[2.0, 1.0])
        
        # Inside the support
        prob_inside = dist.prob([1.0, 0.5])
        expected_prob = 1.0 / dist.volume
        self.assertAlmostEqual(prob_inside, expected_prob)
        
        # Outside the support
        prob_outside = dist.prob([3.0, 0.5])
        self.assertEqual(prob_outside, 0.0)

    @parameterized.expand([
        ("1d_corner_cases", 0.0, 1.0, [[0.0], [1.0], [0.5]], [0.0, 1.0, 0.5]),
        ("2d_corner_cases", [0.0, 0.0], [1.0, 2.0], [[0.0, 0.0], [1.0, 2.0], [0.5, 1.0]], [0.0, 1.0, 0.25]),
    ])
    def test_cdf(self, name, low, high, test_points, expected_cdf):
        """Test cumulative distribution function."""
        dist = BoxUniform(low=low, high=high)
        cdf_values = dist.cdf(test_points)
        
        np.testing.assert_allclose(cdf_values, expected_cdf, rtol=1e-10)

    def test_statistical_properties(self):
        """Test mean, variance, and standard deviation calculations."""
        # 1D case
        dist_1d = BoxUniform(low=2.0, high=8.0)
        self.assertAlmostEqual(dist_1d.mean()[0], 5.0)
        expected_var_1d = (8.0 - 2.0) ** 2 / 12
        self.assertAlmostEqual(dist_1d.variance()[0], expected_var_1d)
        self.assertAlmostEqual(dist_1d.std()[0], np.sqrt(expected_var_1d))
        
        # 2D case
        dist_2d = BoxUniform(low=[0.0, -2.0], high=[4.0, 2.0])
        expected_mean = np.array([2.0, 0.0])
        expected_var = np.array([16.0/12, 16.0/12])
        
        np.testing.assert_allclose(dist_2d.mean(), expected_mean)
        np.testing.assert_allclose(dist_2d.variance(), expected_var)
        np.testing.assert_allclose(dist_2d.std(), np.sqrt(expected_var))

    def test_volume_calculation(self):
        """Test volume calculation for different dimensions."""
        # 1D
        dist_1d = BoxUniform(low=0.0, high=3.0)
        self.assertEqual(dist_1d.volume, 3.0)
        
        # 2D
        dist_2d = BoxUniform(low=[0.0, 0.0], high=[2.0, 3.0])
        self.assertEqual(dist_2d.volume, 6.0)
        
        # 3D
        dist_3d = BoxUniform(low=[0.0, 0.0, 0.0], high=[2.0, 3.0, 4.0])
        self.assertEqual(dist_3d.volume, 24.0)

    def test_support(self):
        """Test support method returns correct bounds."""
        low = np.array([1.0, -2.0])
        high = np.array([3.0, 1.0])
        dist = BoxUniform(low=low, high=high)
        
        support_low, support_high = dist.support()
        
        np.testing.assert_array_equal(support_low, low)
        np.testing.assert_array_equal(support_high, high)
        
        # Check that returned arrays are copies, not references
        support_low[0] = 999
        self.assertNotEqual(dist.low[0], 999)

    def test_repr(self):
        """Test string representation."""
        # Test default dtype
        dist = BoxUniform(low=[0.0, -1.0], high=[1.0, 1.0])
        repr_str = repr(dist)
        
        self.assertIn("BoxUniform", repr_str)
        self.assertIn("low", repr_str)
        self.assertIn("high", repr_str)
        self.assertIn("dtype", repr_str)
        self.assertIn("float64", repr_str)
        
        # Test float32 dtype
        dist_32 = BoxUniform(low=[0.0, -1.0], high=[1.0, 1.0], dtype=np.float32)
        repr_str_32 = repr(dist_32)
        self.assertIn("float32", repr_str_32)

    def test_dtype_consistency(self):
        """Test that all operations maintain dtype consistency."""
        for dtype in [np.float32, np.float64]:
            dist = BoxUniform(low=[0.0, 0.0], high=[1.0, 1.0], dtype=dtype)
            
            # Check initialization
            self.assertEqual(dist.low.dtype, dtype)
            self.assertEqual(dist.high.dtype, dtype)
            
            # Check sampling
            samples = dist.sample(5)
            self.assertEqual(samples.dtype, dtype)
            
            # Check statistical properties
            self.assertEqual(dist.mean().dtype, dtype)
            self.assertEqual(dist.variance().dtype, dtype)
            self.assertEqual(dist.std().dtype, dtype)

    def test_dtype_precision_differences(self):
        """Test numerical differences between float32 and float64."""
        # Create same distribution with different dtypes
        dist_64 = BoxUniform(low=[0.0, 0.0], high=[1.0, 1.0], dtype=np.float64)
        dist_32 = BoxUniform(low=[0.0, 0.0], high=[1.0, 1.0], dtype=np.float32)
        
        # Test that results are close but potentially different due to precision
        test_point = np.array([0.123456789012345, 0.987654321098765])
        
        log_prob_64 = dist_64.log_prob(test_point)
        log_prob_32 = dist_32.log_prob(test_point)
        
        # Both should be finite and close
        self.assertTrue(np.isfinite(log_prob_64))
        self.assertTrue(np.isfinite(log_prob_32))
        np.testing.assert_allclose(log_prob_64, log_prob_32, rtol=1e-6)

    def test_dtype_memory_efficiency(self):
        """Test that float32 uses less memory than float64."""
        n_samples, n_dims = 1000, 10
        
        dist_64 = BoxUniform(
            low=np.zeros(n_dims), 
            high=np.ones(n_dims), 
            dtype=np.float64
        )
        dist_32 = BoxUniform(
            low=np.zeros(n_dims), 
            high=np.ones(n_dims), 
            dtype=np.float32
        )
        
        samples_64 = dist_64.sample(n_samples)
        samples_32 = dist_32.sample(n_samples)
        
        # float64 should use exactly 2x memory as float32
        self.assertEqual(samples_64.nbytes, 2 * samples_32.nbytes)
        
        # Verify shapes are the same
        self.assertEqual(samples_64.shape, samples_32.shape)

    def test_dtype_input_conversion(self):
        """Test that input arrays are converted to the specified dtype."""
        # Test with input arrays of different dtypes
        low_int = np.array([0, -1], dtype=np.int32)
        high_int = np.array([2, 1], dtype=np.int32)
        
        for target_dtype in [np.float32, np.float64]:
            dist = BoxUniform(low=low_int, high=high_int, dtype=target_dtype)
            
            self.assertEqual(dist.low.dtype, target_dtype)
            self.assertEqual(dist.high.dtype, target_dtype)
            
            # Values should be preserved
            np.testing.assert_array_equal(dist.low, [0, -1])
            np.testing.assert_array_equal(dist.high, [2, 1])

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Very small range
        dist_small = BoxUniform(low=0.0, high=1e-10)
        samples = dist_small.sample(10)
        self.assertTrue(np.all(samples >= 0.0))
        self.assertTrue(np.all(samples <= 1e-10))
        
        # Negative ranges
        dist_neg = BoxUniform(low=-5.0, high=-2.0)
        samples = dist_neg.sample(10)
        self.assertTrue(np.all(samples >= -5.0))
        self.assertTrue(np.all(samples <= -2.0))

    def test_random_seed_reproducibility(self):
        """Test that setting random seed produces reproducible results."""
        # Test with our seed mechanism
        dist = BoxUniform(low=0.0, high=1.0, seed=123)
        samples1 = dist.sample(5)
        
        dist.set_seed(123)  # Reset to same seed
        samples2 = dist.sample(5)
        
        np.testing.assert_array_equal(samples1, samples2)
        
        # Test with seed parameter in sample
        samples3 = dist.sample(5, seed=456)
        samples4 = dist.sample(5, seed=456)
        
        np.testing.assert_array_equal(samples3, samples4)

    def test_large_batch_sampling(self):
        """Test sampling with large batch sizes."""
        dist = BoxUniform(low=[0.0, 0.0], high=[1.0, 1.0])
        large_samples = dist.sample(10000)
        
        # Check that samples are approximately uniformly distributed
        mean_samples = np.mean(large_samples, axis=0)
        expected_mean = dist.mean()
        
        # Allow for some statistical variation
        np.testing.assert_allclose(mean_samples, expected_mean, atol=0.05)

    def test_multidimensional_edge_cases(self):
        """Test edge cases for multidimensional distributions."""
        # High dimensional case
        ndims = 10
        low = np.zeros(ndims)
        high = np.ones(ndims)
        dist = BoxUniform(low=low, high=high)
        
        samples = dist.sample(100)
        self.assertEqual(samples.shape, (100, ndims))
        
        # Check all dimensions are within bounds
        self.assertTrue(np.all(samples >= 0.0))
        self.assertTrue(np.all(samples <= 1.0))


@pytest.mark.long
class TestBoxUniformLong(unittest.TestCase):
    """Long-running tests for BoxUniform."""

    def setUp(self):
        """Set up test fixtures."""
        self.seed = 42
        np.random.seed(self.seed)

    def test_statistical_convergence(self):
        """Test that large samples converge to expected statistics."""
        dist = BoxUniform(low=[0.0, -2.0], high=[4.0, 2.0], seed=self.seed)
        large_samples = dist.sample(100000)
        
        # Test mean convergence
        sample_mean = np.mean(large_samples, axis=0)
        expected_mean = dist.mean()
        np.testing.assert_allclose(sample_mean, expected_mean, atol=0.01)
        
        # Test variance convergence
        sample_var = np.var(large_samples, axis=0)
        expected_var = dist.variance()
        np.testing.assert_allclose(sample_var, expected_var, rtol=0.05)

    def test_memory_efficiency(self):
        """Test memory efficiency with different dtypes."""
        n_samples = 100000
        n_dims = 100
        
        # Float64
        dist_64 = BoxUniform(
            low=np.zeros(n_dims), 
            high=np.ones(n_dims), 
            dtype=np.float64
        )
        samples_64 = dist_64.sample(n_samples)
        
        # Float32
        dist_32 = BoxUniform(
            low=np.zeros(n_dims), 
            high=np.ones(n_dims), 
            dtype=np.float32
        )
        samples_32 = dist_32.sample(n_samples)
        
        # Check memory usage difference
        self.assertAlmostEqual(samples_64.nbytes / samples_32.nbytes, 2.0, places=1)


if __name__ == "__main__":
    unittest.main()
