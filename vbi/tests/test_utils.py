import unittest
import numpy as np
import pytest
from parameterized import parameterized
import sys
import os
from scipy.stats import multivariate_normal, norm

# Add the parent directory to the path to import vbi
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from vbi.utils import posterior_peaks_numpy, get_limits_numpy


@pytest.mark.short
@pytest.mark.fast
class TestPosteriorPeaksNumpy(unittest.TestCase):
    """Test suite for posterior_peaks_numpy function."""

    def setUp(self):
        """Set up test fixtures."""
        self.seed = 42
        np.random.seed(self.seed)
        
        # Generate test data with known peaks
        self.mean_1d = 2.5
        self.std_1d = 0.5
        self.samples_1d = np.random.normal(self.mean_1d, self.std_1d, 1000)
        
        self.mean_2d = np.array([1.5, -0.5])
        self.cov_2d = np.array([[0.3, 0.05], [0.05, 0.4]])
        self.samples_2d = np.random.multivariate_normal(self.mean_2d, self.cov_2d, 1000)
        
        self.mean_3d = np.array([0.0, 2.0, -1.0])
        self.cov_3d = np.eye(3) * 0.25
        self.samples_3d = np.random.multivariate_normal(self.mean_3d, self.cov_3d, 1000)

    def test_basic_functionality_1d(self):
        """Test basic functionality with 1D data."""
        peaks = posterior_peaks_numpy(self.samples_1d, return_dict=False)
        
        # Should return a list with one element
        self.assertIsInstance(peaks, list)
        self.assertEqual(len(peaks), 1)
        
        # Peak should be close to true mean (within 3 standard errors of sampling)
        tolerance = 3 * self.std_1d / np.sqrt(len(self.samples_1d))
        self.assertAlmostEqual(peaks[0], self.mean_1d, delta=0.2)

    def test_basic_functionality_2d(self):
        """Test basic functionality with 2D data."""
        peaks = posterior_peaks_numpy(self.samples_2d, return_dict=False)
        
        # Should return a list with two elements
        self.assertIsInstance(peaks, list)
        self.assertEqual(len(peaks), 2)
        
        # Peaks should be close to true means
        peaks_array = np.array(peaks)
        error = np.abs(peaks_array - self.mean_2d)
        self.assertTrue(np.all(error < 0.2))

    def test_return_dict_functionality(self):
        """Test return_dict=True functionality."""
        # Without labels
        peaks_dict = posterior_peaks_numpy(self.samples_2d, return_dict=True)
        
        self.assertIsInstance(peaks_dict, dict)
        self.assertEqual(len(peaks_dict), 2)
        self.assertIn(0, peaks_dict)
        self.assertIn(1, peaks_dict)
        
        # With custom labels
        labels = ['param_A', 'param_B']
        peaks_dict_labeled = posterior_peaks_numpy(
            self.samples_2d, return_dict=True, labels=labels
        )
        
        self.assertIsInstance(peaks_dict_labeled, dict)
        self.assertEqual(len(peaks_dict_labeled), 2)
        self.assertIn('param_A', peaks_dict_labeled)
        self.assertIn('param_B', peaks_dict_labeled)

    @parameterized.expand([
        ("scott",),
        ("silverman",),
        (0.1,),
        (0.5,),
    ])
    def test_different_bandwidth_methods(self, bw_method):
        """Test different bandwidth methods."""
        peaks = posterior_peaks_numpy(
            self.samples_2d, return_dict=False, bw_method=bw_method
        )
        
        # Should still return reasonable results
        self.assertEqual(len(peaks), 2)
        peaks_array = np.array(peaks)
        error = np.abs(peaks_array - self.mean_2d)
        self.assertTrue(np.all(error < 0.5))  # Slightly more tolerant for different BW methods

    @parameterized.expand([
        (50,),
        (100,),
        (200,),
    ])
    def test_different_bins(self, bins):
        """Test different number of bins."""
        peaks = posterior_peaks_numpy(
            self.samples_2d, return_dict=False, bins=bins
        )
        
        # Should still return reasonable results
        self.assertEqual(len(peaks), 2)
        peaks_array = np.array(peaks)
        error = np.abs(peaks_array - self.mean_2d)
        self.assertTrue(np.all(error < 0.3))

    def test_1d_input_as_column_vector(self):
        """Test 1D input reshaped as column vector."""
        samples_1d_reshaped = self.samples_1d.reshape(-1, 1)
        peaks = posterior_peaks_numpy(samples_1d_reshaped, return_dict=False)
        
        self.assertEqual(len(peaks), 1)
        self.assertAlmostEqual(peaks[0], self.mean_1d, delta=0.2)

    def test_3d_functionality(self):
        """Test with 3D data."""
        peaks = posterior_peaks_numpy(self.samples_3d, return_dict=False)
        
        self.assertEqual(len(peaks), 3)
        peaks_array = np.array(peaks)
        error = np.abs(peaks_array - self.mean_3d)
        self.assertTrue(np.all(error < 0.3))

    def test_custom_labels_validation(self):
        """Test validation of custom labels."""
        # Wrong number of labels should raise error
        with self.assertRaises(ValueError):
            posterior_peaks_numpy(
                self.samples_2d, return_dict=True, labels=['only_one_label']
            )

    def test_edge_cases(self):
        """Test edge cases."""
        # Very small sample size
        small_samples = self.samples_2d[:10]
        peaks = posterior_peaks_numpy(small_samples, return_dict=False)
        self.assertEqual(len(peaks), 2)
        
        # Single sample (should raise error)
        single_sample = self.samples_2d[:1]
        with self.assertRaises(ValueError):
            posterior_peaks_numpy(single_sample, return_dict=False)
            
        # Two samples (minimum required)
        two_samples = self.samples_2d[:2]
        peaks = posterior_peaks_numpy(two_samples, return_dict=False)
        self.assertEqual(len(peaks), 2)

    def test_bimodal_distribution(self):
        """Test with bimodal distribution to ensure it finds a peak."""
        # Create bimodal distribution by mixing two normals
        np.random.seed(self.seed)
        samples1 = np.random.normal(-2, 0.5, 500)
        samples2 = np.random.normal(2, 0.5, 500)
        bimodal_samples = np.concatenate([samples1, samples2])
        
        peaks = posterior_peaks_numpy(bimodal_samples, return_dict=False)
        
        # Should find one peak (either at -2 or 2, depending on KDE)
        self.assertEqual(len(peaks), 1)
        # Peak should be near one of the modes
        self.assertTrue(abs(peaks[0] - (-2)) < 1.0 or abs(peaks[0] - 2) < 1.0)

    def test_numpy_array_conversion(self):
        """Test that function properly converts inputs to numpy arrays."""
        # Test with list input
        list_samples = self.samples_2d.tolist()
        peaks = posterior_peaks_numpy(list_samples, return_dict=False)
        self.assertEqual(len(peaks), 2)

    def test_reproducibility_with_kde(self):
        """Test that results are deterministic for KDE."""
        # KDE should give consistent results for same input
        peaks1 = posterior_peaks_numpy(self.samples_2d, return_dict=False)
        peaks2 = posterior_peaks_numpy(self.samples_2d, return_dict=False)
        
        np.testing.assert_array_almost_equal(peaks1, peaks2)


@pytest.mark.short
@pytest.mark.fast
class TestGetLimitsNumpy(unittest.TestCase):
    """Test suite for get_limits_numpy function."""

    def setUp(self):
        """Set up test fixtures."""
        self.seed = 42
        np.random.seed(self.seed)
        
        self.samples_1d = np.random.uniform(-1, 3, 100)
        self.samples_2d = np.random.uniform([-2, 0], [1, 5], (100, 2))

    def test_single_sample_array(self):
        """Test with single sample array."""
        limits = get_limits_numpy(self.samples_2d)
        
        self.assertEqual(limits.shape, (2, 2))
        
        # Check that limits encompass the data
        self.assertLessEqual(limits[0, 0], self.samples_2d[:, 0].min())
        self.assertGreaterEqual(limits[0, 1], self.samples_2d[:, 0].max())
        self.assertLessEqual(limits[1, 0], self.samples_2d[:, 1].min())
        self.assertGreaterEqual(limits[1, 1], self.samples_2d[:, 1].max())

    def test_multiple_sample_arrays(self):
        """Test with list of sample arrays."""
        samples2 = np.random.uniform([-1, -1], [2, 4], (50, 2))
        limits = get_limits_numpy([self.samples_2d, samples2])
        
        self.assertEqual(limits.shape, (2, 2))
        
        # Should encompass both datasets
        all_samples = np.vstack([self.samples_2d, samples2])
        self.assertLessEqual(limits[0, 0], all_samples[:, 0].min())
        self.assertGreaterEqual(limits[0, 1], all_samples[:, 0].max())

    def test_predefined_limits(self):
        """Test with predefined limits."""
        predefined = [[-5, 5], [-10, 10]]
        limits = get_limits_numpy(self.samples_2d, limits=predefined)
        
        expected = np.array(predefined)
        np.testing.assert_array_equal(limits, expected)

    def test_single_limit_broadcast(self):
        """Test broadcasting single limit to all dimensions."""
        single_limit = [[-3, 3]]
        limits = get_limits_numpy(self.samples_2d, limits=single_limit)
        
        expected = np.array([[-3, 3], [-3, 3]])
        np.testing.assert_array_equal(limits, expected)

    def test_1d_input(self):
        """Test with 1D input."""
        limits = get_limits_numpy(self.samples_1d)
        
        self.assertEqual(limits.shape, (1, 2))
        self.assertLessEqual(limits[0, 0], self.samples_1d.min())
        self.assertGreaterEqual(limits[0, 1], self.samples_1d.max())


if __name__ == '__main__':
    unittest.main()