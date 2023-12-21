import unittest
import numpy as np
from vbi.feature_extraction.features import *

class TestAbsEnergy(unittest.TestCase):

    def test_positive_values(self):
        ts = [1, 2, 3, 4, 5]
        expected_values = [55]
        expected_labels = ['abs_energy_0']
    
        values, labels = abs_energy(ts)
    
        self.assertEqual(values, expected_values)
        self.assertEqual(labels, expected_labels)

    def test_negative_values(self):
        ts = [-1, -2, -3, -4, -5]
        expected_values = [55]
        expected_labels = ['abs_energy_0']
    
        values, labels = abs_energy(ts)
    
        self.assertEqual(values, expected_values)
        self.assertEqual(labels, expected_labels)

    def test_mixed_values(self):
        ts = [-1, 2, -3, 4, -5]
        expected_values = [55]
        expected_labels = ['abs_energy_0']
    
        values, labels = abs_energy(ts)
    
        self.assertEqual(values, expected_values)
        self.assertEqual(labels, expected_labels)

    def test_empty_ts(self):
        ts = []
        expected_values = [np.nan]
        expected_labels = ["abs_energy_0"]
    
        values, labels = abs_energy(ts)
    
        self.assertEqual(values, expected_values)
        self.assertEqual(labels, expected_labels)

    def test_nan_values(self):
        ts = [1, np.nan, 3, 4, 5]
        expected_values = [np.nan]
        expected_labels = ['abs_energy_0']
    
        values, labels = abs_energy(ts)
    
        self.assertTrue(np.isnan(values[0]))
        self.assertEqual(labels, expected_labels)

    def test_infinite_values(self):
        ts = [1, np.inf, 3, 4, 5]
        expected_values = [np.nan]
        expected_labels = ['abs_energy_0']
    
        values, labels = abs_energy(ts)
    
        self.assertTrue(np.isnan(values[0]))
        self.assertEqual(labels, expected_labels)

    def test_positive_values_fixed(self):
        ts = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        expected_values = [30, 174]
        expected_labels = ['abs_energy_0', 'abs_energy_1']

        values, labels = abs_energy(ts)

        self.assertEqual(list(values), expected_values)
        self.assertEqual(list(labels), expected_labels)

class TestAuc(unittest.TestCase):

    def test_computes_area_under_curve(self):
        ts = np.array([[1, 2, 3], [4, 5, 6]])
        fs = 2
        expected_values = np.array([2.0, 5.0])
        expected_labels = ["auc_0", "auc_1"]

        values, labels = auc(ts, fs)

        self.assertTrue(np.allclose(values, expected_values))
        self.assertEqual(labels, expected_labels)

    def test_accepts_ndarrays_input(self):
        ts = np.array([[1, 2, 3], [4, 5, 6]])
        fs = 2
        expected_values = np.array([2.0, 5.0])
        expected_labels = ["auc_0", "auc_1"]

        values, labels = auc(ts, fs)

        self.assertTrue(np.allclose(values, expected_values))
        self.assertEqual(labels, expected_labels)
    
    def test_handles_nan_values(self):
        ts = np.array([[1, np.nan, 3], [4, 5, np.nan]])
        fs = 2
        expected_values = np.array([np.nan, np.nan])
        expected_labels = ["auc_0", "auc_1"]
    
        values, labels = auc(ts, fs)
    
        self.assertTrue(np.isnan(values).all())
        self.assertEqual(labels, expected_labels)

class TestCalcVar(unittest.TestCase):

    def test_one_region_one_sample(self):
        ts = np.array([1])
        expected_values = [0]
        expected_labels = ["variance_0"]
        values, labels = calc_var(ts)
        self.assertEqual(values, expected_values)
        self.assertEqual(labels, expected_labels)

    def test_multiple_regions_one_sample(self):
        ts = np.array([[1], [2], [3]])
        expected_values = [0, 0, 0]
        expected_labels = ["variance_0", "variance_1", "variance_2"]
        values, labels = calc_var(ts)
        np.testing.assert_array_equal(values, expected_values)
        np.testing.assert_array_equal(labels, expected_labels)

    def test_one_region_multiple_samples(self):
        ts = np.array([[1, 2, 3]])
        expected_values = [0.66666667]
        expected_labels = ["variance_0"]
        values, labels = calc_var(ts)
        self.assertEqual(np.allclose(values, expected_values), True)
        self.assertEqual(labels, expected_labels)

    def test_empty_list_input(self):
        ts = []
        expected_values = [np.nan]
        expected_labels = ["variance_0"]
        values, labels = calc_var(ts)
        self.assertEqual(values, expected_values)
        self.assertEqual(labels, expected_labels)

        # Returns empty list when input is a list of empty numpy arrays.
    def test_list_of_empty_numpy_arrays_input(self):
        ts = [np.array([]), np.array([]), np.array([])]
        expected_values = [np.nan]
        expected_labels = ["variance_0"]
        values, labels = calc_var(ts)
        self.assertEqual(values, expected_values)
        self.assertEqual(labels, expected_labels)
    
class TestCalcStd(unittest.TestCase):

    def test_handles_empty_input(self):
        ts = []
        expected_values = [np.nan]
        expected_labels = ["std_0"]
    
        values, labels = calc_std(ts)
    
        self.assertEqual(values, expected_values)
        self.assertEqual(labels, expected_labels)

    def test_handles_nan_values(self):
        ts = np.array([[1, 2, np.nan], [4, np.nan, 6]])
        expected_values = [np.nan, np.nan]
        expected_labels = ["std_0", "std_1"]
    
        values, labels = calc_std(ts)
    
        self.assertTrue(np.isnan(values).all())
        self.assertEqual(labels, expected_labels)

    def test_multiple_regions_and_samples(self):
        ts = np.array([[1, 2, 3], [4, 5, 6]])
        expected_values = [0.816497, 0.816497]
        expected_labels = ["std_0", "std_1"]

        values, labels = calc_std(ts)
        self.assertTrue(np.allclose(values, expected_values))
        self.assertEqual(labels, expected_labels)

    def test_handles_infinite_values(self):
        ts = np.array([[1, 2, np.inf], [4, -np.inf, 6]])
        expected_values = [np.nan, np.nan]
        expected_labels = ["std_0", "std_1"]
    
        values, labels = calc_std(ts)
    
        self.assertTrue(np.isnan(values).all())
        self.assertEqual(labels, expected_labels)

class TestFcSum(unittest.TestCase):

    def test_calculate_sum_of_fc(self):
        x = np.array([[1, 2, 3], [4, 5, 6]])
        value, label = fc_sum(x)
        self.assertAlmostEqual(value, 2.0)
        self.assertEqual(label, "fc_sum")
    
    def test_return_zero_for_single_sample_input(self):
        x = np.array([[1], [2]])
        value, label = fc_sum(x)
        self.assertAlmostEqual(value, 0.0)
        
    
if __name__ == '__main__':
    unittest.main()
    # obj = TestModules()
    # obj.test_HH_Solution()