import unittest
import numpy as np
from vbi.feature_extraction.features import (abs_energy, average_power, auc, auc_lim, calc_var, calc_std, calc_mean, calc_centroid, calc_kurtosis, calc_skewness, calc_max, calc_min, calc_median, mean_abs_dev, median_abs_dev, rms, interq_range, zero_crossing)
from parameterized import parameterized


class TestAbsEnergy(unittest.TestCase):

    @parameterized.expand([
        ("positive_values", [1, 2, 3, 4, 5], None, [55], ['abs_energy_0']),
        ("negative_values", [-1, -2, -3, -4, -5], None, [55], ['abs_energy_0']),
        ("mixed_values", [-1, 2, -3, 4, -5], None, [55], ['abs_energy_0']),
        ("empty_ts", [], None, [np.nan], ["abs_energy_0"]),
        ("nan_values", [1, np.nan, 3, 4, 5], None, [np.nan], ['abs_energy_0']),
        ("infinite_values", [1, np.inf, 3, 4, 5], None, [np.nan], ['abs_energy_0']),
        ("positive_values_fixed", np.array([[1, 2, 3, 4], [5, 6, 7, 8]]), [0,1], [30, 174], ['abs_energy_0', 'abs_energy_1'])
    ])
    def test_abs_energy(self, name, ts, indices, expected_values, expected_labels):
        values, labels = abs_energy(ts, indices)
        if np.isnan(expected_values[0]):
            self.assertTrue(np.isnan(values[0]))
        else:
            np.testing.assert_allclose(values, expected_values)
        self.assertEqual(labels, expected_labels)


class TestAveragePower(unittest.TestCase):

    @parameterized.expand([
        ("positive_values", [1, 2, 3, 4, 5], 1, None, [13.75], ['average_power_0']),
        ("empty_ts", [], 1, None, [np.nan], ["average_power_0"]),
        ("mixed_values", [-1, 2, -3, 4, -5], 1, None, [13.75], ['average_power_0']),
        ("nan_values", [1, np.nan, 3, 4, 5], 1, None, [np.nan], ['average_power_0']),
        ("infinite_values", [1, np.inf, 3, 4, 5], 1, None, [np.nan], ['average_power_0']),
        ("positive_values_fixed", np.array([[1, 2, 3, 4], [5, 6, 7, 8]]), 1, [0,1], [10.0, 58.0], ['average_power_0', 'average_power_1'])
    ])
    def test_average_power(self, name, ts, fs, indices, expected_values, expected_labels):
        values, labels = average_power(ts, fs, indices)
        if np.isnan(expected_values[0]):
            self.assertTrue(np.isnan(values[0]))
        else:
            np.testing.assert_allclose(values, expected_values)
        self.assertEqual(labels, expected_labels)


class TestAuc(unittest.TestCase):

    @parameterized.expand([
        ("computes_area_under_curve_with_dx", np.array([[1, 2, 3], [4, 5, 6]]), None, None, None, [4.0, 10.0], ["auc_0", "auc_1"]),
        ("computes_area_under_curve_with_x", np.array([[1, 2, 3], [4, 5, 6]]), None, np.array([0, 1, 2]), None, [4.0, 10.0], ["auc_0", "auc_1"]),
        ("computes_area_under_curve_with_empty_ts", np.array([]), None, None, None, [np.nan], ["auc_0"]),
        ("handles_list_input", [[1, 2, 3], [4, 5, 6]], None, None, None, [4.0, 10.0], ["auc_0", "auc_1"]),
        ("handles_single_value_ts", np.array([1]), None, None, None, [0], ["auc_0"]),
        ("handles_empty_ts", np.array([]), None, None, None, [np.nan], ["auc_0"]),
        ("computes_area_under_curve_with_custom_dx", np.array([[1, 2, 3], [4, 5, 6]]), 0.5, None, [0,1], [2.0, 5.0], ["auc_0", "auc_1"]),
        ("handles_nan_values", np.array([[1, np.nan, 3], [4, 5, np.nan]]), None, None, None, [np.nan], ["auc_0"])
    ])
    def test_auc(self, name, ts, dx, x, indices, expected_values, expected_labels):
        values, labels = auc(ts, dx, x, indices)

        np.testing.assert_allclose(values, expected_values)
        self.assertEqual(labels, expected_labels)


class TestAucLim(unittest.TestCase):

    @parameterized.expand([
        ("computes_area_under_curve_within_limit", np.array([[1, 2, 3], [4, 5, 6]]), None, None, [(0, 2)], None, [4.0, 10.0], ["auc_lim_0", "auc_lim_1"]),
        ("handles_multiple_limits", np.array([[1, 2, 3], [4, 5, 6]]), None, None, [(0, 1), (1, 2)], None, [1.5, 4.5, 2.5, 5.5], ["auc_lim_0", "auc_lim_1", "auc_lim_2", "auc_lim_3"]),
        ("handles_custom_x_values", np.array([[1, 2, 3], [4, 5, 6]]), None, np.array([0, 1, 2]), [(0, 2)], None, [4.0, 10.0], ["auc_lim_0", "auc_lim_1"]),
        ("handles_dx", np.array([[1, 2, 3], [4, 5, 6]]), 0.5, None, [(0, 2)], None, [4.0, 10.0], ["auc_lim_0", "auc_lim_1"]),
        ("handles_empty_ts", np.array([]), None, None, [(0, 2)], None, [np.nan], ["auc_lim_0"]),
        ("handles_nan_values", np.array([[1, np.nan, 3], [4, 5, np.nan]]), None, None, [(0, 2)], [0,1], [np.nan], ["auc_lim_0"])
    ])
    def test_auc_lim(self, name, ts, dx, x, xlim, indices, expected_values, expected_labels):
        values, labels = auc_lim(ts, dx, x, xlim)

        if len(expected_values) == 0:
            self.assertEqual(values, [])
            self.assertEqual(labels, [])
        else:
            np.testing.assert_allclose(values, expected_values)
            self.assertEqual(labels, expected_labels)


class TestCalcVar(unittest.TestCase):

    @parameterized.expand([
        ("computes_variance_single_ts", [[1, 2, 3, 4, 5]], None, [2], ["var_0"]),
        ("computes_variance_multiple_ts", [[1, 2, 3], [4, 5, 6]], [0,1], [0.66666667, 0.66666667], ["var_0", "var_1"]),
        ("handles_empty_ts", [], None, [np.nan], ["var_0"]),
        ("handles_nan_values", [[1, np.nan, 3], [4, 5, np.nan]], None, [np.nan], ["var_0"])
    ])
    def test_calc_var(self, name, ts, indices, expected_values, expected_labels):
        values, labels = calc_var(ts, indices)

        if np.isnan(expected_values[0]):
            self.assertTrue(np.isnan(values[0]))
            self.assertEqual(labels, expected_labels)
        else:
            np.testing.assert_allclose(values, expected_values)
            self.assertEqual(labels, expected_labels)

class TestCalcStd(unittest.TestCase):

    @parameterized.expand([
        ("positive_values", np.array([[1, 2, 3], [4, 5, 6]]), None, [0.81649658, 0.81649658], ['std_0', 'std_1']),
        ("empty_ts", [], None, [np.nan], ["std_0"]),
        ("mixed_values", np.array([[-1, 2, -3], [4, -5, 6]]), None, [2.05480467, 4.78423336], ['std_0', 'std_1']),
        ("nan_values", np.array([[1, np.nan, 3], [4, 5, 6]]), None, [np.nan], ['std_0']),
        ("infinite_values", np.array([[1, np.inf, 3], [4, 5, 6]]), None, [np.nan], ['std_0']),
        ("positive_values_fixed", np.array([[1, 2, 3], [4, 5, 6]]), [0,1], [0.81649658, 0.81649658], ['std_0', 'std_1'])
    ])
    def test_calc_std(self, name, ts, indices, expected_values, expected_labels):
        values, labels = calc_std(ts, indices)
        for i in range(len(values)):
            if np.isnan(expected_values[i]):
                self.assertTrue(np.isnan(values[i]))
            else:
                np.testing.assert_allclose(values[i], expected_values[i])
        self.assertEqual(labels, expected_labels)

class TestCalcMean(unittest.TestCase):

    @parameterized.expand([
        ("positive_values", np.array([[1, 2, 3], [4, 5, 6]]), None, [2., 5.], ['mean_0', 'mean_1']),
        ("empty_ts", [], None, [np.nan], ["mean_0"]),
        ("mixed_values", np.array([[-1, 2, -3], [4, -5, 6]]), None, [-0.66666667,  1.66666667], ['mean_0', 'mean_1']),
        ("nan_values", np.array([[1, np.nan, 3], [4, 5, 6]]), None, [np.nan], ['mean_0']),
        ("infinite_values", np.array([[1, np.inf, 3], [4, 5, 6]]), None, [np.nan], ['mean_0']),
        ("positive_values_fixed", np.array([[1, 2, 3], [4, 5, 6]]), [0,1], [2., 5.], ['mean_0', 'mean_1'])
    ])
    def test_calc_mean(self, name, ts, indices, expected_values, expected_labels):
        values, labels = calc_mean(ts, indices)
        for i in range(len(values)):
            if np.isnan(expected_values[i]):
                self.assertTrue(np.isnan(values[i]))
            else:
                np.testing.assert_allclose(values[i], expected_values[i])
        self.assertEqual(labels, expected_labels)

class TestCalcCentroid(unittest.TestCase):

    @parameterized.expand([
        ("positive_values", np.array([[1, 2, 3], [4, 5, 6]]), 1, None, [1.57142857, 1.25974026], ['centroid_0', 'centroid_1']),
        ("empty_ts", [], 1, None, [np.nan], ["centroid_0"]),
        ("mixed_values", np.array([[-1, 2, -3], [4, -5, 6]]), 1, None, [1.57142857, 1.25974026], ['centroid_0', 'centroid_1']),
        ("nan_values", np.array([[1, np.nan, 3], [4, 5, 6]]), 1, None, [np.nan], ['centroid_0']),
        ("infinite_values", np.array([[1, np.inf, 3], [4, 5, 6]]), 1, None, [np.nan], ['centroid_0']),
        ("positive_values_fixed", np.array([[1, 2, 3], [4, 5, 6]]), 1, [0,1], [1.57142857, 1.25974026], ['centroid_0', 'centroid_1'])
    ])
    def test_calc_centroid(self, name, ts, fs, indices, expected_values, expected_labels):
        values, labels = calc_centroid(ts, fs, indices)
        for i in range(len(values)):
            if np.isnan(expected_values[i]):
                self.assertTrue(np.isnan(values[i]))
            else:
                np.testing.assert_allclose(values[i], expected_values[i])
        self.assertEqual(labels, expected_labels)

class TestCalcKurtosis(unittest.TestCase):

    @parameterized.expand([
        ("positive_values", np.array([[1, 2, 3], [4, 5, 6]]), None, [-1.5, -1.5], ['kurtosis_0', 'kurtosis_1']),
        ("empty_ts", [], None, [np.nan], ["kurtosis_0"]),
        ("mixed_values", np.array([[-1, 2, -3], [4, -5, 6]]), None, [-1.5, -1.5], ['kurtosis_0', 'kurtosis_1']),
        ("nan_values", np.array([[1, np.nan, 3], [4, 5, 6]]), None, [np.nan], ['kurtosis_0']),
        ("infinite_values", np.array([[1, np.inf, 3], [4, 5, 6]]), None, [np.nan], ['kurtosis_0']),
        ("positive_values_fixed", np.array([[1, 2, 3], [4, 5, 6]]), [0,1], [-1.5, -1.5], ['kurtosis_0', 'kurtosis_1'])
    ])
    def test_calc_kurtosis(self, name, ts, indices, expected_values, expected_labels):
        values, labels = calc_kurtosis(ts, indices)
        for i in range(len(values)):
            if np.isnan(expected_values[i]):
                self.assertTrue(np.isnan(values[i]))
            else:
                np.testing.assert_allclose(values[i], expected_values[i])
        self.assertEqual(labels, expected_labels)

class TestSkewness(unittest.TestCase):

    @parameterized.expand([
        ("positive_values", np.array([[1, 2, 3, 1], [4, 5, 6, 0]]), None, [ 0.4933822 , -0.83315041], ['skewness_0', 'skewness_1']),
        ("empty_ts", [], None, [np.nan], ["skewness_0"]),
        ("mixed_values", np.array([[-1, 2, -3, 1], [4, -5, 6, 0]]), None, [-0.27803056, -0.39699236], ['skewness_0', 'skewness_1']),
        ("nan_values", np.array([[1, np.nan, 3], [4, 5, 6]]), None, [np.nan], ['skewness_0']),
        ("infinite_values", np.array([[1, np.inf, 3], [4, 5, 6]]), None, [np.nan], ['skewness_0']),
        ("positive_values_fixed", np.array([[1, 2, 3, 1], [4, 5, 6, 0]]), [0,1], [ 0.4933822 , -0.83315041], ['skewness_0', 'skewness_1'])
    ])
    def test_skewness(self, name, ts, indices, expected_values, expected_labels):
        values, labels = calc_skewness(ts, indices)
        for i in range(len(values)):
            if np.isnan(expected_values[i]):
                self.assertTrue(np.isnan(values[i]))
            else:
                np.testing.assert_allclose(values[i], expected_values[i])
        self.assertEqual(labels, expected_labels)

class TestMax(unittest.TestCase):

    @parameterized.expand([
        ("positive_values", np.array([[1, 2, 3], [4, 5, 6]]), None, [3, 6], ['max_0', 'max_1']),
        ("empty_ts", [], None, [np.nan], ["max_0"]),
        ("mixed_values", np.array([[-1, 2, -3], [4, -5, 6]]), None, [2, 6], ['max_0', 'max_1']),
        ("nan_values", np.array([[1, np.nan, 3], [4, 5, 6]]), None, [np.nan], ['max_0']),
        ("infinite_values", np.array([[1, np.inf, 3], [4, 5, 6]]), None, [np.nan], ['max_0']),
        ("positive_values_fixed", np.array([[1, 2, 3], [4, 5, 6]]), [0,1], [3, 6], ['max_0', 'max_1'])
    ])
    def test_max(self, name, ts, indices, expected_values, expected_labels):
        values, labels = calc_max(ts, indices)
        for i in range(len(values)):
            if np.isnan(expected_values[i]):
                self.assertTrue(np.isnan(values[i]))
            else:
                np.testing.assert_allclose(values[i], expected_values[i])
        self.assertEqual(labels, expected_labels)

class TestMin(unittest.TestCase):

    @parameterized.expand([
        ("positive_values", np.array([[1, 2, 3], [4, 5, 6]]), None, [1, 4], ['min_0', 'min_1']),
        ("empty_ts", [], None, [np.nan], ["min_0"]),
        ("mixed_values", np.array([[-1, 2, -3], [4, -5, 6]]), None, [-3, -5], ['min_0', 'min_1']),
        ("nan_values", np.array([[1, np.nan, 3], [4, 5, 6]]), None, [np.nan], ['min_0']),
        ("infinite_values", np.array([[1, -np.inf, 3], [4, 5, 6]]), None, [np.nan], ['min_0']),
        ("positive_values_fixed", np.array([[1, 2, 3], [4, 5, 6]]), [0,1], [1, 4], ['min_0', 'min_1'])
    ])
    def test_min(self, name, ts, indices, expected_values, expected_labels):
        values, labels = calc_min(ts, indices)
        for i in range(len(values)):
            if np.isnan(expected_values[i]):
                self.assertTrue(np.isnan(values[i]))
            else:
                np.testing.assert_allclose(values[i], expected_values[i])
        self.assertEqual(labels, expected_labels)

class TestMedian(unittest.TestCase):

    @parameterized.expand([
        ("positive_values", np.array([[1, 2, 3], [4, 5, 6]]), None, [2.0, 5.0], ['median_0', 'median_1']),
        ("empty_ts", [], None, [np.nan], ["median_0"]),
        ("mixed_values", np.array([[-1, 2, -3], [4, -5, 6]]), None, [-1.0, 4.0], ['median_0', 'median_1']),
        ("nan_values", np.array([[1, np.nan, 3], [4, 5, 6]]), None, [np.nan], ['median_0']),
        ("infinite_values", np.array([[1, np.inf, 3], [4, 5, 6]]), None, [np.nan], ['median_0']),
        ("positive_values_fixed", np.array([[1, 2, 3], [4, 5, 6]]), [0,1], [2.0, 5.0], ['median_0', 'median_1'])
    ])
    def test_median(self, name, ts, indices, expected_values, expected_labels):
        values, labels = calc_median(ts, indices)
        for i in range(len(values)):
            if np.isnan(expected_values[i]):
                self.assertTrue(np.isnan(values[i]))
            else:
                np.testing.assert_allclose(values[i], expected_values[i])
        self.assertEqual(labels, expected_labels)


class TestMeanAbsDev(unittest.TestCase):

    @parameterized.expand([
        ("positive_values", np.array([[1, 2, 3], [4, 5, 6]]), None, [0.66666667, 0.66666667], ['mean_abs_dev_0', 'mean_abs_dev_1']),
        ("empty_ts", [], None, [np.nan], ["mean_abs_dev_0"]),
        ("mixed_values", np.array([[-1, 2, -3], [4, -5, 6]]), None, [1.77777778, 4.44444444], ['mean_abs_dev_0', 'mean_abs_dev_1']),
        ("nan_values", np.array([[1, np.nan, 3], [4, 5, 6]]), None, [np.nan], ['mean_abs_dev_0']),
        ("infinite_values", np.array([[1, np.inf, 3], [4, 5, 6]]), None, [np.nan], ['mean_abs_dev_0']),
        ("positive_values_fixed", np.array([[1, 2, 3], [4, 5, 6]]), [0,1], [0.66666667, 0.66666667], ['mean_abs_dev_0', 'mean_abs_dev_1'])
    ])
    def test_mean_abs_dev(self, name, ts, indices, expected_values, expected_labels):
        values, labels = mean_abs_dev(ts, indices)
        for i in range(len(values)):
            if np.isnan(expected_values[i]):
                self.assertTrue(np.isnan(values[i]))
            else:
                np.testing.assert_allclose(values[i], expected_values[i])
        self.assertEqual(labels, expected_labels)

class TestMedianAbsDev(unittest.TestCase):

    @parameterized.expand([
        ("positive_values", np.array([[1, 2, 3], [4, 5, 6]]), None, [1.0,1.0], ['median_abs_dev_0', 'median_abs_dev_1']),
        ("empty_ts", [], None, [np.nan], ["median_abs_dev_0"]),
        ("mixed_values", np.array([[-1, 2, -3], [4, -5, 6]]), None, [2.0, 2.0], ['median_abs_dev_0', 'median_abs_dev_1']),
        ("nan_values", np.array([[1, np.nan, 3], [4, 5, 6]]), None, [np.nan], ['median_abs_dev_0']),
        ("infinite_values", np.array([[1, np.inf, 3], [4, 5, 6]]), None, [np.nan], ['median_abs_dev_0']),
        ("positive_values_fixed", np.array([[1, 2, 3], [4, 5, 6]]), [0,1], [1.0,1.0], ['median_abs_dev_0', 'median_abs_dev_1'])
    ])
    def test_median_abs_dev(self, name, ts, indices, expected_values, expected_labels):
        values, labels = median_abs_dev(ts, indices)
        for i in range(len(values)):
            if np.isnan(expected_values[i]):
                self.assertTrue(np.isnan(values[i]))
            else:
                np.testing.assert_allclose(values[i], expected_values[i])
        self.assertEqual(labels, expected_labels)

class TestRms(unittest.TestCase):

    @parameterized.expand([
        ("positive_values", np.array([[1, 2, 3], [4, 5, 6]]), None, [2.1602469, 5.06622805], ['rms_0', 'rms_1']),
        ("empty_ts", [], None, [np.nan], ["rms_0"]),
        ("mixed_values", np.array([[-1, 2, -3], [4, -5, 6]]), None, [2.1602469, 5.06622805], ['rms_0', 'rms_1']),
        ("nan_values", np.array([[1, np.nan, 3], [4, 5, 6]]), None, [np.nan], ['rms_0']),
        ("infinite_values", np.array([[1, np.inf, 3], [4, 5, 6]]), None, [np.nan], ['rms_0']),
        ("positive_values_fixed", np.array([[1, 2, 3], [4, 5, 6]]), [0,1], [2.1602469, 5.06622805], ['rms_0', 'rms_1'])
    ])
    def test_rms(self, name, ts, indices, expected_values, expected_labels):
        values, labels = rms(ts, indices)
        for i in range(len(values)):
            if np.isnan(expected_values[i]):
                self.assertTrue(np.isnan(values[i]))
            else:
                np.testing.assert_allclose(values[i], expected_values[i])
        self.assertEqual(labels, expected_labels)


class TestInterqRange(unittest.TestCase):

    @parameterized.expand([
        (np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]), None, np.array([2, 2]), ["interq_range_0", "interq_range_1"]),
        (np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]), [0, 1], np.array([2, 2]), ["interq_range_0", "interq_range_1"]),
        (np.array([]), None, np.array([np.nan]), ["interq_range_0"]),
    ])
    def test_interq_range(self, ts, indices, expected_values, expected_labels):
        values, labels = interq_range(ts, indices)
        for i in range(len(values)):
            if np.isnan(expected_values[i]):
                self.assertTrue(np.isnan(values[i]))
            else:
                np.testing.assert_allclose(values[i], expected_values[i])
        self.assertEqual(labels, expected_labels)

class TestZeroCrossing(unittest.TestCase):

    @parameterized.expand([
        (np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]), None, np.array([0, 0]), ["zero_crossing_0", "zero_crossing_1"]),
        (np.array([[1, 2, -3, 4, 5], [0, 7, -8, 9, 10]]), [0, 1], np.array([2, 3]), ["zero_crossing_0", "zero_crossing_1"]),
        (np.array([]), None, np.array([np.nan]), ["zero_crossing_0"]),
    ])

    def test_zero_crossing(self, ts, indices, expected_values, expected_labels):
        values, labels = zero_crossing(ts, indices)
        for i in range(len(values)):
            if np.isnan(expected_values[i]):
                self.assertTrue(np.isnan(values[i]))
            else:
                np.testing.assert_allclose(values[i], expected_values[i])
        self.assertEqual(labels, expected_labels)

class TestCalcRess(unittest.TestCase):
    pass


if __name__ == '__main__':
    unittest.main()
    # obj = TestModules()
    # obj.test_HH_Solution()
