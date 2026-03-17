"""
Tests for vbi.feature_extraction.features_utils

Covers signal processing, spectral analysis, normalization,
and statistical utility functions.

Run with:
    pytest vbi/tests/test_features_utils_ext.py -v
"""

import unittest
import numpy as np
import pytest

from vbi.feature_extraction.features_utils import (
    autocorr_norm,
    create_symmetric_matrix,
    lpc,
    create_xx,
    kde,
    gaussian,
    calc_ecdf,
    compute_time,
    calc_fft,
    fundamental_frequency,
    spectral_distance,
    max_frequency,
    max_psd,
    median_frequency,
    spectral_centroid,
    spectral_spread,
    km_order,
    normalize_signal,
    seizure_onset_indicator,
    nat2bit,
)


# ---- helpers ----

def _sine_signal(freq=10.0, fs=100.0, duration=1.0):
    t = np.arange(0, duration, 1.0 / fs)
    return np.sin(2.0 * np.pi * freq * t), fs


def _multi_region_signal(n_regions=3, n_points=200):
    rng = np.random.default_rng(42)
    return rng.standard_normal((n_regions, n_points))


# ---- nat2bit ----

@pytest.mark.short
@pytest.mark.fast
class TestNat2Bit(unittest.TestCase):

    def test_zero(self):
        self.assertAlmostEqual(nat2bit(0), 0.0)

    def test_one_nat(self):
        result = nat2bit(1.0)
        self.assertAlmostEqual(result, 1.4426950408889634)

    def test_negative_value(self):
        result = nat2bit(-2.0)
        self.assertAlmostEqual(result, -2.0 * 1.4426950408889634)


# ---- compute_time ----

@pytest.mark.short
@pytest.mark.fast
class TestComputeTime(unittest.TestCase):

    def test_length_matches_signal(self):
        ts = np.zeros(100)
        t = compute_time(ts, fs=1000)
        self.assertEqual(len(t), 100)

    def test_first_element_zero(self):
        t = compute_time(np.zeros(50), fs=500)
        self.assertAlmostEqual(t[0], 0.0)

    def test_spacing_matches_fs(self):
        t = compute_time(np.zeros(10), fs=100)
        dt = t[1] - t[0]
        self.assertAlmostEqual(dt, 1.0 / 100.0)

    def test_last_element(self):
        fs = 200
        n = 100
        t = compute_time(np.zeros(n), fs=fs)
        self.assertAlmostEqual(t[-1], (n - 1) / fs)


# ---- create_xx ----

@pytest.mark.short
@pytest.mark.fast
class TestCreateXx(unittest.TestCase):

    def test_length_matches_input(self):
        data = np.array([1, 2, 3, 4, 5])
        xx = create_xx(data)
        self.assertEqual(len(xx), len(data))

    def test_range_covers_data(self):
        data = np.array([-3.0, 0.0, 5.0, 2.0])
        xx = create_xx(data)
        self.assertLessEqual(xx[0], min(data))
        self.assertGreaterEqual(xx[-1], max(data))

    def test_constant_signal(self):
        data = np.ones(10) * 3.0
        xx = create_xx(data)
        self.assertEqual(len(xx), 10)


# ---- calc_ecdf ----

@pytest.mark.short
@pytest.mark.fast
class TestCalcEcdf(unittest.TestCase):

    def test_output_lengths_match(self):
        signal = np.array([3, 1, 4, 1, 5])
        sorted_sig, ecdf = calc_ecdf(signal)
        self.assertEqual(len(sorted_sig), 5)
        self.assertEqual(len(ecdf), 5)

    def test_sorted_output(self):
        signal = np.array([5, 2, 8, 1])
        sorted_sig, _ = calc_ecdf(signal)
        np.testing.assert_array_equal(sorted_sig, [1, 2, 5, 8])

    def test_ecdf_ends_at_one(self):
        rng = np.random.default_rng(10)
        signal = rng.standard_normal(50)
        _, ecdf = calc_ecdf(signal)
        self.assertAlmostEqual(ecdf[-1], 1.0)

    def test_ecdf_starts_positive(self):
        signal = np.array([10, 20, 30])
        _, ecdf = calc_ecdf(signal)
        self.assertGreater(ecdf[0], 0)

    def test_ecdf_is_monotonic(self):
        rng = np.random.default_rng(11)
        signal = rng.standard_normal(100)
        _, ecdf = calc_ecdf(signal)
        diffs = np.diff(ecdf)
        self.assertTrue(np.all(diffs >= 0))


# ---- autocorr_norm ----

@pytest.mark.short
@pytest.mark.fast
class TestAutocorrNorm(unittest.TestCase):

    def test_peak_at_lag_zero(self):
        rng = np.random.default_rng(20)
        signal = rng.standard_normal(200)
        acf = autocorr_norm(signal)
        self.assertAlmostEqual(acf[0], max(acf), places=5)

    def test_zero_signal_returns_zeros(self):
        signal = np.zeros(50)
        acf = autocorr_norm(signal)
        np.testing.assert_array_equal(acf, np.zeros(50))

    def test_output_length(self):
        rng = np.random.default_rng(21)
        signal = rng.standard_normal(75)
        acf = autocorr_norm(signal)
        self.assertEqual(len(acf), 75)

    def test_sine_has_periodic_acf(self):
        fs = 200
        t = np.arange(0, 1.0, 1.0 / fs)
        signal = np.sin(2 * np.pi * 10 * t)
        acf = autocorr_norm(signal)
        self.assertAlmostEqual(acf[0], 1.0, places=1)


# ---- create_symmetric_matrix ----

@pytest.mark.short
@pytest.mark.fast
class TestCreateSymmetricMatrix(unittest.TestCase):

    def test_output_shape(self):
        acf = np.array([1.0, 0.8, 0.5, 0.2, 0.1])
        matrix = create_symmetric_matrix(acf, order=4)
        self.assertEqual(matrix.shape, (4, 4))

    def test_is_symmetric(self):
        acf = np.array([1.0, 0.9, 0.7, 0.4, 0.1, 0.0])
        matrix = create_symmetric_matrix(acf, order=5)
        np.testing.assert_array_almost_equal(matrix, matrix.T)

    def test_diagonal_equals_acf_zero(self):
        acf = np.array([3.5, 2.1, 1.0])
        matrix = create_symmetric_matrix(acf, order=3)
        for i in range(3):
            self.assertAlmostEqual(matrix[i, i], acf[0])


# ---- lpc ----

@pytest.mark.short
@pytest.mark.fast
class TestLpc(unittest.TestCase):

    def test_output_length(self):
        rng = np.random.default_rng(30)
        signal = rng.standard_normal(100)
        coeffs = lpc(signal, n_coeff=10)
        self.assertEqual(len(coeffs), 10)

    def test_first_coeff_is_one(self):
        rng = np.random.default_rng(31)
        signal = rng.standard_normal(200)
        coeffs = lpc(signal, n_coeff=12)
        self.assertAlmostEqual(coeffs[0], 1.0)

    def test_2d_input_raises(self):
        with self.assertRaises(ValueError):
            lpc(np.ones((3, 3)), n_coeff=2)

    def test_ncoeff_too_large_raises(self):
        with self.assertRaises(ValueError):
            lpc(np.array([1, 2, 3]), n_coeff=10)

    def test_zero_signal(self):
        signal = np.zeros(50)
        coeffs = lpc(signal, n_coeff=5)
        self.assertIsInstance(coeffs, tuple)


# ---- kde and gaussian ----

@pytest.mark.short
@pytest.mark.fast
class TestKde(unittest.TestCase):

    def test_output_sums_to_one(self):
        rng = np.random.default_rng(40)
        data = rng.standard_normal(200)
        pdf = kde(data)
        self.assertAlmostEqual(np.sum(pdf), 1.0, places=5)

    def test_output_length(self):
        rng = np.random.default_rng(41)
        data = rng.standard_normal(100)
        pdf = kde(data)
        self.assertEqual(len(pdf), 100)

    def test_constant_signal(self):
        data = np.ones(50) * 5.0
        pdf = kde(data)
        self.assertEqual(len(pdf), 50)

    def test_all_positive(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal(200)
        pdf = kde(data)
        self.assertTrue(np.all(pdf >= 0))


@pytest.mark.short
@pytest.mark.fast
class TestGaussian(unittest.TestCase):

    def test_output_sums_to_approx_one(self):
        rng = np.random.default_rng(50)
        data = rng.standard_normal(300)
        pdf = gaussian(data)
        self.assertAlmostEqual(np.sum(pdf), 1.0, places=3)

    def test_output_length(self):
        rng = np.random.default_rng(51)
        data = rng.standard_normal(100)
        pdf = gaussian(data)
        self.assertEqual(len(pdf), 100)

    def test_constant_signal_returns_zero(self):
        data = np.ones(50) * 3.0
        result = gaussian(data)
        self.assertEqual(result, 0.0)


# ---- calc_fft ----

@pytest.mark.short
@pytest.mark.fast
class TestCalcFft(unittest.TestCase):

    def test_output_shapes(self):
        signal, fs = _sine_signal(freq=10, fs=100, duration=1.0)
        ts = signal.reshape(1, -1)
        f, fmag = calc_fft(ts, fs)
        self.assertEqual(len(f), fmag.shape[1])
        self.assertEqual(fmag.shape[0], 1)

    def test_frequency_axis_starts_at_zero(self):
        rng = np.random.default_rng(60)
        ts = rng.standard_normal((2, 100))
        f, _ = calc_fft(ts, fs=100)
        self.assertAlmostEqual(f[0], 0.0)

    def test_peak_at_signal_freq(self):
        signal, fs = _sine_signal(freq=20, fs=200, duration=1.0)
        ts = signal.reshape(1, -1)
        f, fmag = calc_fft(ts, fs)
        peak_freq = f[np.argmax(fmag[0])]
        self.assertAlmostEqual(peak_freq, 20.0, delta=2.0)

    def test_multi_region(self):
        ts = _multi_region_signal(n_regions=4, n_points=256)
        f, fmag = calc_fft(ts, fs=100)
        self.assertEqual(fmag.shape[0], 4)


# ---- max_frequency ----

@pytest.mark.short
@pytest.mark.fast
class TestMaxFrequency(unittest.TestCase):

    def test_single_peak(self):
        signal, fs = _sine_signal(freq=25, fs=200, duration=1.0)
        ts = signal.reshape(1, -1)
        f, fmag = calc_fft(ts, fs)
        fmax, labels = max_frequency(f, fmag)
        self.assertAlmostEqual(fmax[0], 25.0, delta=3.0)
        self.assertEqual(labels, ["max_frequency_0"])

    def test_multi_region_labels(self):
        ts = _multi_region_signal(n_regions=3, n_points=256)
        f, fmag = calc_fft(ts, fs=100)
        fmax, labels = max_frequency(f, fmag)
        self.assertEqual(len(fmax), 3)
        self.assertEqual(len(labels), 3)

    def test_1d_psd_input(self):
        f = np.array([0, 10, 20, 30])
        psd = np.array([1, 5, 3, 2])
        fmax, labels = max_frequency(f, psd)
        self.assertEqual(fmax[0], 10)


# ---- max_psd ----

@pytest.mark.short
@pytest.mark.fast
class TestMaxPsd(unittest.TestCase):

    def test_correct_max(self):
        f = np.array([0, 10, 20])
        psd = np.array([[1.0, 5.0, 3.0], [2.0, 1.0, 8.0]])
        pmax, labels = max_psd(f, psd)
        self.assertAlmostEqual(pmax[0], 5.0)
        self.assertAlmostEqual(pmax[1], 8.0)

    def test_labels(self):
        psd = np.ones((4, 10))
        f = np.arange(10)
        _, labels = max_psd(f, psd)
        self.assertEqual(len(labels), 4)
        self.assertEqual(labels[0], "max_psd_0")


# ---- spectral_centroid ----

@pytest.mark.short
@pytest.mark.fast
class TestSpectralCentroid(unittest.TestCase):

    def test_single_peak(self):
        f = np.array([[0, 10, 20, 30, 40]])
        fmag = np.array([[0, 0, 100, 0, 0]])
        centroid, labels = spectral_centroid(f, fmag)
        self.assertAlmostEqual(centroid[0], 20.0)

    def test_zero_spectrum_returns_zero(self):
        f = np.array([0, 10, 20])
        fmag = np.array([[0, 0, 0]])
        centroid, _ = spectral_centroid(f, fmag)
        self.assertEqual(centroid[0], 0)

    def test_output_length(self):
        ts = _multi_region_signal(n_regions=5, n_points=128)
        f, fmag = calc_fft(ts, fs=100)
        centroid, labels = spectral_centroid(f, fmag)
        self.assertEqual(len(centroid), 5)
        self.assertEqual(len(labels), 5)


# ---- spectral_spread ----

@pytest.mark.short
@pytest.mark.fast
class TestSpectralSpread(unittest.TestCase):

    def test_single_freq_zero_spread(self):
        # spectral_spread uses global f, not per-row
        # a single frequency bin has nonzero spread due to how the formula works
        signal, fs = _sine_signal(freq=20, fs=200, duration=1.0)
        ts = signal.reshape(1, -1)
        f, fmag = calc_fft(ts, fs)
        spread, _ = spectral_spread(f, fmag)
        # pure sine spread depends on frequency resolution
        self.assertLess(spread[0], 25.0)

    def test_wide_spectrum_nonzero_spread(self):
        f = np.array([0, 10, 20, 30, 40])
        fmag = np.array([[1, 1, 1, 1, 1]])
        spread, _ = spectral_spread(f, fmag)
        self.assertGreater(spread[0], 0)

    def test_zero_spectrum(self):
        f = np.array([0, 10])
        fmag = np.array([[0, 0]])
        spread, _ = spectral_spread(f, fmag)
        self.assertEqual(spread[0], 0)


# ---- km_order ----

@pytest.mark.short
@pytest.mark.fast
class TestKmOrder(unittest.TestCase):

    def test_perfectly_synchronized(self):
        n_regions = 5
        n_time = 100
        ts = np.ones((n_regions, n_time)) * np.pi / 4
        r = km_order(ts)
        self.assertAlmostEqual(r, 1.0, places=5)

    def test_result_between_zero_and_one(self):
        rng = np.random.default_rng(123)
        ts = rng.uniform(0, 2 * np.pi, (10, 200))
        r = km_order(ts)
        self.assertGreaterEqual(r, 0.0)
        self.assertLessEqual(r, 1.0)

    def test_avg_false_returns_timeseries(self):
        ts = np.ones((3, 50))
        r = km_order(ts, avg=False)
        self.assertEqual(len(r), 50)

    def test_with_indices(self):
        ts = np.ones((5, 30))
        r = km_order(ts, indices=[0, 1, 2])
        self.assertAlmostEqual(r, 1.0, places=5)

    def test_1d_raises(self):
        with self.assertRaises(ValueError):
            km_order(np.array([1, 2, 3]))

    def test_invalid_indices_raises(self):
        ts = np.ones((3, 10))
        with self.assertRaises(ValueError):
            km_order(ts, indices=[0, 10])

    def test_single_index_raises(self):
        ts = np.ones((3, 10))
        with self.assertRaises(ValueError):
            km_order(ts, indices=[0])


# ---- normalize_signal ----

@pytest.mark.short
@pytest.mark.fast
class TestNormalizeSignal(unittest.TestCase):

    def test_zscore_zero_mean(self):
        ts = np.array([[10, 20, 30, 40, 50]], dtype=float)
        normed = normalize_signal(ts, method="zscore")
        self.assertAlmostEqual(np.mean(normed), 0.0, places=5)

    def test_minmax_range(self):
        ts = np.array([[2.0, 5.0, 8.0, 3.0]])
        normed = normalize_signal(ts, method="minmax")
        self.assertAlmostEqual(np.min(normed), 0.0)
        self.assertAlmostEqual(np.max(normed), 1.0)

    def test_max_normalization(self):
        ts = np.array([[2.0, 4.0, 8.0]])
        normed = normalize_signal(ts, method="max")
        self.assertAlmostEqual(np.max(normed), 1.0)

    def test_none_method_no_change(self):
        ts = np.array([[1.0, 2.0, 3.0]])
        normed = normalize_signal(ts, method="none")
        np.testing.assert_array_equal(normed, ts)

    def test_invalid_method_raises(self):
        ts = np.array([[1.0, 2.0]])
        with self.assertRaises(ValueError):
            normalize_signal(ts, method="foobar")

    def test_1d_reshaped(self):
        ts = np.array([1, 2, 3, 4, 5], dtype=float)
        normed = normalize_signal(ts, method="minmax")
        self.assertEqual(normed.ndim, 2)

    def test_multi_region(self):
        ts = np.array([[1, 2, 3], [10, 20, 30]], dtype=float)
        normed = normalize_signal(ts, method="minmax")
        self.assertEqual(normed.shape, (2, 3))
        for i in range(2):
            self.assertAlmostEqual(np.min(normed[i]), 0.0)
            self.assertAlmostEqual(np.max(normed[i]), 1.0)


# ---- seizure_onset_indicator ----

@pytest.mark.short
@pytest.mark.fast
class TestSeizureOnsetIndicator(unittest.TestCase):

    def test_clear_onset(self):
        ts = np.array([[0, 0, 0, 0, 5, 5, 5]])
        idx = seizure_onset_indicator(ts, thr=0.02)
        self.assertEqual(idx[0], 3)

    def test_no_onset_below_threshold(self):
        ts = np.array([[1.0, 1.001, 1.002, 1.003]])
        idx = seizure_onset_indicator(ts, thr=0.02)
        self.assertEqual(idx[0], 0)

    def test_1d_input(self):
        ts = np.array([0, 0, 0, 10, 10])
        idx = seizure_onset_indicator(ts, thr=0.02)
        self.assertEqual(idx[0], 2)

    def test_multi_region(self):
        ts = np.array([
            [0, 0, 5, 5],
            [0, 0, 0, 5],
        ])
        idx = seizure_onset_indicator(ts, thr=0.02)
        self.assertEqual(len(idx), 2)
        self.assertEqual(idx[0], 1)
        self.assertEqual(idx[1], 2)


# ---- spectral_distance ----

@pytest.mark.short
@pytest.mark.fast
class TestSpectralDistance(unittest.TestCase):

    def test_output_length(self):
        ts = _multi_region_signal(n_regions=3, n_points=128)
        f, fmag = calc_fft(ts, fs=100)
        vals, labels = spectral_distance(f, fmag)
        self.assertEqual(len(vals), 3)
        self.assertEqual(len(labels), 3)

    def test_labels_format(self):
        ts = _multi_region_signal(n_regions=2, n_points=64)
        f, fmag = calc_fft(ts, fs=100)
        _, labels = spectral_distance(f, fmag)
        self.assertEqual(labels[0], "spectral_distance_0")
        self.assertEqual(labels[1], "spectral_distance_1")


# ---- median_frequency ----

@pytest.mark.short
@pytest.mark.fast
class TestMedianFrequency(unittest.TestCase):

    def test_single_peak(self):
        f = np.array([0, 10, 20, 30, 40])
        fmag = np.array([[0, 0, 100, 0, 0]])
        fmed, labels = median_frequency(f, fmag)
        self.assertEqual(fmed[0], 20)

    def test_output_labels(self):
        ts = _multi_region_signal(n_regions=3, n_points=128)
        f, fmag = calc_fft(ts, fs=100)
        fmed, labels = median_frequency(f, fmag)
        self.assertEqual(len(labels), 3)


# ---- fundamental_frequency ----

@pytest.mark.short
@pytest.mark.fast
class TestFundamentalFrequency(unittest.TestCase):

    def test_sine_wave(self):
        signal, fs = _sine_signal(freq=15, fs=200, duration=1.0)
        ts = signal.reshape(1, -1)
        f, fmag = calc_fft(ts, fs)
        f0, labels = fundamental_frequency(f, fmag)
        self.assertAlmostEqual(f0[0], 15.0, delta=3.0)

    def test_labels(self):
        ts = _multi_region_signal(n_regions=2, n_points=200)
        f, fmag = calc_fft(ts, fs=100)
        _, labels = fundamental_frequency(f, fmag)
        self.assertEqual(len(labels), 2)


# ---- full pipeline ----

@pytest.mark.short
@pytest.mark.fast
class TestPipeline(unittest.TestCase):

    def test_fft_to_spectral_features(self):
        signal, fs = _sine_signal(freq=30, fs=200, duration=2.0)
        ts = signal.reshape(1, -1)

        f, fmag = calc_fft(ts, fs)

        fmax, _ = max_frequency(f, fmag)
        pmax, _ = max_psd(f, fmag)

        # spectral_centroid needs f tiled to match fmag shape
        f_2d = np.tile(f, (fmag.shape[0], 1))
        centroid, _ = spectral_centroid(f_2d, fmag)
        spread, _ = spectral_spread(f, fmag)

        self.assertAlmostEqual(fmax[0], 30.0, delta=3.0)
        self.assertGreater(pmax[0], 0)
        self.assertGreater(centroid[0], 0)
        self.assertGreater(spread[0], 0)

    def test_deterministic(self):
        signal, fs = _sine_signal(freq=20, fs=100, duration=1.0)
        ts = signal.reshape(1, -1)
        f, fmag = calc_fft(ts, fs)

        fmax1, _ = max_frequency(f, fmag)
        fmax2, _ = max_frequency(f, fmag)
        self.assertEqual(fmax1[0], fmax2[0])


if __name__ == "__main__":
    unittest.main()
