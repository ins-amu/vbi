"""
Unit tests for Stuart-Landau oscillator model.

These tests verify:
1. Basic integration runs without errors
2. Output shapes and types are correct
3. Numerical stability (no NaN/Inf values)
4. Amplitude converges to theoretical limit cycle: |z| ≈ sqrt(a)
5. Delay handling (zero delays vs. non-zero delays)
6. Coupling effects
7. Deterministic vs. stochastic behavior
8. Parameter validation
"""

import unittest
import numpy as np
import pytest
from copy import deepcopy
from vbi.models.numba.sl import SL_sde


class TestStuartLandauBasic(unittest.TestCase):
    """Basic functionality tests for Stuart-Landau model."""

    def setUp(self):
        """Set up common test parameters."""
        self.nn = 2
        self.weights = np.array([
            [0.0, 0.1],
            [0.1, 0.0],
        ])
        self.base_params = {
            "nn": self.nn,
            "weights": self.weights,
            "a": 0.20,
            "omega": 2.0 * np.pi * 0.040,  # 40 Hz in rad/ms
            "G": 0.5,
            "sigma": 0.01,
            "dt": 0.01,
            "t_end": 100.0,
            "t_cut": 20.0,
            "seed": 42,
            "RECORD_X": True,
            "x_decimate": 1,
        }

    def test_initialization(self):
        """Test model initialization."""
        model = SL_sde(par=self.base_params)
        self.assertIsNotNone(model)
        self.assertEqual(model.P.nn, self.nn)
        self.assertEqual(model.P.a, 0.20)
        self.assertAlmostEqual(model.P.omega, 2.0 * np.pi * 0.040, places=6)

    def test_basic_run(self):
        """Test that model runs without errors."""
        model = SL_sde(par=self.base_params)
        result = model.run()
        
        self.assertIn("X", result)
        self.assertIn("t", result)
        self.assertIsInstance(result["X"], np.ndarray)
        self.assertIsInstance(result["t"], np.ndarray)

    def test_output_shapes(self):
        """Test output array shapes are correct."""
        params = deepcopy(self.base_params)
        params["t_end"] = 100.0
        params["t_cut"] = 20.0
        params["dt"] = 0.01
        params["x_decimate"] = 1
        
        model = SL_sde(par=params)
        result = model.run()
        
        X = result["X"]
        t = result["t"]
        
        # Time points after t_cut
        expected_steps = int((params["t_end"] - params["t_cut"]) / params["dt"])
        
        self.assertEqual(X.shape[1], self.nn, "X should have nn columns")
        self.assertEqual(len(t), len(X), "Time and X should have same length")
        self.assertGreater(len(X), 0, "Should have recorded data")

    def test_complex_output(self):
        """Test that output is complex-valued."""
        model = SL_sde(par=self.base_params)
        result = model.run()
        
        X = result["X"]
        self.assertEqual(X.dtype, np.complex128, "X should be complex128")

    def test_no_nan_values(self):
        """Test that integration produces no NaN values."""
        model = SL_sde(par=self.base_params)
        result = model.run()
        
        X = result["X"]
        self.assertFalse(np.any(np.isnan(X)), "X should not contain NaN values")
        self.assertFalse(np.any(np.isinf(X)), "X should not contain Inf values")

    def test_time_vector(self):
        """Test time vector is correctly generated."""
        params = deepcopy(self.base_params)
        params["t_end"] = 100.0
        params["t_cut"] = 20.0
        
        model = SL_sde(par=params)
        result = model.run()
        
        t = result["t"]
        
        # All times should be > t_cut
        self.assertTrue(np.all(t >= params["t_cut"]), "All times should be >= t_cut")
        self.assertTrue(np.all(t <= params["t_end"]), "All times should be <= t_end")
        
        # Time should be monotonically increasing
        self.assertTrue(np.all(np.diff(t) >= 0), "Time should be monotonically increasing")


class TestStuartLandauDynamics(unittest.TestCase):
    """Tests for Stuart-Landau dynamics and limit cycle behavior."""

    def setUp(self):
        """Set up common test parameters."""
        self.nn = 2
        self.weights = np.array([
            [0.0, 0.1],
            [0.1, 0.0],
        ])

    def test_limit_cycle_amplitude(self):
        """Test that amplitude converges to sqrt(a) for uncoupled oscillator."""
        params = {
            "nn": 1,
            "weights": np.array([[0.0]]),
            "a": 0.20,
            "omega": 2.0 * np.pi * 0.040,
            "G": 0.0,  # No coupling
            "sigma": 0.0,  # Deterministic
            "dt": 0.01,
            "t_end": 500.0,
            "t_cut": 300.0,  # Long transient to reach limit cycle
            "seed": 42,
            "RECORD_X": True,
            "x_decimate": 1,
        }
        
        model = SL_sde(par=params)
        x0 = np.array([0.1 + 0.1j])  # Initial condition
        result = model.run(x0=x0)
        
        X = result["X"]
        amplitude = np.abs(X)
        
        # Theoretical limit cycle amplitude
        expected_amplitude = np.sqrt(params["a"])
        mean_amplitude = amplitude.mean()
        
        # Should be within 5% of theoretical value
        self.assertAlmostEqual(
            mean_amplitude, expected_amplitude, delta=0.05,
            msg=f"Amplitude {mean_amplitude:.4f} should be near sqrt(a)={expected_amplitude:.4f}"
        )

    def test_oscillation_frequency(self):
        """Test that oscillation occurs at the correct frequency."""
        params = {
            "nn": 1,
            "weights": np.array([[0.0]]),
            "a": 0.20,
            "omega": 2.0 * np.pi * 0.040,  # 40 Hz
            "G": 0.0,
            "sigma": 0.0,
            "dt": 0.01,
            "t_end": 500.0,
            "t_cut": 300.0,
            "seed": 42,
            "RECORD_X": True,
            "x_decimate": 1,
        }
        
        model = SL_sde(par=params)
        x0 = np.array([0.1 + 0.1j])
        result = model.run(x0=x0)
        
        X = result["X"]
        t = result["t"]
        
        # Extract phase
        phase = np.angle(X[:, 0])
        
        # Unwrap phase to get continuous phase evolution
        phase_unwrapped = np.unwrap(phase)
        
        # Estimate frequency from phase derivative
        dt_actual = np.mean(np.diff(t))
        dphase_dt = np.gradient(phase_unwrapped, dt_actual)
        
        # omega = dphase/dt
        estimated_omega = np.mean(dphase_dt)
        expected_omega = params["omega"]
        
        # Should be within 10% of expected frequency
        relative_error = np.abs(estimated_omega - expected_omega) / expected_omega
        self.assertLess(
            relative_error, 0.10,
            msg=f"Estimated omega {estimated_omega:.4f} should match {expected_omega:.4f}"
        )

    def test_deterministic_reproducibility(self):
        """Test that deterministic runs are reproducible."""
        params = {
            "nn": 2,
            "weights": np.array([[0.0, 0.1], [0.1, 0.0]]),
            "a": 0.20,
            "omega": 2.0 * np.pi * 0.040,
            "G": 0.5,
            "sigma": 0.0,  # Deterministic
            "dt": 0.01,
            "t_end": 100.0,
            "t_cut": 20.0,
            "seed": 42,
        }
        
        x0 = np.array([0.1 + 0.1j, 0.1 - 0.1j])
        
        # Run 1
        model1 = SL_sde(par=params)
        result1 = model1.run(x0=x0)
        
        # Run 2
        model2 = SL_sde(par=params)
        result2 = model2.run(x0=x0)
        
        # Results should be identical
        np.testing.assert_array_equal(
            result1["X"], result2["X"],
            err_msg="Deterministic runs should produce identical results"
        )

    def test_stochastic_variability(self):
        """Test that stochastic runs with same seed are reproducible."""
        params = {
            "nn": 2,
            "weights": np.array([[0.0, 0.1], [0.1, 0.0]]),
            "a": 0.20,
            "omega": 2.0 * np.pi * 0.040,
            "G": 0.5,
            "sigma": 0.02,  # Stochastic
            "dt": 0.01,
            "t_end": 100.0,
            "t_cut": 20.0,
            "seed": 42,
        }
        
        # Run 1
        model1 = SL_sde(par=params)
        result1 = model1.run()
        
        # Run 2 with same seed
        model2 = SL_sde(par=params)
        result2 = model2.run()
        
        # Results should be identical (same seed)
        np.testing.assert_array_almost_equal(
            result1["X"], result2["X"],
            err_msg="Stochastic runs with same seed should be reproducible"
        )


class TestStuartLandauCoupling(unittest.TestCase):
    """Tests for coupling and delay effects."""

    def test_zero_coupling(self):
        """Test uncoupled oscillators evolve independently."""
        params = {
            "nn": 2,
            "weights": np.array([[0.0, 0.0], [0.0, 0.0]]),  # No connections
            "a": 0.20,
            "omega": 2.0 * np.pi * 0.040,
            "G": 0.5,  # Coupling strength doesn't matter if weights=0
            "sigma": 0.0,
            "dt": 0.01,
            "t_end": 100.0,
            "t_cut": 20.0,
            "seed": 42,
        }
        
        # Different initial conditions
        x0 = np.array([0.1 + 0.1j, 0.2 + 0.05j])
        
        model = SL_sde(par=params)
        result = model.run(x0=x0)
        
        X = result["X"]
        amp = np.abs(X)
        
        # Both should converge to sqrt(a) independently
        expected_amp = np.sqrt(params["a"])
        
        self.assertAlmostEqual(amp[:, 0].mean(), expected_amp, delta=0.05)
        self.assertAlmostEqual(amp[:, 1].mean(), expected_amp, delta=0.05)

    def test_coupling_synchronization(self):
        """Test that coupling can lead to phase synchronization."""
        params = {
            "nn": 2,
            "weights": np.array([[0.0, 0.5], [0.5, 0.0]]),  # Strong coupling
            "a": 0.20,
            "omega": 2.0 * np.pi * 0.040,
            "G": 1.0,  # Strong coupling strength
            "sigma": 0.0,
            "dt": 0.01,
            "t_end": 500.0,
            "t_cut": 400.0,  # Long time to synchronize
            "seed": 42,
        }
        
        # Start with different phases
        x0 = np.array([0.1 + 0.1j, 0.1 - 0.1j])
        
        model = SL_sde(par=params)
        result = model.run(x0=x0)
        
        X = result["X"]
        
        # Check phase difference
        phase1 = np.angle(X[:, 0])
        phase2 = np.angle(X[:, 1])
        phase_diff = np.abs(phase1 - phase2)
        
        # With strong coupling, phase difference should be small or ~pi
        # (could sync in-phase or anti-phase)
        mean_phase_diff = np.mean(phase_diff)
        
        # Either in-phase (diff ≈ 0) or anti-phase (diff ≈ pi)
        in_phase = mean_phase_diff < 0.5
        anti_phase = np.abs(mean_phase_diff - np.pi) < 0.5
        
        self.assertTrue(
            in_phase or anti_phase,
            msg=f"With strong coupling, should sync: phase_diff={mean_phase_diff:.3f}"
        )

    def test_zero_delay(self):
        """Test that zero delay (tr_len=0) works correctly."""
        params = {
            "nn": 2,
            "weights": np.array([[0.0, 0.1], [0.1, 0.0]]),
            "a": 0.20,
            "omega": 2.0 * np.pi * 0.040,
            "G": 0.5,
            "sigma": 0.01,
            "dt": 0.01,
            "t_end": 100.0,
            "t_cut": 20.0,
            "tr_len": np.array([[0.0, 0.0], [0.0, 0.0]]),  # Explicit zero delays
            "seed": 42,
        }
        
        model = SL_sde(par=params)
        result = model.run()
        
        X = result["X"]
        
        # Should run without NaN
        self.assertFalse(np.any(np.isnan(X)))

    def test_scalar_delay(self):
        """Test that scalar tr_len is correctly applied."""
        params = {
            "nn": 2,
            "weights": np.array([[0.0, 0.1], [0.1, 0.0]]),
            "a": 0.20,
            "omega": 2.0 * np.pi * 0.040,
            "G": 0.5,
            "sigma": 0.01,
            "dt": 0.01,
            "t_end": 100.0,
            "t_cut": 20.0,
            "tr_len": 10.0,  # Scalar delay
            "speed": 5.0,
            "seed": 42,
        }
        
        model = SL_sde(par=params)
        result = model.run()
        
        X = result["X"]
        
        # Should run without NaN
        self.assertFalse(np.any(np.isnan(X)))

    def test_matrix_delay(self):
        """Test that matrix tr_len works correctly."""
        params = {
            "nn": 3,
            "weights": np.array([
                [0.0, 0.1, 0.05],
                [0.1, 0.0, 0.1],
                [0.05, 0.1, 0.0],
            ]),
            "a": 0.20,
            "omega": 2.0 * np.pi * 0.040,
            "G": 0.5,
            "sigma": 0.01,
            "dt": 0.01,
            "t_end": 100.0,
            "t_cut": 20.0,
            "tr_len": np.array([
                [0.0, 10.0, 15.0],
                [10.0, 0.0, 12.0],
                [15.0, 12.0, 0.0],
            ]),
            "speed": 5.0,
            "seed": 42,
        }
        
        model = SL_sde(par=params)
        result = model.run()
        
        X = result["X"]
        
        # Should run without NaN
        self.assertFalse(np.any(np.isnan(X)))
        self.assertEqual(X.shape[1], 3, "Should have 3 nodes")


class TestStuartLandauParameters(unittest.TestCase):
    """Tests for parameter handling and validation."""

    def test_example_parameters(self):
        """Test using parameters from sl_example.py."""
        nn = 4
        weights = np.array([
            [0.0, 0.15, 0.05, 0.0],
            [0.15, 0.0, 0.15, 0.05],
            [0.05, 0.15, 0.0, 0.15],
            [0.0, 0.05, 0.15, 0.0],
        ])
        
        params = {
            "nn": nn,
            "weights": weights,
            "a": 0.20,
            "omega": 2.0 * np.pi * 0.040,
            "G": 0.6,
            "sigma": 0.01,
            "dt": 0.01,
            "t_end": 300.0,
            "t_cut": 100.0,
            "speed": 5.0,
            "tr_len": np.zeros_like(weights),
            "seed": 42,
            "RECORD_X": True,
            "x_decimate": 10,
        }
        
        model = SL_sde(par=params)
        result = model.run()
        
        X = result["X"]
        t = result["t"]
        
        # Basic checks
        self.assertEqual(X.shape[1], nn)
        self.assertFalse(np.any(np.isnan(X)))
        self.assertTrue(np.all(t >= params["t_cut"]))
        
        # Check amplitude is reasonable
        amp = np.abs(X)
        expected_amp = np.sqrt(params["a"])
        
        for i in range(nn):
            mean_amp = amp[:, i].mean()
            # With coupling, amplitude might differ slightly from uncoupled case
            self.assertGreater(mean_amp, 0.2, f"Node {i} amplitude too small")
            self.assertLess(mean_amp, 0.8, f"Node {i} amplitude too large")

    def test_decimation(self):
        """Test that x_decimate reduces output size."""
        params_no_decimate = {
            "nn": 2,
            "weights": np.array([[0.0, 0.1], [0.1, 0.0]]),
            "a": 0.20,
            "omega": 2.0 * np.pi * 0.040,
            "G": 0.5,
            "sigma": 0.01,
            "dt": 0.01,
            "t_end": 100.0,
            "t_cut": 20.0,
            "seed": 42,
            "x_decimate": 1,
        }
        
        params_decimate = deepcopy(params_no_decimate)
        params_decimate["x_decimate"] = 10
        
        model1 = SL_sde(par=params_no_decimate)
        result1 = model1.run()
        
        model2 = SL_sde(par=params_decimate)
        result2 = model2.run()
        
        # Decimated output should be ~10x smaller
        ratio = len(result1["X"]) / len(result2["X"])
        self.assertAlmostEqual(ratio, 10.0, delta=1.0)

    def test_different_a_values(self):
        """Test that different bifurcation parameters give different amplitudes."""
        base_params = {
            "nn": 1,
            "weights": np.array([[0.0]]),
            "omega": 2.0 * np.pi * 0.040,
            "G": 0.0,
            "sigma": 0.0,
            "dt": 0.01,
            "t_end": 300.0,
            "t_cut": 200.0,
            "seed": 42,
        }
        
        amplitudes = []
        a_values = [0.1, 0.2, 0.3]
        
        for a in a_values:
            params = deepcopy(base_params)
            params["a"] = a
            
            model = SL_sde(par=params)
            x0 = np.array([0.1 + 0.1j])
            result = model.run(x0=x0)
            
            X = result["X"]
            mean_amp = np.abs(X).mean()
            amplitudes.append(mean_amp)
        
        # Amplitude should increase with a: |z| ≈ sqrt(a)
        self.assertLess(amplitudes[0], amplitudes[1])
        self.assertLess(amplitudes[1], amplitudes[2])
        
        # Check against theoretical values
        for i, a in enumerate(a_values):
            expected = np.sqrt(a)
            self.assertAlmostEqual(amplitudes[i], expected, delta=0.05)


@pytest.mark.long
@pytest.mark.slow
class TestStuartLandauLongRunning(unittest.TestCase):
    """Long-running tests for stability and convergence."""

    def test_long_simulation(self):
        """Test long simulation remains stable."""
        params = {
            "nn": 4,
            "weights": np.array([
                [0.0, 0.15, 0.05, 0.0],
                [0.15, 0.0, 0.15, 0.05],
                [0.05, 0.15, 0.0, 0.15],
                [0.0, 0.05, 0.15, 0.0],
            ]),
            "a": 0.20,
            "omega": 2.0 * np.pi * 0.040,
            "G": 0.6,
            "sigma": 0.02,
            "dt": 0.01,
            "t_end": 5000.0,  # Long simulation
            "t_cut": 1000.0,
            "seed": 42,
            "x_decimate": 10,
        }
        
        model = SL_sde(par=params)
        result = model.run()
        
        X = result["X"]
        
        # Should remain stable (no NaN/Inf)
        self.assertFalse(np.any(np.isnan(X)))
        self.assertFalse(np.any(np.isinf(X)))
        
        # Amplitude should be bounded
        amp = np.abs(X)
        self.assertTrue(np.all(amp < 2.0), "Amplitude should remain bounded")
        self.assertTrue(np.all(amp > 0.01), "Amplitude should not decay to zero")


if __name__ == "__main__":
    unittest.main()
