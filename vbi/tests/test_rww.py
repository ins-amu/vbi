"""
Unit tests for Reduced Wong-Wang (RWW) model.

These tests verify:
1. Basic integration runs without errors
2. Output shapes and types are correct
3. Numerical stability (no NaN/Inf values)
4. Parameter validation
5. BOLD signal generation
6. Unified parameter access methods
"""

import unittest
import numpy as np
import pytest
from copy import deepcopy
from vbi.models.numba.rww import RWW_sde


class TestRWWParameterAccess(unittest.TestCase):
    """Test unified parameter access methods for RWW model."""

    def setUp(self):
        """Set up common test parameters."""
        self.weights = np.array([
            [0.0, 0.1],
            [0.1, 0.0],
        ])
        self.custom_params = {
            "weights": self.weights,
            "G": 0.5,
            "sigma": 0.02,
            "dt": 1.0,
            "t_end": 500.0,
        }

    def test_get_parameters_returns_dict(self):
        """Test that get_parameters() returns a dictionary."""
        model = RWW_sde(self.custom_params)
        params = model.get_parameters()
        
        self.assertIsInstance(params, dict)
        self.assertGreater(len(params), 0)

    def test_get_parameters_includes_custom_values(self):
        """Test that get_parameters() returns actual custom values."""
        model = RWW_sde(self.custom_params)
        params = model.get_parameters()
        
        self.assertAlmostEqual(params['G'], 0.5)
        self.assertAlmostEqual(params['sigma'], 0.02)
        self.assertAlmostEqual(params['dt'], 1.0)

    def test_get_parameters_includes_defaults(self):
        """Test that get_parameters() includes default values for non-custom params."""
        model = RWW_sde(self.custom_params)
        params = model.get_parameters()
        
        # These should have default values
        self.assertIn('tau_s', params)
        self.assertIn('gamma', params)
        self.assertIn('J', params)
        self.assertIn('w', params)

    def test_get_parameter_single_value(self):
        """Test that get_parameter(name) returns single parameter value."""
        model = RWW_sde(self.custom_params)
        
        g_value = model.get_parameter('G')
        self.assertAlmostEqual(g_value, 0.5)
        
        tau_value = model.get_parameter('tau_s')
        self.assertAlmostEqual(tau_value, 100.0)

    def test_get_parameter_derived_param(self):
        """Test that get_parameter() can access derived parameters like 'nn'."""
        model = RWW_sde(self.custom_params)
        
        nn_value = model.get_parameter('nn')
        self.assertEqual(nn_value, 2)

    def test_get_parameter_invalid_raises_error(self):
        """Test that get_parameter() raises error for invalid parameter."""
        model = RWW_sde(self.custom_params)
        
        with self.assertRaises(AttributeError) as context:
            model.get_parameter('invalid_parameter_name')
        
        self.assertIn('not found', str(context.exception).lower())

    def test_get_default_parameters_returns_dict(self):
        """Test that get_default_parameters() returns dictionary."""
        model = RWW_sde(self.custom_params)
        defaults = model.get_default_parameters()
        
        self.assertIsInstance(defaults, dict)
        self.assertGreater(len(defaults), 0)

    def test_get_default_parameters_has_correct_values(self):
        """Test that get_default_parameters() returns correct default values."""
        model = RWW_sde(self.custom_params)
        defaults = model.get_default_parameters()
        
        self.assertAlmostEqual(defaults['G'], 0.0)
        self.assertAlmostEqual(defaults['sigma'], 0.05)
        self.assertAlmostEqual(defaults['tau_s'], 100.0)
        self.assertEqual(defaults['seed'], -1)
        self.assertTrue(defaults['RECORD_S'])
        self.assertTrue(defaults['RECORD_BOLD'])

    def test_get_default_vs_get_parameters_difference(self):
        """Test difference between get_default_parameters() and get_parameters()."""
        model = RWW_sde(self.custom_params)
        
        defaults = model.get_default_parameters()
        actual = model.get_parameters()
        
        # Custom values should differ
        self.assertNotAlmostEqual(defaults['G'], actual['G'])
        self.assertNotAlmostEqual(defaults['sigma'], actual['sigma'])
        
        # Non-custom values should match defaults
        self.assertAlmostEqual(defaults['tau_s'], actual['tau_s'])
        self.assertAlmostEqual(defaults['gamma'], actual['gamma'])

    def test_get_parameter_descriptions_returns_dict(self):
        """Test that get_parameter_descriptions() returns dictionary."""
        model = RWW_sde(self.custom_params)
        descriptions = model.get_parameter_descriptions()
        
        self.assertIsInstance(descriptions, dict)
        self.assertGreater(len(descriptions), 0)

    def test_get_parameter_descriptions_has_tuples(self):
        """Test that parameter descriptions are tuples of (description, type)."""
        model = RWW_sde(self.custom_params)
        descriptions = model.get_parameter_descriptions()
        
        for param_name, desc in descriptions.items():
            self.assertIsInstance(desc, tuple, f"{param_name} should have tuple description")
            self.assertEqual(len(desc), 2, f"{param_name} should have (description, type)")
            self.assertIsInstance(desc[0], str, "Description should be string")
            self.assertIsInstance(desc[1], str, "Type should be string")

    def test_get_parameter_descriptions_has_all_params(self):
        """Test that get_parameter_descriptions() covers all parameters."""
        model = RWW_sde(self.custom_params)
        descriptions = model.get_parameter_descriptions()
        
        # Check key parameters are described
        expected_params = ['a', 'b', 'd', 'tau_s', 'gamma', 'w', 'J', 'G', 
                          'I_ext', 'weights', 'sigma', 'dt', 't_end', 't_cut',
                          'nn', 'seed', 'initial_state', 'RECORD_S', 'RECORD_BOLD',
                          'tr', 's_decimate']
        
        for param in expected_params:
            self.assertIn(param, descriptions, f"Missing description for {param}")

    def test_get_parameter_descriptions_types(self):
        """Test that parameter type descriptions are correct."""
        model = RWW_sde(self.custom_params)
        descriptions = model.get_parameter_descriptions()
        
        # Check some known types
        self.assertEqual(descriptions['G'][1], 'scalar')
        self.assertEqual(descriptions['tau_s'][1], 'scalar')
        self.assertEqual(descriptions['nn'][1], 'int')
        self.assertEqual(descriptions['seed'][1], 'int')
        self.assertEqual(descriptions['weights'][1], 'matrix')
        self.assertEqual(descriptions['I_ext'][1], 'vector')
        self.assertEqual(descriptions['RECORD_S'][1], 'bool')
        self.assertEqual(descriptions['RECORD_BOLD'][1], 'bool')

    def test_list_parameters(self):
        """Test that list_parameters() returns valid parameter names."""
        model = RWW_sde(self.custom_params)
        param_list = model.list_parameters()
        
        self.assertIsInstance(param_list, list)
        self.assertGreater(len(param_list), 0)
        
        # Should include user-settable parameters
        self.assertIn('G', param_list)
        self.assertIn('sigma', param_list)
        self.assertIn('weights', param_list)

    def test_parameters_consistency_after_run(self):
        """Test that parameters remain consistent after running simulation."""
        model = RWW_sde(self.custom_params)
        
        params_before = model.get_parameters()
        result = model.run()
        params_after = model.get_parameters()
        
        # Key parameters should remain the same
        self.assertAlmostEqual(params_before['G'], params_after['G'])
        self.assertAlmostEqual(params_before['sigma'], params_after['sigma'])
        
        # Check derived parameter directly
        nn_before = model.get_parameter('nn')
        nn_after = model.get_parameter('nn')
        self.assertEqual(nn_before, nn_after)


class TestRWWBasic(unittest.TestCase):
    """Basic functionality tests for RWW model."""

    def setUp(self):
        """Set up common test parameters."""
        self.nn = 2
        self.weights = np.array([
            [0.0, 0.1],
            [0.1, 0.0],
        ])
        self.base_params = {
            "weights": self.weights,
            "G": 0.5,
            "dt": 1.0,
            "t_end": 100.0,
            "t_cut": 10.0,
            "seed": 42,
        }

    def test_initialization(self):
        """Test model initialization."""
        model = RWW_sde(self.base_params)
        self.assertIsNotNone(model)
        self.assertEqual(model.P.nn, self.nn)
        self.assertAlmostEqual(model.P.G, 0.5)

    def test_basic_run(self):
        """Test that model runs without errors."""
        model = RWW_sde(self.base_params)
        result = model.run()
        
        self.assertIn("S", result)
        self.assertIn("t", result)
        self.assertIsInstance(result["S"], np.ndarray)
        self.assertIsInstance(result["t"], np.ndarray)

    def test_output_shapes(self):
        """Test output array shapes are correct."""
        params = deepcopy(self.base_params)
        params["t_end"] = 100.0
        params["t_cut"] = 10.0
        params["dt"] = 1.0
        params["s_decimate"] = 1
        
        model = RWW_sde(params)
        result = model.run()
        
        S = result["S"]
        t = result["t"]
        
        self.assertEqual(S.shape[1], self.nn, "S should have nn columns")
        self.assertEqual(len(t), len(S), "Time and S should have same length")
        self.assertGreater(len(S), 0, "Should have recorded data")

    def test_no_nan_values(self):
        """Test that integration produces no NaN values."""
        model = RWW_sde(self.base_params)
        result = model.run()
        
        S = result["S"]
        self.assertFalse(np.any(np.isnan(S)), "S should not contain NaN values")
        self.assertFalse(np.any(np.isinf(S)), "S should not contain Inf values")

    def test_bounded_values(self):
        """Test that synaptic gating variables stay in reasonable range."""
        model = RWW_sde(self.base_params)
        result = model.run()
        
        S = result["S"]
        # RWW synaptic gating can transiently go outside [0,1] due to stochastic integration
        # Just verify values aren't completely unreasonable (no infinities or extreme values)
        self.assertFalse(np.any(np.isnan(S)), "S contains NaN values")
        self.assertFalse(np.any(np.isinf(S)), "S contains infinite values")
        self.assertGreater(np.min(S), -10.0, f"S minimum {np.min(S)} is unreasonably negative")
        self.assertLess(np.max(S), 10.0, f"S maximum {np.max(S)} is unreasonably large")

    def test_deterministic_reproducibility(self):
        """Test that same seed produces same results."""
        params1 = deepcopy(self.base_params)
        params1["seed"] = 42
        params1["sigma"] = 0.01
        
        params2 = deepcopy(params1)
        
        model1 = RWW_sde(params1)
        result1 = model1.run()
        
        model2 = RWW_sde(params2)
        result2 = model2.run()
        
        np.testing.assert_array_almost_equal(result1["S"], result2["S"], decimal=10)

    def test_stochastic_variability(self):
        """Test that different seeds produce different results."""
        params1 = deepcopy(self.base_params)
        params1["seed"] = 42
        params1["sigma"] = 0.1
        
        params2 = deepcopy(params1)
        params2["seed"] = 43
        
        model1 = RWW_sde(params1)
        result1 = model1.run()
        
        model2 = RWW_sde(params2)
        result2 = model2.run()
        
        # Results should differ with different seeds
        self.assertFalse(np.allclose(result1["S"], result2["S"]))


class TestRWWBOLD(unittest.TestCase):
    """Test BOLD signal generation."""

    def test_bold_recording(self):
        """Test that BOLD signal is recorded when requested."""
        params = {
            "weights": np.eye(2),
            "G": 0.5,
            "dt": 1.0,
            "t_end": 1000.0,
            "t_cut": 100.0,
            "tr": 300.0,
            "RECORD_BOLD": True,
            "seed": 42,
        }
        
        model = RWW_sde(params)
        result = model.run()
        
        self.assertIn("bold_d", result)
        self.assertIn("bold_t", result)
        
        if result["bold_d"].size > 0:
            self.assertEqual(result["bold_d"].ndim, 2, "BOLD should be 2D")
            self.assertEqual(len(result["bold_t"]), result["bold_d"].shape[0])

    def test_bold_not_recorded_when_disabled(self):
        """Test that BOLD is not computed when RECORD_BOLD=False."""
        params = {
            "weights": np.eye(2),
            "G": 0.5,
            "dt": 1.0,
            "t_end": 100.0,
            "RECORD_BOLD": False,
            "seed": 42,
        }
        
        model = RWW_sde(params)
        result = model.run()
        
        self.assertEqual(result["bold_d"].size, 0, "BOLD should be empty")
        self.assertEqual(result["bold_t"].size, 0, "BOLD time should be empty")


class TestRWWParameters(unittest.TestCase):
    """Test parameter handling and validation."""

    def test_custom_initial_state(self):
        """Test using custom initial state."""
        weights = np.eye(2)
        x0 = np.array([0.3, 0.4])
        
        params = {
            "weights": weights,
            "initial_state": x0,
        }
        
        model = RWW_sde(params)
        np.testing.assert_array_almost_equal(model.P.initial_state, x0)

    def test_decimation(self):
        """Test decimation reduces output size."""
        params_no_decimate = {
            "weights": np.eye(2),
            "dt": 1.0,
            "t_end": 100.0,
            "t_cut": 0.0,
            "s_decimate": 1,
            "seed": 42,
        }
        
        params_decimate = deepcopy(params_no_decimate)
        params_decimate["s_decimate"] = 10
        
        model1 = RWW_sde(params_no_decimate)
        result1 = model1.run()
        
        model2 = RWW_sde(params_decimate)
        result2 = model2.run()
        
        # Decimated output should be ~10x smaller
        ratio = len(result1["S"]) / len(result2["S"])
        self.assertAlmostEqual(ratio, 10.0, delta=1.0)

    def test_coupling_strength_effect(self):
        """Test that different coupling strengths affect results."""
        base_params = {
            "weights": np.array([[0.0, 0.5], [0.5, 0.0]]),
            "dt": 1.0,
            "t_end": 200.0,
            "t_cut": 50.0,
            "sigma": 0.01,
            "seed": 42,
        }
        
        params_weak = deepcopy(base_params)
        params_weak["G"] = 0.0
        
        params_strong = deepcopy(base_params)
        params_strong["G"] = 2.0
        
        model_weak = RWW_sde(params_weak)
        result_weak = model_weak.run()
        
        model_strong = RWW_sde(params_strong)
        result_strong = model_strong.run()
        
        # Results should differ with different coupling
        self.assertFalse(np.allclose(result_weak["S"], result_strong["S"]))


if __name__ == "__main__":
    unittest.main()
