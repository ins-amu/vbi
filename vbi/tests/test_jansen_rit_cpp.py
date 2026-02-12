"""Tests for C++ Jansen-Rit model implementations.

This module tests the Python wrappers for C++ compiled Jansen-Rit models.
Tests are automatically skipped if C++ modules are not compiled.
"""

import unittest
import numpy as np
import networkx as nx
import pytest
import os
from copy import deepcopy

# Check if C++ modules are available
CPP_JR_AVAILABLE = True
CPP_IMPORT_ERROR = None
try:
    from vbi.models.cpp.jansen_rit import JR_sde, JR_sdde
except ImportError as e:
    CPP_JR_AVAILABLE = False
    CPP_IMPORT_ERROR = str(e)

# Skip decorator for when C++ is not available
skip_if_cpp_unavailable = pytest.mark.skipif(
    not CPP_JR_AVAILABLE,
    reason=f"C++ modules not compiled or not available: {CPP_IMPORT_ERROR}"
)


class TestJRSDE(unittest.TestCase):
    """Test cases for JR_sde C++ implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.nn = 4
        self.weights = nx.to_numpy_array(nx.complete_graph(self.nn))
        
        self.base_params = {
            "weights": self.weights,
            "G": 0.5,
            "A": 3.25,
            "B": 22.0,
            "a": 0.1,
            "b": 0.05,
            "dt": 0.05,
            "t_end": 100.0,
            "t_transition": 50.0,
            "seed": 42,
        }
    
    @skip_if_cpp_unavailable
    def test_initialization(self):
        """Test that JR_sde can be initialized with default parameters."""
        model = JR_sde(self.base_params)
        
        self.assertEqual(model.N, self.nn)
        self.assertEqual(model.num_nodes, self.nn)
        self.assertEqual(model.G, 0.5)
        self.assertEqual(model.A, 3.25)
        self.assertIsNotNone(model.weights)
    
    @skip_if_cpp_unavailable
    def test_valid_params_auto_generation(self):
        """Test that valid_params are automatically generated from get_default_parameters."""
        model = JR_sde(self.base_params)
        
        # Check that valid_params is a list
        self.assertIsInstance(model.valid_params, list)
        
        # Check that it contains expected parameters
        expected_params = ['A', 'B', 'G', 'weights', 'dt', 't_end', 'seed']
        for param in expected_params:
            self.assertIn(param, model.valid_params)
        
        # Check that the number of valid params matches defaults
        defaults = model.get_default_parameters()
        self.assertEqual(len(model.valid_params), len(defaults))
    
    @skip_if_cpp_unavailable
    def test_invalid_parameter_rejection(self):
        """Test that invalid parameters are properly rejected."""
        params = self.base_params.copy()
        params['invalid_param'] = 123
        
        with self.assertRaises(ValueError) as context:
            JR_sde(params)
        
        self.assertIn('Invalid parameter', str(context.exception))
    
    @skip_if_cpp_unavailable
    def test_get_default_parameters(self):
        """Test that get_default_parameters returns a complete dict."""
        model = JR_sde(self.base_params)
        defaults = model.get_default_parameters()
        
        # Check it's a dictionary
        self.assertIsInstance(defaults, dict)
        
        # Check key parameters are present
        key_params = ['G', 'A', 'B', 'a', 'b', 'dt', 't_end', 'weights', 'seed']
        for param in key_params:
            self.assertIn(param, defaults)
    
    @skip_if_cpp_unavailable
    def test_parameter_update(self):
        """Test that parameters are properly updated in __init__."""
        params = self.base_params.copy()
        params['G'] = 1.5
        params['A'] = 5.0
        
        model = JR_sde(params)
        
        self.assertEqual(model.G, 1.5)
        self.assertEqual(model.A, 5.0)
    
    @skip_if_cpp_unavailable
    def test_set_initial_state(self):
        """Test that initial state can be set."""
        model = JR_sde(self.base_params)
        
        # Initially should not be set
        self.assertFalse(model.INITIAL_STATE_SET)
        
        # Set initial state
        model.set_initial_state()
        
        # Should now be set
        self.assertTrue(model.INITIAL_STATE_SET)
        self.assertIsNotNone(model.initial_state)
        
        # Check shape (flattened: 6 state variables per node)
        expected_length = self.nn * 6
        self.assertEqual(model.initial_state.shape, (expected_length,))
    
    @skip_if_cpp_unavailable
    @pytest.mark.slow
    def test_run_simulation(self):
        """Test that simulation runs and produces expected output."""
        params = self.base_params.copy()
        params['t_end'] = 100.0
        params['t_transition'] = 20.0
        
        model = JR_sde(params)
        result = model.run()
        
        # Check that result contains expected keys
        self.assertIn('t', result)
        self.assertIn('x', result)
        
        # Check shapes
        times = result['t']
        states = result['x']
        
        self.assertIsInstance(times, np.ndarray)
        self.assertIsInstance(states, np.ndarray)
        
        # States should have shape (n_nodes, n_timesteps)
        self.assertEqual(states.shape[0], self.nn)
        self.assertEqual(states.shape[1], len(times))
        
        # Check that time is monotonically increasing
        self.assertTrue(np.all(np.diff(times) > 0))
    
    @skip_if_cpp_unavailable
    @pytest.mark.slow
    def test_run_with_parameter_override(self):
        """Test running simulation with parameter override (like in the notebook)."""
        params = self.base_params.copy()
        params['t_end'] = 100.0
        params['t_transition'] = 20.0
        
        model = JR_sde(params)
        
        # Run with parameter override (as done in the notebook example)
        # Override G and C1 values
        C1_override = np.ones(self.nn) * 110.0
        C1_override[0] = 135.0
        theta_dict = {"G": 2.0, "C1": C1_override}
        
        result = model.run(theta_dict)
        
        # Check that result contains expected keys
        self.assertIn('t', result)
        self.assertIn('x', result)
        
        # Check that the model parameters were updated
        self.assertEqual(model.G, 2.0)
        np.testing.assert_array_equal(model.C1, C1_override)
    
    @skip_if_cpp_unavailable
    def test_output_directory_creation(self):
        """Test that output directory is created."""
        params = self.base_params.copy()
        params['output'] = 'test_output_jr_sde'
        
        model = JR_sde(params)
        
        # Check that directory was created
        self.assertTrue(os.path.isdir(params['output']))
        
        # Clean up
        os.rmdir(params['output'])


class TestJRSDDE(unittest.TestCase):
    """Test cases for JR_sdde C++ implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.nn = 3
        self.weights = nx.to_numpy_array(nx.complete_graph(self.nn))
        self.delays = np.ones((self.nn, self.nn)) * 10.0  # 10 ms delays
        
        self.base_params = {
            "weights": self.weights,
            "delays": self.delays,
            "G": 0.01,
            "A": 3.25,
            "B": 22.0,
            "a": 0.1,
            "b": 0.05,
            "dt": 0.01,
            "t_end": 100.0,
            "t_transition": 50.0,
            "seed": 42,
        }
    
    @skip_if_cpp_unavailable
    def test_initialization(self):
        """Test that JR_sdde can be initialized with required parameters."""
        model = JR_sdde(self.base_params)
        
        self.assertEqual(model.N, self.nn)
        self.assertEqual(model.num_nodes, self.nn)
        self.assertIsNotNone(model.weights)
        self.assertIsNotNone(model.delays)
    
    @skip_if_cpp_unavailable
    def test_valid_params_auto_generation(self):
        """Test that valid_params are automatically generated from get_default_parameters."""
        model = JR_sdde(self.base_params)
        
        # Check that valid_params is a list
        self.assertIsInstance(model.valid_params, list)
        
        # Check that it contains expected parameters including weights and delays
        expected_params = ['weights', 'delays', 'A', 'B', 'G', 'dt', 't_end', 'seed']
        for param in expected_params:
            self.assertIn(param, model.valid_params)
        
        # Check that the number of valid params matches defaults
        defaults = model.get_default_parameters()
        self.assertEqual(len(model.valid_params), len(defaults))
    
    @skip_if_cpp_unavailable
    def test_required_parameters(self):
        """Test that weights and delays are required."""
        # Test missing weights
        params = self.base_params.copy()
        params.pop('weights')
        
        with self.assertRaises(AssertionError):
            JR_sdde(params)
        
        # Test missing delays
        params = self.base_params.copy()
        params.pop('delays')
        
        with self.assertRaises(AssertionError):
            JR_sdde(params)
    
    @skip_if_cpp_unavailable
    def test_invalid_parameter_rejection(self):
        """Test that invalid parameters are properly rejected."""
        params = self.base_params.copy()
        params['invalid_param'] = 456
        
        with self.assertRaises(ValueError) as context:
            JR_sdde(params)
        
        self.assertIn('Invalid parameter', str(context.exception))
    
    @skip_if_cpp_unavailable
    def test_get_default_parameters(self):
        """Test that get_default_parameters returns a complete dict."""
        model = JR_sdde(self.base_params)
        defaults = model.get_default_parameters()
        
        # Check it's a dictionary
        self.assertIsInstance(defaults, dict)
        
        # Check key parameters are present (including newly added ones)
        key_params = ['weights', 'delays', 'G', 'A', 'B', 'a', 'b', 'dt', 't_end', 'seed']
        for param in key_params:
            self.assertIn(param, defaults)
    
    @skip_if_cpp_unavailable
    def test_parameter_update(self):
        """Test that parameters are properly updated in __init__."""
        params = self.base_params.copy()
        params['G'] = 0.05
        params['mu'] = 0.5
        
        model = JR_sdde(params)
        
        self.assertEqual(model.G, 0.05)
        self.assertEqual(model.mu, 0.5)
    
    @skip_if_cpp_unavailable
    def test_stimulus_parameters(self):
        """Test that stimulus parameters are properly handled."""
        params = self.base_params.copy()
        params['sti_amplitude'] = 1.0
        params['sti_ti'] = 60.0
        params['sti_duration'] = 10.0
        
        model = JR_sdde(params)
        
        self.assertEqual(model.sti_ti, 60.0)
        self.assertEqual(model.sti_duration, 10.0)
        # sti_amplitude should be expanded to array
        self.assertEqual(len(model.sti_amplitude), self.nn)
    
    @skip_if_cpp_unavailable
    def test_set_initial_state(self):
        """Test that initial state can be set."""
        model = JR_sdde(self.base_params)
        
        # Initially should not be set
        self.assertFalse(model.INITIAL_STATE_SET)
        
        # Set initial state
        model.set_initial_state()
        
        # Should now be set
        self.assertTrue(model.INITIAL_STATE_SET)
        self.assertIsNotNone(model.initial_state)
    
    @skip_if_cpp_unavailable
    @pytest.mark.slow
    def test_run_simulation(self):
        """Test that simulation with delays runs and produces expected output."""
        params = self.base_params.copy()
        params['t_end'] = 100.0
        params['t_transition'] = 30.0
        
        model = JR_sdde(params)
        result = model.run()
        
        # Check that result contains expected keys
        self.assertIn('t', result)
        self.assertIn('x', result)
        
        # Check shapes
        times = result['t']
        states = result['x']
        
        self.assertIsInstance(times, np.ndarray)
        self.assertIsInstance(states, np.ndarray)
        
        # Check that time is monotonically increasing
        self.assertTrue(np.all(np.diff(times) > 0))


class TestJRModelsConsistency(unittest.TestCase):
    """Test consistency between JR_sde and JR_sdde models."""
    
    @skip_if_cpp_unavailable
    def test_default_parameters_structure(self):
        """Test that both models have consistent default parameter structure."""
        nn = 2
        weights = np.ones((nn, nn))
        delays = np.ones((nn, nn))
        
        model_sde = JR_sde({"weights": weights})
        model_sdde = JR_sdde({"weights": weights, "delays": delays})
        
        defaults_sde = model_sde.get_default_parameters()
        defaults_sdde = model_sdde.get_default_parameters()
        
        # Check that common parameters exist in both
        common_params = ['A', 'B', 'a', 'b', 'dt', 'G', 'seed', 'method', 't_end', 't_transition']
        for param in common_params:
            self.assertIn(param, defaults_sde)
            self.assertIn(param, defaults_sdde)
    
    @skip_if_cpp_unavailable
    def test_valid_params_are_keys(self):
        """Test that valid_params match the keys from get_default_parameters."""
        nn = 2
        weights = np.ones((nn, nn))
        delays = np.ones((nn, nn))
        
        model_sde = JR_sde({"weights": weights})
        model_sdde = JR_sdde({"weights": weights, "delays": delays})
        
        # For JR_sde
        defaults_sde = model_sde.get_default_parameters()
        self.assertEqual(set(model_sde.valid_params), set(defaults_sde.keys()))
        
        # For JR_sdde
        defaults_sdde = model_sdde.get_default_parameters()
        self.assertEqual(set(model_sdde.valid_params), set(defaults_sdde.keys()))


if __name__ == '__main__':
    unittest.main()
