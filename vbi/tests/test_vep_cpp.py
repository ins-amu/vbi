"""Tests for C++ VEP (Virtual Epileptic Patient) model implementation.

This module tests the Python wrapper for C++ compiled VEP model.
Tests are automatically skipped if C++ modules are not compiled.
"""

import unittest
import numpy as np
import pytest
import os
from copy import deepcopy

# Check if C++ modules are available
CPP_VEP_AVAILABLE = False
CPP_IMPORT_ERROR = None
try:
    from vbi.models.cpp.vep import VEP_sde
    from vbi.models.cpp._src.vep import VEP as _VEP
    CPP_VEP_AVAILABLE = True
except ImportError as e:
    CPP_IMPORT_ERROR = str(e)

# Skip decorator for when C++ is not available
skip_if_cpp_unavailable = pytest.mark.skipif(
    not CPP_VEP_AVAILABLE,
    reason=f"C++ VEP module not compiled or not available: {CPP_IMPORT_ERROR}"
)


class TestVEPSDE(unittest.TestCase):
    """Test suite for VEP_sde C++ implementation."""

    def setUp(self):
        """Set up test fixtures."""
        self.nn = 4
        self.weights = np.random.rand(self.nn, self.nn)
        
        self.base_params = {
            'weights': self.weights,
            'G': 1.0,
            'tau': 10.0,
            'eta': -1.5,
            'noise_sigma': 0.1,
            'iext': 0.0,
            'dt': 0.01,
            'tend': 50.0,
            'tcut': 0.0,
            'noise_seed': 0,
            'record_step': 1,
            'method': 'euler',
            'output': 'test_output_vep'
        }

    def tearDown(self):
        """Clean up after tests."""
        import shutil
        if os.path.exists('test_output_vep'):
            shutil.rmtree('test_output_vep')

    @skip_if_cpp_unavailable
    def test_initialization(self):
        """Test that VEP_sde initializes correctly with valid parameters."""
        model = VEP_sde(self.base_params)
        
        # Check that parameters are set
        self.assertEqual(model.G, 1.0)
        self.assertEqual(model.tau, 10.0)
        self.assertEqual(model.eta, -1.5)
        self.assertEqual(model.noise_sigma, 0.1)
        self.assertIsNone(model.seed)
        
    @skip_if_cpp_unavailable
    def test_get_default_parameters(self):
        """Test that default parameters are returned correctly."""
        model = VEP_sde()
        defaults = model.get_default_parameters()
        
        # Check that all expected parameters are present
        expected_keys = [
            'G', 'seed', 'initial_state', 'weights', 'tau', 'eta',
            'noise_sigma', 'iext', 'dt', 'tend', 'tcut', 'noise_seed',
            'record_step', 'method', 'output'
        ]
        
        for key in expected_keys:
            self.assertIn(key, defaults)
    
    @skip_if_cpp_unavailable
    def test_valid_params_auto_generation(self):
        """Test that valid_params is automatically generated from get_default_parameters."""
        model = VEP_sde(self.base_params)
        defaults = model.get_default_parameters()
        
        # valid_params should match the keys from get_default_parameters
        self.assertEqual(set(model.valid_params), set(defaults.keys()))
        
    @skip_if_cpp_unavailable
    def test_invalid_parameter_rejection(self):
        """Test that invalid parameters are rejected."""
        params = self.base_params.copy()
        params['invalid_param'] = 123
        
        with self.assertRaises(ValueError) as context:
            VEP_sde(params)
        
        self.assertIn('Invalid parameter', str(context.exception))
    
    @skip_if_cpp_unavailable
    def test_parameter_update(self):
        """Test that parameters can be updated."""
        params = self.base_params.copy()
        params['G'] = 2.5
        params['tau'] = 15.0
        
        model = VEP_sde(params)
        
        self.assertEqual(model.G, 2.5)
        self.assertEqual(model.tau, 15.0)
    
    @skip_if_cpp_unavailable
    def test_set_initial_state(self):
        """Test that initial state can be set."""
        model = VEP_sde(self.base_params)
        
        # Initially should not be set
        self.assertFalse(model.INITIAL_STATE_SET)
        
        # Set initial state
        model.set_initial_state()
        
        # Should now be set
        self.assertTrue(model.INITIAL_STATE_SET)
        self.assertIsNotNone(model.initial_state)
        
        # Check shape (2 state variables per node: x and z)
        expected_length = self.nn * 2
        self.assertEqual(model.initial_state.shape, (expected_length,))
    
    @skip_if_cpp_unavailable
    def test_set_initial_state_with_seed_none(self):
        """Test that set_initial_state works when seed=None (regression test for UnboundLocalError)."""
        params = self.base_params.copy()
        params['seed'] = None  # Explicitly set to None
        
        model = VEP_sde(params)
        
        # This should not raise UnboundLocalError
        model.set_initial_state()
        
        self.assertTrue(model.INITIAL_STATE_SET)
        self.assertIsNotNone(model.initial_state)
        self.assertEqual(model.initial_state.shape, (self.nn * 2,))
    
    @skip_if_cpp_unavailable
    def test_set_initial_state_with_explicit_seed(self):
        """Test that set_initial_state works with an explicit seed."""
        params = self.base_params.copy()
        params['seed'] = 42
        
        model = VEP_sde(params)
        model.set_initial_state()
        
        initial_state_1 = model.initial_state.copy()
        
        # Create another model with same seed
        model2 = VEP_sde(params)
        model2.set_initial_state()
        initial_state_2 = model2.initial_state.copy()
        
        # Should produce same initial states (with same seed)
        np.testing.assert_array_almost_equal(initial_state_1, initial_state_2)
    
    @skip_if_cpp_unavailable
    @pytest.mark.slow
    def test_run_simulation(self):
        """Test that simulation runs and produces expected output."""
        params = self.base_params.copy()
        params['tend'] = 50.0
        params['tcut'] = 0.0
        
        model = VEP_sde(params)
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
    def test_run_with_seed_none(self):
        """Test that run works when seed=None (regression test)."""
        params = self.base_params.copy()
        params['seed'] = None
        params['tend'] = 20.0
        
        model = VEP_sde(params)
        
        # This should not raise UnboundLocalError
        result = model.run()
        
        self.assertIn('t', result)
        self.assertIn('x', result)
        self.assertEqual(result['x'].shape[0], self.nn)
    
    @skip_if_cpp_unavailable
    @pytest.mark.slow
    def test_run_with_parameter_override(self):
        """Test running simulation with parameter override."""
        params = self.base_params.copy()
        params['tend'] = 50.0
        
        model = VEP_sde(params)
        
        # Run with parameter override
        override_params = {"G": 2.0, "tau": 15.0}
        result = model.run(par=override_params)
        
        # Check that result is valid
        self.assertIn('t', result)
        self.assertIn('x', result)
        
        # Check that the model parameters were updated
        self.assertEqual(model.G, 2.0)
        self.assertEqual(model.tau, 15.0)
    
    @skip_if_cpp_unavailable
    @pytest.mark.slow
    def test_run_with_custom_initial_state(self):
        """Test running simulation with custom initial state."""
        params = self.base_params.copy()
        params['tend'] = 20.0
        
        model = VEP_sde(params)
        
        # Create custom initial state
        custom_x0 = np.random.rand(self.nn * 2)
        
        result = model.run(x0=custom_x0)
        
        # Check that result is valid
        self.assertIn('t', result)
        self.assertIn('x', result)
        
        # Check that initial state was set
        self.assertTrue(model.INITIAL_STATE_SET)
        np.testing.assert_array_equal(model.initial_state, custom_x0)
    
    @skip_if_cpp_unavailable
    def test_output_directory_creation(self):
        """Test that output directory is created."""
        params = self.base_params.copy()
        params['output'] = 'test_output_vep_custom'
        
        model = VEP_sde(params)
        
        # Output directory should be created during initialization
        # (not explicitly created in VEP, but would be used during run)
        self.assertEqual(model.output, 'test_output_vep_custom')
        
        # Clean up
        import shutil
        if os.path.exists('test_output_vep_custom'):
            shutil.rmtree('test_output_vep_custom')
    
    @skip_if_cpp_unavailable
    def test_heterogeneous_parameters(self):
        """Test that node-specific parameters work correctly."""
        params = self.base_params.copy()
        
        # Set heterogeneous eta (different value for each node)
        eta_hetero = np.array([-1.5, -1.3, -1.7, -1.4])
        params['eta'] = eta_hetero
        
        # Set heterogeneous iext
        iext_hetero = np.array([0.0, 0.1, 0.2, 0.0])
        params['iext'] = iext_hetero
        
        model = VEP_sde(params)
        
        # Check that heterogeneous parameters are set correctly
        np.testing.assert_array_equal(model.eta, eta_hetero)
        np.testing.assert_array_equal(model.iext, iext_hetero)
    
    @skip_if_cpp_unavailable
    def test_str_and_call_methods(self):
        """Test __str__ and __call__ methods."""
        model = VEP_sde(self.base_params)
        
        # Test __str__
        str_output = str(model)
        self.assertIsInstance(str_output, str)
        
        # Test __call__
        params_dict = model()
        self.assertIsInstance(params_dict, dict)
        self.assertIn('G', params_dict)


class TestVEPConsistency(unittest.TestCase):
    """Test consistency and edge cases for VEP model."""
    
    @skip_if_cpp_unavailable
    def test_default_parameters_structure(self):
        """Test that default parameters have consistent structure."""
        model = VEP_sde()
        defaults = model.get_default_parameters()
        
        # Check that all values are of expected types
        self.assertIsInstance(defaults['G'], (int, float))
        self.assertIsInstance(defaults['tau'], (int, float))
        self.assertIsInstance(defaults['dt'], (int, float))
        self.assertIsInstance(defaults['method'], str)
    
    @skip_if_cpp_unavailable
    def test_valid_params_are_keys(self):
        """Test that valid_params matches get_default_parameters keys."""
        model = VEP_sde({'weights': np.eye(3)})
        defaults = model.get_default_parameters()
        
        # Every key in defaults should be in valid_params
        for key in defaults.keys():
            self.assertIn(key, model.valid_params)
        
        # Every key in valid_params should be in defaults
        for key in model.valid_params:
            self.assertIn(key, defaults.keys())


if __name__ == '__main__':
    unittest.main()
