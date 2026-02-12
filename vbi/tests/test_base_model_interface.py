"""Tests for the unified BaseModel interface across C++ models.

This module tests that all C++ models properly implement the BaseModel interface
and provide consistent methods for parameter management.
"""

import unittest
import numpy as np
import pytest
import os
import sys
import importlib
import inspect
from pathlib import Path

# Add the vbi module to path if needed
vbi_path = Path(__file__).parent.parent
if str(vbi_path) not in sys.path:
    sys.path.insert(0, str(vbi_path))

from vbi.models.cpp.base import BaseModel


def discover_model_classes():
    """
    Automatically discover all model classes in vbi/models/cpp/.
    
    Returns
    -------
    dict
        Dictionary mapping model class names to their import paths and availability.
    """
    models_dir = Path(__file__).parent.parent / 'models' / 'cpp'
    discovered_models = {}
    
    # Get all Python files in the cpp models directory
    python_files = [f for f in models_dir.glob('*.py') 
                   if f.name not in ['__init__.py', 'base.py', '__pycache__']]
    
    for py_file in python_files:
        module_name = py_file.stem
        module_path = f'vbi.models.cpp.{module_name}'
        
        try:
            module = importlib.import_module(module_path)
            
            # Find all classes in the module that inherit from BaseModel
            for name, obj in inspect.getmembers(module, inspect.isclass):
                # Skip BaseModel itself and non-BaseModel classes
                if obj is BaseModel:
                    continue
                    
                # Check if it's defined in this module (not imported)
                if obj.__module__ != module_path:
                    continue
                
                # Check if it inherits from BaseModel or is a model-like class
                # (has common model methods even if not inheriting from BaseModel)
                is_base_model = issubclass(obj, BaseModel)
                has_model_methods = (
                    hasattr(obj, 'get_default_parameters') and
                    hasattr(obj, 'run') and
                    hasattr(obj, 'check_parameters')
                )
                
                if is_base_model or has_model_methods:
                    discovered_models[name] = {
                        'module_path': module_path,
                        'class': obj,
                        'available': True,
                        'inherits_base': is_base_model,
                        'error': None
                    }
        except ImportError as e:
            # Module not available (C++ not compiled, etc.)
            print(f"Could not import {module_path}: {e}")
            
            # Try to get the class name from the module name
            # Most modules have a class with the same name as the module
            class_name = module_name.upper()  # e.g., 'vep' -> 'VEP'
            if module_name == 'vep':
                class_name = 'VEP_sde'
            elif module_name == 'jansen_rit':
                class_name = 'JR_sde'
            elif module_name == 'mpr':
                class_name = 'MPR_sde'
            elif module_name == 'wc':
                class_name = 'WC_ode'
            elif module_name == 'km':
                class_name = 'KM_sde'
            elif module_name == 'damp_oscillator':
                class_name = 'DO'
            
            # Add the model as unavailable
            discovered_models[class_name] = {
                'module_path': module_path,
                'class': None,
                'available': False,
                'inherits_base': False,  # We don't know, but assume not since import failed
                'error': str(e)
            }
    
    return discovered_models


# Discover all models
DISCOVERED_MODELS = discover_model_classes()

# Create CPP_AVAILABLE and CPP_ERRORS dictionaries from discovered models
CPP_AVAILABLE = {}
CPP_ERRORS = {}

for model_name, info in DISCOVERED_MODELS.items():
    CPP_AVAILABLE[model_name] = info['available']
    CPP_ERRORS[model_name] = str(info.get('error', 'Unknown error')) if not info['available'] else None

print(f"\n{'='*70}")
print(f"Discovered {len(DISCOVERED_MODELS)} model classes:")
for name, info in DISCOVERED_MODELS.items():
    inheritance = "✓ BaseModel" if info['inherits_base'] else "✗ No BaseModel"
    availability = "✓ Available" if info['available'] else "✗ Not Available"
    print(f"  - {name:20s} from {info['module_path']:40s} [{inheritance}] [{availability}]")
print(f"{'='*70}\n")


class TestBaseModelInheritance(unittest.TestCase):
    """Test that all discovered models inherit from BaseModel."""
    
    def test_all_models_inherit_from_base(self):
        """Verify all model classes inherit from BaseModel."""
        non_inheriting = []
        
        for model_name, info in DISCOVERED_MODELS.items():
            if not info['inherits_base']:
                non_inheriting.append(model_name)
        
        if non_inheriting:
            error_msg = (
                f"The following models do not inherit from BaseModel:\n"
                f"  {', '.join(non_inheriting)}\n"
                f"Please update them to inherit from BaseModel."
            )
            self.fail(error_msg)
    
    def test_all_models_have_required_methods(self):
        """Verify all models have required BaseModel methods."""
        required_methods = [
            'get_parameters',
            'set_parameters',
            'get_parameter',
            'list_parameters',
            'check_parameters',
            'get_default_parameters',
            'run'
        ]
        
        missing_methods = {}
        
        for model_name, info in DISCOVERED_MODELS.items():
            model_class = info['class']
            missing = []
            
            for method in required_methods:
                if not hasattr(model_class, method):
                    missing.append(method)
            
            if missing:
                missing_methods[model_name] = missing
        
        if missing_methods:
            error_msg = "The following models are missing required methods:\n"
            for model, methods in missing_methods.items():
                error_msg += f"  {model}: {', '.join(methods)}\n"
            self.fail(error_msg)


class TestModelSpecificFunctionality(unittest.TestCase):
    """Test that each model's interface methods work correctly."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.nn = 2
        self.weights = np.array([[0, 1], [1, 0]], dtype=np.float32)
    
    def _get_model_specific_params(self, model_name):
        """Get model-specific initialization parameters."""
        base_params = {'weights': self.weights}
        
        # Model-specific requirements
        if model_name == 'KM_sde':
            base_params['omega'] = np.array([1.0, 1.5])
        elif model_name == 'JR_sdde':
            base_params['delays'] = np.ones((self.nn, self.nn)) * 10.0
        elif model_name == 'DO':
            # Damped oscillator doesn't need weights
            base_params = {'a': 1.0, 'b': 2.0}
        
        return base_params
    
    def _test_model_interface(self, model_name, model_info):
        """Test a specific model's interface functionality."""
        if not model_info['available']:
            pytest.skip(f"Model {model_name} not available")
        
        model_class = model_info['class']
        init_params = self._get_model_specific_params(model_name)
        
        try:
            # Test initialization
            model = model_class(par=init_params)
            
            # Test get_parameters()
            params = model.get_parameters()
            self.assertIsInstance(params, dict, 
                                 f"{model_name}.get_parameters() should return dict")
            self.assertTrue(len(params) > 0,
                          f"{model_name}.get_parameters() returned empty dict")
            
            # Test list_parameters()
            param_list = model.list_parameters()
            self.assertIsInstance(param_list, list,
                                f"{model_name}.list_parameters() should return list")
            self.assertTrue(len(param_list) > 0,
                          f"{model_name}.list_parameters() returned empty list")
            
            # Test get_parameter() with a known parameter
            # Most models have 'G' parameter
            test_params = ['G', 'dt', 'seed']
            found_param = None
            for param_name in test_params:
                if param_name in param_list:
                    found_param = param_name
                    break
            
            if found_param:
                value = model.get_parameter(found_param)
                self.assertIsNotNone(value,
                                   f"{model_name}.get_parameter('{found_param}') returned None")
            
            # Test set_parameters()
            if found_param:
                original_value = model.get_parameter(found_param)
                if isinstance(original_value, (int, float)):
                    new_value = original_value + 1.0 if original_value != -1 else 0.0
                    model.set_parameters({found_param: new_value})
                    updated_value = model.get_parameter(found_param)
                    self.assertEqual(updated_value, new_value,
                                   f"{model_name}.set_parameters() did not update parameter")
            
            # Test backward compatibility with _par
            self.assertTrue(hasattr(model, '_par'),
                          f"{model_name} should have _par attribute")
            self.assertIsInstance(model._par, dict,
                                f"{model_name}._par should be a dict")
            
            # Test __call__ method
            call_result = model()
            self.assertIsInstance(call_result, dict,
                                f"{model_name}() should return dict")
            
            # Test invalid parameter raises error
            with self.assertRaises(ValueError,
                                 msg=f"{model_name} should raise ValueError for invalid parameter"):
                model.set_parameters({'completely_invalid_param_xyz': 123})
            
            # Test get_parameter with invalid name raises error
            with self.assertRaises(KeyError,
                                 msg=f"{model_name} should raise KeyError for invalid parameter"):
                model.get_parameter('completely_invalid_param_xyz')
            
            return True
            
        except Exception as e:
            self.fail(f"{model_name} interface test failed: {str(e)}")
    
    def test_generate_tests_for_all_models(self):
        """Dynamically generate and run tests for all discovered models."""
        failed_models = []
        
        for model_name, model_info in DISCOVERED_MODELS.items():
            if not model_info['inherits_base']:
                # Skip models that don't inherit from BaseModel
                # They are already flagged in test_all_models_inherit_from_base
                continue
            
            try:
                print(f"\nTesting {model_name}...")
                self._test_model_interface(model_name, model_info)
                print(f"✓ {model_name} passed")
            except Exception as e:
                print(f"✗ {model_name} failed: {e}")
                failed_models.append((model_name, str(e)))
        
        if failed_models:
            error_msg = "The following models failed interface tests:\n"
            for model, error in failed_models:
                error_msg += f"  {model}: {error}\n"
            self.fail(error_msg)


class TestBaseModelInterface(unittest.TestCase):
    """Test suite for unified BaseModel interface."""

    def setUp(self):
        """Set up test fixtures."""
        self.nn = 2
        self.weights = np.array([[0, 1], [1, 0]], dtype=np.float32)

    @pytest.mark.skipif(
        not CPP_AVAILABLE.get('VEP_sde', False),
        reason=f"C++ VEP module not available: {CPP_ERRORS.get('VEP_sde', 'Unknown')}"
    )
    def test_vep_unified_interface(self):
        """Test VEP_sde implements BaseModel interface."""
        from vbi.models.cpp.vep import VEP_sde
        
        model = VEP_sde(par={'weights': self.weights, 'G': 2.0, 'tau': 15.0})
        
        # Test get_parameters() method
        params = model.get_parameters()
        self.assertIsInstance(params, dict)
        self.assertIn('G', params)
        self.assertEqual(params['G'], 2.0)
        self.assertEqual(params['tau'], 15.0)
        
        # Test get_parameter() method
        g_value = model.get_parameter('G')
        self.assertEqual(g_value, 2.0)
        
        # Test set_parameters() method
        model.set_parameters({'G': 3.0})
        self.assertEqual(model.get_parameter('G'), 3.0)
        
        # Test list_parameters() method
        param_list = model.list_parameters()
        self.assertIsInstance(param_list, list)
        self.assertIn('G', param_list)
        self.assertIn('tau', param_list)
        
        # Test backward compatibility with _par
        self.assertEqual(model._par['G'], 3.0)
        
        # Test __call__ method
        params_call = model()
        self.assertIsInstance(params_call, dict)
        self.assertEqual(params_call['G'], 3.0)
        
        # Test invalid parameter raises error
        with self.assertRaises(KeyError):
            model.get_parameter('invalid_param')

    @pytest.mark.skipif(
        not CPP_AVAILABLE.get('JR_sde', False),
        reason=f"C++ JR module not available: {CPP_ERRORS.get('JR_sde', 'Unknown')}"
    )
    def test_jr_unified_interface(self):
        """Test JR_sde implements BaseModel interface."""
        from vbi.models.cpp.jansen_rit import JR_sde
        
        model = JR_sde(par={'weights': self.weights, 'G': 0.5})
        
        # Test get_parameters() method
        params = model.get_parameters()
        self.assertIsInstance(params, dict)
        self.assertIn('G', params)
        self.assertEqual(params['G'], 0.5)
        self.assertIn('A', params)
        
        # Test get_parameter() method
        a_value = model.get_parameter('A')
        self.assertEqual(a_value, 3.25)  # default value
        
        # Test set_parameters() method
        model.set_parameters({'G': 1.0, 'A': 4.0})
        self.assertEqual(model.get_parameter('G'), 1.0)
        self.assertEqual(model.get_parameter('A'), 4.0)
        
        # Test backward compatibility
        self.assertEqual(model._par['G'], 1.0)

    @pytest.mark.skipif(
        not CPP_AVAILABLE.get('MPR_sde', False),
        reason=f"C++ MPR module not available: {CPP_ERRORS.get('MPR_sde', 'Unknown')}"
    )
    def test_mpr_unified_interface(self):
        """Test MPR_sde implements BaseModel interface."""
        from vbi.models.cpp.mpr import MPR_sde
        
        model = MPR_sde(par={'weights': self.weights, 'G': 0.733})
        
        # Test get_parameters() method
        params = model.get_parameters()
        self.assertIsInstance(params, dict)
        self.assertIn('G', params)
        self.assertEqual(params['G'], 0.733)
        self.assertIn('J', params)
        
        # Test set_parameters() method
        model.set_parameters({'G': 1.0})
        self.assertEqual(model.get_parameter('G'), 1.0)
        
        # Test list_parameters() method
        param_list = model.list_parameters()
        self.assertIn('G', param_list)
        self.assertIn('J', param_list)
        self.assertIn('eta', param_list)

    @pytest.mark.skipif(
        not CPP_AVAILABLE.get('WC_ode', False),
        reason=f"C++ WC module not available: {CPP_ERRORS.get('WC_ode', 'Unknown')}"
    )
    def test_wc_unified_interface(self):
        """Test WC_ode implements BaseModel interface."""
        from vbi.models.cpp.wc import WC_ode
        
        model = WC_ode(par={'weights': self.weights})
        
        # Test get_parameters() method
        params = model.get_parameters()
        self.assertIsInstance(params, dict)
        self.assertIn('c_ee', params)
        self.assertIn('tau_e', params)
        
        # Test get_parameter() method
        c_ee = model.get_parameter('c_ee')
        self.assertEqual(c_ee, 16.0)  # default value
        
        # Test list_parameters() method
        param_list = model.list_parameters()
        self.assertIsInstance(param_list, list)
        self.assertTrue(len(param_list) > 0)

    @pytest.mark.skipif(
        not CPP_AVAILABLE.get('KM_sde', False),
        reason=f"C++ KM module not available: {CPP_ERRORS.get('KM_sde', 'Unknown')}"
    )
    def test_km_unified_interface(self):
        """Test KM_sde implements BaseModel interface."""
        from vbi.models.cpp.km import KM_sde
        
        omega = np.array([1.0, 1.5])
        model = KM_sde(par={'omega': omega, 'G': 1.0})
        
        # Test get_parameters() method
        params = model.get_parameters()
        self.assertIsInstance(params, dict)
        self.assertIn('G', params)
        
        # Test set_parameters() method
        model.set_parameters({'G': 2.0})
        self.assertEqual(model.get_parameter('G'), 2.0)

    @pytest.mark.skipif(
        not CPP_AVAILABLE.get('DO', False),
        reason=f"C++ DO module not available: {CPP_ERRORS.get('DO', 'Unknown')}"
    )
    def test_do_unified_interface(self):
        """Test DO (Damped Oscillator) implements BaseModel interface."""
        from vbi.models.cpp.damp_oscillator import DO
        
        model = DO(par={'a': 1.0, 'b': 2.0})
        
        # Test get_parameters() method
        params = model.get_parameters()
        self.assertIsInstance(params, dict)
        self.assertIn('a', params)
        self.assertEqual(params['a'], 1.0)
        
        # Test set_parameters() method
        model.set_parameters({'a': 1.5})
        self.assertEqual(model.get_parameter('a'), 1.5)
        
        # Test list_parameters() method
        param_list = model.list_parameters()
        self.assertIn('a', param_list)
        self.assertIn('b', param_list)

    def test_invalid_parameter_error(self):
        """Test that invalid parameters raise appropriate errors."""
        # This test can run even without C++ modules
        if CPP_AVAILABLE.get('VEP_sde', False):
            from vbi.models.cpp.vep import VEP_sde
            
            # Test invalid parameter in initialization
            with self.assertRaises(ValueError):
                model = VEP_sde(par={'invalid_param': 123})
            
            # Test invalid parameter in set_parameters
            model = VEP_sde(par={'weights': self.weights})
            with self.assertRaises(ValueError):
                model.set_parameters({'invalid_param': 123})


class TestBackwardCompatibility(unittest.TestCase):
    """Test backward compatibility with old parameter access patterns."""

    def setUp(self):
        """Set up test fixtures."""
        self.weights = np.array([[0, 1], [1, 0]], dtype=np.float32)

    @pytest.mark.skipif(
        not CPP_AVAILABLE.get('VEP_sde', False),
        reason=f"C++ VEP module not available: {CPP_ERRORS.get('VEP_sde', 'Unknown')}"
    )
    def test_old_par_access(self):
        """Test that old ._par access pattern still works."""
        from vbi.models.cpp.vep import VEP_sde
        
        model = VEP_sde(par={'weights': self.weights, 'G': 2.0})
        
        # Old way should still work
        self.assertEqual(model._par['G'], 2.0)
        
        # Direct attribute access should work
        self.assertEqual(model.G, 2.0)
        
        # Can still modify _par directly (though not recommended)
        model._par['G'] = 3.0
        self.assertEqual(model._par['G'], 3.0)

    @pytest.mark.skipif(
        not CPP_AVAILABLE.get('JR_sde', False),
        reason=f"C++ JR module not available: {CPP_ERRORS.get('JR_sde', 'Unknown')}"
    )
    def test_call_method_compatibility(self):
        """Test that __call__ method returns parameters."""
        from vbi.models.cpp.jansen_rit import JR_sde
        
        model = JR_sde(par={'weights': self.weights, 'G': 0.5})
        
        # Calling model() should return parameters
        params = model()
        self.assertIsInstance(params, dict)
        self.assertEqual(params['G'], 0.5)


if __name__ == '__main__':
    unittest.main()
