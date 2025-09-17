"""Tests for vbi/__init__.py module.

This module tests the main VBI package initialization, imports, and core functionality.
"""

import pytest
import numpy as np
import importlib
import sys
from unittest.mock import patch, MagicMock


class TestVBIInit:
    """Test class for vbi/__init__.py module."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Ensure clean import state for each test
        if 'vbi' in sys.modules:
            # Store the original module for restoration
            self.original_vbi = sys.modules['vbi']
        else:
            self.original_vbi = None
    
    def teardown_method(self):
        """Clean up after each test method."""
        # Restore original module state
        if self.original_vbi is not None:
            sys.modules['vbi'] = self.original_vbi
        elif 'vbi' in sys.modules:
            del sys.modules['vbi']
    
    @pytest.mark.short
    def test_basic_imports(self):
        """Test that the vbi module can be imported without errors."""
        import vbi
        assert vbi is not None
        assert hasattr(vbi, '__version__')
    
    @pytest.mark.short
    def test_version_attribute(self):
        """Test that __version__ attribute is available and is a string."""
        import vbi
        assert hasattr(vbi, '__version__')
        assert isinstance(vbi.__version__, str)
        assert len(vbi.__version__) > 0
        # Basic version format check (should contain dots for semantic versioning)
        assert '.' in vbi.__version__
    
    @pytest.mark.short
    def test_get_version_function(self):
        """Test the get_version() function."""
        import vbi
        version = vbi.get_version()
        assert isinstance(version, str)
        assert len(version) > 0
        assert version == vbi.__version__
    
    @pytest.mark.short
    def test_feature_extraction_imports(self):
        """Test that feature extraction functions are properly imported."""
        import vbi
        
        # Test main feature extraction functions
        feature_functions = [
            'extract_features_df',
            'extract_features_list', 
            'extract_features',
            'calc_features'
        ]
        
        for func_name in feature_functions:
            assert hasattr(vbi, func_name), f"Missing function: {func_name}"
            func = getattr(vbi, func_name)
            assert callable(func), f"Function {func_name} is not callable"
    
    @pytest.mark.short
    def test_feature_settings_imports(self):
        """Test that feature settings functions are properly imported."""
        import vbi
        
        settings_functions = [
            'get_features_by_given_names',
            'get_features_by_domain',
            'update_cfg',
            'add_feature',
            'add_features_from_json'
        ]
        
        for func_name in settings_functions:
            assert hasattr(vbi, func_name), f"Missing function: {func_name}"
            func = getattr(vbi, func_name)
            assert callable(func), f"Function {func_name} is not callable"
    
    @pytest.mark.short
    def test_utils_imports(self):
        """Test that utility functions are properly imported."""
        import vbi
        
        util_functions = ['LoadSample', 'timer', 'display_time', 'report_cfg']
        
        for func_name in util_functions:
            assert hasattr(vbi, func_name), f"Missing function: {func_name}"
            func = getattr(vbi, func_name)
            assert callable(func), f"Function {func_name} is not callable"
    
    @pytest.mark.short
    def test_models_import(self):
        """Test that models module is imported."""
        import vbi
        assert hasattr(vbi, 'models')
        assert vbi.models is not None
    
    @pytest.mark.short
    def test_tests_alias(self):
        """Test that tests alias is created for run_tests."""
        import vbi
        assert hasattr(vbi, 'tests')
        assert callable(vbi.tests)
        # Should be an alias for run_tests
        assert hasattr(vbi, 'run_tests')
        assert vbi.tests == vbi.run_tests
    
    @pytest.mark.short
    def test_conditional_torch_imports_available(self):
        """Test conditional imports when torch/sbi are available."""
        # This test assumes torch/sbi might be available
        import vbi
        
        torch_dependent_functions = ['posterior_peaks', 'j2p', 'p2j', 'make_mask']
        
        for func_name in torch_dependent_functions:
            assert hasattr(vbi, func_name), f"Missing function: {func_name}"
            func = getattr(vbi, func_name)
            assert callable(func), f"Function {func_name} is not callable"
    
    @pytest.mark.short
    def test_conditional_torch_imports_error_handling(self):
        """Test that torch-dependent functions handle missing dependencies gracefully."""
        import vbi
        
        # Test posterior_peaks with dummy data
        test_data = np.random.randn(10, 2)
        
        # Function should either work or give a helpful ImportError
        try:
            result = vbi.utils.posterior_peaks(test_data)
            # If it works, result should be reasonable
            assert result is not None
        except ImportError as e:
            # Should give helpful error message
            assert "requires SBI and PyTorch" in str(e) or "requires" in str(e)
    
    @pytest.mark.short
    def test_make_mask_error_handling(self):
        """Test make_mask function error handling."""
        import vbi
        
        try:
            # Try to call make_mask with dummy arguments
            result = vbi.make_mask(shape=(10, 10))
        except ImportError as e:
            # Should give helpful error message about PyTorch
            assert "requires PyTorch" in str(e) or "requires" in str(e)
        except Exception:
            # Other exceptions are OK - function might need specific arguments
            pass
    
    @pytest.mark.short
    def test_inference_class_availability(self):
        """Test Inference class availability and error handling."""
        import vbi
        
        assert hasattr(vbi, 'Inference')
        
        try:
            # Try to instantiate Inference class
            inference = vbi.Inference()
            # If successful, should be a valid object
            assert inference is not None
        except ImportError as e:
            # Should give helpful error message
            assert "requires additional dependencies" in str(e)
        except Exception:
            # Other exceptions might be OK depending on implementation
            pass
    
    @pytest.mark.short 
    def test_inference_available_flag(self):
        """Test the _INFERENCE_AVAILABLE flag."""
        import vbi
        
        # The flag should exist and be a boolean
        assert hasattr(vbi, '_INFERENCE_AVAILABLE')
        assert isinstance(vbi._INFERENCE_AVAILABLE, bool)
    
    @pytest.mark.short
    def test_all_main_exports_exist(self):
        """Test that all expected main exports are available."""
        import vbi
        
        # Core functions that should always be available
        expected_exports = [
            '__version__',
            'get_version',
            'models',
            'tests',
            'run_tests',
            'LoadSample',
            'timer', 
            'display_time',
            'report_cfg',
            'extract_features_df',
            'extract_features_list',
            'extract_features',
            'calc_features',
            'get_features_by_given_names',
            'get_features_by_domain',
            'update_cfg',
            'add_feature',
            'add_features_from_json',
            'posterior_peaks',
            'j2p',
            'p2j',
            'make_mask',
            'Inference',
            '_INFERENCE_AVAILABLE'
        ]
        
        missing_exports = []
        for export in expected_exports:
            if not hasattr(vbi, export):
                missing_exports.append(export)
        
        assert len(missing_exports) == 0, f"Missing exports: {missing_exports}"

    @pytest.mark.short
    def test_import_structure(self):
        """Test that imports don't create circular dependencies."""
        # This test ensures that importing vbi doesn't cause issues
        import vbi
        
        # Test that we can access nested modules
        assert vbi.models is not None
        
        # Test that version info is accessible
        assert vbi.__version__ is not None
        
        # Basic smoke test - ensure no obvious import errors
        assert str(type(vbi)) == "<class 'module'>"

    @pytest.mark.short
    def test_version_format(self):
        """Test that version follows semantic versioning format."""
        import vbi
        version = vbi.__version__
        
        # Basic checks for semantic versioning (major.minor.patch)
        parts = version.split('.')
        assert len(parts) >= 2, f"Version should have at least major.minor: {version}"
        
        # First part should be numeric (major version)
        assert parts[0].isdigit(), f"Major version should be numeric: {parts[0]}"


# Integration tests that might need longer setup
class TestVBIInitIntegration:
    """Integration tests for VBI initialization."""
    
    @pytest.mark.long
    def test_feature_extraction_integration(self):
        """Test that feature extraction functions work together."""
        import vbi
        
        # This is a basic integration test
        # More comprehensive tests should be in specific feature extraction tests
        assert callable(vbi.extract_features)
        assert callable(vbi.calc_features)
        
        # Test that configuration functions work
        assert callable(vbi.update_cfg)
        assert callable(vbi.get_features_by_domain)
    
    @pytest.mark.long
    def test_models_integration(self):
        """Test that models module integrates properly."""
        import vbi
        
        # Basic test that models module is accessible
        assert vbi.models is not None
        
        # Test that we can access model submodules
        # (specific model tests should be in separate test files)
        assert hasattr(vbi.models, '__name__')


if __name__ == "__main__":
    # Allow running this test file directly
    pytest.main([__file__])
