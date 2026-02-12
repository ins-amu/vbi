"""Tests for optional import functionality in VBI.

This module tests that VBI works correctly when optional dependencies
(such as JAX, PyTorch, CuPy, SBI) are not available. It verifies that:
1. Core VBI functionality is accessible without optional dependencies
2. Appropriate error messages are raised when optional features are used
3. Optional imports are properly handled
"""

import pytest
import sys
from unittest.mock import patch, MagicMock
import importlib


class TestOptionalImports:
    """Test class for optional import handling."""
    
    def test_optional_import_available(self):
        """Test optional_import returns module when available."""
        from vbi.optional_deps import optional_import
        
        # Test with a known available module
        torch = optional_import('torch')
        # torch might not be installed, so just check it returns None or a module
        assert torch is None or hasattr(torch, '__name__')
    
    def test_optional_import_unavailable(self):
        """Test optional_import returns None for unavailable modules."""
        from vbi.optional_deps import optional_import
        
        # Test with a definitely non-existent module
        result = optional_import('nonexistent_module_12345')
        assert result is None
    
    def test_require_optional_available(self):
        """Test require_optional successfully imports available modules."""
        from vbi.optional_deps import require_optional
        
        # Test with a standard library module
        result = require_optional('os')
        assert hasattr(result, 'path')
    
    def test_require_optional_unavailable(self):
        """Test require_optional raises OptionalDependencyError for missing modules."""
        from vbi.optional_deps import require_optional, OptionalDependencyError
        
        with pytest.raises(OptionalDependencyError):
            require_optional('nonexistent_module_12345')
    
    def test_require_optional_error_message(self):
        """Test that OptionalDependencyError has helpful message."""
        from vbi.optional_deps import require_optional, OptionalDependencyError
        
        with pytest.raises(OptionalDependencyError) as exc_info:
            require_optional('fake_module', install_name='fake_install')
        
        error_msg = str(exc_info.value)
        assert 'fake_install' in error_msg
        assert 'pip install' in error_msg
    
    def test_require_optional_with_extra(self):
        """Test require_optional error message includes extra information."""
        from vbi.optional_deps import require_optional, OptionalDependencyError
        
        with pytest.raises(OptionalDependencyError) as exc_info:
            require_optional('fake_module', install_name='fake_install', extra='myextra')
        
        error_msg = str(exc_info.value)
        assert 'vbi[myextra]' in error_msg


@pytest.mark.skip(reason="JAX accessibility tests are deactivated")
class TestJAXOptional:
    """Test class for JAX optional functionality."""
    
    def test_jax_module_conditionally_available(self):
        """Test that vbi.jax is available when JAX is installed."""
        import vbi
        
        # The jax module should always be accessible from vbi
        assert hasattr(vbi, 'jax')
    
    def test_jax_availability_flag(self):
        """Test that JAX availability is properly flagged."""
        import vbi
        
        # The _JAX_AVAILABLE flag should exist
        assert hasattr(vbi, '_JAX_AVAILABLE')
        assert isinstance(vbi._JAX_AVAILABLE, bool)
    
    @pytest.mark.skipif(sys.modules.get('jax') is None, reason="JAX not installed")
    def test_jax_imports_when_available(self):
        """Test JAX module is properly imported when JAX is available."""
        import vbi
        
        # If JAX is available, the jax module should have neural_mass
        if vbi._JAX_AVAILABLE:
            assert hasattr(vbi.jax, 'neural_mass')
    
    def test_jax_placeholder_when_unavailable(self):
        """Test JAX placeholder classes exist."""
        import vbi
        
        # The jax class should be accessible
        assert hasattr(vbi, 'jax')
        # The neural_mass should be accessible (either real or placeholder)
        assert hasattr(vbi.jax, 'neural_mass')


class TestVBIWithoutJAX:
    """Test VBI core functionality without JAX."""
    
    def test_vbi_imports_without_jax(self):
        """Test that VBI main module imports without JAX."""
        # This test verifies vbi can be imported
        import vbi
        assert vbi is not None
    
    def test_vbi_core_modules_available(self):
        """Test that core VBI modules are available without JAX."""
        import vbi
        
        # Core modules should be available
        assert hasattr(vbi, 'utils')
        assert hasattr(vbi, 'models')
    
    def test_optional_deps_module_available(self):
        """Test that optional_deps module is always available."""
        from vbi import optional_deps
        
        # Check key functions and classes are available
        assert hasattr(optional_deps, 'optional_import')
        assert hasattr(optional_deps, 'require_optional')
        assert hasattr(optional_deps, 'OptionalDependencyError')
    
    def test_check_availability_functions(self):
        """Test availability check functions."""
        from vbi.optional_deps import (
            check_torch_available,
            check_sbi_available,
            check_cupy_available
        )
        
        # These should return boolean values
        torch_available = check_torch_available()
        sbi_available = check_sbi_available()
        cupy_available = check_cupy_available()
        
        assert isinstance(torch_available, bool)
        assert isinstance(sbi_available, bool)
        assert isinstance(cupy_available, bool)


class TestOptionalDependencyError:
    """Test OptionalDependencyError exception class."""
    
    def test_optional_dependency_error_inherits_import_error(self):
        """Test that OptionalDependencyError is an ImportError."""
        from vbi.optional_deps import OptionalDependencyError
        
        err = OptionalDependencyError("test message")
        assert isinstance(err, ImportError)
    
    def test_optional_dependency_error_message(self):
        """Test OptionalDependencyError message."""
        from vbi.optional_deps import OptionalDependencyError
        
        msg = "test error message"
        err = OptionalDependencyError(msg)
        assert str(err) == msg
    
    def test_optional_dependency_error_can_be_raised(self):
        """Test that OptionalDependencyError can be raised and caught."""
        from vbi.optional_deps import OptionalDependencyError
        
        with pytest.raises(OptionalDependencyError):
            raise OptionalDependencyError("test")
        
        with pytest.raises(ImportError):
            raise OptionalDependencyError("test")


class TestRequiresOptionalDecorator:
    """Test the requires_optional decorator."""
    
    def test_requires_optional_decorator_exists(self):
        """Test that requires_optional decorator is available."""
        from vbi.optional_deps import requires_optional
        
        assert callable(requires_optional)
    
    def test_requires_optional_decorator_with_available_dependency(self):
        """Test decorator passes with available dependency."""
        from vbi.optional_deps import requires_optional
        
        @requires_optional(('sys', 'sys', None))
        def test_func():
            return "success"
        
        # sys is always available
        result = test_func()
        assert result == "success"
    
    def test_requires_optional_decorator_with_unavailable_dependency(self):
        """Test decorator raises error with unavailable dependency."""
        from vbi.optional_deps import requires_optional, OptionalDependencyError
        
        @requires_optional(('nonexistent_module_xyz', 'fake', None))
        def test_func():
            return "success"
        
        with pytest.raises(OptionalDependencyError):
            test_func()


@pytest.mark.skip(reason="JAX accessibility tests are deactivated")
class TestJAXModule:
    """Test JAX module structure and imports."""
    
    def test_jax_neural_mass_accessible(self):
        """Test that jax.neural_mass is accessible from vbi."""
        import vbi
        
        # Should not raise an error to access the module structure
        nm = vbi.jax.neural_mass
        assert nm is not None
    
    def test_jax_module_attributes(self):
        """Test JAX module has expected attributes."""
        from vbi.models import jax
        
        # These attributes should exist
        assert hasattr(jax, 'neural_mass')


class TestIntegrationWithoutOptionalDeps:
    """Integration tests for VBI without optional dependencies."""
    
    def test_vbi_can_be_instantiated_basic_objects(self):
        """Test basic VBI object creation without optional deps."""
        import vbi
        
        # Test that core imports work
        from vbi.optional_deps import OptionalDependencyError
        assert OptionalDependencyError is not None
    
    def test_optional_import_with_install_name(self):
        """Test optional_import with different install names."""
        from vbi.optional_deps import optional_import
        
        # Test that install_name parameter is accepted
        result = optional_import('nonexistent', install_name='my_package')
        assert result is None
    
    def test_vbi_version_accessible(self):
        """Test that VBI version is accessible."""
        import vbi
        
        # Version should be defined
        assert hasattr(vbi, '__version__')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
