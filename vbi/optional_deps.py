"""
Optional dependency handling for VBI.

This module provides utilities for gracefully handling optional dependencies
and provides informative error messages when they're missing.
"""

import importlib
import functools
from typing import Optional, Any, Callable


class OptionalDependencyError(ImportError):
    """Raised when an optional dependency is required but not available."""
    pass


def optional_import(module_name: str, install_name: Optional[str] = None) -> Any:
    """
    Import a module if available, otherwise return None.
    
    Parameters
    ----------
    module_name : str
        Name of the module to import
    install_name : str, optional
        Name to use in installation instructions (if different from module_name)
        
    Returns
    -------
    module or None
        The imported module if successful, None if not available
    """
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None


def require_optional(module_name: str, install_name: Optional[str] = None, 
                    extra: Optional[str] = None) -> Any:
    """
    Import a required optional dependency with helpful error message.
    
    Parameters
    ----------
    module_name : str
        Name of the module to import
    install_name : str, optional
        Name to use in installation instructions
    extra : str, optional
        VBI extra that provides this dependency
        
    Returns
    -------
    module
        The imported module
        
    Raises
    ------
    OptionalDependencyError
        If the module cannot be imported
    """
    try:
        return importlib.import_module(module_name)
    except ImportError as e:
        install_name = install_name or module_name
        extra_hint = f" or 'pip install vbi[{extra}]'" if extra else ""
        
        raise OptionalDependencyError(
            f"The '{install_name}' package is required for this functionality. "
            f"Install it with 'pip install {install_name}'{extra_hint}"
        ) from e


def requires_optional(*dependencies):
    """
    Decorator to check for optional dependencies before function execution.
    
    Parameters
    ----------
    *dependencies : tuples
        Each tuple should be (module_name, install_name, extra)
        
    Examples
    --------
    @requires_optional(('torch', 'torch', 'inference'))
    def inference_function():
        import torch
        # function implementation
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for dep in dependencies:
                if len(dep) == 3:
                    module_name, install_name, extra = dep
                elif len(dep) == 2:
                    module_name, install_name = dep
                    extra = None
                else:
                    module_name = dep[0]
                    install_name = None
                    extra = None
                    
                require_optional(module_name, install_name, extra)
            return func(*args, **kwargs)
        return wrapper
    return decorator


# Pre-import commonly used optional dependencies
torch = optional_import('torch')
cupy = optional_import('cupy')
sbi = optional_import('sbi')


def check_torch_available():
    """Check if PyTorch is available."""
    return torch is not None


def check_sbi_available():
    """Check if SBI is available."""
    return sbi is not None


def check_cupy_available():
    """Check if CuPy is available."""
    return cupy is not None
