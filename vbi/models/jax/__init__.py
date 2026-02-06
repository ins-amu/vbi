"""
JAX-based neural mass models and utilities.

This module provides local neural mass models and also re-exports
models from vbjax package.

Requirements:
    pip install vbi[jax]
"""

try:
    from . import neural_mass
    
    # Also import and re-export models from vbjax
    try:
        import vbjax
        # Re-export vbjax neural_mass models
        from vbjax import neural_mass as vbjax_neural_mass
    except ImportError:
        vbjax_neural_mass = None
    
    __all__ = ['neural_mass', 'vbjax_neural_mass']
    
except ImportError as e:
    raise ImportError(
        "JAX models require JAX to be installed. "
        "Install with: pip install vbi[jax]"
    ) from e