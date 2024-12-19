import warnings
from vbi.models.cupy.mpr import MPR_sde

warnings.warn(
    "The module 'vbi.models.cupy.mpr_modified_bold' is deprecated and will be removed in a future version. "
    "Please use 'vbi.models.cupy.mpr' instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export the object(s) from the new module for backward compatibility
__all__ = ["MPR_sde"]
