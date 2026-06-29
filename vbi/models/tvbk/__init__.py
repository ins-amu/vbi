"""The Virtual Brain Kernel (TVBK) model backend for VBI.

Provides an interface to models implemented with the TVB-Kernel library.

.. note::
    Requires ``tvbk`` to be installed separately.
"""


try:
    import tvbk as m
    from .tvbk_wrapper import MPR
    TVBK_AVAILABLE = True
except ImportError:
    TVBK_AVAILABLE = False
    print("Tvbtk not available. Please install it.")
