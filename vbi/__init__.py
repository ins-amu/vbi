import os
from .tests.test_suite import run_tests
from .utils import test_imports
from . import models
from ._version import __version__

# Create alias for backward compatibility
tests = run_tests


from .feature_extraction.calc_features import (
    extract_features_df,
    extract_features_list,
    extract_features,
    calc_features,
)

from .feature_extraction.features_settings import (
    get_features_by_given_names,
    get_features_by_domain,
    update_cfg,
    add_feature,
    add_features_from_json,
)

from .feature_extraction.features_utils import report_cfg
from .utils import LoadSample, timer, display_time

# Conditionally import torch/sbi dependent functions
try:
    from .utils import posterior_peaks as _posterior_peaks, j2p, p2j
    # Test if the functions can actually run without torch/sbi
    try:
        import numpy as np
        test_samples = np.random.randn(10, 2)
        _posterior_peaks(test_samples)
        posterior_peaks = _posterior_peaks
    except (ImportError, NameError):
        def posterior_peaks(*args, **kwargs):
            raise ImportError(
                "posterior_peaks requires SBI and PyTorch. Install with: pip install vbi[inference]"
            )
except ImportError:
    # Create placeholder functions that give helpful error messages
    def posterior_peaks(*args, **kwargs):
        raise ImportError(
            "posterior_peaks requires SBI and PyTorch. Install with: pip install vbi[inference]"
        )
    
    def j2p(*args, **kwargs):
        raise ImportError(
            "j2p requires additional dependencies. Install with: pip install vbi[inference]"
        )
    
    def p2j(*args, **kwargs):
        raise ImportError(
            "p2j requires additional dependencies. Install with: pip install vbi[inference]"
        )

# Conditionally import torch-dependent utility functions
try:
    from .feature_extraction.utility import make_mask
except ImportError:
    def make_mask(*args, **kwargs):
        raise ImportError(
            "make_mask requires PyTorch. Install with: pip install vbi[inference]"
        )

# Conditionally import inference functionality
try:
    from .inference import Inference
    _INFERENCE_AVAILABLE = True
except ImportError:
    _INFERENCE_AVAILABLE = False
    
    class Inference:
        """Placeholder for Inference class when dependencies are not available."""
        def __init__(self):
            raise ImportError(
                "Inference functionality requires additional dependencies. "
                "Install with: pip install vbi[inference]"
            )



def get_version():
    version_file = os.path.join(os.path.dirname(__file__), '_version.py')
    with open(version_file) as f:
        for line in f:
            if line.startswith('__version__'):
                return line.split('=')[1].strip().strip('"\'')
