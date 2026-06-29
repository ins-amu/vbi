"""Feature-extraction sub-package for VBI.

Provides summary statistics and feature pipelines for converting simulated
and empirical neuroimaging time series into compact feature vectors suitable
for simulation-based inference.

Key modules
-----------
features
    Core feature-extraction functions (FC, FCD, spectral features, etc.).
features_utils
    Utility helpers including JIDT-based information-theoretic measures.
calc_features
    High-level interface for batch feature computation.
features_settings
    JSON-based feature configuration loading and management.
"""
# from .tests.all_tests import tests
