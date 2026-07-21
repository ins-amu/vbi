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

from .pipeline import FeaturePipeline
from .pruner import FeaturePruner
from .features_settings import (
    get_features_by_domain,
    get_features_by_given_names,
    update_cfg,
    Data_F,
)

__all__ = [
    "FeaturePipeline",
    "FeaturePruner",
    "get_features_by_domain",
    "get_features_by_given_names",
    "update_cfg",
    "Data_F",
]
