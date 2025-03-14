import os
from .tests.test_suite import tests
from .utils import test_imports
from . import models
from ._version import __version__


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
from .utils import LoadSample, timer, display_time, posterior_peaks

from .feature_extraction.utility import make_mask

from .utils import j2p, p2j



def get_version():
    version_file = os.path.join(os.path.dirname(__file__), '_version.py')
    with open(version_file) as f:
        for line in f:
            if line.startswith('__version__'):
                return line.split('=')[1].strip().strip('"\'')
