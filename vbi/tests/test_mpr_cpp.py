import torch
import unittest
import numpy as np
import networkx as nx
import warnings

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message="builtin type .* has no __module__ attribute",
)

MPR_AVAILABLE = True
try:
    from vbi.models.cpp.mpr import MPR_sde
except ImportError:
    MPR_AVAILABLE = False


SEED = 2
np.random.seed(SEED)
torch.manual_seed(SEED)

nn = 3
g = nx.complete_graph(nn)
sc = nx.to_numpy_array(g) / 10.0


@unittest.skipIf(not MPR_AVAILABLE, "vbi.models.cpp.mpr.MPR_sde module not available")
class testMPRSDE(unittest.TestCase):

    mpr = MPR_sde()
    p = mpr.get_default_parameters()
    p["weights"] = sc
    p["seed"] = SEED
    p["t_cut"] = 0.01 * 60 * 1000
    p["t_end"] = 0.02 * 60 * 1000

    def test_invalid_parameter_raises_value_error(self):
        invalid_params = {"invalid_param": 42}
        with self.assertRaises(ValueError):
            MPR_sde(par=invalid_params)

    def test_run(self):

        control = {"G": 0.1, "eta": -4.7}
        mpr = MPR_sde(self.p)
        sol = mpr.run(par=control)
        x = sol["bold_d"]
        t = sol["bold_t"]
        self.assertEqual(x.shape[0], nn)
