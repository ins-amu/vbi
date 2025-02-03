import torch
import unittest
import numpy as np
import networkx as nx
from copy import deepcopy

MPR_AVAILABLE = True
try:
    from vbi.models.cupy.mpr import MPR_sde
except ImportError:
    MPR_AVAILABLE = False


seed = 2
np.random.seed(seed)
torch.manual_seed(seed)

nn = 3
g = nx.complete_graph(nn)
sc = nx.to_numpy_array(g) / 10.0


@unittest.skipIf(not MPR_AVAILABLE, "vbi.models.cupy.mpr.MPR_sde module not available")
class testMPRSDE(unittest.TestCase):
    
    mpr = MPR_sde()
    p = mpr.get_default_parameters()
    p['weights'] = sc
    p['seed'] = seed
    p['t_cut'] = 0.01 * 60 * 1000
    p['t_end'] = 0.02 * 60 * 1000
    p['engine'] = "cpu"
    
    def test_invalid_parameter_raises_value_error(self):
        invalid_params = {"invalid_param": 42}
        with self.assertRaises(ValueError):
            MPR_sde(par=invalid_params)

    def test_run(self):
        
        par = deepcopy(self.p)
        par['G'] = 0.1
        par['eta'] = -4.7
        mpr = MPR_sde(self.p)
        sol = mpr.run()
        x = sol["fmri_d"]
        t = sol["fmri_t"]
        self.assertEqual(x.shape[1], nn)
    