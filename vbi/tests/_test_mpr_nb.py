import torch
import unittest
import numpy as np
import networkx as nx
from vbi.models.numba.mpr import MPR_sde

seed = 2
np.random.seed(seed)
torch.manual_seed(seed)

nn = 3
g = nx.complete_graph(nn)
sc = nx.to_numpy_array(g)/ 10.0

class testMPRSDE(unittest.TestCase):
    
    mpr = MPR_sde()
    p = mpr.get_default_parameters()
    p['weights'] = sc
    p['seed'] = seed
    p['t_cut'] = 0.01 * 60 * 1000
    p['t_end'] = 0.02 * 60 * 1000
    
    def test_invalid_parameter_raises_value_error(self):
        invalid_params = {"invalid_param": 42}
        with self.assertRaises(ValueError):
            MPR_sde(par=invalid_params)

    def test_run(self):
        
        control = {"G": 0.1, "eta": -4.7}
        mpr = MPR_sde(self.p)
        sol = mpr.run(par=control)
        x = sol["x"]
        t = sol["t"]
        self.assertEqual(x.shape[0], nn)