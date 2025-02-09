import torch
import unittest
import numpy as np
import networkx as nx
from copy import deepcopy
from vbi import LoadSample
from numpy.random import uniform
import matplotlib.pyplot as plt

GHB_AVAILABLE = True
try:
    from vbi.models.cupy.ghb import GHB_sde
except ImportError:
    GHB_AVAILABLE = False

# if cupy.cuda is avalable set engine to "gpu" else "cpu"
engin = "cpu"
try:
    import cupy
    if cupy.cuda.is_available():
        engin = "gpu"
except ImportError:
    pass


seed = 2
np.random.seed(seed)
torch.manual_seed(seed)

weights = LoadSample(nn=84).get_weights()
nn = len(weights)
freq = uniform(0.02, 0.04, nn)
omega = 2 * np.pi * freq

eta_mu = -1.
eta_std = 1.
eta_heter_rnd = np.random.randn(nn)
eta = eta_mu+eta_std * eta_heter_rnd


params = {
    "eta": eta,
    "dt": 0.01,
    "num_sim": 1,
    "sigma": 0.1,
    "t_cut": 4.0,
    "t_end": 15.0,
    "seed": seed,
    "G": 10.0,
    "decimate": 10,
    "omega": omega,
    "engine": engin,
    "weights": weights,
    "initial_state": uniform(0, 1, (2 * nn, 1)),
}


@unittest.skipIf(not GHB_AVAILABLE, "vbi.models.cupy.ghb.GHB_sde module not available")
class testGHBSDE(unittest.TestCase):
    def test_GHB_sde_cupy(self):
        ghb = GHB_sde(params)
        data = ghb.run()
        t = data['t']
        bold = data['bold']
        fc = np.corrcoef(bold.squeeze())
        # print(fc.mean(), fc.flatten().std())
        self.assertEqual(fc.shape, (nn, nn))
        self.assertAlmostEqual(fc.mean(), 0.63, delta=0.1)
        self.assertAlmostEqual(fc.flatten().std(), 0.3, delta=0.1)
    def test_GHB_sde_numpy(self):
    
        par = deepcopy(params)
        par['engine'] = "cpu"
        ghb = GHB_sde(par)
        data = ghb.run()
        t = data['t']
        bold = data['bold']
        fc = np.corrcoef(bold.squeeze())
        # print(fc.mean(), fc.flatten().std())
        self.assertEqual(fc.shape, (nn, nn))
        self.assertAlmostEqual(fc.mean(), 0.27, delta=0.1)
        self.assertAlmostEqual(fc.flatten().std(), 0.4, delta=0.1)



if __name__ == '__main__':
    unittest.main()
    # test = testGHBSDE()
    # test.test_GHB_sde_cupy()
    # test.test_GHB_sde_numpy()