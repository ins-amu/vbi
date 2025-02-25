import unittest
import numpy as np
import networkx as nx
from vbi.utils import timer
from copy import deepcopy
import matplotlib.pyplot as plt
from vbi.models.numba.mpr import MPR_sde

seed = 2
np.random.seed(seed)

nn = 6
weights = nx.to_numpy_array(nx.complete_graph(nn))
params = {
    "G": 0.33,
    "weights": weights,
    "t_end": 2000,
    "dt": 0.01,
    "tau": 1.0,
    "eta": np.array([-4.6]),
    "rv_decimate": 10,  # in time steps
    "noise_amp": 0.037,
    "tr": 300.0,  # in [ms]
    "seed": 42,
    "RECORD_BOLD": True,
    "RECORD_RV": True,
}


def wrapper(g, par):
    par = deepcopy(par)
    sde = MPR_sde(par)
    control = {"G": g}
    data = sde.run(control)
    rv_t = data["rv_t"]
    rv_d = data["rv_d"]
    nn = par["weights"].shape[0]
    r = rv_d[:, :nn]
    v = rv_d[:, nn:]

    bold_d = data["bold_d"]
    bold_t = data["bold_t"]

    return rv_t, r, v, bold_t, bold_d


def plot(rv_t, r, v, bold_d, bold_t, g, close=True):
    step = 10
    fig, ax = plt.subplots(3, 1, figsize=(12, 6))
    ax[0].plot(rv_t[::step], r[::step, :], lw=0.1)
    ax[1].plot(rv_t[::step], v[::step, :], lw=0.1)
    ax[2].plot(bold_t, bold_d, lw=0.1)
    ax[0].set_ylabel("r")
    ax[1].set_ylabel("v")
    ax[2].set_ylabel("BOLD")


class testMPRSDE(unittest.TestCase):

    # @timer
    def test_run(self):

        # warm up
        wrapper(0.1, params)

        params["t_end"] = 30_000
        rv_t, r, v, bold_t, bold_d = wrapper(0.1, params)
        fc = np.corrcoef(bold_d.T)
        print(fc.mean())
        # plot(rv_t, r, v, bold_d, bold_t, 0.33)

        self.assertEqual(r.shape[1], nn)
        self.assertEqual(v.shape[1], nn)
        self.assertEqual(bold_d.shape[1], nn)
        self.assertTrue((fc.mean() - 0.99) < 0.01)
        


if __name__ == "__main__":
    unittest.main()
    # obj = testMPRSDE()
    # obj.test_run()
    # plt.show()
    # obj.test_run()
