import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from vbi.models.cpp.wc import WC_ode

from helpers import *

seed = 2
np.random.seed(seed)
torch.manual_seed(seed)

LABESSIZE = 12
plt.rcParams['axes.labelsize'] = LABESSIZE
plt.rcParams['xtick.labelsize'] = LABESSIZE
plt.rcParams['ytick.labelsize'] = LABESSIZE

nn = 6
SC = nx.to_numpy_array(nx.complete_graph(nn))



par = {
    "dt": 0.05,
    "weights": SC,
    "output": "output",
    "t_end": 1000.0,
    "t_cut": 0.0,
    "method": "heun",
    "seed": 2,
    "noise_seed": True,
}

obj = WC_ode(par)
# print(obj)
data = obj.run()
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
plot_ts_pxx_wc(data, par, ax, alpha=0.6, lw=1)
plt.savefig("output/wc_ode_cpp.png")
