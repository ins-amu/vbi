import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from vbi.models.cpp.jansen_rit import JR_sde_cpp
from helpers import *

seed = 2
np.random.seed(seed)

LABESSIZE = 14
plt.rcParams['axes.labelsize'] = LABESSIZE
plt.rcParams['xtick.labelsize'] = LABESSIZE
plt.rcParams['ytick.labelsize'] = LABESSIZE

nn = 6
SC = nx.to_numpy_array(nx.complete_graph(nn))

par_dict = {
    "G": 1.0,
    "noise_mu": 0.24,
    "noise_std": 0.1,
    "dt": 0.05,
    "C0": 135.0 * 1.0,
    "C1": 135.0 * 0.8,
    "C2": 135.0 * 0.25,
    "C3": 135.0 * 0.25,
    "adj": SC,
    "t_transition": 500.0,      # ms
    "t_end": 2501.0,            # ms
    "data_path": "output",
}

control_dict = {
    "C1": {"indices": [[0], [2, 3]], "value": [135.0, 155.0]},
    "G": {"value": 1.0},
}

obj = JR_sde_cpp(par_dict)
print(obj())
data = obj.run(control_dict)

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
plot_ts_pxx_jr(data, par_dict, [ax[0], ax[1]], alpha=0.6, lw=1)
ax[0].set_xlim(2000, 2500)
plt.tight_layout()
plt.show()
