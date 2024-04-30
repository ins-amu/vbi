import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
# from vbi.models.cpp.km import KM_sde
from vbi.models.cupy.km import KM_sde
from helpers import *


seed = 2
np.random.seed(seed)

LABESSIZE = 14
plt.rcParams['axes.labelsize'] = LABESSIZE
plt.rcParams['xtick.labelsize'] = LABESSIZE
plt.rcParams['ytick.labelsize'] = LABESSIZE

nn = 6
SC = nx.to_numpy_array(nx.complete_graph(nn), dtype=np.float64)

G = np.linspace(0, 0.1, 20)
num_sim = len(G)

par_dict = {
    "G": G,
    "dt": 0.05,
    "num_sim": num_sim,
    "weights": SC,
    "noise_amp": 0.2,
    "t_transition": 100.0,
    "t_end": 500.0,
    "alpha": None,
    "engine": "gpu",
    "seed": seed,
    "output": "output",
    "omega": 2 * np.pi * np.random.normal(0.1, 0.01, nn),
}

obj = KM_sde(par_dict)
print(obj())
data = obj.run(verbose=False)

t = data['t']
x = data['x']  # (num_steps, num_nodes, num_sim)
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
plot_ts_pxx_km_cupy(data, par_dict, ax, alpha=0.6, lw=1)
ax[0].set_xlim(450, 500)
plt.tight_layout()
plt.show()
