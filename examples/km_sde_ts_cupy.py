import tqdm
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from multiprocessing import Pool
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

par_dict = {
    "G": 0.0,
    "dt": 0.05,     # [s]
    "num_sim": 1,
    "weights": SC,
    "noise_amp": 0.2,
    "t_transition": 0.0,      # ms
    "t_end": 500.0,            # ms
    "alpha": None,
    "engine": "cpu",
    "output": "output",
    "omega": 2 * np.pi* np.random.normal(0.1, 0.01, nn),
}

obj = KM_sde(par_dict)
print(obj())
data = obj.run()

t = data['t']
x = np.sin(data['x'][:,:,0].T)
data = {'t': t, 'x': x}
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
plot_ts_pxx_km(data, par_dict, [ax[0], ax[1]], alpha=0.6, lw=1)
ax[0].set_xlim(400, 500)
plt.tight_layout()
plt.show()