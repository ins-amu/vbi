import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from vbi.models.cupy.jansen_rit import JR_sde

seed = 2
np.random.seed(seed)

LABESSIZE = 14
plt.rcParams['axes.labelsize'] = LABESSIZE
plt.rcParams['xtick.labelsize'] = LABESSIZE
plt.rcParams['ytick.labelsize'] = LABESSIZE

nn = 6
weights = nx.to_numpy_array(nx.complete_graph(nn))

par = {
    "weights": weights,
    "G": 0.25,
    "t_cut": 500,
    "t_end": 2000,
    "noise_amp": 0.02,
    "dt": 0.02,
}

obj = JR_sde(par)
# print(obj())
data = obj.run()

t = data['t']
x = data['x'][:, :, 0].T 
data = {"t": t, "x": x}

info = np.isnan(x).sum()
print(t.shape, x.shape)

if info == 0:
    from helpers import plot_ts_pxx_jr
    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    plot_ts_pxx_jr(data, par, ax, alpha=0.5)
    plt.show()