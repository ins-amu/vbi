import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from vbi.models.cupy.mpr import MPR_sde

seed = 2
np.random.seed(seed)

LABESSIZE = 14
plt.rcParams['axes.labelsize'] = LABESSIZE
plt.rcParams['xtick.labelsize'] = LABESSIZE
plt.rcParams['ytick.labelsize'] = LABESSIZE

nn = 6
weights = nx.to_numpy_array(nx.complete_graph(nn))

par = {
    "G": 0.25,              # global coupling strength
    "weights": weights,     # connection matrix
    "method": "heun",       # integration method
    "t_transition": 2000,   # [ms]
    "t_end": 30_000,        # [ms]
    "num_sim": 1,           # number of simulations
    "engine": "cpu",        # cpu or gpu
    "seed": seed,           # seed for random number generator
    "RECORD_TS": True,
}
obj = MPR_sde(par)
# print(obj())
sol = obj.run()

rv_t = sol["rv_t"]
rv_d = sol["rv_d"]
fmri_d = sol["fmri_d"]
fmri_t = sol["fmri_t"]

print(f"rv_t.shape = {rv_t.shape}")
print(f"rv_d.shape = {rv_d.shape}")
print(f"fmri_d.shape = {fmri_d.shape}")
print(f"fmri_t.shape = {fmri_t.shape}")
# print(f"rv_t = {rv_t}")
# print(f"fmri_t = {fmri_t}")


if fmri_d.ndim == 3:
    fig, ax = plt.subplots(3, figsize=(10, 5), sharex=True)
    ax[0].set_ylabel("BOLD")
    ax[0].plot(fmri_t/1000, fmri_d[:,:,0], alpha=0.8, lw=2)
    ax[0].margins(0, 0.1)

    ax[1].plot(rv_t/1000, rv_d[:, :nn, 0], alpha=0.8, lw=0.5)
    ax[2].plot(rv_t/1000, rv_d[:, nn:, 0], alpha=0.8, lw=0.5)
    ax[1].set_ylabel("r")
    ax[2].set_ylabel("v")
    ax[2].set_xlabel("Time [s]")
    ax[1].margins(0, 0.01)
    plt.tight_layout()
    plt.show()