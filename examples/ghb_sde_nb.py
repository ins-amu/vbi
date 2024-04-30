import numpy as np
from vbi import LoadSample
import matplotlib.pyplot as plt
from numpy.random import uniform
from vbi.models.numba.ghb import GHB_sde

seed = 2
np.random.seed(seed)

LABESSIZE = 14
plt.rcParams["axes.labelsize"] = LABESSIZE
plt.rcParams["xtick.labelsize"] = LABESSIZE
plt.rcParams["ytick.labelsize"] = LABESSIZE

weights = LoadSample(nn=84).get_weights()
nn = len(weights)

freq = uniform(0.02, 0.04, nn)
omega = 2 * np.pi * freq

eta_mu = -1.
eta_std = 1.
eta_heter_rnd = np.random.randn(nn)
eta = eta_mu+eta_std * eta_heter_rnd

params = {
    "G": 0.25*100,
    "dt": 0.001,
    "tcut": 10.0,
    "tend": 120.0,
    "sigma": 0.1,
    "decimate": 100,
    "eta": eta,
    "omega": omega,
    "weights": weights,
    "init_state": uniform(0, 1, 2 * nn),
}

def wrapper(params, tspan=[0, 120]):
    obj = GHB_sde(params)
    return obj.run(tspan=tspan)

from timeit import timeit

# warm up
wrapper(params, tspan=[0,1])
tic = timeit(lambda: wrapper(params, tspan=[0,120]), number=1)
print(f"Elapsed time: {tic:.2f} s")

data = wrapper(params)
bold = data["bold"]
t = data["t"]
print(bold.shape, t.shape)

fig, ax = plt.subplots(1, figsize=(12, 3))
ax.plot(t, bold.T, lw=1)
ax.set_title("Simulated BOLD signals", fontsize=18)
ax.set_xlabel("Time", fontsize=18)
ax.set_ylabel("Activity", fontsize=18)
ax.margins(x=0)
plt.tight_layout()
# plt.savefig("ghb_sde_nb.png")
plt.show()
