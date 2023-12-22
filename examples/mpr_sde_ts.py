import numpy as np 
import networkx as nx
import matplotlib.pyplot as plt
from vbi.models.cpp.mpr import MPR_sde

seed = 2
np.random.seed(seed)

LABESSIZE = 14
plt.rcParams['axes.labelsize'] = LABESSIZE
plt.rcParams['xtick.labelsize'] = LABESSIZE
plt.rcParams['ytick.labelsize'] = LABESSIZE

nn = 6
weights = nx.to_numpy_array(nx.complete_graph(nn))

parameters = {
    "G": 0.55,                         # global coupling strength
    "dt": 0.01,                         # for mpr model [ms]
    "dt_bold": 0.001,                   # for Balloon model [s]
    "J": 14.5,                          # model parameter
    "eta": -4.6*np.ones(nn),                        # model parameter
    "tau": 1.0,                         # model parameter
    "delta": 0.7,                       # model parameter
    "decimate": 500,                    # sampling from mpr time series
    "noise_amp": 0.037,                 # amplitude of noise
    "iapp": 0.0,                        # constant applyed current
    "t_initial": 0.0,                   # initial time * 10[ms]
    "t_transition": 20_000.0,           # transition time * 10 [ms]
    "t_end": 250_000.0,                 # end time * 10 [ms]
    "weights": weights,                 # weighted connection matrix
    "seed": seed,                       # seed for random number generator
    "noise_seed": True,                 # fix seed for noise
    "record_step": 10,                  # sampling every n step from mpr time series
    "output": "output",                 # output directory
    "RECORD_AVG": 0                     # true to store large time series in file
}

control_dict = {
    "eta": {"indices": [[0], [2, 3]], "value": [-4.3, -4.5]}, # set different values for eta
    "G": {"value": 0.5},
    }

obj = MPR_sde(parameters)
# print(obj())
sol = obj.run(par=control_dict)
print(obj.eta)

t = sol["t"]
x = sol["x"]

print(f"t.shape = {t.shape}")
print(f"x.shape = {x.shape}")

if x.ndim == 2:
    pass
    fig, ax = plt.subplots(1, figsize=(10, 3))
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("BOLD")
    plt.plot(t/1000, x.T, alpha=0.8, lw=2)
    plt.margins(0,0.1)
    plt.tight_layout()
    plt.savefig("output/mpr_sde_ts.png", dpi=300)
    plt.close()
else:
    exit(0)


# Feature extraction ------------------------------------------------
from vbi.feature_extraction.features_settings import *
from vbi.feature_extraction.calc_features import *

fs = 1/(parameters["dt_bold"]) / 1000
cfg = get_features_by_domain(domain="statistical")
# report_cfg(cfg)
data = dataframe_feature_extractor([x], fs, fea_dict=cfg, n_workers=1)
print(data.values)


