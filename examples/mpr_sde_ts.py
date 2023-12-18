import numpy as np 
import networkx as nx
import matplotlib.pyplot as plt

seed = 2
np.random.seed(seed)

LABESSIZE = 14
plt.rcParams['axes.labelsize'] = LABESSIZE
plt.rcParams['xtick.labelsize'] = LABESSIZE
plt.rcParams['ytick.labelsize'] = LABESSIZE

nn = 6
SC = nx.to_numpy_array(nx.complete_graph(nn))

parameters = {
    "G": 0.733,                         # global coupling strength
    "dt": 0.01,                         # for mpr model [ms]
    "dt_bold": 0.001,                   # for Balloon model [s]
    "J": 14.5,                          # model parameter
    "eta": -4.6*np.ones(nn),                        # model parameter
    "tau": 1.0,                         # model parameter
    "delta": 0.7,                       # model parameter
    "decimate": 500,                    # sampling from mpr time series
    "noise_amp": 0.037,                 # amplitude of noise
    "fix_seed": 0,                      # fix seed for noise
    "iapp": 0.0,                        # constant applyed current
    "t_initial": 0.0,                   # initial time * 10[ms]
    "t_transition": 5000.0,             # transition time * 10 [ms]
    "t_final": 25_000.0,                # end time * 10 [ms]
    "adj": SC,                          # weighted connection matrix
    "record_step": 10,                  # sampling every n step from mpr time series
    "data_path": "output",              # output directory
    "control": ["G"],                   # control parameters need to estimate
    "RECORD_AVG": 0                     # true to store large time series in file
}
