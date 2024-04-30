import torch
import numpy as np
from helpers import plot_mat
import matplotlib.pyplot as plt
from vbi.utils import LoadSample
from vbi.models.numba.ww import WW_sde
from vbi.feature_extraction.features_utils import get_fc, get_fcd2
import networkx as nx
import vbi

seed = 2
np.random.seed(seed)
torch.manual_seed(seed)

LABESSIZE = 12
plt.rcParams['axes.labelsize'] = LABESSIZE
plt.rcParams['xtick.labelsize'] = LABESSIZE
plt.rcParams['ytick.labelsize'] = LABESSIZE

# weights = vbi.LoadSample(84).get_weights()
nn = 6
weights = nx.to_numpy_array(nx.complete_graph(nn))

par = {
    "G": 2.0,
    "dt": 0.01,
    "t_cut": 10.0,
    "t_end": 500.0,
    "weights": weights,
    "seed": seed,
    "sigma_noise": 0.1,
    "ts_decimate": 10,
    "fmri_decimate": 20,
    "method": "heun",
    "RECORD_TS": True,
    "RECORD_FMRI": True,
}

# obj = WW_sde(par)
# # print(obj)
# data = obj.run()

# t = data['t']
# s = data['s']
# t_fmri = data['t_fmri']
# d_fmri = data['d_fmri']

# fig, ax = plt.subplots(2, figsize=(12, 6), sharex=True)
# for i in range(6):
#     ax[0].plot(t, s[:, i]+i, label=f'x_{i}', color="teal")
#     ax[1].plot(t_fmri, d_fmri[:, i]+i*2, label=f'd_fmri_{i}', color="teal")
# ax[0].margins(x=0)
# ax[1].set_xlabel("Time [ms]")
# ax[0].set_ylabel("Activity")
# ax[1].set_ylabel("BOLD")
# plt.show()





# if 1:
#     fc = get_fc(d_fmri.T)['full']
#     fcd = get_fcd2(d_fmri.T, 30, 200, 0.95)
#     fig, ax = plt.subplots(2, 2, figsize=(10, 4))
#     ax = ax.flatten()
    
#     for i in range(6):
#         ax[0].plot(t, s[:, i]+i, color="teal")
#         ax[2].plot(t_fmri, d_fmri[:, i]+i*2, label=f'd_fmri_{i}', color="teal")

#     ax[0].set_xlabel('Time [ms]')
#     ax[0].set_ylabel('Activity')
#     ax[2].set_xlabel('Time [ms]')
#     ax[2].set_ylabel('BOLD signal')
#     ax[2].margins(x=0)
#     ax[0].margins(x=0)
#     plot_mat(fc, ax=ax[1], vmin=-1, vmax=1)
#     plot_mat(fcd, ax=ax[3], vmin=-1, vmax=1)
#     ax[1].set_title('FC')
#     ax[3].set_title('FCD')
#     plt.tight_layout()
#     plt.show()


import multiprocessing as mp 

def wrapper(par):
    obj = WW_sde(par)
    data = obj.run()
    return data


with mp.Pool(10) as p:
    data = p.map(wrapper, [par]*20)
    

