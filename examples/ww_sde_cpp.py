import torch
import pickle
import numpy as np
from tqdm import tqdm
import networkx as nx
import sbi.utils as utils
import matplotlib.pyplot as plt
from multiprocessing import Pool
from sbi.analysis import pairplot
from vbi.inference import Inference
from vbi.models.cpp.ww import WW_sde 
from sklearn.preprocessing import StandardScaler

from vbi import report_cfg
from vbi import list_feature_extractor
from vbi import get_features_by_domain, get_features_by_given_names

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
    "G": 0.5,
    "dt": 0.001,
    "t_cut": 2000.0,
    "t_end": 20_000.0,
    "weights": SC,
    "seed": seed,
    "sigma_noise": 0.001,
    "ts_decimate": 1000,
    "fmri_decimate": 500,
    "RECORD_TS": True,
    "RECORD_FMRI": True,
}


from time import time
obj = WW_sde(par)
# print(obj)
tic = time()
data = obj.run()
print(f"Elapsed time: {time() - tic:.2f} s")

t = data['t']
s = data['s']
t_fmri = data['t_fmri']
d_fmri = data['d_fmri']

print(t.shape, s.shape, t_fmri.shape, d_fmri.shape)
# print(d_fmri)

fig, ax = plt.subplots(2, 1, figsize=(10, 4))

ax[0].plot(t[::5]/1000, s[::5, :])
ax[0].set_xlabel('Time (s)')
ax[0].set_ylabel('Activity')
ax[1].plot(t_fmri/1000, d_fmri)
ax[1].set_xlabel('Time (s)')
ax[1].set_ylabel('BOLD signal')
plt.tight_layout()
plt.show()