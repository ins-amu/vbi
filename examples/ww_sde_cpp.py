import torch
import pickle
import numpy as np
from time import time
from tqdm import tqdm
import networkx as nx
import sbi.utils as utils
import scipy.stats as stats
import matplotlib.pyplot as plt
from multiprocessing import Pool
from sbi.analysis import pairplot
from vbi.inference import Inference
from vbi.models.cpp.ww import WW_sde 
from sklearn.preprocessing import StandardScaler
from vbi.feature_extraction.features_utils import get_fc, get_fcd2

import vbi
from vbi import report_cfg
from vbi import list_feature_extractor
from vbi import get_features_by_domain, get_features_by_given_names
from helpers import plot_mat

seed = 2 
np.random.seed(seed)
torch.manual_seed(seed)

LABESSIZE = 12
plt.rcParams['axes.labelsize'] = LABESSIZE
plt.rcParams['xtick.labelsize'] = LABESSIZE
plt.rcParams['ytick.labelsize'] = LABESSIZE

nn = 50
weights = vbi.Load_Data_Sample(84).get_weights()

par = {
    "G": 0.65,
    "dt": 0.05,
    "t_cut": 2000.0,
    "t_end": 20_000.0,
    "weights": weights,
    "seed": seed,
    "sigma_noise": 0.001,
    "ts_decimate": 10,
    "fmri_decimate": 500,
    "RECORD_TS": 1,
    "RECORD_FMRI": 1,
    "SPARSE": False,
}


obj = WW_sde(par)
# print(obj)
tic = time()
data = obj.run()
print(f"Elapsed time: {time() - tic:.2f} s")

t = data['t']
s = data['s']
t_fmri = data['t_fmri']
d_fmri = data['d_fmri']

fc = get_fc(d_fmri.T)['full']
fcd = get_fcd2(d_fmri.T, 30, 200, 0.95)

print(t.shape, s.shape, t_fmri.shape, d_fmri.shape)
# print(d_fmri)

fig, ax = plt.subplots(2, 2, figsize=(10, 4))
ax = ax.flatten()
ax[0].plot(t[::1]/1000, s[::1, :], color="teal", alpha=0.3, lw=0.5)
ax[0].set_xlabel('Time (s)')
ax[0].set_ylabel('Activity')
ax[2].plot(t_fmri/1000, stats.zscore(d_fmri, axis=0), color="teal", alpha=0.3, lw=0.5)
ax[2].set_xlabel('Time (s)')
ax[2].set_ylabel('BOLD signal')
ax[2].margins(x=0)
ax[0].margins(x=0)

plot_mat(fc, ax=ax[1], vmin=-1, vmax=1)
plot_mat(fcd, ax=ax[3], vmin=-1, vmax=1)

plt.tight_layout()
plt.show()
