import torch
import pickle
import numpy as np
from time import time
from tqdm import tqdm
import sbi.utils as utils
import scipy.stats as stats
from helpers import plot_mat
import matplotlib.pyplot as plt
from multiprocessing import Pool
from sbi.analysis import pairplot
from vbi.inference import Inference
from vbi.models.cpp.ww import WW_sde
from sklearn.preprocessing import StandardScaler
from vbi.feature_extraction.features_utils import get_fc, get_fcd2

import vbi
from vbi import extract_features_list
from vbi import get_features_by_domain, get_features_by_given_names

seed = 2
np.random.seed(seed)
torch.manual_seed(seed)

LABESSIZE = 12
plt.rcParams['axes.labelsize'] = LABESSIZE
plt.rcParams['xtick.labelsize'] = LABESSIZE
plt.rcParams['ytick.labelsize'] = LABESSIZE

weights = vbi.Load_Data_Sample(84).get_weights()

par = {
    "G": 0.65,
    "dt": 0.05,
    "t_cut": 500.0,
    "t_end": 10_000.0,
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

if 0:
    fc = get_fc(d_fmri.T)['full']
    fcd = get_fcd2(d_fmri.T, 30, 200, 0.95)
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
    ax[1].set_title('FC')
    ax[3].set_title('FCD')
    plt.tight_layout()
    plt.show()


cfg = get_features_by_domain(domain="connectivity")
cfg = get_features_by_given_names(cfg, names=['fc_stat'])
# report_cfg(cfg)

def wrapper(par, control, cfg, verbose=False):
    ode = WW_sde(par)
    sol = ode.run(control)

    # extract features
    fs = 1.0 / par['dt'] * 1000  # [Hz]
    stat_vec = extract_features_list(ts=[sol['d_fmri'].T],
                                      cfg=cfg,
                                      fs=fs,
                                      verbose=verbose).values
    return stat_vec[0]

def batch_run(par, control_list, cfg, n_workers=1):
    stat_vec = []
    n = len(control_list)
    def update_bar(_):
        pbar.update()
    with Pool(processes=n_workers) as pool:
        with tqdm(total=n) as pbar:
            async_results = [pool.apply_async(wrapper,
                                              args=(
                                                  par, control_list[i], cfg, False),
                                              callback=update_bar)
                             for i in range(n)]
            stat_vec = [res.get() for res in async_results]
    return stat_vec


theta_true = {
    "G": {"value": 0.65},
}
# tic = time()
# x_ = wrapper(par, theta_true, cfg)
# print(f"Elapsed time: {time() - tic:.2f} s")
# print(x_)

num_sim = 200
num_workers = 10
G_min, G_max = 0.0, 1.5
prior_min = [G_min]
prior_max = [G_max]
prior = utils.BoxUniform(low=torch.tensor(prior_min),
                         high=torch.tensor(prior_max))

obj = Inference()
theta = obj.sample_prior(prior, num_sim)
theta_np = theta.numpy().astype(float)
control_list = [{'G': {"value": theta_np[i, 0]}} for i in range(num_sim)]

stat_vec = batch_run(par, control_list, cfg, num_workers)

scaler = StandardScaler()
stat_vec_st = scaler.fit_transform(np.array(stat_vec))
stat_vec_st = torch.tensor(stat_vec_st, dtype=torch.float32)
torch.save(theta, 'output/theta.pt')
torch.save(stat_vec, 'output/stat_vec.pt')

print(theta.shape, stat_vec_st.shape)

posterior = obj.train(theta, stat_vec_st, prior, method="SNPE", density_estimator="maf")

with open('output/posterior.pkl', 'wb') as f:
    pickle.dump(posterior, f)

with open('output/posterior.pkl', 'rb') as f:
    posterior = pickle.load(f)

xo = wrapper(par, theta_true, cfg)
xo_st = scaler.transform(xo.reshape(1, -1))

samples = obj.sample_posterior(xo_st, 10000, posterior)
torch.save(samples, 'output/samples.pt')

limits = [[i, j] for i, j in zip(prior_min, prior_max)]
points = [[theta_true['G']['value']]]
fig, ax = pairplot(
    samples,
    limits=limits,
    figsize=(4, 4),
    points=points,
    labels=["G"],
    offdiag='kde',
    diag='kde',
    points_colors="r",
    samples_colors="k",
    points_offdiag={'markersize': 10})
ax[0,0].tick_params(labelsize=14)
ax[0,0].margins(y=0)
plt.tight_layout()
fig.savefig("output/tri.jpeg", dpi=300)
