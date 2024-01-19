import torch
import pickle
import numpy as np
from tqdm import tqdm
from copy import copy
import networkx as nx
import sbi.utils as utils
import matplotlib.pyplot as plt
from multiprocessing import Pool
from sbi.analysis import pairplot
from vbi.inference import Inference
from sklearn.preprocessing import StandardScaler
from vbi.models.cpp.jansen_rit import JR_sdde_cpp
from helpers import *

from vbi import report_cfg
from vbi import list_feature_extractor
from vbi import get_features_by_domain, get_features_by_given_names

LABESSIZE = 12
plt.rcParams['axes.labelsize'] = LABESSIZE
plt.rcParams['xtick.labelsize'] = LABESSIZE
plt.rcParams['ytick.labelsize'] = LABESSIZE

seed = 2
np.random.seed(seed)
torch.manual_seed(seed)

N = 8
adj = nx.to_numpy_array(nx.complete_graph(N), dtype=float)
delays = np.ones_like(adj) * 2  # arbitrary delay
np.fill_diagonal(delays, 0.0)

param = {
    "dt": 0.1,                     # time step
    "G": 0.1,                       # coupling strength
    "mu": 0.22,                     # mean of the input
    "sigma": 0.005,                 # noise amplitude
    "C0": 135.0 * 1.0,
    "C1": 135.0 * 0.8,
    "C2": 135.0 * 0.25,
    "C3": 135.0 * 0.25,
    "t_end": 2000.0,                # simulation time
    "t_transition": 1000.0,          # transition time
    "weights": adj,                 # adjacency matrix
    "delays": delays,               # delay matrix
    "method": "heun",

    # stimulation
    # "sti_ti": 1000 + 100,
    # "sti_duration": 10,
    # "sti_gain": 0.5,
    # "sti_amplitude": 0.1* np.ones(8), # or just 0.1
}

theta_true = {
    "G": {"value": 1.0},
    "C1": {"indices": [[0], [2, 3]], "value": [150.0, 160.0]}
}

obj = JR_sdde_cpp(param)
data = obj.run(par=theta_true)
t = data['t']
x = data['x']
sti = data['sti']
# print(obj.C1, obj.sti_amplitude)

if 0:  # plot stimulation signal
    plt.figure(figsize=(6, 2.5))
    plt.plot(data['t'], data['sti'])
    plt.xlabel('time(ms)')
    plt.ylabel('sti amplitude')
    plt.tight_layout()
    plt.show()


if 1:  # plot time series
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    plot_ts_jr(data, param, [ax[0], ax[1]], alpha=0.6, lw=1)
    plt.savefig("output/jr_sdde_cpu.jpeg", dpi=300)

cfg = get_features_by_domain(domain="statistical")
cfg = get_features_by_given_names(cfg, names=['calc_std', 'calc_mean'])
# report_cfg(cfg)


def wrapper(par, control, cfg, verbose=False):
    ode = JR_sdde_cpp(par)
    sol = ode.run(control)

    # extract features
    fs = 1.0 / par['dt'] * 1000  # [Hz]
    stat_vec = list_feature_extractor(ts=[sol['x']],
                                      fea_dict=cfg,
                                      fs=fs,
                                      n_workers=1,
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


num_sim = 200
num_workers = 10
C11_min, C11_max = 130.0, 300.0
C12_min, C12_max = 130.0, 300.0
G_min, G_max = 0.0, 5.0
prior_min = [G_min, C11_min, C12_min]
prior_max = [G_max, C11_max, C12_max]
prior = utils.BoxUniform(low=torch.tensor(prior_min),
                         high=torch.tensor(prior_max))

obj = Inference()
theta = obj.sample_prior(prior, num_sim)
theta_np = theta.numpy().astype(float)
control_list = [{"G": {"value": theta_np[i, 0]},
                 "C1": {"indices": [[0], [2, 3]], "value": [theta_np[i, 1], theta_np[i, 2]]}}
                for i in range(num_sim)]

stat_vec = batch_run(param, control_list, cfg, num_workers)

scaler = StandardScaler()
stat_vec_st = scaler.fit_transform(np.array(stat_vec))
stat_vec_st = torch.tensor(stat_vec_st, dtype=torch.float32)
torch.save(theta, 'output/theta.pt')
torch.save(stat_vec, 'output/stat_vec.pt')

print(theta.shape, stat_vec_st.shape)

posterior = obj.train(theta, stat_vec_st, prior, method='SNPE', density_estimator='maf')

with open('output/posterior.pkl', 'wb') as f:
    pickle.dump(posterior, f)

with open('output/posterior.pkl', 'rb') as f:
    posterior = pickle.load(f)

xo = wrapper(param, theta_true, cfg)
xo_st = scaler.transform(xo.reshape(1, -1))

samples = obj.sample_posterior(xo_st, 10000, posterior)
torch.save(samples, 'output/samples.pt')

limits = [[i, j] for i, j in zip(prior_min, prior_max)]
points = [[theta_true['G']['value'], theta_true['C1']['value'][0], theta_true['C1']['value'][1]]]
fig, ax = pairplot(
    samples,
    limits=limits,
    fig_size=(5, 5),
    points=points,
    labels=["G", "C11", "C12"],
    offdiag='kde',
    diag='kde',
    points_colors="r",
    samples_colors="k",
    points_offdiag={'markersize': 10})
ax[0,0].tick_params(labelsize=14)
ax[0,0].margins(y=0)
plt.tight_layout()
fig.savefig("output/tri_jr_sdde_cpu.jpeg", dpi=300);
