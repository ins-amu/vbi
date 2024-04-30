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
from vbi.models.cpp.jansen_rit import JR_sde
from sklearn.preprocessing import StandardScaler

from vbi import report_cfg
from vbi import extract_features_list
from vbi import get_features_by_domain, get_features_by_given_names
from helpers import *

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
    "G": 1.0,
    "noise_mu": 0.24,
    "noise_std": 0.1,
    "dt": 0.05,
    "C0": 135.0 * 1.0,
    "C1": 135.0 * 0.8,
    "C2": 135.0 * 0.25,
    "C3": 135.0 * 0.25,
    "weights": SC,
    "t_transition": 500.0,      # ms
    "t_end": 2501.0,            # ms
    "output": "output",
}

# value of C1 for node 0 is 135.0 and for nodes 2,3 are 155.0
theta_true = {
    "C1": {"indices": [[0], [2, 3]], "value": [135.0, 155.0]},
    "G": {"value": 1.0},
}

obj = JR_sde(par)
print(obj())

data = obj.run(theta_true)

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
plot_ts_pxx_jr(data, par, [ax[0], ax[1]], alpha=0.6, lw=1)
ax[0].set_xlim(2000, 2500)
plt.tight_layout()
plt.savefig("output/jansen_rit_ts_psd.jpeg", dpi=300)

cfg = get_features_by_domain(domain="statistical")
cfg = get_features_by_given_names(cfg, names=['calc_std', 'calc_mean'])
report_cfg(cfg)

def wrapper(par, control, cfg, verbose=False):
    ode = JR_sde(par)
    sol = ode.run(control)

    # extract features
    fs = 1.0 / par['dt'] * 1000  # [Hz]
    stat_vec = extract_features_list(ts=[sol['x']],
                                      cfg=cfg,
                                      fs=fs,
                                      n_workers=1,
                                      verbose=verbose).values
    return stat_vec[0]


def batch_run(par, control_list, cfg, n_workers=1):
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

x_ = wrapper(par, theta_true, cfg)
print(x_)

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
control_list = [{'C1': {"indices": [[0], [2, 3]], "value": [theta_np[i, 1], theta_np[i, 2]]},
                 'G': {"value": theta_np[i, 0]}} for i in range(num_sim)]

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
points = [[theta_true['G']['value'], theta_true['C1']['value'][0], theta_true['C1']['value'][1]]]
fig, ax = pairplot(
    samples,
    limits=limits,
    figsize=(5, 5),
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
fig.savefig("output/triangleplot.jpeg", dpi=300)
