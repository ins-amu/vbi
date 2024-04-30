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
from vbi.models.cpp.km import KM_sde
from sklearn.preprocessing import StandardScaler
from helpers import *

from vbi import extract_features_list
from vbi import get_features_by_domain, get_features_by_given_names

seed = 2
np.random.seed(seed)
torch.manual_seed(seed)

LABESSIZE = 12
plt.rcParams['axes.labelsize'] = LABESSIZE
plt.rcParams['xtick.labelsize'] = LABESSIZE
plt.rcParams['ytick.labelsize'] = LABESSIZE

nn = 6
SC = nx.to_numpy_array(nx.complete_graph(nn), dtype=np.float64)

par_dict = {
    "G": 0.01,
    "dt": 0.01,
    "noise_amp": 0.05,
    "omega": 2 * np.pi * 1.0 + np.random.normal(0.1, 0.5, nn),
    "weights": SC,
    "t_transition": 500.0,      # ms
    "t_end": 2001.0,            # ms
    "alpha": None,
    "output": "output",
}

theta_true = {"G": {"value": 0.5}}

obj = KM_sde(par_dict)
# print(obj())
data = obj.run(par=theta_true)

if 0:  # plot time series
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    plot_ts_pxx_km(data, par_dict, [ax[0], ax[1]], alpha=0.6, lw=1)
    ax[0].set_xlim(1990, 2000)
    plt.tight_layout()
    plt.show()


def wrapper(par, control, cfg, verbose=False):
    sde = KM_sde(par)
    sol = sde.run(control)

    # extract features
    fs = 1 / par['dt']
    stat_vec = extract_features_list(ts=[sol['x']],
                                      cfg=cfg,
                                      fs=fs,
                                      n_workers=1,
                                      verbose=verbose).values
    return stat_vec[0]


def batch_run(par, control_list, cfg, n_workers=1):
    stat_vec = []
    n = len(control_list)

    def update_bar(_):
        pbar.update()
    with Pool(n_workers) as p:
        with tqdm(total=n) as pbar:
            async_results = [p.apply_async(wrapper,
                                           args=(
                                               par, control_list[i], cfg, False),
                                           callback=update_bar)
                             for i in range(n)]
            stat_vec = [res.get() for res in async_results]
    return stat_vec


cfg = get_features_by_domain(domain="statistical")
cfg = get_features_by_given_names(cfg, names=['kop', 'calc_std', 'calc_mean'])
# report_cfg(cfg)

num_sim = 200
num_workers = 10
G_min, G_max = 0, 1.0
prior_min = [G_min]
prior_max = [G_max]
prior = utils.BoxUniform(low=torch.tensor(prior_min),
                         high=torch.tensor(prior_max))

obj = Inference()
theta = obj.sample_prior(prior, num_sim)
theta_np = theta.numpy().astype(float)
control_list = [{'G': {'value': theta_np[i, 0]}} for i in range(num_sim)]

stat_vec = batch_run(par_dict, control_list, cfg, num_workers)

scalar = StandardScaler()
stat_vec_st = scalar.fit_transform(np.array(stat_vec))
stat_vec_st = torch.tensor(stat_vec_st, dtype=torch.float32)
torch.save(stat_vec_st, "output/stat_vec_st.pt")
torch.save(theta, "output/theta.pt")

print(theta.shape, stat_vec_st.shape)

posterior = obj.train(theta, stat_vec_st, prior, method='SNPE', density_estimator="maf")

with open('output/posterior.pkl', 'wb') as f:
    pickle.dump(posterior, f)

with open('output/posterior.pkl', 'rb') as f:
    posterior = pickle.load(f)

xo = wrapper(par_dict, theta_true, cfg)
xo_st = scalar.transform(np.array(xo).reshape(1, -1))

samples = obj.sample_posterior(xo_st, 10000, posterior)
torch.save(samples, 'output/samples.pt')

limits = [[i, j] for i, j in zip(prior_min, prior_max)]
points = [[theta_true['G']['value']]]
fig, ax = pairplot(
    samples,
    limits=limits,
    fig_size=(4,4),
    points=points,
    labels=['G'],
    offdiag='kde',
    diag='kde',
    points_colors="r",
    samples_colors="k",
    points_offdiag={'markersize': 10})
plt.tight_layout()
fig.savefig("output/tri_km_cpu.jpeg", dpi=300)



#################### Sweep G ####################

def kop(ts):  # order parameter
    nn, nt = ts.shape
    r = np.abs(np.sum(np.exp(1j * ts), axis=0) / nn)
    return r


def run(g):
    par = deepcopy(par_dict)
    par['G'] = g
    obj = KM_sde(par)
    data = obj.run()
    r = np.mean(kop(data['x']))
    return r


K = np.arange(0, 3, 0.1)
R = np.zeros(len(K))
fig, ax = plt.subplots(1, figsize=(6, 3))

with Pool(10) as p:
    R = np.array(list(tqdm(p.imap(run, K), total=len(K))))

plt.plot(K, R)
plt.xlabel("G")
plt.ylabel(r"$\langle r \rangle_{t}$")
plt.tight_layout()
plt.show()
