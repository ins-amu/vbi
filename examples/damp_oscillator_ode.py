import torch
import pickle
import numpy as np
from tqdm import tqdm
import sbi.utils as utils 
import matplotlib.pyplot as plt
from multiprocessing import Pool
from sbi.analysis import pairplot
from vbi.inference import Inference
from sklearn.preprocessing import StandardScaler
from vbi.models.cpp.damp_oscillator import DO_cpp

from vbi import report_cfg
from vbi import list_feature_extractor
from vbi import get_features_by_domain, get_features_by_given_names

seed = 2 
np.random.seed(seed)
torch.manual_seed(seed)

parameters = {
        "a": 0.1,
        "b": 0.05,
        "dt": 0.01,
        "t_start": 0,
        "method": "rk4",
        "t_end": 100.0,
        "t_transition": 20,
        "output": "output",
        "initial_state": [0.5, 1.0],
    }

ode = DO_cpp(parameters)
print(ode())

sol = ode.run()
t = sol["t"]
x = sol["x"]
plt.figure(figsize=(4, 3))
plt.plot(t, x[:, 0], label='$\\theta$')
plt.plot(t, x[:, 1], label='$\omega$')
plt.xlabel("t")
plt.ylabel("x")
plt.legend()
plt.tight_layout()
plt.savefig("output/damp_oscillator_ts.jpeg", dpi=300)

cfg = get_features_by_domain(domain="statistical")
cfg = get_features_by_given_names(cfg, names=['calc_std', 'calc_mean'])
report_cfg(cfg)

def wrapper(par, control, cfg, verbose=False):
    ode = DO_cpp(par)
    sol = ode.run(control)

    # extract features 
    fs = 1.0 / par['dt'] * 1000 # [Hz]
    stat_vec = list_feature_extractor(ts=[sol['x'].T], 
                                      fea_dict=cfg, 
                                      fs=fs, 
                                      n_workers=1, 
                                      verbose=verbose).values 
    return stat_vec[0]
        
def batch_run(par, control_list, cfg, n_workers=1):
    stat_vec = []
    with Pool(processes=n_workers) as pool:
        stat_vec = pool.starmap(wrapper, [(par, control, cfg) for control in control_list])

    return stat_vec


control = {"a": 0.11, "b": 0.06}
x_ = wrapper(parameters, control, cfg)
print(x_)

num_sim = 200
num_workers = 10
a_min, a_max = 0.0, 1.0
b_min, b_max = 0.0, 1.0
prior_min = [a_min, b_min]
prior_max = [a_max, b_max]
theta_true = {"a": 0.1, "b": 0.05}

prior = utils.torchutils.BoxUniform(
    low=torch.as_tensor(prior_min),
    high=torch.as_tensor(prior_max))


obj = Inference()
theta = obj.sample_prior(prior, num_sim)
theta_np = theta.numpy().astype(float)
control_list = [{'a': theta_np[i, 0], 'b': theta_np[i, 1]} for i in range(num_sim)]

stat_vec = batch_run(parameters, control_list, cfg)

scaler = StandardScaler()
stat_vec_st = scaler.fit_transform(np.array(stat_vec))
stat_vec_st = torch.tensor(stat_vec_st, dtype=torch.float32)
torch.save(theta, 'output/theta.pt')
torch.save(stat_vec_st, 'output/stat_vec_st.pt')

theta.shape, stat_vec_st.shape

posterior = obj.train(theta, stat_vec_st, prior, method="SNPE", density_estimator="maf")

with open('output/posterior.pkl', 'wb') as f:
    pickle.dump(posterior, f)

with open('output/posterior.pkl', 'rb') as f:
    posterior = pickle.load(f)

xo = wrapper(parameters, theta_true, cfg)
xo_st = scaler.transform(xo.reshape(1, -1))

samples = obj.sample_posterior(xo_st, 10000, posterior)
torch.save(samples, 'output/samples.pt')

limits = [[i, j] for i, j in zip(prior_min, prior_max)]
fig, ax = pairplot(
    samples,
    points=[list(theta_true.values())],
    figsize=(5, 5),
    limits=limits,
    labels=["a", "b"],
    offdiag='kde',
    diag='kde',
    points_colors="r",
    samples_colors="k",
    points_offdiag={'markersize': 10})
ax[0,0].tick_params(labelsize=14)
ax[0,0].margins(y=0)
plt.tight_layout()
fig.savefig("output/triangleplot.jpeg", dpi=300);