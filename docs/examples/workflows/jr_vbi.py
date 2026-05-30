"""
Jansen-Rit whole-brain — end-to-end SBI with VBIInference
==========================================================

Uses the ``jansen_rit`` ModelSpec with the VBI simulator pipeline.
Reproduces the inference setup from ``jansen_rit_sde_numba_cde.ipynb``
using the modern ``VBIInference`` API.

The model
---------
Six-dimensional JR neural-mass model per node (Jansen & Rit 1995).
Stochastic (additive noise on y4, the excitatory interneuron velocity).
Long-range coupling enters the excitatory drive of each node via
G * S(W @ (y1-y2)), using CouplingSpec(kind="difference").

Parameters to infer
-------------------
* ``G``   — global coupling strength (scales all white-matter weights).
* ``a_2`` — slow-excitatory synaptic contact probability; maps to C1 in
  classic JR notation via ``C1 = J * a_2`` where ``J = 135``.
  Notebook reference: ``TRUE_THETA = [G=1.5, C1=135.0]``, i.e. ``a_2=1.0``.
  Prior equivalent to C1 ∈ [130, 300].

Observable
----------
``y1`` (excitatory dendritic potential, VOI index 1) via temporal-average
monitor at 1 ms resolution (1 kHz).  Spectral features are averaged
across all 84 cortical nodes, matching the notebook's ``average=True``.

Features
--------
Spectral domain (Welch PSD): ``spectrum_stats``, ``spectrum_auc``,
``spectrum_moments``.  Built by ``helpers.build_jr_spectral_pipeline``.

Simulation parameters (matching jansen_rit_sde_numba_cde.ipynb)
----------------------------------------------------------------
* dt          = 0.05 ms
* noise_amp   = 0.1  (noise_nsig on y4)
* mu          = 0.24
* t_cut       = 500  ms
* duration    = 2501 ms

Workflow
--------
1. Load 84-node Hagmann SC and build ``SimulationSpec`` (stochastic Heun).
2. Run one simulation at the true parameters for visual inspection.
3. Define prior and feature pipeline.
4. Build ``VBIInference``.
5. Single-round: simulate → train MAF → build posterior.
6. Plot loss, pairplot.
7. Save checkpoint.

Expected runtime: ~8–15 min  (numba backend, 84 nodes, 1000 sims × 2501 ms).
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vbi.simulator import Simulator
from vbi.simulator.spec import (
    SimulationSpec, IntegratorSpec, CouplingSpec, MonitorSpec,
    prepare_connectivity,
)
from vbi.simulator.models.jansen_rit import jansen_rit
from vbi.inference import (
    VBIInference, BoxUniform,
    pairplot, plot_loss,
)

from helpers import build_jr_spectral_pipeline, plot_jr_timeseries_psd

# ── Paths ──────────────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_DIR  = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

print("=" * 62)
print("Jansen-Rit  —  VBIInference end-to-end pipeline")
print("=" * 62)

# ── 1 - Connectivity ───────────────────────────────────────────────────────────

W, D = prepare_connectivity(
    weights       = DATA_DIR / "weights.txt",
    tract_lengths = DATA_DIR / "tract_lengths.txt",
    normalize     = True,
)
N_NODES = W.shape[0]
print(f"\n  Connectivity : {N_NODES} nodes   "
      f"W ∈ [{W.min():.4f}, {W.max():.4f}]")

# ── 2 - Simulation settings (match jansen_rit_sde_numba_cde.ipynb) ─────────────

DT          = 0.05     # ms
PERIOD      = 1.0      # ms  (tavg → fs = 1000 Hz)
FS_HZ       = 1000.0 / PERIOD
DURATION    = 2500.0   # ms
T_CUT       = 0.0    # ms
SIM_BACKEND = "numba"

# True parameters: G=1.5, C1=135 (= J * a_2 = 135 * 1.0)
G_TRUE  = 0.1
A2_TRUE = 1.0          # a_2 = C1 / J = 135 / 135

# Prior: G ∈ [0, 5], C1 ∈ [130, 300] → a_2 ∈ [130/135, 300/135]
G_LOW,  G_HIGH  = 0.0, 5.0
A2_LOW, A2_HIGH = 130.0 / 135.0, 300.0 / 135.0   # ≈ [0.963, 2.222]
MU          = 0.24
NOISE_AMP   = 0.1      

# ── 3 - Integrator and base SimulationSpec ─────────────────────────────────────

integrator = IntegratorSpec(
    method      = "heun",
    dt          = DT,
    stochastic  = True,
    noise_nsig  = np.array([NOISE_AMP]),
    noise_style = "amplitude",
    noise_seed  = 42,
)

sim_spec = SimulationSpec(
    model         = jansen_rit,
    integrator    = integrator,
    coupling      = CouplingSpec("difference"),
    monitors      = (MonitorSpec("tavg", period=PERIOD),),
    weights       = W,
    tract_lengths = D,
    speed         = 4.0,
    node_params   = {"mu": np.full(N_NODES, MU)},
)

# ── 4 - Reference run at true parameters ───────────────────────────────────────

sim_spec_true = SimulationSpec(
    model         = jansen_rit,
    integrator    = integrator,
    coupling      = CouplingSpec("difference"),
    monitors      = (MonitorSpec("tavg", period=PERIOD),),
    weights       = W,
    tract_lengths = D,
    speed         = 4.0,
    node_params   = {
        "mu":  np.full(N_NODES, MU),
        "a_2": np.full(N_NODES, A2_TRUE),
        "G":   G_TRUE,
    },
)

print(f"\n  Reference simulation  G={G_TRUE}, a_2={A2_TRUE} "
      f"(C1={A2_TRUE * 135:.0f})…", flush=True)
result_true = Simulator(sim_spec_true, backend=SIM_BACKEND).run(DURATION)
t_obs, ts_obs = result_true["tavg"]   # (T, n_sv, N)

plot_jr_timeseries_psd(
    t_obs, ts_obs, FS_HZ,
    title       = f"True dynamics  G={G_TRUE}, a_2={A2_TRUE} (C1={A2_TRUE*135:.0f})",
    out_path    = OUT_DIR / "jr_vbi_timeseries.png",
    t_window_ms = (1500, 2501),
)
print(f"  Time-series → {OUT_DIR/'jr_vbi_timeseries.png'}")
exit(0)
# ── 5 - Feature pipeline ───────────────────────────────────────────────────────

pipeline = build_jr_spectral_pipeline(FS_HZ, t_cut=T_CUT, voi=1)

labels, values = pipeline.extract(result_true)
print(f"\n  Feature dim    : {len(labels)}")
print(f"  Feature labels : {labels}")
print(f"  x_obs          : {np.round(values, 4)}")

x_obs = values

# ── 6 - Prior ──────────────────────────────────────────────────────────────────

prior = BoxUniform(
    low         = np.array([G_LOW,  A2_LOW]),
    high        = np.array([G_HIGH, A2_HIGH]),
    param_names = ["G", "a_2"],
)

# ── 7 - VBIInference ───────────────────────────────────────────────────────────

inf = VBIInference(
    sim_spec           = sim_spec,
    prior              = prior,
    pipeline           = pipeline,
    density_estimator  = "maf",
    sim_backend        = SIM_BACKEND,
    backend            = "auto",
    show_progress_bars = True,
)
print(f"\n  {inf}")

# ── 8 - Simulate + train + posterior ───────────────────────────────────────────

N_SIM = 1000

print(f"\n  Simulating {N_SIM} × {DURATION} ms …", flush=True)
theta, x = inf.simulate(N_SIM, DURATION, seed=0)
print(f"  theta {theta.shape}   x {x.shape}")

print("  Training MAF …", flush=True)
estimator = inf.train(
    training_batch_size = 256,
    learning_rate       = 5e-4,
    stop_after_epochs   = 30,
    max_num_epochs      = 500,
)
print(f"  Best val loss : {estimator.best_val_loss:.4f}  (epoch {estimator.best_epoch})")

fig_loss = inf.plot_loss()
fig_loss.savefig(OUT_DIR / "jr_vbi_loss.png", dpi=120, bbox_inches="tight")
plt.close(fig_loss)
print(f"  Loss curve    → {OUT_DIR/'jr_vbi_loss.png'}")

posterior = inf.build_posterior(estimator)
samples   = posterior.sample((5000,), x=x_obs[None], seed=0)

print(f"\n  Posterior samples : {samples.shape}")
print(f"  True θ            : G={G_TRUE}, a_2={A2_TRUE} (C1={A2_TRUE*135:.0f})")
print(f"  Posterior mean    : {np.round(samples.mean(0), 4)}")
print(f"  Posterior std     : {np.round(samples.std(0),  4)}")

fig_pair = pairplot(
    samples,
    labels = ["G", r"$a_2$  (C1=135·$a_2$)"],
    points = np.array([[G_TRUE, A2_TRUE]]),
)
fig_pair.savefig(OUT_DIR / "jr_vbi_pairplot.png", dpi=120, bbox_inches="tight")
plt.close(fig_pair)
print(f"  Pairplot          → {OUT_DIR/'jr_vbi_pairplot.png'}")

# ── 9 - Save checkpoint ────────────────────────────────────────────────────────

ckpt = OUT_DIR / "jr_vbi.npz"
inf.save(ckpt)
print(f"\n  Checkpoint → {ckpt}")

# ── Summary ────────────────────────────────────────────────────────────────────

print("\n" + "=" * 62)
print("Summary")
print("=" * 62)
print(f"  Model    : JansenRit  ({N_NODES} nodes, stochastic)")
print(f"  Backend  : {SIM_BACKEND} (sim) + auto (inference)")
print(f"  Monitor  : tavg  period={PERIOD} ms  (fs={FS_HZ:.0f} Hz)")
print(f"  Noise    : amplitude={NOISE_AMP} on y4,  mu={MU}")
print(f"  Features : {len(labels)}-D spectral  (voi=y1, avg across nodes)")
print(f"  N sims   : {N_SIM}  ×  {DURATION} ms")
print(f"  True θ   : G={G_TRUE},  a_2={A2_TRUE}  (C1={A2_TRUE*135:.0f})")
print(f"  Post mean: {np.round(samples.mean(0), 4)}")
print(f"  Outputs  : {OUT_DIR}/")
