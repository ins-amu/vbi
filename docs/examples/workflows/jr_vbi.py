"""
Jansen-Rit whole-brain — end-to-end SBI with VBIInference
==========================================================

Uses the ``jansen_rit`` ModelSpec with the VBI simulator pipeline.
Reproduces the inference setup from ``jansen_rit_sde_numba_cde.ipynb``
using the modern ``VBIInference`` API with upstream ``sbi`` as the inference
backend.

The model
---------
Six-dimensional JR neural-mass model per node (Jansen & Rit 1995).
Stochastic (additive noise on y4, the excitatory interneuron velocity).
Long-range coupling enters the excitatory drive of each node via
G * W @ S(y1-y2), using CouplingSpec(kind="jr_sigmoidal").

Parameter to infer
------------------
* ``a_2`` — slow-excitatory synaptic contact probability; maps to C1 in
  classic JR notation via ``C1 = J * a_2``.
  This script uses ``G=1.5`` and ``a_2=1.0`` for the reference run.
  Prior equivalent to C1 ∈ [130, 300], provided J=135.
  ``G`` is fixed at 1.5.

Observable
----------
``y1 - y2`` (excitatory minus inhibitory dendritic potential, VOI indices
1 and 2) via raw monitor at integration-step resolution.  Spectral
features are averaged across all 84 cortical nodes, matching the notebook's
``average=True``.

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
* duration    = 2500 ms

Workflow
--------
1. Load 84-node Hagmann SC and build ``SimulationSpec`` (stochastic Heun).
2. Run one simulation at the true parameters for visual inspection.
3. Define prior and feature pipeline.
4. Build ``VBIInference`` with ``inference_backend="sbi"``.
5. Single-round: simulate → train MAF → build posterior.
6. Plot loss, pairplot.
7. Save checkpoint.

Expected runtime: ~8–15 min  (numba backend, 84 nodes, 1000 sims × 2500 ms).
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vbi.simulator import Simulator
from vbi.simulator.spec import (
    SimulationSpec,
    IntegratorSpec,
    CouplingSpec,
    MonitorSpec,
    prepare_connectivity,
)
from vbi.simulator.models.jansen_rit import jansen_rit
from vbi.inference import (
    VBIInference,
    BoxUniform,
    pairplot,
)

from vbi.feature_extraction import FeaturePruner
from helpers import (
    build_jr_spectral_pipeline,
    plot_jr_timeseries_psd,
    plot_feature_scatter,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reuse-simulations",
        action="store_true",
        help="Load saved theta/x from the checkpoint and skip the training sweep.",
    )
    return parser.parse_args()


ARGS = parse_args()

# ── Paths ──────────────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)
CKPT = OUT_DIR / "jr_vbi.npz"
CACHE_DIR = OUT_DIR / "jr_vbi_sim_cache"

print("=" * 62)
print("Jansen-Rit  —  VBIInference + sbi end-to-end pipeline")
print("=" * 62)

# ── 1 - Connectivity ───────────────────────────────────────────────────────────

W, D = prepare_connectivity(
    weights=DATA_DIR / "weights.txt",
    tract_lengths=DATA_DIR / "tract_lengths.txt",
    normalize=True,
)
D = np.zeros_like(D)
N_NODES = W.shape[0]
print(
    f"\n  Connectivity : {N_NODES} nodes   "
    f"W ∈ [{W.min():.4f}, {W.max():.4f}]   delays ignored"
)

# ── 2 - Simulation settings (match jansen_rit_sde_numba_cde.ipynb) ─────────────

DT = 0.1  # ms
FS_HZ = 1000.0 / DT
DURATION = 2500.0  # ms
T_CUT = 500.0  # ms
SIM_BACKEND = "numba"
INFERENCE_BACKEND = "sbi"

# True parameters for the reference run: G=1.5, a_2=1.0 (C1=135)
G_TRUE = 1.5
A2_TRUE = 1.0  # a_2 = C1 / J
J = 135.0

# Prior: C1 ∈ [130, 300] → a_2 ∈ [130/135, 300/135]
A2_LOW, A2_HIGH = 130.0 / J, 300.0 / J  # ≈ [0.963, 2.222]
MU = 0.24
NOISE_AMP = 0.1

N_SIM = 32  # number of simulations

# ── 3 - Integrator and base SimulationSpec ─────────────────────────────────────

integrator = IntegratorSpec(
    method="heun",
    dt=DT,
    stochastic=True,
    noise_nsig=np.array([NOISE_AMP]),
    noise_style="amplitude",
    noise_seed=42,
)

sim_spec = SimulationSpec(
    model=jansen_rit,
    integrator=integrator,
    coupling=CouplingSpec("jr_sigmoidal"),
    monitors=(MonitorSpec("raw"),),
    weights=W,
    tract_lengths=D,
    speed=4.0,
    node_params={
        "mu": np.full(N_NODES, MU),
        "G": G_TRUE,
    },
)

# ── 4 - Reference run at true parameters ───────────────────────────────────────

sim_spec_true = SimulationSpec(
    model=jansen_rit,
    integrator=integrator,
    coupling=CouplingSpec("jr_sigmoidal"),
    monitors=(MonitorSpec("raw"),),
    weights=W,
    tract_lengths=D,
    speed=4.0,
    node_params={
        "mu": np.full(N_NODES, MU),
        "a_2": np.full(N_NODES, A2_TRUE),
        "G": G_TRUE,
    },
)

print(
    f"\n  Reference simulation  G={G_TRUE}, a_2={A2_TRUE} " f"(C1={A2_TRUE * J:.0f})…",
    flush=True,
)
result_true = Simulator(sim_spec_true, backend=SIM_BACKEND).run(DURATION)
t_obs, ts_obs = result_true["raw"]  # (T, n_sv, N)

plot_jr_timeseries_psd(
    t_obs,
    ts_obs,
    FS_HZ,
    title=f"True dynamics  G={G_TRUE}, a_2={A2_TRUE} (C1={A2_TRUE*J:.0f})",
    out_path=OUT_DIR / "jr_vbi_timeseries.png",
    t_window_ms=(DURATION - 500, DURATION),
)
print(f"  Time-series → {OUT_DIR/'jr_vbi_timeseries.png'}")
# ── 5 - Feature pipeline ───────────────────────────────────────────────────────

pipeline = build_jr_spectral_pipeline(
    FS_HZ,
    t_cut=T_CUT,
    voi=(1, 2),
    signal="raw",
    # pruner=FeaturePruner(min_std=1e-4, max_corr=0.98),
)

labels, values = pipeline.extract(result_true)
print(f"\n  Feature dim    : {len(labels)}")
print(f"  Feature labels : {labels}")
print(f"  x_obs          : {np.round(values, 4)}")
x_obs = values

# ── 6 - Prior ──────────────────────────────────────────────────────────────────

prior = BoxUniform(
    low=np.array([A2_LOW]),
    high=np.array([A2_HIGH]),
    param_names=["a_2"],
)

# ── 7 - VBIInference ───────────────────────────────────────────────────────────

if ARGS.reuse_simulations:
    if not (CACHE_DIR / "metadata.json").exists():
        raise FileNotFoundError(
            f"Cannot reuse simulations because raw simulation cache does not exist: "
            f"{CACHE_DIR}"
        )
    inf = VBIInference(
        sim_spec=sim_spec,
        prior=prior,
        pipeline=pipeline,
        density_estimator="maf",
        sim_backend=SIM_BACKEND,
        backend="auto",
        inference_backend=INFERENCE_BACKEND,
        show_progress_bars=True,
    )
    print(f"\n  Re-extracting features from cached recordings in {CACHE_DIR} …", flush=True)
    theta, x = VBIInference.extract_from_cache(CACHE_DIR, pipeline)
    print(f"  theta {theta.shape}   x {x.shape}")

    if pipeline.pruner is not None and pipeline.pruner.kept_mask_ is not None:
        x_obs = pipeline.pruner.transform(x_obs)
        feature_labels = list(pipeline.pruner.kept_labels_)
        print(f"  x_obs (pruned with fitted mask) : {np.round(x_obs, 4)}")
    elif pipeline.pruner is not None:
        x, feature_labels = pipeline.pruner.fit_transform(x, labels)
        x_obs = pipeline.pruner.transform(x_obs)
        print(f"\n{pipeline.pruner.summary()}")
        print(f"  x_obs (pruned) : {np.round(x_obs, 4)}")
    else:
        feature_labels = labels

    inf.append_simulations(
        theta,
        x,
        param_names=prior._resolved_param_names,
        feature_labels=feature_labels,
    )
else:
    inf = VBIInference(
        sim_spec=sim_spec,
        prior=prior,
        pipeline=pipeline,
        density_estimator="maf",
        sim_backend=SIM_BACKEND,
        backend="auto",
        inference_backend=INFERENCE_BACKEND,
        show_progress_bars=True,
    )
    print(f"\n  {inf}")

    # ── 8 - Simulate + train + posterior ───────────────────────────────────────

    print(f"\n  Simulating {N_SIM} × {DURATION} ms …", flush=True)
    theta, x = inf.simulate(
        N_SIM,
        DURATION,
        seed=0,
        n_workers=8,
        cache_dir=CACHE_DIR,
        chunk_size=8,
    )
    print(f"  theta {theta.shape}   x {x.shape}")

    # Prune x_obs with the same mask fitted on the sweep (x already pruned inside simulate())
    if pipeline.pruner is not None and x_obs is not None:
        x_obs = pipeline.pruner.transform(x_obs)
        print(f"\n{pipeline.pruner.summary()}")
        print(f"  x_obs (pruned) : {np.round(x_obs, 4)}")

feature_labels = list(inf._feature_labels or [])
feature_scatter_path = OUT_DIR / "jr_vbi_feature_scatter.png"
plot_feature_scatter(theta, x, x_obs, feature_labels, A2_TRUE, feature_scatter_path)
print(f"  Feature scatter → {feature_scatter_path}")

print("  Training MAF …", flush=True)
estimator = inf.train(
    # training_batch_size=256,
    # learning_rate=5e-4,
    # stop_after_epochs=30,
    # max_num_epochs=500,
)
best_val_loss = getattr(estimator, "best_val_loss", None)
best_epoch = getattr(estimator, "best_epoch", None)
if best_val_loss is not None and best_epoch is not None:
    print(f"  Best val loss : {best_val_loss:.4f}  (epoch {best_epoch})")
else:
    print(f"  Trained estimator : {type(estimator).__name__}")

try:
    fig_loss = inf.plot_loss()
except RuntimeError as err:
    print(f"  Loss curve    : skipped ({err})")
else:
    fig_loss.savefig(OUT_DIR / "jr_vbi_loss.png", dpi=120, bbox_inches="tight")
    plt.close(fig_loss)
    print(f"  Loss curve    → {OUT_DIR/'jr_vbi_loss.png'}")

posterior = inf.build_posterior(estimator)
if x_obs is None:
    samples = None
    print("\n  Posterior diagnostics skipped because x_obs was not generated.")
else:
    samples = posterior.sample((5000,), x=x_obs[None], seed=0)

    print(f"\n  Posterior samples : {samples.shape}")
    print(f"  Fixed G           : {G_TRUE}")
    print(f"  True θ            : a_2={A2_TRUE} (C1={A2_TRUE*J:.0f})")
    print(f"  Posterior mean    : a_2={samples.mean(0)[0]:.4f}")
    print(f"  Posterior std     : a_2={samples.std(0)[0]:.4f}")

    fig_pair = pairplot(
        samples,
        labels=[r"$a_2$"],
        points=np.array([[A2_TRUE]]),
    )
    fig_pair.savefig(OUT_DIR / "jr_vbi_pairplot.png", dpi=120, bbox_inches="tight")
    plt.close(fig_pair)
    print(f"  Pairplot          → {OUT_DIR/'jr_vbi_pairplot.png'}")

# ── 9 - Save checkpoint ────────────────────────────────────────────────────────

inf.save(CKPT)
print(f"\n  Checkpoint → {CKPT}")

# ── Summary ────────────────────────────────────────────────────────────────────

print("\n" + "=" * 62)
print("Summary")
print("=" * 62)
print(f"  Model    : JansenRit  ({N_NODES} nodes, stochastic)")
print(f"  Backend  : {SIM_BACKEND} (sim) + {INFERENCE_BACKEND} (inference)")
print(f"  Monitor  : raw  dt={DT} ms  (fs={FS_HZ:.0f} Hz)")
print(f"  Noise    : amplitude={NOISE_AMP} on y4,  mu={MU}")
n_feat_final = x.shape[1]
print(f"  Features : {n_feat_final}-D spectral  (voi=y1-y2, avg across nodes)")
print(f"  N sims   : {theta.shape[0]}  ×  {DURATION} ms")
print(f"  True θ   : G={G_TRUE},  a_2={A2_TRUE}  (C1={A2_TRUE*J:.0f})")
if samples is not None:
    print(f"  Post mean: {np.round(samples.mean(0), 4)}")
print(f"  Outputs  : {OUT_DIR}/")
