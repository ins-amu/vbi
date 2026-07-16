import argparse
import os
from pathlib import Path
from time import perf_counter
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vbi.simulator import Simulator, Sweeper
from vbi.simulator.spec import (
    SimulationSpec,
    IntegratorSpec,
    CouplingSpec,
    MonitorSpec,
    Connectivity,
    SweepSpec,
)
from vbi.simulator.models.jansen_rit import jansen_rit
from vbi.inference import (
    VBIInference,
    BoxUniform,
    pairplot,
    simulate_for_vbi_sweep_cached,
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
    parser.add_argument(
        "--simulate-only",
        action="store_true",
        help=(
            "Run the cached simulation sweep, skip feature extraction/training, "
            "and exit. Useful for timing simulation plus cache writing."
        ),
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

conn = Connectivity.from_file(
    weights=DATA_DIR / "weights.txt",
    tract_lengths=None,  # delays ignored for JR
    normalize=True,
)
N_NODES = conn.n_nodes
print(
    f"\n  Connectivity : {N_NODES} nodes   "
    f"W ∈ [{conn.weights.min():.4f}, {conn.weights.max():.4f}]   delays ignored"
)

# ── 2 - Simulation settings (match jansen_rit_sde_numba_cde.ipynb) ─────────────

DT = 0.1  # ms
SUBSAMPLE_PERIOD = 1.0  # ms
FS_HZ = 1000.0 / DT
DURATION = 2500.0  # ms
T_CUT = 500.0  # ms
INTEGRATOR_BACKEND = "numba"
INFERENCE_BACKEND = "vbi"
if INTEGRATOR_BACKEND == "numba":
    os.environ.setdefault("VBI_NB_CACHE", str(OUT_DIR / "numba_cache"))

# True parameters for the reference run: G=1.5, a_2=1.0 (C1=135)
G_TRUE = 1.5
A2_TRUE = 1.5  # a_2 = C1 / J
J = 135.0

# Prior: C1 ∈ [130, 300] → a_2 ∈ [130/135, 300/135]
A2_LOW, A2_HIGH = 130.0 / J, 300.0 / J  # ≈ [0.963, 2.222]
MU = 0.24
NOISE_AMP = 0.1

N_SIM = 2  # number of simulations
N_WORKERS = 1

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
    monitors=(MonitorSpec("subsample", period=SUBSAMPLE_PERIOD),),
    connectivity=conn,
    node_params={
        "mu": np.full(N_NODES, MU),
        "G": G_TRUE,
    },
)
# ── 4 - Feature pipeline ───────────────────────────────────────────────────────

pipeline = build_jr_spectral_pipeline(
    FS_HZ,
    t_cut=T_CUT,
    voi=(1, 2),
    signal="raw",
    pruner=FeaturePruner(min_std=1e-4, max_corr=0.98),
)

labels = []
x_obs = None

# ── 6 - Prior ──────────────────────────────────────────────────────────────────

prior = BoxUniform(
    low=np.array([A2_LOW]),
    high=np.array([A2_HIGH]),
    param_names=["a_2"],
)

# ── 7 - VBIInference ───────────────────────────────────────────────────────────

inf = VBIInference(
    sim_spec=sim_spec,
    prior=prior,
    pipeline=pipeline,
    density_estimator="maf",
    integrator_backend=INTEGRATOR_BACKEND,
    estimator_backend="auto",
    inference_backend=INFERENCE_BACKEND,
    show_progress_bars=True,
)
print(f"\n  {inf}")

#     # ── 8 - Simulate + train + posterior ───────────────────────────────────────

#     print(f"\n  Simulating {N_SIM} × {DURATION} ms …", flush=True)
#     if ARGS.simulate_only:
#         theta, x, _, _ = simulate_for_vbi_sweep_cached(
#             sim_spec=sim_spec,
#             prior=prior,
#             pipeline=pipeline,
#             num_simulations=N_SIM,
#             duration=DURATION,
#             seed=0,
#             n_workers=8,
#             cache_dir=CACHE_DIR,
#             chunk_size=8,
#             integrator_backend=INTEGRATOR_BACKEND,
#             show_progress_bars=True,
#             extract_features_after=False,
#         )
#         print(f"  theta {theta.shape}   x {x.shape}  (features skipped)")
#         print("  Exiting after cached simulation sweep.", flush=True)
#         raise SystemExit(0)

#     theta, x = inf.simulate(
#         N_SIM,
#         DURATION,
#         seed=0,
#         n_workers=8,
#         cache_dir=CACHE_DIR,
#         chunk_size=8,
#     )
#     print(f"  theta {theta.shape}   x {x.shape}")

#     # Prune x_obs with the same mask fitted on the sweep (x already pruned inside simulate())
#     if pipeline.pruner is not None and x_obs is not None:
#         x_obs = pipeline.pruner.transform(x_obs)
#         print(f"\n{pipeline.pruner.summary()}")
#         print(f"  x_obs (pruned) : {np.round(x_obs, 4)}")

# feature_labels = list(inf._feature_labels or [])
# feature_scatter_path = OUT_DIR / "jr_vbi_feature_scatter.png"
# plot_feature_scatter(
#     theta,
#     x,
#     x_obs,
#     feature_labels=feature_labels,
#     true_params=A2_TRUE,
#     out_path=feature_scatter_path,
# )
# print(f"  Feature scatter → {feature_scatter_path}")

# print("  Training MAF …", flush=True)
# estimator = inf.train(
#     # training_batch_size=256,
#     # learning_rate=5e-4,
#     # stop_after_epochs=100,
#     # max_num_epochs=1000,
# )
# best_val_loss = getattr(estimator, "best_val_loss", None)
# best_epoch = getattr(estimator, "best_epoch", None)
# if best_val_loss is not None and best_epoch is not None:
#     print(f"  Best val loss : {best_val_loss:.4f}  (epoch {best_epoch})")
# else:
#     print(f"  Trained estimator : {type(estimator).__name__}")

# try:
#     fig_loss = inf.plot_loss()
# except RuntimeError as err:
#     print(f"  Loss curve    : skipped ({err})")
# else:
#     fig_loss.savefig(OUT_DIR / "jr_vbi_loss.png", dpi=120, bbox_inches="tight")
#     plt.close(fig_loss)
#     print(f"  Loss curve    → {OUT_DIR/'jr_vbi_loss.png'}")

# posterior = inf.build_posterior(estimator)
# if x_obs is None:
#     samples = None
#     print("\n  Posterior diagnostics skipped because x_obs was not generated.")
# else:
#     samples = posterior.sample((5000,), x=x_obs[None], seed=0)

#     print(f"\n  Posterior samples : {samples.shape}")
#     print(f"  Fixed G           : {G_TRUE}")
#     print(f"  True θ            : a_2={A2_TRUE} (C1={A2_TRUE*J:.0f})")
#     print(f"  Posterior mean    : a_2={samples.mean(0)[0]:.4f}")
#     print(f"  Posterior std     : a_2={samples.std(0)[0]:.4f}")

#     fig_pair = pairplot(
#         samples,
#         labels=[r"$a_2$"],
#         points=np.array([[A2_TRUE]]),
#     )
#     fig_pair.savefig(OUT_DIR / "jr_vbi_pairplot.png", dpi=120, bbox_inches="tight")
#     plt.close(fig_pair)
#     print(f"  Pairplot          → {OUT_DIR/'jr_vbi_pairplot.png'}")

# # ── 9 - Save checkpoint ────────────────────────────────────────────────────────

# inf.save(CKPT)
# print(f"\n  Checkpoint → {CKPT}")

# # ── Summary ────────────────────────────────────────────────────────────────────

# print("\n" + "=" * 62)
# print("Summary")
# print("=" * 62)
# print(f"  Model    : JansenRit  ({N_NODES} nodes, stochastic)")
# print(f"  Backend  : {INTEGRATOR_BACKEND} (sim) + {INFERENCE_BACKEND} (inference)")
# print(f"  Monitor  : raw  dt={DT} ms  (fs={FS_HZ:.0f} Hz)")
# print(f"  Noise    : amplitude={NOISE_AMP} on y4,  mu={MU}")
# n_feat_final = x.shape[1]
# print(f"  Features : {n_feat_final}-D spectral  (voi=y1-y2, avg across nodes)")
# print(f"  N sims   : {theta.shape[0]}  ×  {DURATION} ms")
# print(f"  True θ   : G={G_TRUE},  a_2={A2_TRUE}  (C1={A2_TRUE*J:.0f})")
# if samples is not None:
#     print(f"  Post mean: {np.round(samples.mean(0), 4)}")
# print(f"  Outputs  : {OUT_DIR}/")
