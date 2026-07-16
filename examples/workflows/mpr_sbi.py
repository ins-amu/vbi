"""
MPR model — end-to-end SBI with VBIInference
=============================================

Reproduces ``examples/inference/vbi_inference_mpr.ipynb`` as a
runnable Python script using the full ``VBIInference`` pipeline.

Workflow
--------
1. Load the 84-node Hagmann SC and build a ``SimulationSpec``.
2. Define a ``BoxUniform`` prior over G (global coupling) and η (excitability).
3. Define an FC feature pipeline.
4. **Round 1** – 300 simulations from the prior → train MAF → build posterior.
5. Plot training loss and posterior pairplot.
6. **Cache demo** – simulate with ``cache_dir``, re-extract with new features.
7. Save / load checkpoint.
8. **Round 2** – 200 simulations from the round-1 posterior (sequential SNPE).
9. Compare round-1 vs round-2 scatter plots.
10. ``from_config`` – drive the same workflow from a JSON config dict.

``VBIInference`` vs ``SNPE`` directly
--------------------------------------
Use ``VBIInference`` when the simulator is a VBI brain model (MPR, JR, WC …)
that runs through ``SimulationSpec`` + ``Sweeper`` + ``FeaturePipeline``.
For standalone ODE/SDE models (DampedOscillator, toy problems) use ``SNPE``
directly — see ``workflows/damp_oscillator.py``.

Expected runtime: ~5–10 min  (numba backend, 84 nodes, 500 sims total).
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vbi.simulator import Simulator
from vbi.simulator.spec import (
    SimulationSpec, IntegratorSpec, CouplingSpec, MonitorSpec,
    Connectivity,
)
from vbi.simulator.models.mpr import mpr
from vbi.feature_extraction import (
    FeaturePipeline, get_features_by_domain, get_features_by_given_names,
)
from vbi.inference import (
    VBIInference, BoxUniform,
    pairplot, plot_loss,
)

# ── Paths ─────────────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_DIR  = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

print("=" * 62)
print("MPR model  —  end-to-end SBI with VBIInference")
print("=" * 62)

# ── 1 - Connectivity ──────────────────────────────────────────────────────────

conn = Connectivity.from_file(
    weights       = DATA_DIR / "weights.txt",
    tract_lengths = DATA_DIR / "tract_lengths.txt",
    normalize     = True,
)
n_nodes = conn.n_nodes
print(f"\n  Connectivity : {n_nodes} nodes   "
      f"W ∈ [{conn.weights.min():.4f}, {conn.weights.max():.4f}]")

# Save as .npz for the from_config demo
conn.save(OUT_DIR / "sc_84.npz")

# ── 2 - SimulationSpec ────────────────────────────────────────────────────────

sim_spec = SimulationSpec(
    model         = mpr,
    integrator    = IntegratorSpec(method="heun", dt=0.1),
    coupling      = CouplingSpec("linear", a=1.0),
    monitors      = (MonitorSpec("tavg", period=1.0),),
    connectivity  = conn,
    node_params   = {"eta": np.full(n_nodes, -4.6)},
)

# ── 3 - Prior and feature pipeline ────────────────────────────────────────────

prior = BoxUniform(
    low         = np.array([0.5, -5.5]),
    high        = np.array([4.0, -3.0]),
    param_names = ["G", "eta"],
)

cfg_fc      = get_features_by_domain("connectivity")
cfg_fc      = get_features_by_given_names(cfg_fc, ["calc_fc"])
pipeline_fc = FeaturePipeline(cfg_fc, signal="tavg", t_cut=500.0)

# Quick single run to confirm feature shape
_result  = Simulator(sim_spec, backend="numpy").run(600.0)
_labels, _values = pipeline_fc.extract(_result)
print(f"  Feature dim  : {len(_labels)}  (FC upper triangle)")

# ── 4 - Build VBIInference ────────────────────────────────────────────────────

inf = VBIInference(
    sim_spec          = sim_spec,
    prior             = prior,
    pipeline          = pipeline_fc,
    density_estimator = "maf",
    integrator_backend = "numba",
    estimator_backend  = "auto",
    show_progress_bars = True,
)
print(f"\n  {inf}")

# ── 5 - Round 1: simulate + train + posterior ─────────────────────────────────

N_R1     = 300
DURATION = 2000.0   # ms

print(f"\n  Round 1: {N_R1} simulations × {DURATION} ms … ", flush=True)
theta_r1, x_r1 = inf.simulate(N_R1, DURATION, seed=0)
print(f"  theta {theta_r1.shape}   x {x_r1.shape}")

print("  Training MAF …", flush=True)
est_r1 = inf.train(
    training_batch_size = 128,
    stop_after_epochs   = 30,
    max_num_epochs      = 300,
)
print(f"  Best val loss : {est_r1.best_val_loss:.4f}  (epoch {est_r1.best_epoch})")

fig_loss = inf.plot_loss()
fig_loss.savefig(OUT_DIR / "mpr_r1_loss.png", dpi=120, bbox_inches="tight")
plt.close(fig_loss)
print(f"  Loss curve → {OUT_DIR/'mpr_r1_loss.png'}")

post_r1 = inf.build_posterior(est_r1)

# Use the first simulated feature vector as a proxy for x_obs
x_obs = x_r1[0]

samples_r1 = post_r1.sample((1000,), x=x_obs)
print(f"  Posterior samples : {samples_r1.shape}")

fig_pair_r1 = inf.pairplot(x_obs, num_samples=1000)
fig_pair_r1.savefig(OUT_DIR / "mpr_r1_pairplot.png", dpi=120, bbox_inches="tight")
plt.close(fig_pair_r1)
print(f"  Pairplot → {OUT_DIR/'mpr_r1_pairplot.png'}")

# ── 6 - Cache demo: simulate → store raw → re-extract features ────────────────

CACHE_DIR = OUT_DIR / "sim_cache_mpr"
print(f"\n  Cache demo: 200 sims → {CACHE_DIR} … ", flush=True)

inf_cache = VBIInference(
    sim_spec   = sim_spec, prior=prior, pipeline=pipeline_fc,
    integrator_backend="numba", estimator_backend="auto", show_progress_bars=False,
)
theta_c, x_fc = inf_cache.simulate(
    200, DURATION, seed=1,
    cache_dir  = CACHE_DIR,
    chunk_size = 100,
)
print(f"  Cached   : theta {theta_c.shape}  x_fc {x_fc.shape}")
print(f"  Files    : {[f.name for f in sorted(CACHE_DIR.iterdir())]}")

# Re-extract with statistical features — no re-simulation needed
cfg_stat   = get_features_by_domain("statistical")
cfg_stat   = get_features_by_given_names(cfg_stat, ["calc_mean", "calc_std"])
pipe_stat  = FeaturePipeline(cfg_stat, signal="tavg", t_cut=500.0)

theta_re, x_stat = VBIInference.extract_from_cache(CACHE_DIR, pipe_stat)
print(f"  Re-extracted (stat features): theta {theta_re.shape}  x {x_stat.shape}")
assert x_fc.shape[1] != x_stat.shape[1], "Feature dims should differ"
print("  ✓ Feature dims differ after re-extraction with new pipeline")

# ── 7 - Save / load checkpoint ────────────────────────────────────────────────

ckpt = OUT_DIR / "mpr_r1.npz"
inf.save(ckpt)
print(f"\n  Checkpoint saved → {ckpt}")

inf_loaded = VBIInference.load(
    ckpt,
    sim_spec = sim_spec,
    pipeline = pipeline_fc,
    prior    = prior,
)
print(f"  Loaded   : {inf_loaded}")

post_loaded = inf_loaded.build_posterior()
s_loaded    = post_loaded.sample((200,), x=x_obs)
assert s_loaded.shape == (200, 2)
print("  ✓ Post-load posterior samples: OK")

# ── 8 - Round 2: posterior-focused simulation (sequential SNPE) ───────────────

N_R2 = 200
print(f"\n  Round 2: {N_R2} sims from round-1 posterior … ", flush=True)

theta_r2, x_r2 = inf.simulate(
    N_R2, DURATION,
    proposal = post_r1,
    x_obs    = x_obs,
    seed     = 2,
)
print(f"  theta {theta_r2.shape}   x {x_r2.shape}")
print(f"  Total sims so far : {inf._snpe.n_simulations}")

est_r2 = inf.train(
    training_batch_size = 128,
    stop_after_epochs   = 30,
    max_num_epochs      = 300,
)
print(f"  Best val loss (r2): {est_r2.best_val_loss:.4f}  (epoch {est_r2.best_epoch})")

post_r2    = inf.build_posterior(est_r2)
samples_r2 = post_r2.sample((1000,), x=x_obs)

# Side-by-side scatter: round-1 vs round-2
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, samp, label in zip(
    axes,
    [samples_r1, samples_r2],
    [f"Round 1  (n={N_R1} prior sims)",
     f"Round 2  (+{N_R2} posterior sims)"],
):
    ax.scatter(samp[:, 0], samp[:, 1], s=3, alpha=0.35, color="steelblue")
    ax.set_xlabel("G"); ax.set_ylabel("η")
    ax.set_title(label, fontsize=10)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

fig.suptitle("Round-1 vs Round-2 posterior samples", fontsize=11)
fig.tight_layout()
fig.savefig(OUT_DIR / "mpr_r1_vs_r2.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"  Comparison plot → {OUT_DIR/'mpr_r1_vs_r2.png'}")

# ── 9 - from_config: same workflow driven by a config dict ────────────────────

import json

config = {
    "sim": {
        "model": "mpr",
        "connectivity": str(OUT_DIR / "sc_84.npz"),
        "node_params": {"eta": -4.6},
        "dt": 0.1,
        "method": "heun",
        "monitors": [{"kind": "tavg", "period": 1.0}],
        "coupling": {"kind": "linear", "a": 1.0},
        "speed": 4.0,
    },
    "prior": {
        "type": "BoxUniform",
        "low":  [0.5, -5.5],
        "high": [4.0, -3.0],
        "param_names": ["G", "eta"],
    },
    "pipeline": {
        "features": ["calc_fc"],
        "signal":   "tavg",
        "t_cut":    500.0,
    },
    "inference": {
        "density_estimator": "maf",
        "integrator_backend": "numba",
        "estimator_backend":  "auto",
        "training": {
            "training_batch_size": 128,
            "stop_after_epochs":   30,
            "max_num_epochs":      300,
        },
    },
}

cfg_path = OUT_DIR / "mpr_config.json"
with open(cfg_path, "w") as f:
    json.dump(config, f, indent=2)

inf_cfg = VBIInference.from_config(config)
print(f"\n  from_config: {inf_cfg}")
print(f"  Default train kwargs: {inf_cfg._default_train_kwargs}")
print(f"  Config saved → {cfg_path}")

# ── Summary ───────────────────────────────────────────────────────────────────

print("\n" + "=" * 62)
print("Summary")
print("=" * 62)
print(f"  Round 1 sims : {N_R1}  ({DURATION} ms each)")
print(f"  Round 2 sims : {N_R2}  (proposal = round-1 posterior)")
print(f"  Density est. : MAF  |  features: FC ({x_r1.shape[1]}-D)")
print(f"  Checkpoint   : {ckpt}")
print(f"  Outputs      : {OUT_DIR}/")
print("  Files generated:")
for f in sorted(OUT_DIR.glob("mpr_*")):
    print(f"    {f.name}")
