"""
Feature Extraction Pipeline Demo
===================================

Demonstrates the feature-extraction pipeline on the MPR model:

1. Single run - extract statistical features from one simulation.
2. 1-D sweep - vary the coupling weight ``cr`` in [0, 1] and collect
   features into a DataFrame.
3. 2-D sweep - joint grid over (``cr``, ``eta``); saves a mean heat-map.

Global coupling strength is set via ``CouplingSpec.a`` (same convention as
TVB); per-variable coupling weights ``cr``/``cv`` live on the model.

Run
---
::

    python feature_pipeline_demo.py
"""

# %%
# Setup
# -----

from __future__ import annotations

from pathlib import Path
import sys

sys.dont_write_bytecode = True

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    _SCRIPT_PATH = Path(__file__)
except NameError:
    # sphinx-gallery execs this file without setting __file__; it already
    # chdirs into the script's own directory first, so cwd is equivalent.
    _SCRIPT_PATH = Path.cwd() / "feature_pipeline_demo.py"

# Prefer the vbi living in this checkout over any other version already
# installed (e.g. a different vbi checkout installed editable elsewhere).
# Downloaded standalone copies of this script should `pip install vbi`
# instead - see the first notebook cell.
_repo_root = _SCRIPT_PATH.resolve().parents[2]
sys.path.insert(0, str(_repo_root))

from vbi.simulator import Simulator, Sweeper
from vbi.simulator.models.mpr import mpr
from vbi.simulator.spec import (
    Connectivity, CouplingSpec, IntegratorSpec, MonitorSpec, SimulationSpec,
)
from vbi.simulator.spec.sweep import SweepSpec
from vbi.feature_extraction import (
    FeaturePipeline, get_features_by_domain, get_features_by_given_names, update_cfg,
)

N_NODES = 8
DT = 0.01           # ms
TRACT_LENGTH = 4.0  # mm
SPEED = 4.0         # mm/ms
COUPLING_A = 0.2    # global coupling strength via CouplingSpec.a (TVB convention)
DURATION = 2_000.0  # ms (single run)
SWEEP_DUR = 1_000.0  # ms (sweeps - shorter for speed)
T_CUT = 500.0       # ms burn-in to discard
BACKEND = "numba"


def complete_graph_weights(n_nodes: int) -> np.ndarray:
    """Return a dense complete-graph weight matrix with zero diagonal."""
    weights = np.ones((n_nodes, n_nodes), dtype=np.float64)
    np.fill_diagonal(weights, 0.0)
    return weights


# %%
# VBI simulator
# -------------
# Every MPR parameter besides ``I`` uses the model's own default.

def make_spec(a: float = COUPLING_A) -> SimulationSpec:
    weights = complete_graph_weights(N_NODES)
    tract_lengths = np.full_like(weights, TRACT_LENGTH)
    np.fill_diagonal(tract_lengths, 0.0)

    return SimulationSpec(
        model=mpr,
        integrator=IntegratorSpec(method="heun", dt=DT),
        coupling=CouplingSpec(kind="linear", a=a, b=0.0),
        monitors=(MonitorSpec(kind="tavg", period=1.0),),
        connectivity=Connectivity(weights, tract_lengths, speed=SPEED),
        node_params={"I": 2.0},
    )


def stat_pipeline(t_cut: float = T_CUT) -> FeaturePipeline:
    """Statistical summary features (mean, std) for every node."""
    cfg = get_features_by_domain("statistical")
    cfg = get_features_by_given_names(cfg, ["calc_mean", "calc_std"])
    return FeaturePipeline(cfg, signal="tavg", t_cut=t_cut)


# %%
# Demo 1: single run
# -------------------

def demo_single_run() -> None:
    print("\n=== Demo 1: single-run feature extraction ===")

    spec = make_spec()
    result = Simulator(spec, backend="numpy").run(DURATION)
    pipeline = stat_pipeline()

    labels, values = pipeline.extract(result)
    print(f"\n  statistical features  (shape {values.shape})")
    for lbl, val in zip(labels, values):
        print(f"    {lbl:<35s}  {val:.6g}")

    print("\n  as DataFrame:")
    print(pipeline.extract_df(result).to_string(index=False))


# %%
# Demo 2: 1-D sweep over cr
# ---------------------------

def demo_1d_sweep() -> None:
    print(f"\n=== Demo 2: 1-D sweep over cr  (backend={BACKEND}) ===")

    cr_values = np.linspace(0.0, 1.0, 10)
    sweep_spec = SweepSpec(params={"cr": cr_values}, pipeline=stat_pipeline())

    df = Sweeper(make_spec(), sweep_spec, backend=BACKEND).run_df(SWEEP_DUR)
    print(f"  DataFrame shape: {df.shape}  columns: {list(df.columns)}")
    print(df.to_string(index=False))


# %%
# Demo 3: 2-D sweep over (cr, eta)
# -----------------------------------
# Saves a mean-feature heat-map over the joint grid.

def demo_2d_sweep(out_path: Path) -> None:
    print(f"\n=== Demo 3: 2-D sweep over (cr, eta)  (backend={BACKEND}) ===")

    cr_range = np.linspace(0.0, 1.0, 5)
    eta_range = np.linspace(-6.0, -3.6, 5)
    sweep_spec = SweepSpec(params={"cr": cr_range, "eta": eta_range}, pipeline=stat_pipeline())

    labels, values = Sweeper(make_spec(), sweep_spec, backend=BACKEND).run(SWEEP_DUR)
    print(f"  labels : {labels}")
    print(f"  values : shape {values.shape}")

    mean_col = next(i for i, l in enumerate(labels) if "mean" in l.lower())
    feat_vals = values[:, mean_col].reshape(len(cr_range), len(eta_range))

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(feat_vals, origin="lower", aspect="auto",
                   extent=[eta_range[0], eta_range[-1], cr_range[0], cr_range[-1]])
    fig.colorbar(im, ax=ax, label=labels[mean_col])
    ax.set_xlabel("eta")
    ax.set_ylabel("cr")
    ax.set_title(f"MPR: {labels[mean_col]} over (cr, eta)")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    print(f"  saved heat-map: {out_path}")


# %%
# Bonus: update_cfg
# ------------------
# Tune per-feature parameters via ``update_cfg``.

def demo_update_cfg() -> None:
    print("\n=== Bonus: update_cfg - tune per-feature parameters ===")

    cfg = get_features_by_domain("statistical")
    cfg = get_features_by_given_names(cfg, ["calc_mean"])
    cfg = update_cfg(cfg, "calc_mean", {"indices": None})

    pipeline = FeaturePipeline(cfg, signal="tavg", t_cut=T_CUT)
    result = Simulator(make_spec(), backend="numpy").run(DURATION)
    labels, values = pipeline.extract(result)
    print(f"  labels : {labels}")
    print(f"  values : {values}")


# %%
# Run the demo
# -------------

def main() -> None:
    demo_single_run()
    demo_1d_sweep()
    demo_2d_sweep(_SCRIPT_PATH.with_name("outputs") / "feature_pipeline_heatmap.png")
    demo_update_cfg()
    print("\nDone.")


if __name__ == "__main__":
    main()
