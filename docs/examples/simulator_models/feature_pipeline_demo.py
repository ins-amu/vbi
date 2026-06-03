"""
Feature extraction pipeline demo - MPR model.

Demonstrates three use cases:

1. Single run  - extract statistical features from one simulation.
2. 1-D sweep   - vary the coupling weight cr in [0, 1] and collect features into a DataFrame.
3. 2-D sweep   - joint grid over (cr, eta); saves a mean heat-map.

Global coupling strength is set via CouplingSpec.a (same convention as TVB).
Per-variable coupling weights cr and cv live on the model.

Run:
    python feature_pipeline_demo.py                 # all three, numpy backend
    python feature_pipeline_demo.py --backend numba # use Numba for the sweep
    python feature_pipeline_demo.py --no-plot       # skip matplotlib output
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.dont_write_bytecode = True

import numpy as np

from helpers import (
    complete_graph_weights,
    constant_tract_lengths,
    ensure_repo_on_path,
    homogeneous_node_params,
    quiet_optional_imports,
)

ensure_repo_on_path(__file__)

with quiet_optional_imports():
    from vbi.simulator import Simulator, Sweeper
    from vbi.simulator.models.mpr import mpr
    from vbi.simulator.spec import (
        Connectivity,
        CouplingSpec,
        IntegratorSpec,
        MonitorSpec,
        SimulationSpec,
    )
    from vbi.simulator.spec.sweep import SweepSpec
    from vbi.feature_extraction import (
        FeaturePipeline,
        get_features_by_domain,
        get_features_by_given_names,
        update_cfg,
    )


# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------

N_NODES      = 8
DT           = 0.01          # ms
TRACT_LENGTH = 4.0           # mm
SPEED        = 4.0           # mm/ms
COUPLING_A   = 0.2           # global coupling strength via CouplingSpec.a (TVB convention)
DURATION     = 2_000.0       # ms  (single run)
SWEEP_DUR    = 1_000.0       # ms  (sweeps - shorter for speed)
T_CUT        = 500.0         # ms  burn-in to discard


# ---------------------------------------------------------------------------
# Build a reusable SimulationSpec
# ---------------------------------------------------------------------------

def make_spec(a: float = COUPLING_A) -> SimulationSpec:
    weights       = complete_graph_weights(N_NODES)
    tract_lengths = constant_tract_lengths(weights, TRACT_LENGTH)
    node_params   = homogeneous_node_params(
        n_nodes=N_NODES,
        params={
            "tau":   1.0,
            "I":     2.0,
            "Delta": 0.7,
            "J":     14.5,
            "eta":  -4.6,
            "Gamma": 0.0,
            "cr":    1.0,
            "cv":    0.0,
        },
    )
    return SimulationSpec(
        model=mpr,
        integrator=IntegratorSpec(method="heun", dt=DT),
        coupling=CouplingSpec(kind="linear", a=a, b=0.0),
        monitors=(MonitorSpec(kind="tavg", period=1.0),),
        connectivity=Connectivity(weights, tract_lengths, speed=SPEED),
        node_params=node_params,
    )


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------

def stat_pipeline(t_cut: float = T_CUT) -> FeaturePipeline:
    """Statistical summary features (mean, std) for every node."""
    cfg = get_features_by_domain("statistical")
    cfg = get_features_by_given_names(cfg, ["calc_mean", "calc_std"])
    return FeaturePipeline(cfg, signal="tavg", t_cut=t_cut)


# ---------------------------------------------------------------------------
# Demo 1: single run
# ---------------------------------------------------------------------------

def demo_single_run() -> None:
    print("\n=== Demo 1: single-run feature extraction ===")

    spec     = make_spec()
    result   = Simulator(spec, backend="numpy").run(DURATION)
    pipeline = stat_pipeline()

    labels, values = pipeline.extract(result)
    print(f"\n  statistical features  (shape {values.shape})")
    for lbl, val in zip(labels, values):
        print(f"    {lbl:<35s}  {val:.6g}")

    print("\n  as DataFrame:")
    df = pipeline.extract_df(result)
    print(df.to_string(index=False))


# ---------------------------------------------------------------------------
# Demo 2: 1-D sweep over cr
# ---------------------------------------------------------------------------

def demo_1d_sweep(backend: str = "numpy") -> None:
    print(f"\n=== Demo 2: 1-D sweep over cr  (backend={backend}) ===")

    cr_values  = np.linspace(0.0, 1.0, 10)
    spec       = make_spec()
    pipeline   = stat_pipeline()
    sweep_spec = SweepSpec(
        params={"cr": cr_values},
        pipeline=pipeline,
    )

    df = Sweeper(spec, sweep_spec, backend=backend).run_df(SWEEP_DUR)
    print(f"  DataFrame shape: {df.shape}  columns: {list(df.columns)}")
    print(df.to_string(index=False))


# ---------------------------------------------------------------------------
# Demo 3: 2-D sweep over (cr, eta) - saves a mean-FC heat-map
# ---------------------------------------------------------------------------

def demo_2d_sweep(backend: str = "numpy", output: Path | None = None) -> None:
    print(f"\n=== Demo 3: 2-D sweep over (cr, eta)  (backend={backend}) ===")

    cr_range  = np.linspace(0.0, 1.0, 5)
    eta_range = np.linspace(-6.0, -3.6, 5)

    spec       = make_spec()
    pipeline   = stat_pipeline()
    sweep_spec = SweepSpec(
        params={"cr": cr_range, "eta": eta_range},
        pipeline=pipeline,
    )

    labels, values = Sweeper(spec, sweep_spec, backend=backend).run(SWEEP_DUR)
    print(f"  labels : {labels}")
    print(f"  values : shape {values.shape}")

    mean_col = next((i for i, l in enumerate(labels) if "mean" in l.lower()), None)
    if mean_col is None or output is None:
        return

    feat_vals = values[:, mean_col].reshape(len(cr_range), len(eta_range))
    _save_heatmap(feat_vals, cr_range, eta_range, labels[mean_col], output)
    print(f"  saved heat-map: {output}")


def _save_heatmap(
    data: np.ndarray,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    feature_name: str,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(
        data,
        origin="lower",
        aspect="auto",
        extent=[y_vals[0], y_vals[-1], x_vals[0], x_vals[-1]],
    )
    fig.colorbar(im, ax=ax, label=feature_name)
    ax.set_xlabel("eta")
    ax.set_ylabel("cr")
    ax.set_title(f"MPR: {feature_name} over (cr, eta)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Bonus: update_cfg demo
# ---------------------------------------------------------------------------

def demo_update_cfg() -> None:
    """Show how to tune per-feature parameters via update_cfg."""
    print("\n=== Bonus: update_cfg - tune per-feature parameters ===")

    cfg = get_features_by_domain("statistical")
    cfg = get_features_by_given_names(cfg, ["calc_mean"])
    cfg = update_cfg(cfg, "calc_mean", {"indices": None})

    pipeline = FeaturePipeline(cfg, signal="tavg", t_cut=T_CUT)
    spec     = make_spec()
    result   = Simulator(spec, backend="numpy").run(DURATION)
    labels, values = pipeline.extract(result)
    print(f"  labels : {labels}")
    print(f"  values : {values}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=("numpy", "numba"),
        default="numba",
        help="simulator backend for sweep demos (default: numba)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="skip saving the heat-map figure",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("outputs") / "feature_pipeline_heatmap.png",
        help="output path for the 2-D sweep heat-map",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    demo_single_run()
    demo_1d_sweep(backend=args.backend)
    demo_2d_sweep(
        backend=args.backend,
        output=None if args.no_plot else args.output,
    )
    demo_update_cfg()

    print("\nDone.")


if __name__ == "__main__":
    main()
