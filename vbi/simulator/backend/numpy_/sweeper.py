from __future__ import annotations
import copy
import numpy as np

from vbi.simulator.spec.simulation import SimulationSpec
from vbi.simulator.spec.sweep import SweepSpec
from .simulator import NumpySimulator


def _patch_spec(base: SimulationSpec, param_names: list[str],
                theta: np.ndarray) -> SimulationSpec:
    """Return a SimulationSpec with swept parameters overridden."""
    new_node_params = dict(base.node_params)
    new_integrator = base.integrator

    # We need to patch model default_params — we do this via node_params
    # so the simulator's _build_params picks them up (node_params override defaults).
    for name, val in zip(param_names, theta):
        new_node_params[name] = float(val)

    patched = SimulationSpec(
        model=base.model,
        integrator=new_integrator,
        coupling=base.coupling,
        monitors=base.monitors,
        weights=base.weights,
        tract_lengths=base.tract_lengths,
        speed=base.speed,
        node_params=new_node_params,
    )
    return patched


class NumpySweeper:
    """
    Reference sweep backend — sequential Python loop over parameter sets.
    Not optimised; used for validation of faster backends.
    """

    def __init__(self, spec: SimulationSpec, sweep_spec: SweepSpec):
        self.spec = spec
        self.sweep = sweep_spec

    def run(self, duration: float) -> dict | tuple:
        """
        Run all parameter sets sequentially.

        Returns
        -------
        If sweep_spec.pipeline is None:
            dict mapping monitor kind -> list of (t, data) per run.
        If pipeline is set:
            (labels, values) where values shape is (n_samples, n_features+n_params).
        """
        param_names = self.sweep._param_names_list
        param_sets = self.sweep.param_sets       # (n_samples, n_params)
        n = param_sets.shape[0]
        pipeline = self.sweep.pipeline

        if pipeline is None:
            # Return raw monitor output — list per run
            all_results: list[dict] = []
            for i in range(n):
                patched = _patch_spec(self.spec, param_names, param_sets[i])
                sim = NumpySimulator()
                sim.build(patched)
                all_results.append(sim.run(duration))
            return all_results

        # Pipeline mode: extract features per run; accumulate into arrays
        labels_set = False
        feat_labels: list[str] = []
        rows: list[np.ndarray] = []

        for i in range(n):
            patched = _patch_spec(self.spec, param_names, param_sets[i])
            sim = NumpySimulator()
            sim.build(patched)
            result = sim.run(duration)

            feat_labels, feat_vals = pipeline.extract(result)
            if not labels_set:
                all_labels = param_names + feat_labels
                labels_set = True

            row = np.concatenate([param_sets[i], feat_vals])
            rows.append(row)

        values = np.stack(rows)    # (n_samples, n_params + n_features)
        return all_labels, values

    def run_df(self, duration: float):
        """Return a pandas DataFrame (requires pandas)."""
        import pandas as pd
        labels, values = self.run(duration)
        return pd.DataFrame(values, columns=labels)
