from __future__ import annotations
import dataclasses
import numpy as np

from vbi.simulator.spec.simulation import SimulationSpec
from vbi.simulator.spec.sweep import SweepSpec
from .simulator import NumpySimulator


def _patch_spec(base: SimulationSpec, param_names: list[str],
                theta: np.ndarray,
                run_index: int = 0,
                same_noise: bool = True) -> SimulationSpec:
    """Return a SimulationSpec with swept parameters overridden."""
    new_node_params = dict(base.node_params)

    # node_params overrides take priority over model.default_params in build()
    for name, val in zip(param_names, theta):
        new_node_params[name] = float(val)

    # For stochastic sweeps with same_noise=False derive a unique per-run seed.
    integrator = base.integrator
    if base.integrator.stochastic and not same_noise:
        integrator = dataclasses.replace(
            base.integrator,
            noise_seed=base.integrator.noise_seed + run_index,
        )

    patched = SimulationSpec(
        model=base.model,
        integrator=integrator,
        coupling=base.coupling,
        monitors=base.monitors,
        weights=base.weights,
        tract_lengths=base.tract_lengths,
        speed=base.speed,
        node_params=new_node_params,
        stimuli=base.stimuli,   # preserve stimuli from base spec
    )
    return patched


class NumpySweeper:
    """
    Reference sweep backend - sequential Python loop over parameter sets.
    Not optimised; used for validation of faster backends.

    Notes
    -----
    ``same_noise`` behaviour (from ``SweepSpec.same_noise``):
      True  (default) - all sweep runs share ``base.integrator.noise_seed``,
            giving identical stochastic forcing across parameter sets.
      False - each run uses ``base_seed + run_index`` for independent noise.
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
            list of dicts mapping monitor kind -> (t, data) per run.
        If pipeline is set:
            (labels, values) where values shape is (n_samples, n_features+n_params).
        """
        param_names = self.sweep._param_names_list
        param_sets  = self.sweep.param_sets       # (n_samples, n_params)
        n           = param_sets.shape[0]
        pipeline    = self.sweep.pipeline
        same_noise  = self.sweep.same_noise

        if pipeline is None:
            all_results: list[dict] = []
            for i in range(n):
                patched = _patch_spec(self.spec, param_names, param_sets[i],
                                      run_index=i, same_noise=same_noise)
                sim = NumpySimulator()
                sim.build(patched)
                all_results.append(sim.run(duration))
            return all_results

        # Pipeline mode: extract features per run; accumulate into arrays
        labels_set = False
        all_labels: list[str] = []
        rows: list[np.ndarray] = []

        for i in range(n):
            patched = _patch_spec(self.spec, param_names, param_sets[i],
                                  run_index=i, same_noise=same_noise)
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
