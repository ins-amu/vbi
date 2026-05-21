from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from vbi.simulator.spec.simulation import SimulationSpec
from vbi.simulator.spec.sweep import SweepSpec
from .codegen import build_jax_dfun
from .simulator import (
    _build_jax_params,
    _build_initial_state,
    _build_bounds,
    _build_noise_amp,
    _resolve_dtype,
    _make_step_fn,
    _run_monitor,
)


class JaxSweeper:
    """
    JAX sweep backend — jit + vmap over parameter sets.

    The swept parameter values are batched into a dict of shape-(n_samples,)
    arrays.  jax.vmap maps over axis-0 of each value, giving each simulation
    its own scalar.  A single jax.jit call compiles the full batch.
    """

    def __init__(self, spec: SimulationSpec, sweep_spec: SweepSpec) -> None:
        self.spec = spec
        self.sweep = sweep_spec

        dt = spec.integrator.dt
        n_nodes = spec.n_nodes

        self._dt = dt
        self._base_params = _build_jax_params(spec)
        self._state0 = _build_initial_state(spec)
        self._lo, self._hi = _build_bounds(spec)
        self._horizon = spec.horizon(dt)

        G = float(self._base_params.get("G", 1.0))
        self._dtype = _resolve_dtype(spec)
        self._weights = jnp.array(spec.weights, dtype=self._dtype)
        self._delay_steps = jnp.array(spec.delay_steps(dt), dtype=jnp.int32)

        stochastic = spec.integrator.stochastic
        self._noise_amp = _build_noise_amp(spec) if stochastic else jnp.zeros(spec.model.n_sv)
        self._master_key = jax.random.PRNGKey(spec.integrator.noise_seed)

        dfun = build_jax_dfun(spec.model)

        self._step_fn = _make_step_fn(
            dfun=dfun,
            weights=self._weights,
            delay_steps=self._delay_steps,
            has_delays=spec.has_delays,
            horizon=self._horizon,
            n_nodes=n_nodes,
            cvar_indices=spec.model.cvar_indices,
            lo_bounds=self._lo,
            hi_bounds=self._hi,
            dt=dt,
            G_default=G,
            coup_a=float(spec.coupling.a),
            coup_b=float(spec.coupling.b),
            coup_kind=spec.coupling.kind,
            coup_midpoint=float(getattr(spec.coupling, "midpoint", 0.0)),
            coup_sigma=float(getattr(spec.coupling, "sigma", 1.0)),
            use_heun=(spec.integrator.method == "heun"),
            stochastic=stochastic,
            noise_amp=self._noise_amp,
            master_key=self._master_key,
        )

        self._param_names = sweep_spec._param_names_list
        self._n_nodes = n_nodes

        self._run_batch_jit = None  # built lazily on first .run() call

    def _build_batch_runner(self, n_steps: int, mon_spec,
                            same_noise: bool) -> Callable:
        """
        Build jit+vmap runner for a specific (n_steps, monitor_kind, same_noise).

        same_noise=True  : all runs share the same master_key (vmap broadcasts it)
                           → identical stochastic forcing across sweep; parameter
                             effects visible without noise variability.
        same_noise=False : each run gets a unique key from split(master_key, N)
                           → statistically independent noise realizations.
        """
        dt = self._dt
        horizon = self._horizon
        state0 = self._state0
        base_params = self._base_params
        cvar_indices = list(self.spec.model.cvar_indices)
        cvar_idx_jnp = jnp.array(cvar_indices, dtype=jnp.int32)

        spec = self.spec
        step_kwargs = dict(
            dfun=build_jax_dfun(spec.model),
            weights=self._weights,
            delay_steps=self._delay_steps,
            has_delays=spec.has_delays,
            horizon=horizon,
            n_nodes=self._n_nodes,
            cvar_indices=spec.model.cvar_indices,
            lo_bounds=self._lo,
            hi_bounds=self._hi,
            dt=dt,
            G_default=float(base_params.get("G", 1.0)),
            coup_a=float(spec.coupling.a),
            coup_b=float(spec.coupling.b),
            coup_kind=spec.coupling.kind,
            coup_midpoint=float(getattr(spec.coupling, "midpoint", 0.0)),
            coup_sigma=float(getattr(spec.coupling, "sigma", 1.0)),
            use_heun=(spec.integrator.method == "heun"),
            stochastic=spec.integrator.stochastic,
            noise_amp=self._noise_amp,
        )

        def simulate_one(swept_params_one: dict,
                         run_key) -> tuple[jnp.ndarray, jnp.ndarray]:
            """One simulation run; run_key controls noise realization."""
            params = {**base_params, **swept_params_one}
            buf = jnp.zeros(
                (horizon, len(cvar_indices), self._n_nodes), dtype=self._dtype)
            buf = buf.at[:].set(state0[cvar_idx_jnp][None, :, :])

            step_fn_i = _make_step_fn(**step_kwargs, master_key=run_key)

            carry = (state0, buf, jnp.int32(0), params)
            times, data = _run_monitor(step_fn_i, carry, mon_spec, n_steps, dt)
            return times, data

        if same_noise:
            simulate_batch = jax.vmap(simulate_one, in_axes=(0, None))
        else:
            simulate_batch = jax.vmap(simulate_one, in_axes=(0, 0))

        return jax.jit(simulate_batch)

    def run(self, duration: float) -> list[dict] | tuple:
        """
        Run parameter sweep.

        Returns
        -------
        If sweep_spec.pipeline is None:
            list of monitor-result dicts, one per parameter set.
        If pipeline is set:
            (labels, values) where values is shape (n_samples, n_features+n_params).
        """
        param_sets = self.sweep.param_sets      # (n_samples, n_params)
        n_samples = param_sets.shape[0]
        n_steps = round(duration / self._dt)
        pipeline = self.sweep.pipeline

        swept_batch = {
            name: jnp.array(param_sets[:, i], dtype=self._dtype)
            for i, name in enumerate(self._param_names)
        }

        same_noise = getattr(self.sweep, "same_noise", True)

        if same_noise:
            run_keys = self._master_key
        else:
            run_keys = jax.random.split(self._master_key, n_samples)

        all_results: list[dict] = []
        for mon_spec in self.spec.monitors:
            runner = self._build_batch_runner(n_steps, mon_spec, same_noise)
            times_batch, data_batch = runner(swept_batch, run_keys)
            if not all_results:
                all_results = [{} for _ in range(n_samples)]
            for i in range(n_samples):
                all_results[i][mon_spec.kind] = (
                    np.array(times_batch[i]),
                    np.array(data_batch[i]),
                )

        if pipeline is None:
            return all_results

        labels_set = False
        feat_labels: list[str] = []
        rows: list[np.ndarray] = []

        for i, result in enumerate(all_results):
            feat_lab, feat_val = pipeline.extract(result)
            if not labels_set:
                feat_labels = feat_lab
                labels_set = True
            param_vals = param_sets[i]
            rows.append(np.concatenate([param_vals, feat_val]))

        param_labels = self._param_names
        labels = list(param_labels) + feat_labels
        values = np.stack(rows)
        return labels, values

    def run_df(self, duration: float) -> "pd.DataFrame":
        labels, values = self.run(duration)
        return pd.DataFrame(values, columns=labels)
