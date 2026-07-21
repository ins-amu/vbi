"""
NumbaSweeperCPU - parallel parameter sweep using @njit(parallel=True) + prange.

Each prange iteration runs one independent simulation with its own state copy,
ring buffer, and (for stochastic) its own random seed.  The output is a
(n_samples, n_record, n_voi, n_nodes) array that the pipeline then reduces
to feature vectors.
"""
from __future__ import annotations

import numpy as np

from vbi.simulator.spec.simulation import SimulationSpec
from vbi.simulator.spec.sweep import SweepSpec

from .codegen import (
    build_numba_dfun,
    build_params,
    build_srcbuf,
    build_initial_state,
    get_G_idx,
    get_bounds,
    get_noise_params,
)
from ._nb_sim import (
    nb_sweep_det, nb_sweep_stoch,
    nb_sweep_det_feat, nb_sweep_stoch_feat,
)
from vbi.simulator.spec.stimulus import build_stim_data as _build_stim_data
from .simulator import _apply_monitor


def _count_records(n_steps: int, t_cut_step: int, record_period: int) -> int:
    if n_steps <= t_cut_step:
        return 0
    return (n_steps - t_cut_step + record_period - 1) // record_period


class NumbaSweeperCPU:
    """
    Parallel sweep backend for SBI training data generation.

    run(duration)  → (labels, values)  shape (n_samples, n_params + n_features)
                   OR list of raw monitor dicts if sweep_spec.pipeline is None
    run_df(duration) → pandas DataFrame
    """

    def __init__(self, spec: SimulationSpec, sweep_spec: SweepSpec):
        self.spec  = spec
        self.sweep = sweep_spec

        dt      = spec.integrator.dt
        n_nodes = spec.n_nodes

        if spec.coupling.kind not in ("linear", "kuramoto", "jr_sigmoidal"):
            raise NotImplementedError(
                f"Numba backend supports 'linear', 'kuramoto', and 'jr_sigmoidal' coupling; "
                f"got {spec.coupling.kind!r}."
            )
        self._use_kuramoto    = spec.coupling.kind == "kuramoto"
        self._use_jr_sigmoidal = spec.coupling.kind == "jr_sigmoidal"
        self._alpha = np.float64(spec.coupling.alpha)

        self._dfun   = build_numba_dfun(spec.model)
        self._params = build_params(spec)
        self._G_idx  = get_G_idx(spec.model)
        if self._G_idx < 0:
            G = np.asarray(spec.node_params.get("G", 1.0), dtype=np.float64)
            extra = np.empty((1, n_nodes), dtype=np.float64)
            if G.ndim == 0:
                extra[0, :] = float(G)
            elif G.shape == (n_nodes,):
                extra[0, :] = G
            else:
                extra[0, :] = float(G.flat[0])
            self._params = np.vstack([self._params, extra])
            self._G_idx  = self._params.shape[0] - 1

        self._a = np.float64(spec.coupling.a)
        self._b = np.float64(spec.coupling.b)

        if self._use_jr_sigmoidal:
            _pnames = list(spec.model.param_names)
            for _pn in ("nu_max", "r", "v0"):
                if _pn not in _pnames:
                    raise ValueError(
                        f"CouplingSpec(kind='jr_sigmoidal') requires model parameters "
                        f"'nu_max', 'r', 'v0'. Model {spec.model.name!r} is missing {_pn!r}."
                    )
            self._nu_max_jr = np.float64(self._params[_pnames.index("nu_max"), 0])
            self._r_jr      = np.float64(self._params[_pnames.index("r"),      0])
            self._v0_jr     = np.float64(self._params[_pnames.index("v0"),     0])
        else:
            self._nu_max_jr = np.float64(0.0)
            self._r_jr      = np.float64(0.0)
            self._v0_jr     = np.float64(0.0)

        self._state0       = build_initial_state(spec)
        self._cvar_indices = np.array(spec.model.cvar_indices, dtype=np.int64)
        self._delay_steps  = spec.delay_steps(dt).astype(np.int32)
        self._horizon      = spec.horizon(dt)
        self._has_delays   = bool(spec.has_delays)
        self._srcbuf0      = build_srcbuf(
            len(spec.model.cvar), n_nodes, self._horizon,
            self._state0[list(spec.model.cvar_indices)])

        self._lower_bounds, self._has_lower, \
            self._upper_bounds, self._has_upper = get_bounds(spec.model)

        self._use_heun   = (spec.integrator.method == "heun")
        self._stochastic = spec.integrator.stochastic
        if self._stochastic:
            self._eff_noise_amp, self._noise_mask = get_noise_params(spec)
        else:
            self._eff_noise_amp = np.empty(0, dtype=np.float64)
            self._noise_mask    = np.zeros(spec.model.n_sv, dtype=np.bool_)

        self._weights = np.ascontiguousarray(spec.weights, dtype=np.float64)

        # Pre-resolve sweep param indices in the packed params array.
        # If G was injected (not declared in model), it lives at self._G_idx.
        model_param_names = list(spec.model.param_names)
        sweep_names       = self.sweep._param_names_list

        def _resolve_idx(name: str) -> int:
            if name in model_param_names:
                return model_param_names.index(name)
            if name == "G" and self._G_idx >= 0:
                return self._G_idx
            raise ValueError(
                f"Sweep parameter {name!r} not found in model param_names "
                f"{model_param_names!r} and is not an injected 'G' parameter."
            )

        self._sweep_param_indices = np.array(
            [_resolve_idx(n) for n in sweep_names],
            dtype=np.int64,
        )

    def _run_raw(self, duration: float) -> np.ndarray:
        """
        Run the sweep and return (n_samples, n_record, n_sv, n_nodes) raw output.
        record_period is determined by the pipeline's signal or the first monitor.
        """
        spec     = self.spec
        dt       = spec.integrator.dt
        n_steps  = round(duration / dt)

        # record_period: use the first monitor's period (default: every step)
        if spec.monitors:
            m0 = spec.monitors[0]
            if m0.period:
                record_period = max(1, round(m0.period / dt))
            else:
                record_period = 1
        else:
            record_period = 1

        # burn-in: only applied inside @njit when a pipeline is set.
        # Without a pipeline the full simulation is returned (t_cut applied later
        # by the caller, matching NumpySweeper semantics).
        pipeline      = self.sweep.pipeline
        t_cut_step    = round(pipeline.t_cut / dt) if pipeline is not None else 0
        n_record      = _count_records(n_steps, t_cut_step, record_period)

        n_sv   = spec.model.n_sv
        n_voi  = n_sv
        voi_indices = np.arange(n_sv, dtype=np.int64)

        param_sets = self.sweep.param_sets.astype(np.float64)

        # Stimulus is identical for every sample in the sweep
        stim_data, has_stimulus = _build_stim_data(spec, n_steps, dt)

        if self._stochastic:
            base_seed = spec.integrator.noise_seed
            n_ps = param_sets.shape[0]
            if self.sweep.same_noise:
                seeds = np.full(n_ps, base_seed, dtype=np.int64)
            else:
                seeds = np.arange(n_ps, dtype=np.int64) + base_seed
            raw = nb_sweep_stoch(
                param_sets, self._params, self._state0, self._srcbuf0,
                self._weights, self._delay_steps, self._horizon,
                self._G_idx, self._a, self._b, dt, n_steps, n_record,
                self._has_delays, self._cvar_indices,
                self._lower_bounds, self._has_lower,
                self._upper_bounds, self._has_upper,
                self._eff_noise_amp, self._noise_mask,
                record_period, t_cut_step, n_voi, voi_indices,
                self._use_heun, self._sweep_param_indices, seeds,
                self._dfun, self._use_kuramoto, self._use_jr_sigmoidal,
                self._nu_max_jr, self._r_jr, self._v0_jr, self._alpha,
                stim_data, has_stimulus,
            )
        else:
            raw = nb_sweep_det(
                param_sets, self._params, self._state0, self._srcbuf0,
                self._weights, self._delay_steps, self._horizon,
                self._G_idx, self._a, self._b, dt, n_steps, n_record,
                self._has_delays, self._cvar_indices,
                self._lower_bounds, self._has_lower,
                self._upper_bounds, self._has_upper,
                record_period, t_cut_step, n_voi, voi_indices,
                self._use_heun, self._sweep_param_indices, self._dfun,
                self._use_kuramoto, self._use_jr_sigmoidal,
                self._nu_max_jr, self._r_jr, self._v0_jr, self._alpha,
                stim_data, has_stimulus,
            )

        return raw, record_period  # (n_samples, n_record, n_sv, n_nodes)

    def _sweep_params(self, duration: float):
        """Return commonly-needed loop params without running the simulation."""
        dt            = self.spec.integrator.dt
        n_steps       = round(duration / dt)
        m0            = self.spec.monitors[0] if self.spec.monitors else None
        record_period = max(1, round(m0.period / dt)) if (m0 and m0.period) else 1
        pipeline      = self.sweep.pipeline
        t_cut_step    = round(pipeline.t_cut / dt) if pipeline is not None else 0
        n_record      = _count_records(n_steps, t_cut_step, record_period)
        n_sv          = self.spec.model.n_sv
        voi_indices   = np.arange(n_sv, dtype=np.int64)
        return dt, n_steps, record_period, t_cut_step, n_record, n_sv, voi_indices

    def run(self, duration: float):
        """
        Run all parameter sets in parallel.

        Returns
        -------
        If sweep_spec.pipeline is None:
            list of raw monitor dicts (one per run) - format matches NumpySweeper.
        If pipeline is set:
            (all_labels, values) where values shape (n_samples, n_params + n_features).
        """
        pipeline    = self.sweep.pipeline
        param_names = self.sweep._param_names_list
        param_sets  = self.sweep.param_sets
        n_samples   = param_sets.shape[0]

        # ------------------------------------------------------------------
        # Tier-2: JIT inline feature extraction - skip full raw materialisation
        # ------------------------------------------------------------------
        nb_spec = getattr(pipeline, "nb_extractor", None) if pipeline else None

        if nb_spec is not None:
            dt, n_steps, record_period, t_cut_step, n_record, n_sv, voi_indices = \
                self._sweep_params(duration)

            n_nodes = self.spec.n_nodes
            # Resolve n_voi_feat: -1 sentinel means "use all VOIs"
            n_voi_feat  = n_sv if nb_spec.n_voi_feat == -1 else nb_spec.n_voi_feat
            n_channels  = n_voi_feat * n_nodes
            n_feat      = nb_spec.n_features(n_channels)
            feat_labels = nb_spec.labels(n_channels)
            ps          = param_sets.astype(np.float64)

            stim_data, has_stimulus = _build_stim_data(self.spec, n_steps, dt)

            common_args = (
                ps, self._params, self._state0, self._srcbuf0,
                self._weights, self._delay_steps, self._horizon,
                self._G_idx, self._a, self._b,
                np.float64(dt), n_steps, n_record,
                self._has_delays, self._cvar_indices,
                self._lower_bounds, self._has_lower,
                self._upper_bounds, self._has_upper,
                record_period, t_cut_step, n_sv, voi_indices,
                self._use_heun, self._sweep_param_indices,
                nb_spec.do_mean, nb_spec.do_std,
                nb_spec.do_fc, nb_spec.do_fcd,
                np.int64(nb_spec.fcd_window), np.int64(n_feat),
                np.int64(n_voi_feat),
                np.int64(nb_spec.voi_diff_pos), np.int64(nb_spec.voi_diff_neg),
                self._dfun,
            )

            if self._stochastic:
                base_seed = self.spec.integrator.noise_seed
                if self.sweep.same_noise:
                    seeds = np.full(n_samples, base_seed, dtype=np.int64)
                else:
                    seeds = np.arange(n_samples, dtype=np.int64) + base_seed
                feat_vals = nb_sweep_stoch_feat(
                    ps, self._params, self._state0, self._srcbuf0,
                    self._weights, self._delay_steps, self._horizon,
                    self._G_idx, self._a, self._b,
                    np.float64(dt), n_steps, n_record,
                    self._has_delays, self._cvar_indices,
                    self._lower_bounds, self._has_lower,
                    self._upper_bounds, self._has_upper,
                    self._eff_noise_amp, self._noise_mask,
                    record_period, t_cut_step, n_sv, voi_indices,
                    self._use_heun, self._sweep_param_indices, seeds,
                    nb_spec.do_mean, nb_spec.do_std,
                    nb_spec.do_fc, nb_spec.do_fcd,
                    np.int64(nb_spec.fcd_window), np.int64(n_feat),
                    np.int64(n_voi_feat),
                    np.int64(nb_spec.voi_diff_pos), np.int64(nb_spec.voi_diff_neg),
                    self._dfun, self._use_kuramoto, self._use_jr_sigmoidal,
                    self._nu_max_jr, self._r_jr, self._v0_jr, self._alpha,
                    stim_data, has_stimulus,
                )
            else:
                feat_vals = nb_sweep_det_feat(
                    *common_args, self._use_kuramoto, self._use_jr_sigmoidal,
                    self._nu_max_jr, self._r_jr, self._v0_jr, self._alpha,
                    stim_data, has_stimulus,
                )

            values = np.concatenate([ps, feat_vals], axis=1)
            return param_names + feat_labels, values

        # ------------------------------------------------------------------
        # Tier-1 / no-pipeline: run full simulation, wrap or reduce in Python
        # ------------------------------------------------------------------
        raw, record_period = self._run_raw(duration)
        dt = self.spec.integrator.dt

        if pipeline is None:
            t_raw = np.arange(raw.shape[1], dtype=np.float64) * record_period * dt
            results = []
            for i in range(n_samples):
                sample_result = {}
                for m in self.spec.monitors:
                    sample_result[m.kind] = _apply_monitor(
                        m.kind, m, raw[i], t_raw, self.spec.model
                    )
                results.append(sample_result)
            return results

        # Tier-1 pipeline loop
        t_cut_step   = round(pipeline.t_cut / dt)
        t_base       = t_cut_step * dt
        labels_set   = False
        all_labels: list[str] = []
        rows: list[np.ndarray] = []

        # Find the MonitorSpec that matches pipeline.signal
        m_spec = next(
            (m for m in self.spec.monitors if m.kind == pipeline.signal), None
        )
        if m_spec is None:
            raise ValueError(
                f"pipeline.signal={pipeline.signal!r} has no matching monitor in "
                f"spec.monitors. Available: {[m.kind for m in self.spec.monitors]}"
            )

        t_raw = t_base + np.arange(raw.shape[1]) * record_period * dt

        for i in range(n_samples):
            t_proc, data_proc = _apply_monitor(
                pipeline.signal, m_spec, raw[i], t_raw, self.spec.model
            )
            monitor_result = {pipeline.signal: (t_proc, data_proc)}
            feat_labels_i, feat_vals_i = pipeline.extract(monitor_result)
            if not labels_set:
                all_labels = param_names + feat_labels_i
                labels_set = True
            rows.append(np.concatenate(
                [param_sets[i].astype(np.float64), feat_vals_i]
            ))

        return all_labels, np.stack(rows)

    def run_df(self, duration: float):
        """Return sweep results as a pandas DataFrame."""
        import pandas as pd
        labels, values = self.run(duration)
        return pd.DataFrame(values, columns=labels)
