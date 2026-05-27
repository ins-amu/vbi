"""
NumbaSimulator — single-run Numba CPU backend.

The @njit simulation loop returns a raw subsampled array.
Python post-processing converts it to the same {kind: (t, data)} dict
that the NumPy backend returns, so all existing tests and pipelines work
unchanged.
"""
from __future__ import annotations

import numpy as np

from vbi.simulator.spec.simulation import SimulationSpec
from vbi.simulator.spec.stimulus import build_stim_data
from vbi.simulator.backend.numpy_.monitors import _resolve_voi, _bw_init, _bw_step, _bw_bold, _BW_DEFAULTS

from .codegen import (
    build_numba_dfun,
    build_params,
    build_srcbuf,
    build_initial_state,
    get_G_idx,
    get_bounds,
    get_noise_params,
)
from ._nb_sim import nb_simulate_det, nb_simulate_stoch


def _count_records(n_steps: int, t_cut_step: int, record_period: int) -> int:
    if n_steps <= t_cut_step:
        return 0
    return (n_steps - t_cut_step + record_period - 1) // record_period


def _apply_monitor(kind, m_spec, raw_data, raw_times, model):
    """
    Convert raw subsampled array to a single monitor's (t, data).

    raw_data  : (n_record, n_sv, n_nodes)
    raw_times : (n_record,) — time of each recorded step in ms
    """
    voi = _resolve_voi(m_spec.variables, model)

    if kind == "raw":
        return raw_times.copy(), raw_data[:, voi, :]

    elif kind == "subsample":
        return raw_times.copy(), raw_data[:, voi, :]

    elif kind == "tavg":
        n_record, n_sv, n_nodes = raw_data.shape
        dt_raw = raw_times[1] - raw_times[0] if len(raw_times) > 1 else 1.0
        period = m_spec.period if m_spec.period else dt_raw
        istep  = max(1, round(period / dt_raw))
        n_win  = n_record // istep
        if n_win == 0:
            raise ValueError(
                f"TemporalAverageMonitor (Numba) collected no samples. "
                f"duration must exceed period ({period} ms)."
            )
        win = raw_data[:n_win * istep].reshape(n_win, istep, n_sv, n_nodes)
        avg = win.mean(axis=1)[:, voi, :]
        # midpoint timing matching NumPy TemporalAverageMonitor
        t0  = raw_times[0]
        t_avg = t0 + (np.arange(n_win) * istep + (istep - 1) * 0.5) * dt_raw
        return t_avg, avg

    elif kind == "gavg":
        n_record, n_sv, n_nodes = raw_data.shape
        dt_raw = raw_times[1] - raw_times[0] if len(raw_times) > 1 else 1.0
        period = m_spec.period if m_spec.period else dt_raw
        istep  = max(1, round(period / dt_raw))
        data   = raw_data[::istep, voi, :].mean(axis=2, keepdims=True)
        t      = raw_times[::istep]
        return t, data

    elif kind == "bold":
        # BW integration applied post-hoc to the first VOI (raw per step needed)
        n_record, n_sv, n_nodes = raw_data.shape
        dt_raw = raw_times[1] - raw_times[0] if len(raw_times) > 1 else 1.0
        tr      = m_spec.tr
        tr_steps = max(1, round(tr / dt_raw))
        bw_state = _bw_init(n_nodes)
        times_bold, data_bold = [], []
        for i in range(n_record):
            neural = raw_data[i, voi[0] if len(voi) > 0 else 0]
            bw_state = _bw_step(bw_state, neural, dt_raw * 1e-3, _BW_DEFAULTS)
            if i > 0 and i % tr_steps == 0:
                times_bold.append(raw_times[i])
                data_bold.append(_bw_bold(bw_state, _BW_DEFAULTS)[np.newaxis, :].copy())
        if not times_bold:
            raise ValueError(
                f"BoldMonitor (Numba) collected no samples. "
                f"duration must be > tr ({tr} ms)."
            )
        return np.array(times_bold), np.stack(data_bold)

    raise ValueError(f"Unsupported monitor kind: {kind!r}")


class NumbaSimulator:
    """
    Single-run Numba CPU backend.

    Implements the same interface as NumpySimulator:
        build(spec)  → None
        run(duration) → dict[str, (t, data)]
    """

    def build(self, spec: SimulationSpec) -> None:
        self.spec = spec
        dt      = spec.integrator.dt
        n_nodes = spec.n_nodes

        if spec.coupling.kind not in ("linear", "kuramoto"):
            raise NotImplementedError(
                f"Numba backend supports 'linear' and 'kuramoto' coupling; "
                f"got {spec.coupling.kind!r}."
            )
        self._use_kuramoto = spec.coupling.kind == "kuramoto"
        self._alpha = np.float64(spec.coupling.alpha)

        self._dfun  = build_numba_dfun(spec.model)
        self._params = build_params(spec)
        self._G_idx  = get_G_idx(spec.model)
        if self._G_idx < 0:
            # No G in model — inject a constant row at the end
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

        self._state0      = build_initial_state(spec)
        self._cvar_indices = np.array(spec.model.cvar_indices, dtype=np.int64)
        self._delay_steps  = spec.delay_steps(dt).astype(np.int32)
        self._horizon      = spec.horizon(dt)
        self._has_delays   = bool(spec.has_delays)
        self._srcbuf0      = build_srcbuf(
            len(spec.model.cvar), n_nodes, self._horizon,
            self._state0[list(spec.model.cvar_indices)])

        self._lower_bounds, self._has_lower, \
            self._upper_bounds, self._has_upper = get_bounds(spec.model)

        self._use_heun    = (spec.integrator.method == "heun")
        self._stochastic  = spec.integrator.stochastic
        if self._stochastic:
            self._eff_noise_amp, self._noise_mask = get_noise_params(spec)
        else:
            self._eff_noise_amp = np.empty(0, dtype=np.float64)
            self._noise_mask    = np.zeros(spec.model.n_sv, dtype=np.bool_)

        self._weights = np.ascontiguousarray(spec.weights, dtype=np.float64)

    def run(self, duration: float) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        spec  = self.spec
        dt    = spec.integrator.dt
        n_steps = round(duration / dt)

        # For single-run: record every step (raw). Monitors are applied post-hoc.
        # For a 'raw' monitor we record all SVs; for others it's the same data
        # — monitor post-processing selects VOI and temporal aggregation.
        record_period = 1   # always record every step for full monitor fidelity
        t_cut_step    = 0
        n_sv          = spec.model.n_sv
        voi_indices   = np.arange(n_sv, dtype=np.int64)
        n_record      = _count_records(n_steps, t_cut_step, record_period)

        # Pre-sample stimuli for this run duration
        stim_data, has_stimulus = build_stim_data(spec, n_steps, dt)

        if self._stochastic:
            raw_data = nb_simulate_stoch(
                self._state0, self._srcbuf0,
                self._weights, self._delay_steps, self._horizon,
                self._params, self._G_idx, self._a, self._b,
                dt, n_steps, n_record,
                self._has_delays, self._cvar_indices,
                self._lower_bounds, self._has_lower,
                self._upper_bounds, self._has_upper,
                self._eff_noise_amp, self._noise_mask,
                record_period, t_cut_step, n_sv, voi_indices,
                self._use_heun, np.int64(spec.integrator.noise_seed),
                self._dfun, self._use_kuramoto, self._alpha,
                stim_data, has_stimulus,
            )
        else:
            raw_data = nb_simulate_det(
                self._state0, self._srcbuf0,
                self._weights, self._delay_steps, self._horizon,
                self._params, self._G_idx, self._a, self._b,
                dt, n_steps, n_record,
                self._has_delays, self._cvar_indices,
                self._lower_bounds, self._has_lower,
                self._upper_bounds, self._has_upper,
                record_period, t_cut_step, n_sv, voi_indices,
                self._use_heun, self._dfun, self._use_kuramoto, self._alpha,
                stim_data, has_stimulus,
            )

        # raw_data: (n_record, n_sv, n_nodes)
        raw_times = np.arange(n_record, dtype=np.float64) * dt

        result = {}
        for m in spec.monitors:
            result[m.kind] = _apply_monitor(m.kind, m, raw_data, raw_times, spec.model)
        return result
