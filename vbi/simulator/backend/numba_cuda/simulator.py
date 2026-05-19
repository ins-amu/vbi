"""
CudaSimulator — single-run GPU backend (batch=1 wrapper around CudaSweeperGPU).

The CUDA backend is batch-first by design: the sweep is the primary use case.
A single-run simulator is provided for debugging and single-trajectory use;
it simply calls the sweeper with a 1-element sweep over a dummy parameter.
"""
from __future__ import annotations

import numpy as np

from vbi.simulator.spec.simulation import SimulationSpec
from vbi.simulator.spec.sweep import SweepSpec
from vbi.simulator.backend.numba_.simulator import _apply_monitor

from .sweeper import CudaSweeperGPU, _require_cuda
from .codegen import (
    build_cuda_module,
    build_params_matrix,
    build_initial_state,
    build_ring_buffer,
    get_bounds_arrays,
    generate_noise,
)


def _count_records(n_steps: int, t_cut: int, period: int) -> int:
    if n_steps <= t_cut:
        return 0
    return (n_steps - t_cut + period - 1) // period

try:
    from numba import cuda as _cuda
    CUDA_AVAILABLE = _cuda.is_available()
except Exception:
    CUDA_AVAILABLE = False

_TPB = 256


class CudaSimulator:
    """
    Single-run Numba-CUDA backend.

    Interface mirrors NumpySimulator / NumbaSimulator:
        build(spec)   → None
        run(duration) → dict[str, (t, data)]
    """

    def build(self, spec: SimulationSpec, verbose: bool = False) -> None:
        _require_cuda()
        if spec.coupling.kind != "linear":
            raise NotImplementedError(
                f"CUDA backend supports 'linear' coupling; got {spec.coupling.kind!r}."
            )
        self.spec = spec
        self._mod = build_cuda_module(spec.model)

        dt      = spec.integrator.dt
        model   = spec.model
        n_nodes = spec.n_nodes

        self._weights_f32  = np.ascontiguousarray(spec.weights, dtype=np.float32)
        self._delay_steps  = np.ascontiguousarray(spec.delay_steps(dt), dtype=np.int32)
        self._horizon      = spec.horizon(dt)
        self._has_delays   = bool(spec.has_delays)

        G = float(np.asarray(spec.node_params.get(
            "G", model.default_params.get("G", 1.0))).mean())
        self._coup_a = np.float32(spec.coupling.a * G)
        self._coup_b = np.float32(spec.coupling.b)

        self._lo, self._hlo, self._hi, self._hhi = get_bounds_arrays(model)
        self._state0  = build_initial_state(spec)
        self._use_heun   = (spec.integrator.method == "heun")
        self._stochastic = spec.integrator.stochastic
        self._seed_base  = int(spec.integrator.noise_seed)

    def run(self, duration: float) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        from numba import cuda

        spec    = self.spec
        dt      = np.float32(spec.integrator.dt)
        n_steps = int(round(duration / float(dt)))
        n_sv    = spec.model.n_sv
        n_cvar  = len(spec.model.cvar)
        n_nodes = spec.n_nodes

        # Single simulation: batch=1
        n_samples     = 1
        record_period = 1
        t_cut_step    = 0
        n_record      = n_steps

        params_h   = build_params_matrix(spec, n_samples)
        state_h    = self._state0[np.newaxis, :, :].astype(np.float32)
        cvar_idx   = np.array(spec.model.cvar_indices, dtype=np.int32)
        init_cvar  = self._state0[list(spec.model.cvar_indices)].astype(np.float32)
        buf_h      = build_ring_buffer(
            n_samples, n_cvar, n_nodes, self._horizon, init_cvar)
        ts_out_h   = np.zeros(
            (n_samples, n_record, n_sv, n_nodes), dtype=np.float32)

        # Transfer
        state_d    = cuda.to_device(state_h)
        buf_d      = cuda.to_device(buf_h)
        weights_d  = cuda.to_device(self._weights_f32)
        delays_d   = cuda.to_device(self._delay_steps)
        params_d   = cuda.to_device(params_h)
        cvar_idx_d = cuda.to_device(cvar_idx)
        lo_d       = cuda.to_device(self._lo)
        hlo_d      = cuda.to_device(self._hlo)
        hi_d       = cuda.to_device(self._hi)
        hhi_d      = cuda.to_device(self._hhi)
        ts_out_d   = cuda.to_device(ts_out_h)

        blocks = 1  # single simulation → one block

        if self._stochastic:
            noise_h = generate_noise(spec, n_steps, n_samples, self._seed_base)
            noise_d = cuda.to_device(noise_h)
            self._mod.cuda_sweep_stoch[blocks, _TPB](
                state_d, buf_d, weights_d, delays_d,
                params_d, cvar_idx_d,
                lo_d, hlo_d, hi_d, hhi_d,
                np.int32(self._horizon), dt, np.int32(n_steps),
                np.int32(n_record), np.int32(t_cut_step),
                np.int32(record_period),
                self._coup_a, self._coup_b,
                self._has_delays, self._use_heun,
                noise_d, ts_out_d,
            )
        else:
            self._mod.cuda_sweep_det[blocks, _TPB](
                state_d, buf_d, weights_d, delays_d,
                params_d, cvar_idx_d,
                lo_d, hlo_d, hi_d, hhi_d,
                np.int32(self._horizon), dt, np.int32(n_steps),
                np.int32(n_record), np.int32(t_cut_step),
                np.int32(record_period),
                self._coup_a, self._coup_b,
                self._has_delays, self._use_heun,
                ts_out_d,
            )

        cuda.synchronize()
        raw = ts_out_d.copy_to_host().astype(np.float64)[0]  # (n_record, n_sv, n_nodes)
        raw_times = np.arange(n_record, dtype=np.float64) * float(dt)

        result = {}
        for m in spec.monitors:
            result[m.kind] = _apply_monitor(m.kind, m, raw, raw_times, spec.model)
        return result
