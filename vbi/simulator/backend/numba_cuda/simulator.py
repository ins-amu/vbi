"""
CudaSimulator - single-run GPU backend (batch=1 wrapper using coalesced layout).
"""
from __future__ import annotations

import numpy as np

from vbi.simulator.spec.simulation import SimulationSpec
from vbi.simulator.spec.stimulus import build_stim_data
from vbi.simulator.backend.numba_.simulator import _apply_monitor

from .codegen import (
    build_cuda_module,
    build_params_matrix,
    build_initial_state,
    build_ring_buffer,
    get_bounds_arrays,
    generate_noise,
    to_csr,
)
from .sweeper import _require_cuda, _auto_sparse

try:
    from numba import cuda as _cuda
    CUDA_AVAILABLE = _cuda.is_available()
except Exception:
    CUDA_AVAILABLE = False

_TPB = 256


def _count_records(n_steps: int, t_cut: int, period: int) -> int:
    if n_steps <= t_cut:
        return 0
    return (n_steps - t_cut + period - 1) // period


class CudaSimulator:
    """Single-run Numba-CUDA backend (coalesced, optional sparse)."""

    def build(self, spec: SimulationSpec,
              connectivity: str = "auto",
              verbose: bool = False) -> None:
        _require_cuda()
        if spec.coupling.kind not in ("linear", "kuramoto"):
            raise NotImplementedError(
                f"CUDA backend supports 'linear' and 'kuramoto' coupling; "
                f"got {spec.coupling.kind!r}."
            )
        _use_kuramoto = spec.coupling.kind == "kuramoto"
        _alpha        = float(spec.coupling.alpha)
        self.spec = spec

        if connectivity == "auto":
            self._sparse = _auto_sparse(spec.weights)
        else:
            self._sparse = (connectivity == "sparse")

        self._mod = build_cuda_module(spec.model, sparse=self._sparse,
                                      use_kuramoto=_use_kuramoto, alpha=_alpha)

        dt      = spec.integrator.dt
        model   = spec.model
        n_nodes = spec.n_nodes

        self._weights_f32  = np.ascontiguousarray(spec.weights, dtype=np.float32)
        ds = spec.delay_steps(dt).astype(np.int32)
        self._delay_steps  = np.ascontiguousarray(ds)
        self._horizon      = spec.horizon(dt)
        self._has_delays   = bool(spec.has_delays)

        if self._sparse:
            (self._w_data, self._w_indices, self._w_indptr,
             self._idelays_csr, _, _) = to_csr(
                 spec.weights, ds if self._has_delays else None,
                 has_delays=self._has_delays)

        G = float(np.asarray(
            spec.node_params.get("G", model.default_params.get("G", 1.0))
        ).mean())
        if _use_kuramoto:
            self._coup_a = np.float32(G / n_nodes)
            self._coup_b = np.float32(_alpha)
        else:
            self._coup_a = np.float32(spec.coupling.a * G)
            self._coup_b = np.float32(spec.coupling.b)

        self._lo, self._hlo, self._hi, self._hhi = get_bounds_arrays(model)
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
        n_samp  = 1

        n_record = n_steps
        t_cut    = 0
        period   = 1

        params_h = build_params_matrix(spec, n_samp)
        cvar_idx = np.array(spec.model.cvar_indices, dtype=np.int32)
        init_cvar = np.zeros((n_cvar, n_nodes), dtype=np.float32)
        for ci_i, ci in enumerate(spec.model.cvar_indices):
            init_cvar[ci_i, :] = spec.model.state_variables[ci].default_init

        state_h  = build_initial_state(spec, n_samp)
        buf_h    = build_ring_buffer(n_samp, n_cvar, n_nodes, self._horizon, init_cvar)
        ts_out_h = np.zeros((n_record, n_sv, n_nodes, n_samp), dtype=np.float32)

        state_d    = cuda.to_device(state_h)
        buf_d      = cuda.to_device(buf_h)
        params_d   = cuda.to_device(params_h)
        cvar_idx_d = cuda.to_device(cvar_idx)
        lo_d       = cuda.to_device(self._lo)
        hlo_d      = cuda.to_device(self._hlo)
        hi_d       = cuda.to_device(self._hi)
        hhi_d      = cuda.to_device(self._hhi)
        ts_out_d   = cuda.to_device(ts_out_h)

        if self._sparse:
            conn = [cuda.to_device(self._w_data), cuda.to_device(self._w_indices),
                    cuda.to_device(self._w_indptr), cuda.to_device(self._idelays_csr)]
        else:
            conn = [cuda.to_device(self._weights_f32),
                    cuda.to_device(self._delay_steps)]

        # Pre-sample stimuli and transfer to device
        stim_np, has_stimulus = build_stim_data(spec, n_steps, float(dt))
        stim_flat = np.ascontiguousarray(stim_np.ravel(), dtype=np.float32)
        stim_d    = cuda.to_device(stim_flat)

        # Kernel indexes coup_a by [tid]; wrap scalar in a 1-element device array.
        coup_a_d = cuda.to_device(np.array([self._coup_a], dtype=np.float32))

        common = [state_d, buf_d] + conn + [
            params_d, cvar_idx_d,
            lo_d, hlo_d, hi_d, hhi_d,
            np.int32(self._horizon), dt,
            np.int32(n_steps), np.int32(n_record),
            np.int32(t_cut), np.int32(period),
            coup_a_d, self._coup_b,
            self._has_delays, self._use_heun,
            stim_d, has_stimulus,
        ]

        if self._stochastic:
            noise_h = generate_noise(spec, n_steps, n_samp, self._seed_base)
            noise_d = cuda.to_device(noise_h)
            self._mod.cuda_sweep_stoch[1, _TPB](*common, noise_d, ts_out_d)
        else:
            self._mod.cuda_sweep_det[1, _TPB](*common, ts_out_d)

        cuda.synchronize()

        ts_dev = ts_out_d.copy_to_host()         # (nr, n_sv, nn, 1)
        raw    = ts_dev[:, :, :, 0].astype(np.float64)  # (nr, n_sv, nn)
        raw_times = np.arange(n_steps, dtype=np.float64) * float(dt)

        result = {}
        for m in spec.monitors:
            result[m.kind] = _apply_monitor(m.kind, m, raw, raw_times, spec.model)
        return result
