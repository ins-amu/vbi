"""
CudaSweeperGPU — parallel parameter sweep on GPU.

One GPU thread = one complete simulation.
float32 precision on GPU; results returned as float64 numpy arrays.

Usage
-----
    sweeper = CudaSweeperGPU(spec, sweep_spec)
    labels, values = sweeper.run(duration=5000.0)
    # or
    results = sweeper.run(duration=5000.0)  # list of dicts when no pipeline
"""
from __future__ import annotations

import numpy as np

from vbi.simulator.spec.simulation import SimulationSpec
from vbi.simulator.spec.sweep import SweepSpec
from vbi.simulator.backend.numba_.simulator import _apply_monitor  # reuse post-processing

from .codegen import (
    build_cuda_module,
    build_params_matrix,
    build_initial_state,
    build_ring_buffer,
    get_bounds_arrays,
    generate_noise,
)

try:
    from numba import cuda as _cuda
    CUDA_AVAILABLE = _cuda.is_available()
except Exception:
    CUDA_AVAILABLE = False

_TPB = 256   # threads per block


def _require_cuda():
    if not CUDA_AVAILABLE:
        raise RuntimeError(
            "CUDA backend unavailable. Ensure:\n"
            "  1. An NVIDIA GPU is present and the CUDA toolkit is installed.\n"
            "  2. `numba[cuda]` is installed: pip install numba[cuda]\n"
            "  3. CUDA_HOME or PATH includes nvcc.\n"
            "Alternatively, use backend='numba' (CPU) or backend='cpp'."
        )


def _count_records(n_steps: int, t_cut: int, period: int) -> int:
    if n_steps <= t_cut:
        return 0
    return (n_steps - t_cut + period - 1) // period


class CudaSweeperGPU:
    """
    GPU parameter sweep backend.

    Parameters
    ----------
    spec        : SimulationSpec   base simulation spec
    sweep_spec  : SweepSpec        which params to vary
    """

    def __init__(self, spec: SimulationSpec, sweep_spec: SweepSpec):
        _require_cuda()
        if spec.coupling.kind != "linear":
            raise NotImplementedError(
                f"CUDA backend supports 'linear' coupling; got {spec.coupling.kind!r}."
            )

        self.spec  = spec
        self.sweep = sweep_spec

        dt      = spec.integrator.dt
        n_nodes = spec.n_nodes
        model   = spec.model

        # Compile CUDA module (dfun + kernels generated together)
        self._mod = build_cuda_module(model)

        # Connectivity
        self._weights_f32 = np.ascontiguousarray(spec.weights, dtype=np.float32)
        ds = spec.delay_steps(dt).astype(np.int32)
        self._delay_steps  = np.ascontiguousarray(ds)
        self._horizon      = spec.horizon(dt)
        self._has_delays   = bool(spec.has_delays)

        # Coupling scale
        G = float(np.asarray(spec.node_params.get("G",
                  model.default_params.get("G", 1.0))).mean())
        self._coup_a = np.float32(spec.coupling.a * G)
        self._coup_b = np.float32(spec.coupling.b)

        # Bounds
        self._lo, self._hlo, self._hi, self._hhi = get_bounds_arrays(model)

        # Initial state
        self._state0 = build_initial_state(spec)

        # Integration flags
        self._use_heun   = (spec.integrator.method == "heun")
        self._stochastic = spec.integrator.stochastic
        self._seed_base  = int(spec.integrator.noise_seed)

        # Sweep metadata
        self._model_param_names = list(model.param_names)
        self._sweep_names = sweep_spec._param_names_list

    def run(self, duration: float):
        """
        Run all parameter sets on GPU.

        Returns
        -------
        If sweep_spec.pipeline is None:
            list[dict]  — one {monitor_kind: (t, data)} per run (float64)
        If pipeline is set:
            (labels, values)  shape (n_samples, n_params + n_features) float64
        """
        from numba import cuda

        spec        = self.spec
        dt          = np.float32(spec.integrator.dt)
        n_steps     = int(round(duration / float(dt)))
        pipeline    = self.sweep.pipeline
        param_sets  = self.sweep.param_sets        # (n_samples, n_sweep)
        n_samples   = param_sets.shape[0]
        param_names = self.sweep._param_names_list

        # record_period from first monitor
        if spec.monitors and spec.monitors[0].period:
            record_period = max(1, round(float(spec.monitors[0].period) / float(dt)))
        else:
            record_period = 1

        t_cut_step = (round(pipeline.t_cut / float(dt))
                      if pipeline is not None else 0)
        n_record   = _count_records(n_steps, t_cut_step, record_period)

        n_sv    = spec.model.n_sv
        n_cvar  = len(spec.model.cvar)
        n_nodes = spec.n_nodes

        # ---- Build host arrays ----
        params_h = build_params_matrix(
            spec, n_samples,
            sweep_names=param_names,
            sweep_sets=param_sets.astype(np.float64),
        )

        state_h = np.tile(
            self._state0[np.newaxis, :, :], (n_samples, 1, 1)
        ).astype(np.float32)

        cvar_idx = np.array(spec.model.cvar_indices, dtype=np.int32)
        init_cvar = self._state0[list(spec.model.cvar_indices)].astype(np.float32)
        buf_h = build_ring_buffer(
            n_samples, n_cvar, n_nodes, self._horizon, init_cvar
        )

        ts_out_h = np.zeros(
            (n_samples, n_record, n_sv, n_nodes), dtype=np.float32
        )

        # ---- Transfer to device ----
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

        # ---- Launch ----
        blocks = (n_samples + _TPB - 1) // _TPB

        if self._stochastic:
            noise_h = generate_noise(spec, n_steps, n_samples, self._seed_base)
            noise_d = cuda.to_device(noise_h)
            self._mod.cuda_sweep_stoch[blocks, _TPB](
                state_d, buf_d,
                weights_d, delays_d,
                params_d, cvar_idx_d,
                lo_d, hlo_d, hi_d, hhi_d,
                np.int32(self._horizon),
                dt,
                np.int32(n_steps),
                np.int32(n_record),
                np.int32(t_cut_step),
                np.int32(record_period),
                self._coup_a, self._coup_b,
                self._has_delays,
                self._use_heun,
                noise_d,
                ts_out_d,
            )
        else:
            self._mod.cuda_sweep_det[blocks, _TPB](
                state_d, buf_d,
                weights_d, delays_d,
                params_d, cvar_idx_d,
                lo_d, hlo_d, hi_d, hhi_d,
                np.int32(self._horizon),
                dt,
                np.int32(n_steps),
                np.int32(n_record),
                np.int32(t_cut_step),
                np.int32(record_period),
                self._coup_a, self._coup_b,
                self._has_delays,
                self._use_heun,
                ts_out_d,
            )

        cuda.synchronize()

        # ---- Copy back ----
        ts_out_h = ts_out_d.copy_to_host().astype(np.float64)

        # ---- Post-process ----
        rec_dt = float(dt) * record_period
        t_base = t_cut_step * float(dt)

        if pipeline is None:
            mon_kind = spec.monitors[0].kind if spec.monitors else "raw"
            results = []
            for i in range(n_samples):
                t_arr = t_base + np.arange(n_record, dtype=np.float64) * rec_dt
                mon_dict = {}
                for m in spec.monitors:
                    mon_dict[m.kind] = _apply_monitor(
                        m.kind, m, ts_out_h[i], t_arr, spec.model)
                results.append(mon_dict)
            return results

        # Pipeline mode
        labels_set = False
        all_labels: list[str] = []
        rows: list[np.ndarray] = []

        for i in range(n_samples):
            t_arr = t_base + np.arange(n_record, dtype=np.float64) * rec_dt
            monitor_result = {pipeline.signal: (t_arr, ts_out_h[i])}
            feat_labels, feat_vals = pipeline.extract(monitor_result)
            if not labels_set:
                all_labels = param_names + feat_labels
                labels_set = True
            rows.append(np.concatenate(
                [param_sets[i].astype(np.float64), feat_vals]
            ))

        return all_labels, np.stack(rows)

    def run_df(self, duration: float):
        """Return pipeline sweep results as a pandas DataFrame."""
        import pandas as pd
        labels, values = self.run(duration)
        return pd.DataFrame(values, columns=labels)
