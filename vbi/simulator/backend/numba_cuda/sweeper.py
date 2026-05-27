"""
CudaSweeperGPU — parallel parameter sweep on GPU.

Memory layout: coalesced (sample index is LAST / innermost dimension).
Connectivity:  dense (default for small N) or CSR sparse (default for N > 64
               or user-specified via connectivity="sparse"|"dense"|"auto").

Usage
-----
    # auto-detect sparse vs dense:
    sweeper = CudaSweeperGPU(spec, sweep_spec)

    # force sparse (good for real connectomes with ~20-40% density):
    sweeper = CudaSweeperGPU(spec, sweep_spec, connectivity="sparse")

    labels, values = sweeper.run(5000.0)   # pipeline mode
    results        = sweeper.run(5000.0)   # list[dict] when no pipeline
"""
from __future__ import annotations

import numpy as np

from vbi.simulator.spec.simulation import SimulationSpec
from vbi.simulator.spec.stimulus import build_stim_data
from vbi.simulator.spec.sweep import SweepSpec
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

try:
    from numba import cuda as _cuda
    CUDA_AVAILABLE = _cuda.is_available()
except Exception:
    CUDA_AVAILABLE = False

_TPB = 256   # threads per block
_AUTO_SPARSE_THRESHOLD = 0.5   # use CSR when density < this


def _require_cuda():
    if not CUDA_AVAILABLE:
        raise RuntimeError(
            "CUDA backend unavailable.  Ensure an NVIDIA GPU is present, "
            "the CUDA toolkit is installed, and `numba[cuda]` is installed:\n"
            "    pip install numba[cuda]\n"
            "Use backend='numba' (CPU) or backend='cpp' instead."
        )


def _count_records(n_steps: int, t_cut: int, period: int) -> int:
    if n_steps <= t_cut:
        return 0
    return (n_steps - t_cut + period - 1) // period


def _auto_sparse(weights: np.ndarray) -> bool:
    """Decide sparse vs dense based on matrix density."""
    nz      = np.count_nonzero(weights)
    total   = weights.size
    density = nz / total if total > 0 else 0.0
    return density < _AUTO_SPARSE_THRESHOLD


class CudaSweeperGPU:
    """
    GPU parameter sweep backend.

    Parameters
    ----------
    spec          : SimulationSpec
    sweep_spec    : SweepSpec
    connectivity  : "auto" | "dense" | "sparse"
        "auto"   — use CSR sparse when weight matrix density < 50 %
        "dense"  — always use the full (n_nodes × n_nodes) float32 matrix
        "sparse" — always use CSR (w_data / w_indices / w_indptr)
    """

    def __init__(self, spec: SimulationSpec, sweep_spec: SweepSpec,
                 connectivity: str = "auto"):
        _require_cuda()
        if spec.coupling.kind not in ("linear", "kuramoto"):
            raise NotImplementedError(
                f"CUDA backend supports 'linear' and 'kuramoto' coupling; "
                f"got {spec.coupling.kind!r}."
            )
        _use_kuramoto = spec.coupling.kind == "kuramoto"
        _alpha        = float(spec.coupling.alpha)

        self.spec          = spec
        self.sweep         = sweep_spec
        self._use_kuramoto = _use_kuramoto

        dt      = spec.integrator.dt
        n_nodes = spec.n_nodes
        model   = spec.model

        # Sparse / dense decision
        if connectivity not in {"auto", "dense", "sparse"}:
            raise ValueError(
                f"connectivity must be 'auto', 'dense', or 'sparse'; got {connectivity!r}."
            )
        if connectivity == "auto":
            self._sparse = _auto_sparse(spec.weights)
        elif connectivity == "sparse":
            self._sparse = True
        else:
            self._sparse = False

        # Compile CUDA module (kernel source depends on sparse flag)
        self._mod = build_cuda_module(model, sparse=self._sparse,
                                      use_kuramoto=_use_kuramoto, alpha=_alpha)

        # Connectivity arrays
        self._weights_f32  = np.ascontiguousarray(spec.weights, dtype=np.float32)
        ds = spec.delay_steps(dt).astype(np.int32)
        self._delay_steps  = np.ascontiguousarray(ds)
        self._horizon      = spec.horizon(dt)
        self._has_delays   = bool(spec.has_delays)

        if self._sparse:
            (self._w_data, self._w_indices, self._w_indptr,
             self._idelays_csr, self._nnz, density) = to_csr(
                 spec.weights, ds if self._has_delays else None,
                 has_delays=self._has_delays)
            self._conn_info = f"CSR sparse  nnz={self._nnz}  density={density:.1%}"
        else:
            density = np.count_nonzero(spec.weights) / max(1, spec.weights.size)
            self._conn_info = f"dense  N={n_nodes}  density={density:.1%}"

        # Coupling scale
        G = float(np.asarray(
            spec.node_params.get("G", model.default_params.get("G", 1.0))
        ).mean())
        if _use_kuramoto:
            self._coup_a = np.float32(G / n_nodes)
            self._coup_b = np.float32(_alpha)
        else:
            self._coup_a = np.float32(spec.coupling.a * G)
            self._coup_b = np.float32(spec.coupling.b)

        # State / bounds
        self._lo, self._hlo, self._hi, self._hhi = get_bounds_arrays(model)
        self._use_heun   = (spec.integrator.method == "heun")
        self._stochastic = spec.integrator.stochastic
        self._seed_base  = int(spec.integrator.noise_seed)
        self._sweep_names = sweep_spec._param_names_list

        # Validate sweep parameter names early.
        pnames = list(spec.model.param_names)
        _supported = set(pnames) | {"G"}
        for name in self._sweep_names:
            if name not in _supported:
                raise ValueError(
                    f"CUDA backend: unknown sweep parameter {name!r}. "
                    f"Supported: model parameters {pnames} or 'G' (global coupling scale)."
                )

    def _transfer_connectivity(self):
        """Transfer connectivity arrays to device; returns dict of device arrays."""
        from numba import cuda
        if self._sparse:
            return dict(
                w_data_d    = cuda.to_device(self._w_data),
                w_indices_d = cuda.to_device(self._w_indices),
                w_indptr_d  = cuda.to_device(self._w_indptr),
                idelays_d   = cuda.to_device(self._idelays_csr),
            )
        else:
            return dict(
                weights_d = cuda.to_device(self._weights_f32),
                delays_d  = cuda.to_device(self._delay_steps),
            )

    def run(self, duration: float):
        """
        Run all parameter sets on GPU.

        Returns
        -------
        list[dict]        — when sweep_spec.pipeline is None
        (labels, values)  — when pipeline is set
        """
        from numba import cuda

        spec        = self.spec
        dt          = np.float32(spec.integrator.dt)
        n_steps     = int(round(duration / float(dt)))
        pipeline    = self.sweep.pipeline
        param_sets  = self.sweep.param_sets
        n_samples   = param_sets.shape[0]
        param_names = self.sweep._param_names_list

        _NEEDS_FULL_RES = {"tavg", "bold"}
        if spec.monitors and any(m.kind in _NEEDS_FULL_RES for m in spec.monitors):
            # tavg/bold post-processing requires raw-resolution samples.
            record_period = 1
        elif spec.monitors and spec.monitors[0].period:
            record_period = max(1, round(float(spec.monitors[0].period) / float(dt)))
        else:
            record_period = 1

        t_cut_step = (round(pipeline.t_cut / float(dt))
                      if pipeline is not None else 0)
        n_record   = _count_records(n_steps, t_cut_step, record_period)

        n_sv    = spec.model.n_sv
        n_cvar  = len(spec.model.cvar)
        n_nodes = spec.n_nodes

        # ---- Host arrays (coalesced layout: tid last) ----
        params_h = build_params_matrix(
            spec, n_samples,
            sweep_names=param_names,
            sweep_sets=param_sets.astype(np.float64),
        )   # (n_params, n_samples)

        cvar_idx  = np.array(spec.model.cvar_indices, dtype=np.int32)
        init_cvar = np.zeros((n_cvar, n_nodes), dtype=np.float32)
        for ci_i, ci in enumerate(spec.model.cvar_indices):
            init_cvar[ci_i, :] = spec.model.state_variables[ci].default_init

        state_h  = build_initial_state(spec, n_samples)   # (n_sv, n_nodes, n_samples)
        buf_h    = build_ring_buffer(n_samples, n_cvar, n_nodes,
                                     self._horizon, init_cvar)  # (n_cvar, n_nodes, hor, n_samp)
        # Memory guard: warn if ts_out would exceed GPU memory
        ts_bytes = n_record * n_sv * n_nodes * n_samples * 4
        try:
            from numba import cuda as _c
            free_mem, total_mem = _c.current_context().get_memory_info()
            if ts_bytes > free_mem * 0.8:
                import warnings
                warnings.warn(
                    f"CUDA ts_out array ({ts_bytes/1e9:.2f} GB) may exceed available "
                    f"GPU memory ({free_mem/1e9:.2f} GB free). "
                    "Use a larger monitor period (e.g. MonitorSpec('tavg', period=1.0)) "
                    "to reduce output size, or reduce n_samples.",
                    RuntimeWarning, stacklevel=2,
                )
        except Exception:
            pass
        ts_out_h = np.zeros((n_record, n_sv, n_nodes, n_samples), dtype=np.float32)

        # ---- Transfer to device ----
        state_d    = cuda.to_device(state_h)
        buf_d      = cuda.to_device(buf_h)
        params_d   = cuda.to_device(params_h)
        cvar_idx_d = cuda.to_device(cvar_idx)
        lo_d       = cuda.to_device(self._lo)
        hlo_d      = cuda.to_device(self._hlo)
        hi_d       = cuda.to_device(self._hi)
        hhi_d      = cuda.to_device(self._hhi)
        ts_out_d   = cuda.to_device(ts_out_h)
        conn       = self._transfer_connectivity()

        blocks = (n_samples + _TPB - 1) // _TPB

        # Per-sample coupling scale — base value broadcast; overridden if G is swept.
        coup_a_h = np.full(n_samples, self._coup_a, dtype=np.float32)
        if "G" in param_names:
            g_idx = param_names.index("G")
            G_vals = param_sets[:, g_idx].astype(np.float32)
            if self._use_kuramoto:
                coup_a_h = (G_vals / spec.n_nodes).astype(np.float32)
            else:
                coup_a_h = (np.float32(spec.coupling.a) * G_vals).astype(np.float32)
        coup_a_d = cuda.to_device(coup_a_h)

        # Pre-sample stimuli (same for all sweep samples) and transfer to device
        stim_np, has_stimulus = build_stim_data(spec, n_steps, float(dt))
        stim_flat = np.ascontiguousarray(stim_np.ravel(), dtype=np.float32)
        stim_d    = cuda.to_device(stim_flat)

        # ---- Common kernel args (order matches kernel signature) ----
        common = [
            state_d, buf_d,
        ]
        if self._sparse:
            common += [conn["w_data_d"], conn["w_indices_d"],
                       conn["w_indptr_d"], conn["idelays_d"]]
        else:
            common += [conn["weights_d"], conn["delays_d"]]
        common += [
            params_d, cvar_idx_d,
            lo_d, hlo_d, hi_d, hhi_d,
            np.int32(self._horizon),
            dt,
            np.int32(n_steps),
            np.int32(n_record),
            np.int32(t_cut_step),
            np.int32(record_period),
            coup_a_d, self._coup_b,
            self._has_delays,
            self._use_heun,
            stim_d, has_stimulus,
        ]

        if self._stochastic:
            noise_h = generate_noise(spec, n_steps, n_samples, self._seed_base,
                                     same_noise=self.sweep.same_noise)
            noise_d = cuda.to_device(noise_h)
            self._mod.cuda_sweep_stoch[blocks, _TPB](*common, noise_d, ts_out_d)
        else:
            self._mod.cuda_sweep_det[blocks, _TPB](*common, ts_out_d)

        cuda.synchronize()

        # ---- Copy back; transpose to (n_samples, n_record, n_sv, n_nodes) ----
        # ts_out on device is (n_record, n_sv, n_nodes, n_samples)
        ts_dev = ts_out_d.copy_to_host()                     # (nr, n_sv, nn, ns)
        ts     = np.ascontiguousarray(
                     ts_dev.transpose(3, 0, 1, 2)            # (ns, nr, n_sv, nn)
                 ).astype(np.float64)

        # ---- Post-process ----
        rec_dt = float(dt) * record_period
        t_base = t_cut_step * float(dt)

        if pipeline is None:
            results = []
            for i in range(n_samples):
                t_arr = t_base + np.arange(n_record, dtype=np.float64) * rec_dt
                mon_dict = {}
                for m in spec.monitors:
                    mon_dict[m.kind] = _apply_monitor(
                        m.kind, m, ts[i], t_arr, spec.model)
                results.append(mon_dict)
            return results

        _pipeline_mon = next(
            (m for m in spec.monitors if m.kind == pipeline.signal), None)
        labels_set = False
        all_labels: list[str] = []
        rows: list[np.ndarray] = []
        for i in range(n_samples):
            t_arr = t_base + np.arange(n_record, dtype=np.float64) * rec_dt
            if _pipeline_mon is not None:
                mon_t, mon_data = _apply_monitor(
                    pipeline.signal, _pipeline_mon, ts[i], t_arr, spec.model)
            else:
                mon_t, mon_data = t_arr, ts[i]
            feat_labels, feat_vals = pipeline.extract(
                {pipeline.signal: (mon_t, mon_data)})
            if not labels_set:
                all_labels = param_names + feat_labels
                labels_set = True
            rows.append(np.concatenate(
                [param_sets[i].astype(np.float64), feat_vals]))
        return all_labels, np.stack(rows)

    def run_df(self, duration: float):
        import pandas as pd
        labels, values = self.run(duration)
        return pd.DataFrame(values, columns=labels)
