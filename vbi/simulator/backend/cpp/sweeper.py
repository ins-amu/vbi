"""
CppSweeper — serial and parallel parameter sweep using the C++ backend.

The compiled .so is loaded once per SimulationSpec (cache key).  Each sweep
point runs run_simulation() with its own params copy; the C++ function releases
the GIL, so Python ThreadPoolExecutor gives true parallelism.

Interface matches NumpySweeper and NumbaSweeperCPU:
    run(duration)   → list[dict] | (labels, values)
    run_df(duration)→ pd.DataFrame
"""
from __future__ import annotations

import concurrent.futures
import os
import numpy as np

from vbi.simulator.spec.simulation import SimulationSpec
from vbi.simulator.spec.sweep import SweepSpec
from vbi.simulator.backend.numba_.simulator import _apply_monitor

from .build import build_or_load
from .codegen import build_params_array, get_G, get_noise_data
from vbi.simulator.spec.stimulus import build_stim_data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _patch_params(base_params: np.ndarray,
                  param_names: list[str],
                  model_param_names: list[str],
                  theta: np.ndarray,
                  n_nodes: int) -> np.ndarray:
    """Return a copy of base_params with swept parameters overridden."""
    p = base_params.copy()
    for name, val in zip(param_names, theta):
        idx = model_param_names.index(name)
        p[idx * n_nodes: idx * n_nodes + n_nodes] = float(val)
    return p


def _run_one(
    mod,
    params_flat: np.ndarray,
    state0: np.ndarray,
    weights_flat: np.ndarray,
    idelays_flat: np.ndarray,
    horizon: int,
    coup_a: float,
    coup_b: float,
    has_delays: bool,
    noise_flat: np.ndarray,
    noise_sv_idx: np.ndarray,
    n_steps: int,
    record_every: int,
    t_cut_steps: int,
    stim_flat: np.ndarray,
    has_stimulus: bool,
) -> np.ndarray:
    """Run one simulation point and return raw_data (n_record, n_sv, n_nodes)."""
    return mod.run_simulation(
        state0, weights_flat, idelays_flat, horizon,
        params_flat, coup_a, coup_b, has_delays,
        noise_flat, noise_sv_idx,
        n_steps, record_every, t_cut_steps,
        stim_data=stim_flat, has_stimulus=has_stimulus,
    )


# ---------------------------------------------------------------------------
# CppSweeper
# ---------------------------------------------------------------------------

class CppSweeper:
    """
    C++ parameter sweep — serial or parallel via threads.

    Parameters
    ----------
    spec        : SimulationSpec   base simulation
    sweep_spec  : SweepSpec        which params to vary
    n_workers   : int | None       thread-pool size; None = serial;
                                   0 = os.cpu_count() threads
    """

    def __init__(self, spec: SimulationSpec, sweep_spec: SweepSpec,
                 n_workers: int | None = None):
        if spec.coupling.kind not in ("linear", "kuramoto"):
            raise NotImplementedError(
                f"C++ sweeper supports 'linear' and 'kuramoto' coupling; "
                f"got {spec.coupling.kind!r}.")

        self.spec       = spec
        self.sweep      = sweep_spec
        self.n_workers  = n_workers

        dt      = spec.integrator.dt
        n_nodes = spec.n_nodes

        self._mod = build_or_load(spec)

        # Flat (n_params * n_nodes,) base params — sweep overrides individual rows
        self._base_params = np.ascontiguousarray(
            build_params_array(spec).ravel(), dtype=np.float64)

        G = get_G(spec)
        if spec.coupling.kind == "kuramoto":
            self._coup_a = float(G / n_nodes)
            self._coup_b = float(spec.coupling.alpha)
        else:
            self._coup_a = float(spec.coupling.a * G)
            self._coup_b = float(spec.coupling.b)

        self._weights   = np.ascontiguousarray(spec.weights.ravel(), dtype=np.float64)
        self._idelays   = np.ascontiguousarray(spec.delay_steps(dt).ravel(), dtype=np.int32)
        self._horizon   = spec.horizon(dt)
        self._has_delays= bool(spec.has_delays)

        state0 = np.zeros((spec.model.n_sv, n_nodes), dtype=np.float64)
        for i, sv in enumerate(spec.model.state_variables):
            state0[i] = sv.default_init
        self._state0 = np.ascontiguousarray(state0.ravel(), dtype=np.float64)

        # Sweep param metadata
        self._model_param_names = list(spec.model.param_names)
        self._n_nodes = n_nodes

    def _build_run_args(self, duration: float, param_set: np.ndarray,
                        param_names: list[str], seed_offset: int):
        """Build all arguments needed to call _run_one for one sweep point."""
        spec   = self.spec
        dt     = spec.integrator.dt
        n_steps = round(duration / dt)

        # record_every from first monitor period
        if spec.monitors and spec.monitors[0].period:
            record_every = max(1, round(spec.monitors[0].period / dt))
        else:
            record_every = 1

        pipeline   = self.sweep.pipeline
        t_cut_steps = round(pipeline.t_cut / dt) if pipeline is not None else 0

        # Patched params for this sweep point
        params_flat = _patch_params(
            self._base_params, param_names, self._model_param_names,
            param_set, self._n_nodes)

        # Noise: use seed_offset to give each run an independent sequence
        if spec.integrator.stochastic:
            from vbi.simulator.spec.integrator import IntegratorSpec
            # Temporarily patch seed to get per-run noise
            patched_int = IntegratorSpec(
                method=spec.integrator.method,
                dt=spec.integrator.dt,
                stochastic=True,
                noise_nsig=spec.integrator.noise_nsig,
                noise_style=getattr(spec.integrator, "noise_style", "amplitude"),
                noise_seed=spec.integrator.noise_seed + seed_offset,
            )
            patched_spec = SimulationSpec(
                model=spec.model,
                integrator=patched_int,
                coupling=spec.coupling,
                monitors=spec.monitors,
                weights=spec.weights,
                tract_lengths=spec.tract_lengths,
                speed=spec.speed,
                node_params=spec.node_params,
            )
            noise_data, noise_sv_idx = get_noise_data(patched_spec, n_steps)
            noise_flat = np.ascontiguousarray(noise_data.ravel(), dtype=np.float64)
        else:
            noise_flat   = np.empty(0, dtype=np.float64)
            noise_sv_idx = np.empty(0, dtype=np.int32)

        # Pre-sample stimuli (same for all sweep points)
        stim_data, has_stimulus = build_stim_data(spec, n_steps, dt)
        stim_flat = np.ascontiguousarray(stim_data.ravel(), dtype=np.float64)

        return (
            self._mod, params_flat, self._state0, self._weights,
            self._idelays, self._horizon, self._coup_a, self._coup_b,
            self._has_delays, noise_flat, noise_sv_idx,
            n_steps, record_every, t_cut_steps,
            stim_flat, has_stimulus,
        )

    def _raw_to_monitor(self, raw: np.ndarray) -> dict:
        spec = self.spec
        dt   = spec.integrator.dt
        record_every = 1 if not (spec.monitors and spec.monitors[0].period) else \
            max(1, round(spec.monitors[0].period / dt))
        raw_times = np.arange(raw.shape[0], dtype=np.float64) * record_every * dt
        return {m.kind: _apply_monitor(m.kind, m, raw, raw_times, spec.model)
                for m in spec.monitors}

    # ------------------------------------------------------------------
    # run — returns either list[dict] or (labels, values)
    # ------------------------------------------------------------------

    def run(self, duration: float, parallel: bool | None = None,
            batch_size: int | None = None):
        """
        Run all parameter sets.

        Parameters
        ----------
        duration   : float        simulation length in ms
        parallel   : bool | None
            True  → thread pool (n_workers threads)
            False → serial Python loop
            None  → use self.n_workers (None = serial, int = parallel)
        batch_size : int | None
            Max number of sweep points whose args are live in memory at once.
            None means process all points in one shot (original behaviour).
            Set to e.g. 64 or 128 when sweeping thousands of points to avoid
            OOM from pre-allocating every params_flat array simultaneously.

        Returns
        -------
        If sweep_spec.pipeline is None:
            list[dict]  — one monitor-result dict per run
        If pipeline is set:
            (labels, values)  shape (n_samples, n_params + n_features)
        """
        if parallel is None:
            use_parallel = self.n_workers is not None
        else:
            use_parallel = parallel

        param_names  = self.sweep._param_names_list
        param_sets   = self.sweep.param_sets   # (n_samples, n_params)
        n_samples    = param_sets.shape[0]
        pipeline     = self.sweep.pipeline

        # Resolved thread count — used both for the pool and as the default batch size.
        resolved_workers = (self.n_workers if self.n_workers and self.n_workers > 0
                            else os.cpu_count() or 4)
        workers = resolved_workers if use_parallel else None
        effective_batch = batch_size if batch_size and batch_size > 0 else resolved_workers

        # --- execute in batches ---
        raw_list: list[np.ndarray] = []

        if use_parallel:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                for start in range(0, n_samples, effective_batch):
                    batch_args = [
                        self._build_run_args(duration, param_sets[i], param_names, i)
                        for i in range(start, min(start + effective_batch, n_samples))
                    ]
                    raw_list.extend(pool.map(lambda args: _run_one(*args), batch_args))
        else:
            for start in range(0, n_samples, effective_batch):
                batch_args = [
                    self._build_run_args(duration, param_sets[i], param_names, i)
                    for i in range(start, min(start + effective_batch, n_samples))
                ]
                raw_list.extend(_run_one(*args) for args in batch_args)

        # --- post-process ---
        spec = self.spec
        dt   = spec.integrator.dt
        record_every = 1 if not (spec.monitors and spec.monitors[0].period) else \
            max(1, round(spec.monitors[0].period / dt))

        if pipeline is None:
            results = []
            for raw in raw_list:
                raw_times = np.arange(raw.shape[0], dtype=np.float64) * record_every * dt
                res = {m.kind: _apply_monitor(m.kind, m, raw, raw_times, spec.model)
                       for m in spec.monitors}
                results.append(res)
            return results

        # Pipeline mode
        labels_set = False
        all_labels: list[str] = []
        rows: list[np.ndarray] = []

        for i, raw in enumerate(raw_list):
            pipeline_t_cut = round(pipeline.t_cut / dt)
            t0 = pipeline_t_cut * dt
            raw_times = t0 + np.arange(raw.shape[0], dtype=np.float64) * record_every * dt
            monitor_result = {pipeline.signal: (raw_times, raw)}
            feat_labels, feat_vals = pipeline.extract(monitor_result)
            if not labels_set:
                all_labels = param_names + feat_labels
                labels_set = True
            rows.append(np.concatenate([param_sets[i].astype(np.float64), feat_vals]))

        return all_labels, np.stack(rows)

    def run_serial(self, duration: float, batch_size: int | None = None):
        """Convenience: force serial execution (Python loop, C++ per run)."""
        return self.run(duration, parallel=False, batch_size=batch_size)

    def run_parallel(self, duration: float, n_workers: int | None = None,
                     batch_size: int | None = None):
        """Parallel via Python ThreadPoolExecutor — n_workers threads, GIL released in C++."""
        old = self.n_workers
        if n_workers is not None:
            self.n_workers = n_workers
        result = self.run(duration, parallel=True, batch_size=batch_size)
        self.n_workers = old
        return result

    def run_df(self, duration: float, parallel: bool | None = None,
               batch_size: int | None = None):
        """Return sweep results as a pandas DataFrame (ThreadPoolExecutor mode)."""
        import pandas as pd
        labels, values = self.run(duration, parallel=parallel, batch_size=batch_size)
        return pd.DataFrame(values, columns=labels)
