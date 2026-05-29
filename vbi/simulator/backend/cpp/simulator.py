"""
CppSimulator - C++ pybind11 backend for VBI.

Compiles a model-specific .so on first use (cache in ~/.cache/vbi/cpp).
The run() interface is identical to NumpySimulator and NumbaSimulator.
"""
from __future__ import annotations

import numpy as np

from vbi.simulator.spec.simulation import SimulationSpec
from vbi.simulator.backend.numba_.simulator import _apply_monitor  # reuse post-processing

from .build import build_or_load
from .codegen import build_params_array, get_G, get_noise_data
from vbi.simulator.spec.stimulus import build_stim_data


class CppSimulator:
    """
    Single-run C++ backend.

    Interface
    ---------
    build(spec)   → None
    run(duration) → dict[str, (t_array, data_array)]
    """

    def build(self, spec: SimulationSpec, verbose: bool = False) -> None:
        if spec.coupling.kind not in ("linear", "kuramoto"):
            raise NotImplementedError(
                f"C++ backend supports 'linear' and 'kuramoto' coupling; "
                f"got {spec.coupling.kind!r}.")

        self.spec = spec
        self._mod  = build_or_load(spec, verbose=verbose)

        dt      = spec.integrator.dt
        n_nodes = spec.n_nodes

        self._params = np.ascontiguousarray(
            build_params_array(spec).ravel(), dtype=np.float64)

        G = get_G(spec)
        if spec.coupling.kind == "kuramoto":
            self._coup_a = float(G / n_nodes)          # G/N
            self._coup_b = float(spec.coupling.alpha)  # frustration angle
        else:
            self._coup_a = float(spec.coupling.a * G)
            self._coup_b = float(spec.coupling.b)

        # Connectivity
        self._weights = np.ascontiguousarray(spec.weights, dtype=np.float64)
        delay_steps   = spec.delay_steps(dt)
        self._idelays = np.ascontiguousarray(delay_steps, dtype=np.int32)
        self._horizon = spec.horizon(dt)
        self._has_delays = bool(spec.has_delays)

        # Initial state  (n_sv, n_nodes)
        state0 = np.zeros((spec.model.n_sv, n_nodes), dtype=np.float64)
        for i, sv in enumerate(spec.model.state_variables):
            state0[i] = sv.default_init
        self._state0 = np.ascontiguousarray(state0.ravel(), dtype=np.float64)

    def run(self, duration: float) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        spec    = self.spec
        dt      = spec.integrator.dt
        n_steps = round(duration / dt)

        # Noise
        if spec.integrator.stochastic:
            noise_data, noise_sv_idx = get_noise_data(spec, n_steps)
            noise_flat = np.ascontiguousarray(noise_data.ravel(), dtype=np.float64)
        else:
            noise_flat   = np.empty(0, dtype=np.float64)
            noise_sv_idx = np.empty(0, dtype=np.int32)

        # Pre-sample stimuli for this run duration
        stim_flat, has_stimulus = build_stim_data(spec, n_steps, dt)

        # Call C++
        raw_data = self._mod.run_simulation(
            self._state0,
            self._weights.ravel(),
            self._idelays.ravel(),
            self._horizon,
            self._params,
            self._coup_a,
            self._coup_b,
            self._has_delays,
            noise_flat,
            noise_sv_idx,
            n_steps,
            record_every=1,
            t_cut_steps=0,
            stim_data=np.ascontiguousarray(stim_flat.ravel(), dtype=np.float64),
            has_stimulus=has_stimulus,
        )
        # raw_data: (n_record, n_sv, n_nodes)
        raw_times = np.arange(raw_data.shape[0], dtype=np.float64) * dt

        result = {}
        for m in spec.monitors:
            result[m.kind] = _apply_monitor(m.kind, m, raw_data, raw_times, spec.model)
        return result
