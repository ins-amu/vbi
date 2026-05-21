from __future__ import annotations
from typing import Callable
import numpy as np

from vbi.simulator.spec.simulation import SimulationSpec
from vbi.simulator.spec.model import ModelSpec
from vbi.simulator.spec.stimulus import StimSpec, build_stim_data
from .history import History
from .coupling import build_coupling
from .integrators import build_integrator
from .monitors import build_monitor


# ---------------------------------------------------------------------------
# dfun builder
# ---------------------------------------------------------------------------

def build_dfun(spec: ModelSpec) -> Callable:
    """
    Compile dfun_str expressions into a vectorized NumPy function.

    Generated signature:
        fn(state, coupling, params) -> np.ndarray  shape (n_sv, n_nodes)

    where:
        state    : (n_sv, n_nodes)
        coupling : (n_cvar, n_nodes)  — one row per coupling variable
        params   : dict[str, scalar | (n_nodes,)]

    For each cvar name, injects  c_{name}  as a local (n_nodes,) variable.
    Also injects  c = coupling[0]  for single-cvar backward compatibility.

    Uses exec() on our own spec strings — not user-supplied input.
    Compiled once at build() time, not per step.
    """
    sv = spec.sv_names
    param_names = spec.param_names
    cvar_names = spec.cvar

    lines = [
        "import numpy as _np",
        "from numpy import pi, exp, log, sin, cos, tanh, sqrt",
        "def _dfun(state, coupling, params):",
    ]
    # Unpack state variables
    for i, name in enumerate(sv):
        lines.append(f"    {name} = state[{i}]")
    # Unpack params
    for name in param_names:
        lines.append(f"    {name} = params['{name}']")
    # Inject coupling by cvar name: c_r, c_V, etc.
    for i, cname in enumerate(cvar_names):
        lines.append(f"    c_{cname} = coupling[{i}]")
    # Also expose c = first coupling term for single-cvar models
    lines.append("    c = coupling[0]")
    # Compute derivatives
    lines.append("    _out = _np.empty_like(state)")
    for i, name in enumerate(sv):
        lines.append(f"    _out[{i}] = {spec.dfun_str[name]}")
    lines.append("    return _out")

    src = "\n".join(lines)
    globs: dict = {}
    exec(compile(src, "<dfun>", "exec"), globs)
    return globs["_dfun"]


# ---------------------------------------------------------------------------
# Parameter helpers
# ---------------------------------------------------------------------------

def _build_params(spec: SimulationSpec) -> dict:
    params = dict(spec.model.default_params)
    params.update(spec.node_params)
    return params


def _build_initial_state(spec: SimulationSpec) -> np.ndarray:
    n_nodes = spec.n_nodes
    state = np.zeros((spec.model.n_sv, n_nodes))
    for i, sv in enumerate(spec.model.state_variables):
        state[i] = sv.default_init
    return state


def _apply_bounds(state: np.ndarray, spec: SimulationSpec) -> None:
    for i, sv in enumerate(spec.model.state_variables):
        if sv.lower_bound is not None:
            np.maximum(state[i], sv.lower_bound, out=state[i])
        if sv.upper_bound is not None:
            np.minimum(state[i], sv.upper_bound, out=state[i])


def _resolve_noise_amplitude(spec: SimulationSpec, nsig: np.ndarray) -> np.ndarray:
    """Return direct additive noise amplitude for the chosen noise convention."""
    style = getattr(spec.integrator, "noise_style", "amplitude")
    if style == "amplitude":
        return nsig
    if style == "tvb":
        return np.sqrt(2.0 * nsig)
    raise ValueError(f"Unknown noise_style: {style!r}")


# ---------------------------------------------------------------------------
# Main simulator
# ---------------------------------------------------------------------------

class NumpySimulator:
    """Pure-NumPy reference backend — used for correctness validation."""

    def build(self, spec: SimulationSpec) -> None:
        self.spec = spec
        dt = spec.integrator.dt
        n_nodes = spec.n_nodes

        self._params = _build_params(spec)
        G = float(self._params.get("G", 1.0))

        self._has_delays = spec.has_delays
        self._delay_steps = spec.delay_steps(dt)
        self._history = History(spec.horizon(dt), len(spec.model.cvar), n_nodes)
        self._coupling = build_coupling(spec.coupling, spec.weights, G)
        self._integrator = build_integrator(spec.integrator.method,
                                            spec.integrator.stochastic)
        self._monitors = [build_monitor(m, spec.model, dt)
                          for m in spec.monitors]
        self._dfun = build_dfun(spec.model)
        self._state = _build_initial_state(spec)
        self._history.initialize(self._state[list(spec.model.cvar_indices)])

        # Stimuli: resolve sv_name → cvar index, broadcast amplitude to (n_nodes,)
        self._stimuli: list[tuple[int, np.ndarray, StimSpec]] = []
        for stim in spec.stimuli:
            if stim.sv_name not in spec.model.cvar:
                raise ValueError(
                    f"StimSpec.sv_name {stim.sv_name!r} is not a coupling variable "
                    f"(model.cvar = {spec.model.cvar}).  Stimulus is injected via "
                    "the coupling array; only coupling variables are supported."
                )
            cvar_idx = spec.model.cvar.index(stim.sv_name)
            amplitude = np.broadcast_to(
                np.asarray(stim.amplitude, dtype=np.float64), (n_nodes,)
            ).copy()
            self._stimuli.append((cvar_idx, amplitude, stim))

        # Noise setup
        if spec.integrator.stochastic:
            ni = spec.model.noise_indices
            self._noise_mask = np.zeros(spec.model.n_sv, dtype=bool)
            self._noise_mask[list(ni)] = True
            nsig = spec.integrator.noise_nsig
            if nsig is None:
                nsig = np.ones(len(ni)) * 1e-3
            nsig = np.asarray(nsig, dtype=np.float64)
            self._noise_amp = _resolve_noise_amplitude(spec, nsig)
        else:
            self._noise_mask = None
            self._noise_amp = None

    def run(self, duration: float) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        spec = self.spec
        dt = spec.integrator.dt
        n_steps = round(duration / dt)
        stochastic = spec.integrator.stochastic
        rng = np.random.default_rng(spec.integrator.noise_seed) if stochastic else None

        cvar_idx = list(spec.model.cvar_indices)
        params = self._params

        def dfun_fn(state, coupling):
            # coupling: (n_cvar, n_nodes) — passed directly; build_dfun unpacks by name
            return self._dfun(state, coupling, params)

        has_delays = self._has_delays

        for step in range(n_steps):
            # 1. Coupling — fast instant path when all delays are zero
            if has_delays:
                delayed = self._history.read_delayed(self._delay_steps)
                coupling = self._coupling.compute(delayed, current_state=self._state)
            else:
                coupling = self._coupling.compute_instant(self._state[cvar_idx])

            # 1b. Stimulus — additive injection into coupling (same as TVB hybrid)
            if self._stimuli:
                t_ms = step * dt
                for ci, amplitude, stim in self._stimuli:
                    val = stim.evaluate(step, t_ms)
                    if val != 0.0:
                        coupling[ci] += val * amplitude

            # 2. Integrate
            if stochastic:
                self._state = self._integrator.step(
                    self._state, dfun_fn, coupling, dt,
                    self._noise_amp, self._noise_mask, rng)
            else:
                self._state = self._integrator.step(
                    self._state, dfun_fn, coupling, dt)

            # 3. Clamp state bounds
            _apply_bounds(self._state, spec)

            # 4. Update history (only needed for DDE)
            if has_delays:
                self._history.write(self._state[cvar_idx])

            # 5. Sample all monitors
            for mon in self._monitors:
                mon.sample(step, self._state)

        return {spec.monitors[i].kind: mon.result()
                for i, mon in enumerate(self._monitors)}
