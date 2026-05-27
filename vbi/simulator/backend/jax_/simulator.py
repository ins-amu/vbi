"""
JAX backend — differentiable simulator using jax.lax.scan + jax.vmap.

Design notes
------------
Ring buffer
    Shape (horizon, n_cvar, n_nodes); stored in lax.scan carry.
    Write: buf.at[step % horizon].set(new_cvar)
    Read:  flat-index trick → buf[:, cv, :].reshape(-1)[idx_time*N + j]
           avoids Python loops inside traced code.

Time loop
    Nested scan:
        outer scan  → n_record iterations, one monitor sample per iteration
        inner scan  → record_period steps each (subsample / tavg)
    For raw monitor: record_period = 1, inner scan is trivial.
    All carry shapes are static (JAX requirement).

Noise
    jax.random.fold_in(master_key, step) — no key splitting, cheap,
    reproducible across runs; adapted from bold_delay_gpu.py.
    master_key is static (not part of JIT-recompile trigger).

Sweep (JaxSweeper)
    Swept params are collected into a dict of shape-(n_samples,) arrays.
    jax.vmap maps axis-0 of those arrays → one scalar per run inside the
    vmapped function.  jax.jit wraps the vmapped call.
    Non-swept data (weights, delays, base_params, etc.) goes into a
    static pytree shared across all runs.
"""
from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from vbi.simulator.spec.simulation import SimulationSpec
from vbi.simulator.spec.stimulus import build_stim_data
from .codegen import build_jax_dfun


# ---------------------------------------------------------------------------
# Balloon-Windkessel helpers (2-state, matching numpy/C++ backends)
# State per node: [s, f]  (vasodilatory signal, blood inflow)
# ---------------------------------------------------------------------------

_BW_DEFAULTS = dict(rho=0.8, e=0.02, taus=0.8, tauf=0.4, k1=5.6, eps=0.5)


def _bw_init_jax(n_nodes: int) -> jnp.ndarray:
    """Initial BW state (2, n_nodes): s=0, f=1."""
    return jnp.stack([jnp.zeros(n_nodes), jnp.ones(n_nodes)])


def _bw_step_jax(bw: jnp.ndarray, neural: jnp.ndarray, dt_sec: float,
                 p: dict) -> jnp.ndarray:
    """
    Euler step of the simplified Balloon-Windkessel ODE.

    ds/dt = neural * eps - s/taus - (f-1)/tauf
    df/dt = s
    """
    s, f = bw[0], bw[1]
    ds = neural * p["eps"] - s / p["taus"] - (f - 1.0) / p["tauf"]
    df = s
    return jnp.stack([s + dt_sec * ds, f + dt_sec * df])


def _bw_bold_jax(bw: jnp.ndarray, p: dict) -> jnp.ndarray:
    """BOLD signal: coef * (f - 1),  shape (n_nodes,)."""
    coef = (100.0 / p["rho"]) * p["e"] * p["k1"]
    return coef * (bw[1] - 1.0)


# ---------------------------------------------------------------------------
# Parameter helpers
# ---------------------------------------------------------------------------

def _build_jax_params(spec: SimulationSpec) -> dict:
    """Return a plain dict of scalar/array params (numpy → python floats)."""
    params = dict(spec.model.default_params)
    params.update(spec.node_params)
    return params


def _resolve_dtype(spec: SimulationSpec):
    """Return the numpy/jax dtype from IntegratorSpec.jax_dtype."""
    jax_dtype = getattr(spec.integrator, "jax_dtype", "float32")
    if jax_dtype == "float64":
        import jax
        jax.config.update("jax_enable_x64", True)
        return np.float64
    return np.float32


def _build_initial_state(spec: SimulationSpec) -> jnp.ndarray:
    dtype = _resolve_dtype(spec)
    n_nodes = spec.n_nodes
    state = np.zeros((spec.model.n_sv, n_nodes), dtype=dtype)
    for i, sv in enumerate(spec.model.state_variables):
        state[i] = sv.default_init
    return jnp.array(state)


def _build_bounds(spec: SimulationSpec) -> tuple[jnp.ndarray, jnp.ndarray]:
    dtype = _resolve_dtype(spec)
    lo = np.array(
        [sv.lower_bound if sv.lower_bound is not None else -np.inf
         for sv in spec.model.state_variables],
        dtype=dtype,
    )
    hi = np.array(
        [sv.upper_bound if sv.upper_bound is not None else np.inf
         for sv in spec.model.state_variables],
        dtype=dtype,
    )
    return jnp.array(lo), jnp.array(hi)


def _build_noise_amp(spec: SimulationSpec) -> jnp.ndarray:
    """(n_sv,) amplitude array — zero for non-noisy state vars."""
    dtype = _resolve_dtype(spec)
    n_sv = spec.model.n_sv
    amp = np.zeros(n_sv, dtype=dtype)
    ni = list(spec.model.noise_indices)
    nsig = spec.integrator.noise_nsig
    if nsig is None:
        nsig = np.ones(len(ni), dtype=dtype) * 1e-3
    nsig = np.asarray(nsig, dtype=dtype)
    style = getattr(spec.integrator, "noise_style", "amplitude")
    if style == "tvb":
        nsig = np.sqrt(2.0 * nsig)
    amp[ni] = nsig
    return jnp.array(amp)


# ---------------------------------------------------------------------------
# Ring-buffer primitives
# ---------------------------------------------------------------------------

def _write_ring(buf: jnp.ndarray, step: int, cvar_state: jnp.ndarray,
                horizon: int) -> jnp.ndarray:
    """buf: (horizon, n_cvar, n_nodes) → updated buf."""
    return buf.at[step % horizon].set(cvar_state)


def _read_delayed_coupling(
        buf: jnp.ndarray, step: int,
        delay_steps: jnp.ndarray, weights: jnp.ndarray,
        G: float, a: float, b: float,
        horizon: int, n_nodes: int) -> jnp.ndarray:
    """
    Returns coupling (n_cvar, n_nodes).

    delay_steps : (n_nodes, n_nodes) int32  — delay_steps[tgt, src]
    weights     : (n_nodes, n_nodes)        — weights[tgt, src]

    Uses the flat-index trick to turn a 2-D dynamic gather into a 1-D gather:
        delayed[tgt, src] = buf[idx_time[tgt,src], cvar, src]
        flat_idx = idx_time * n_nodes + src_idx
    """
    idx_time = (step - delay_steps) % horizon        # (N, N)
    src_idx = jnp.arange(n_nodes, dtype=jnp.int32)  # (N,)
    flat_idx = idx_time * n_nodes + src_idx[None, :] # (N, N)

    n_cvar = buf.shape[1]
    cvars = []
    for cv in range(n_cvar):
        buf_cv = buf[:, cv, :]                       # (horizon, N)
        delayed = buf_cv.reshape(-1)[flat_idx]       # (N, N)
        cvars.append(G * a * jnp.sum(weights * delayed, axis=1) + b)
    return jnp.stack(cvars)                          # (n_cvar, N)


def _instant_coupling(
        cvar_state: jnp.ndarray, weights: jnp.ndarray,
        G: float, a: float, b: float) -> jnp.ndarray:
    """No-delay path.  cvar_state: (n_cvar, N)  →  coupling: (n_cvar, N)."""
    # coupling[c, tgt] = G*a * sum_src(weights[tgt,src] * cvar_state[c,src]) + b
    return G * a * jnp.einsum("ts,cs->ct", weights, cvar_state) + b


def _sigmoidal_coupling(
        cvar_state: jnp.ndarray, weights: jnp.ndarray,
        G: float, a: float, b: float,
        midpoint: float, sigma: float) -> jnp.ndarray:
    """Sigmoidal pre-synaptic coupling: 1/(1+exp(-(x-midpoint)/sigma))."""
    sig = 1.0 / (1.0 + jnp.exp(-(cvar_state - midpoint) / sigma))
    return G * a * jnp.einsum("ts,cs->ct", weights, sig) + b


def _kuramoto_coupling_instant(
        cvar_state: jnp.ndarray, weights: jnp.ndarray,
        G: float, n_nodes: int, alpha: float = 0.0) -> jnp.ndarray:
    """c[tgt] = (G/N) Σ_src W[tgt,src] sin(θ_src − θ_tgt + alpha)."""
    theta = cvar_state[0]
    diff  = theta[:, None] - theta[None, :] + alpha
    c = (G / n_nodes) * jnp.einsum("ts,st->t", weights, jnp.sin(diff))
    return c[None, :]


def _read_delayed_kuramoto_coupling(
        buf, step, delay_steps, weights, G, horizon, n_nodes,
        current_state, alpha: float = 0.0) -> jnp.ndarray:
    """Delayed Kuramoto: c[tgt] = (G/N) Σ_src W[tgt,src] sin(θ_src(t-τ) − θ_tgt(t) + alpha)."""
    idx_time = (step - delay_steps) % horizon
    src_idx  = jnp.arange(n_nodes, dtype=jnp.int32)
    flat_idx = idx_time * n_nodes + src_idx[None, :]

    buf_theta = buf[:, 0, :]
    theta_src_delayed = buf_theta.reshape(-1)[flat_idx]

    theta_tgt = current_state[0]
    diff = theta_src_delayed - theta_tgt[:, None] + alpha
    c = (G / n_nodes) * jnp.einsum("ts,ts->t", weights, jnp.sin(diff))
    return c[None, :]


# ---------------------------------------------------------------------------
# Integrators (pure JAX, traceable)
# ---------------------------------------------------------------------------

def _heun_det(state, dfun, coupling, dt, params):
    k1 = dfun(state, coupling, params)
    k2 = dfun(state + dt * k1, coupling, params)
    return state + 0.5 * dt * (k1 + k2)


def _euler_det(state, dfun, coupling, dt, params):
    return state + dt * dfun(state, coupling, params)


def _heun_stoch(state, dfun, coupling, dt, params, noise):
    k1 = dfun(state, coupling, params)
    x_pred = state + dt * k1 + noise
    k2 = dfun(x_pred, coupling, params)
    return state + 0.5 * dt * (k1 + k2) + noise


def _euler_stoch(state, dfun, coupling, dt, params, noise):
    return state + dt * dfun(state, coupling, params) + noise


# ---------------------------------------------------------------------------
# Core scan-body factory
# ---------------------------------------------------------------------------

def _make_step_fn(
        dfun: Callable,
        weights: jnp.ndarray,
        delay_steps: jnp.ndarray,
        has_delays: bool,
        horizon: int,
        n_nodes: int,
        cvar_indices: tuple[int, ...],
        lo_bounds: jnp.ndarray,
        hi_bounds: jnp.ndarray,
        dt: float,
        G_default: float,
        coup_a: float,
        coup_b: float,
        coup_kind: str,
        coup_midpoint: float,
        coup_sigma: float,
        coup_alpha: float,
        use_heun: bool,
        stochastic: bool,
        noise_amp: jnp.ndarray,
        master_key,
) -> Callable:
    """
    Returns step(carry, _) → (carry, state)
    carry = (state, buf, step_int32, params)

    params is part of the carry so that vmap can inject swept values.
    G is read from params["G"] at every step so that parameter sweeps and
    jax.grad work correctly — it is never baked into a closure constant.
    """
    cvar_idx = jnp.array(list(cvar_indices), dtype=jnp.int32)
    sqrt_dt = jnp.float32(jnp.sqrt(dt))

    def _coupling(buf, step, state, params):
        # G from params so that swept G (vmap) and grad both work correctly
        G = params.get("G", G_default)
        if coup_kind == "kuramoto":
            if has_delays:
                return _read_delayed_kuramoto_coupling(
                    buf, step, delay_steps, weights,
                    G, horizon, n_nodes, state, coup_alpha)
            return _kuramoto_coupling_instant(state[cvar_idx], weights, G, n_nodes, coup_alpha)
        if has_delays:
            return _read_delayed_coupling(
                buf, step, delay_steps, weights,
                G, coup_a, coup_b, horizon, n_nodes)
        cvar_state = state[cvar_idx]   # (n_cvar, N)
        if coup_kind == "sigmoidal":
            return _sigmoidal_coupling(
                cvar_state, weights, G, coup_a, coup_b,
                coup_midpoint, coup_sigma)
        return _instant_coupling(cvar_state, weights, G, coup_a, coup_b)

    def step(carry, _):
        state, buf, t, params, stim_data = carry
        coup = _coupling(buf, t, state, params)
        coup = coup + stim_data[t]   # stim_data[t]: dynamic JAX gather, zero when no stimulus

        if stochastic:
            step_key = jax.random.fold_in(master_key, t.astype(jnp.uint32))
            z = jax.random.normal(step_key, state.shape)
            noise = noise_amp[:, None] * sqrt_dt * z
            new_state = (_heun_stoch if use_heun else _euler_stoch)(
                state, dfun, coup, dt, params, noise)
        else:
            new_state = (_heun_det if use_heun else _euler_det)(
                state, dfun, coup, dt, params)

        new_state = jnp.clip(new_state, lo_bounds[:, None], hi_bounds[:, None])

        if has_delays:
            new_buf = _write_ring(buf, t, new_state[cvar_idx], horizon)
        else:
            new_buf = buf  # no-op: buf unused without delays

        return (new_state, new_buf, t + 1, params, stim_data), new_state

    return step


# ---------------------------------------------------------------------------
# Monitor runners  (nested-scan strategy for subsample / tavg)
# ---------------------------------------------------------------------------

def _run_monitor(
        step_fn: Callable,
        init_carry,
        monitor_spec,
        n_steps: int,
        dt: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Run simulation and return (times, data) for the given monitor.

    For raw (period == dt):   scan length = n_steps, output every step.
    For subsample / tavg:     nested scan — inner = istep steps, outer = n_record.
    For gavg:                 spatial mean applied to subsample output.
    """
    kind = monitor_spec.kind
    period = monitor_spec.period if monitor_spec.period is not None else dt
    istep = max(1, round(period / dt))

    if kind == "bold":
        tr = getattr(monitor_spec, "tr", 2000.0)   # ms
        tr_steps = round(tr / dt)
        n_record = n_steps // tr_steps
        n_nodes = init_carry[0].shape[1]            # state: (n_sv, n_nodes)
        bw0 = _bw_init_jax(n_nodes)
        dt_sec = jnp.float32(dt * 1e-3)            # ms → s for BW ODE
        p = _BW_DEFAULTS

        def bold_outer(carry, _):
            neural_carry, bw = carry

            def inner_bold(c, __):
                nc, bw_i = c
                new_nc, new_state = step_fn(nc, None)
                # First state variable (r) drives the BW ODE
                new_bw = _bw_step_jax(bw_i, new_state[0], dt_sec, p)
                return (new_nc, new_bw), None

            (new_nc, new_bw), _ = jax.lax.scan(
                inner_bold, (neural_carry, bw), None, length=tr_steps)

            bold_signal = _bw_bold_jax(new_bw, p)[None, :]   # (1, n_nodes)
            _, _, t, _, _ = new_nc
            t_out = jnp.float32(t) * dt
            return (new_nc, new_bw), (t_out, bold_signal)

        _, (times, bolds) = jax.lax.scan(
            bold_outer, (init_carry, bw0), None, length=n_record)
        # times: (n_record,)   bolds: (n_record, 1, n_nodes)
        return times, bolds

    if kind == "raw":
        # One output per integration step — simple scan
        _, all_states = jax.lax.scan(
            step_fn, init_carry, None, length=n_steps)
        # all_states: (n_steps, n_sv, n_nodes) — dtype follows state0
        times = jnp.arange(n_steps, dtype=all_states.dtype) * dt
        return times, all_states

    # subsample / tavg / gavg — outer scan, inner scan of istep steps
    n_record = n_steps // istep

    def outer_body(carry, _):
        inner_carry, inner_ts = jax.lax.scan(
            step_fn, carry, None, length=istep)
        # inner_ts: (istep, n_sv, n_nodes)

        if kind in ("subsample", "gavg"):
            recorded = inner_ts[0]   # 1st step of window = step % istep == 0
        else:  # tavg
            recorded = inner_ts.mean(axis=0)

        if kind == "gavg":
            recorded = recorded.mean(axis=-1, keepdims=True)  # (n_sv, 1)

        # Time: step counter is in carry after inner scan
        _, _, t, _, _ = inner_carry
        t_out = jnp.float32(t - istep + 1) * dt
        return inner_carry, (t_out, recorded)

    _, (times, data) = jax.lax.scan(
        outer_body, init_carry, None, length=n_record)
    # times: (n_record,)
    # data:  (n_record, n_sv, n_nodes) [or (n_record, n_sv, 1) for gavg]
    return times, data


# ---------------------------------------------------------------------------
# JaxSimulator
# ---------------------------------------------------------------------------

class JaxSimulator:
    """JAX backend — single run, jit-compiled, differentiable."""

    def build(self, spec: SimulationSpec) -> None:
        self.spec = spec
        dt = spec.integrator.dt
        n_nodes = spec.n_nodes

        self._dt = dt
        self._params = _build_jax_params(spec)
        self._state0 = _build_initial_state(spec)
        self._lo, self._hi = _build_bounds(spec)

        G = float(self._params.get("G", 1.0))
        self._dtype = _resolve_dtype(spec)
        self._weights = jnp.array(spec.weights, dtype=self._dtype)
        self._delay_steps = jnp.array(spec.delay_steps(dt), dtype=jnp.int32)
        self._horizon = spec.horizon(dt)

        stochastic = spec.integrator.stochastic
        self._noise_amp = _build_noise_amp(spec) if stochastic else jnp.zeros(spec.model.n_sv)
        self._master_key = jax.random.PRNGKey(spec.integrator.noise_seed)

        self._dfun = build_jax_dfun(spec.model)

        self._step_fn = _make_step_fn(
            dfun=self._dfun,
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
            coup_alpha=float(getattr(spec.coupling, "alpha", 0.0)),
            use_heun=(spec.integrator.method == "heun"),
            stochastic=stochastic,
            noise_amp=self._noise_amp,
            master_key=self._master_key,
        )

        # static_argnums=(1,): n_steps is a Python int → concrete in lax.scan
        self._run_jit = jax.jit(self._run_core, static_argnums=(1,))

    def _run_core(self, params: dict, n_steps: int,
                  stim_data: jnp.ndarray | None = None) -> dict:
        if stim_data is None:
            n_cvar  = len(self.spec.model.cvar)
            stim_data = jnp.zeros(
                (n_steps, n_cvar, self.spec.n_nodes), dtype=self._dtype)
        horizon = self._horizon

        # Initialise ring buffer
        cvar_idx = list(self.spec.model.cvar_indices)
        cvar_idx_jnp = jnp.array(cvar_idx, dtype=jnp.int32)
        buf = jnp.zeros(
            (horizon, len(cvar_idx), self.spec.n_nodes), dtype=self._dtype)
        buf = buf.at[:].set(self._state0[cvar_idx_jnp][None, :, :])

        carry = (self._state0, buf, jnp.int32(0), params, stim_data)

        results = {}
        for mon_spec in self.spec.monitors:
            # Each monitor gets its own run — trade compile time for simplicity
            # (for multiple monitors on one run, nest them in one scan body)
            times, data = _run_monitor(
                self._step_fn, carry, mon_spec, n_steps, self._dt)
            results[mon_spec.kind] = (times, data)

        return results

    def run(self, duration: float) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        n_steps = round(duration / self._dt)
        stim_np, _ = build_stim_data(self.spec, n_steps, self._dt)
        stim_jax = jnp.array(stim_np, dtype=self._dtype)   # (n_steps, n_cvar, n_nodes)
        jax_results = self._run_jit(self._params, n_steps, stim_jax)
        # Convert JAX arrays to numpy after JIT boundary
        return {kind: (np.array(t), np.array(d))
                for kind, (t, d) in jax_results.items()}
