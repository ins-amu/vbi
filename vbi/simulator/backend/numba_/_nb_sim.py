"""
Fixed @njit utility functions for the Numba CPU backend.

The model-specific dfun is passed as a first-class @njit function argument,
following the pattern used in the TVB hybrid Numba backend.  Everything else
(coupling, integrators, simulate loop, parallel sweep) lives here so Numba
can load these functions from disk (cache=True is possible for the pieces
that do not take a dynamic dfun argument).

Ring-buffer layout  : srcbuf[cv, node, step % horizon]
Delayed-read index  : (step - 1 - delay + horizon) % horizon
This matches TVB DenseHistory semantics exactly (validated in test_mpr_numpy.py).

Parameter layout    : params[param_idx, node]  float64  (n_params, n_nodes)
All params are broadcast to n_nodes columns; scalar params have equal columns.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

from vbi.feature_extraction.features_utils_nb import nb_extract


# ---------------------------------------------------------------------------
# Coupling — Linear (instant, no delays)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _coup_linear_instant(cvar_state, weights, G, a, b):
    """
    c[cv, tgt] = G * a * sum_src(w[tgt, src] * cvar_state[cv, src]) + b

    cvar_state : (n_cvar, n_nodes)
    weights    : (n_nodes, n_nodes)   weights[tgt, src]
    Returns    : (n_cvar, n_nodes)
    """
    n_cvar = cvar_state.shape[0]
    n_nodes = cvar_state.shape[1]
    out = np.zeros((n_cvar, n_nodes))
    for cv in range(n_cvar):
        for tgt in range(n_nodes):
            s = 0.0
            for src in range(n_nodes):
                s += weights[tgt, src] * cvar_state[cv, src]
            out[cv, tgt] = G * a * s + b
    return out


# ---------------------------------------------------------------------------
# Coupling — Kuramoto sinusoidal (instant, no delays)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _coup_kuramoto_instant(cvar_state, weights, G, n_nodes, alpha):
    """
    c[tgt] = (G/N) Σ_src W[tgt,src] sin(theta[src] - theta[tgt] + alpha)
    alpha=0 → standard Kuramoto; alpha≠0 → frustrated Kuramoto.
    cvar_state : (1, n_nodes)
    """
    out = np.zeros((1, n_nodes))
    theta = cvar_state[0]
    for tgt in range(n_nodes):
        s = 0.0
        for src in range(n_nodes):
            s += weights[tgt, src] * np.sin(theta[src] - theta[tgt] + alpha)
        out[0, tgt] = (G / n_nodes) * s
    return out


# ---------------------------------------------------------------------------
# Coupling — Kuramoto sinusoidal (delayed, ring-buffer read)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _coup_kuramoto_delayed(srcbuf, step, horizon, weights, delay_steps, G, n_nodes,
                           current_theta, alpha):
    """
    c[tgt] = (G/N) Σ_src W[tgt,src] sin(theta_src(t-τ) - theta_tgt(t) + alpha)
    srcbuf        : (1, n_nodes, horizon)
    current_theta : (n_nodes,)
    """
    out = np.zeros((1, n_nodes))
    t = step - 1
    for tgt in range(n_nodes):
        s = 0.0
        for src in range(n_nodes):
            d = delay_steps[src, tgt]
            idx = (t - d + horizon) % horizon
            theta_src_delayed = srcbuf[0, src, idx]
            s += weights[tgt, src] * np.sin(theta_src_delayed - current_theta[tgt] + alpha)
        out[0, tgt] = (G / n_nodes) * s
    return out


# ---------------------------------------------------------------------------
# Coupling — Linear (delayed, ring-buffer read)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _coup_linear_delayed(srcbuf, step, horizon, weights, delay_steps, G, a, b):
    """
    c[cv, tgt] = G * a * sum_src(w[tgt,src] * srcbuf[cv, src, delayed_idx]) + b

    srcbuf      : (n_cvar, n_nodes, horizon)
    delay_steps : (n_nodes, n_nodes) int32   delay_steps[src, tgt]
    step        : current loop counter (read happens before write of this step)
    Returns     : (n_cvar, n_nodes)
    """
    n_cvar = srcbuf.shape[0]
    n_nodes = srcbuf.shape[1]
    out = np.zeros((n_cvar, n_nodes))
    t = step - 1   # last-written logical step
    for cv in range(n_cvar):
        for tgt in range(n_nodes):
            s = 0.0
            for src in range(n_nodes):
                d = delay_steps[src, tgt]
                idx = (t - d + horizon) % horizon
                s += weights[tgt, src] * srcbuf[cv, src, idx]
            out[cv, tgt] = G * a * s + b
    return out


# ---------------------------------------------------------------------------
# Integrators  (dfun_fn passed as first-class @njit function)
# ---------------------------------------------------------------------------

@njit(cache=False)
def _heun_det(state, coupling, params, dt, dfun_fn):
    k1 = dfun_fn(state, coupling, params)
    k2 = dfun_fn(state + dt * k1, coupling, params)
    return state + 0.5 * dt * (k1 + k2)


@njit(cache=False)
def _euler_det(state, coupling, params, dt, dfun_fn):
    return state + dt * dfun_fn(state, coupling, params)


@njit(cache=False)
def _heun_stoch(state, coupling, params, dt, dfun_fn, dW):
    """Stochastic Heun: dW = pre-generated (n_sv, n_nodes) noise for this step."""
    k1 = dfun_fn(state, coupling, params)
    x1 = state + dt * k1 + dW
    k2 = dfun_fn(x1, coupling, params)
    return state + 0.5 * dt * (k1 + k2) + dW


@njit(cache=False)
def _euler_stoch(state, coupling, params, dt, dfun_fn, dW):
    return state + dt * dfun_fn(state, coupling, params) + dW


# ---------------------------------------------------------------------------
# Bounds clamping  (in-place, no return)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _apply_bounds(state, lower_bounds, has_lower, upper_bounds, has_upper):
    n_sv = state.shape[0]
    n_nodes = state.shape[1]
    for i in range(n_sv):
        if has_lower[i]:
            lb = lower_bounds[i]
            for j in range(n_nodes):
                if state[i, j] < lb:
                    state[i, j] = lb
        if has_upper[i]:
            ub = upper_bounds[i]
            for j in range(n_nodes):
                if state[i, j] > ub:
                    state[i, j] = ub


# ---------------------------------------------------------------------------
# Deterministic simulation loop
# ---------------------------------------------------------------------------

@njit(cache=False)
def nb_simulate_det(
    state0, srcbuf0, weights, delay_steps, horizon,
    params, G_idx, a, b, dt, n_steps, n_record,
    has_delays, cvar_indices,
    lower_bounds, has_lower, upper_bounds, has_upper,
    record_period, t_cut_step, n_voi, voi_indices,
    use_heun, dfun_fn, use_kuramoto, alpha, stim_data, has_stimulus,
):
    """
    Deterministic (no noise) simulation loop.

    params      : (n_params, n_nodes) float64
    srcbuf0     : (n_cvar, n_nodes, horizon) float64 — initial ring buffer
    use_kuramoto: bool — use sinusoidal Kuramoto coupling instead of linear
    Returns     : (n_record, n_voi, n_nodes) float64
    """
    n_sv   = state0.shape[0]
    n_nodes = state0.shape[1]
    n_cvar  = cvar_indices.shape[0]

    state  = state0.copy()
    srcbuf = srcbuf0.copy()

    G = params[G_idx, 0]

    out     = np.empty((n_record, n_voi, n_nodes))
    rec_idx = 0

    for step in range(n_steps):
        # --- Coupling ---
        if has_delays:
            if use_kuramoto:
                coupling = _coup_kuramoto_delayed(
                    srcbuf, step, horizon, weights, delay_steps, G, n_nodes,
                    state[cvar_indices[0]], alpha)
            else:
                coupling = _coup_linear_delayed(
                    srcbuf, step, horizon, weights, delay_steps, G, a, b)
        else:
            cvar_state = np.empty((n_cvar, n_nodes))
            for cv in range(n_cvar):
                cvar_state[cv] = state[cvar_indices[cv]]
            if use_kuramoto:
                coupling = _coup_kuramoto_instant(cvar_state, weights, G, n_nodes, alpha)
            else:
                coupling = _coup_linear_instant(cvar_state, weights, G, a, b)

        # --- Stimulus injection ---
        if has_stimulus:
            for _cv in range(n_cvar):
                for _nd in range(n_nodes):
                    coupling[_cv, _nd] += stim_data[step, _cv, _nd]

        # --- Integrate ---
        if use_heun:
            new_state = _heun_det(state, coupling, params, dt, dfun_fn)
        else:
            new_state = _euler_det(state, coupling, params, dt, dfun_fn)

        # --- Bounds ---
        _apply_bounds(new_state, lower_bounds, has_lower, upper_bounds, has_upper)

        state = new_state

        # --- Update history ---
        if has_delays:
            h = step % horizon
            for cv in range(n_cvar):
                ci = cvar_indices[cv]
                for node in range(n_nodes):
                    srcbuf[cv, node, h] = state[ci, node]

        # --- Record ---
        if step >= t_cut_step and (step - t_cut_step) % record_period == 0:
            if rec_idx < n_record:
                for vi in range(n_voi):
                    out[rec_idx, vi] = state[voi_indices[vi]]
                rec_idx += 1

    return out


# ---------------------------------------------------------------------------
# Stochastic simulation loop  (noise generated per step inside @njit)
# ---------------------------------------------------------------------------

@njit(cache=False)
def nb_simulate_stoch(
    state0, srcbuf0, weights, delay_steps, horizon,
    params, G_idx, a, b, dt, n_steps, n_record,
    has_delays, cvar_indices,
    lower_bounds, has_lower, upper_bounds, has_upper,
    eff_noise_amp, noise_mask,
    record_period, t_cut_step, n_voi, voi_indices,
    use_heun, seed, dfun_fn, use_kuramoto, alpha, stim_data, has_stimulus,
):
    """
    Stochastic simulation loop.  Noise is generated inside the @njit function
    using Numba's thread-local RNG (seeded per call for reproducibility).

    eff_noise_amp : (n_noise_vars,) float64 — amplitude * sqrt(dt) already resolved
    noise_mask    : (n_sv,) bool — which state vars receive additive noise
    seed          : int64 — seed for this run
    use_kuramoto  : bool — use sinusoidal Kuramoto coupling instead of linear
    Returns       : (n_record, n_voi, n_nodes) float64
    """
    np.random.seed(seed)

    n_sv    = state0.shape[0]
    n_nodes = state0.shape[1]
    n_cvar  = cvar_indices.shape[0]

    state  = state0.copy()
    srcbuf = srcbuf0.copy()

    G = params[G_idx, 0]

    out     = np.empty((n_record, n_voi, n_nodes))
    rec_idx = 0

    for step in range(n_steps):
        # --- Coupling ---
        if has_delays:
            if use_kuramoto:
                coupling = _coup_kuramoto_delayed(
                    srcbuf, step, horizon, weights, delay_steps, G, n_nodes,
                    state[cvar_indices[0]], alpha)
            else:
                coupling = _coup_linear_delayed(
                    srcbuf, step, horizon, weights, delay_steps, G, a, b)
        else:
            cvar_state = np.empty((n_cvar, n_nodes))
            for cv in range(n_cvar):
                cvar_state[cv] = state[cvar_indices[cv]]
            if use_kuramoto:
                coupling = _coup_kuramoto_instant(cvar_state, weights, G, n_nodes, alpha)
            else:
                coupling = _coup_linear_instant(cvar_state, weights, G, a, b)

        # --- Stimulus injection ---
        if has_stimulus:
            for _cv in range(n_cvar):
                for _nd in range(n_nodes):
                    coupling[_cv, _nd] += stim_data[step, _cv, _nd]

        # --- Noise for this step ---
        dW = np.zeros((n_sv, n_nodes))
        noise_j = 0
        for sv_i in range(n_sv):
            if noise_mask[sv_i]:
                dW[sv_i] = eff_noise_amp[noise_j] * np.random.randn(n_nodes)
                noise_j += 1

        # --- Integrate ---
        if use_heun:
            new_state = _heun_stoch(state, coupling, params, dt, dfun_fn, dW)
        else:
            new_state = _euler_stoch(state, coupling, params, dt, dfun_fn, dW)

        # --- Bounds ---
        _apply_bounds(new_state, lower_bounds, has_lower, upper_bounds, has_upper)

        state = new_state

        # --- Update history ---
        if has_delays:
            h = step % horizon
            for cv in range(n_cvar):
                ci = cvar_indices[cv]
                for node in range(n_nodes):
                    srcbuf[cv, node, h] = state[ci, node]

        # --- Record ---
        if step >= t_cut_step and (step - t_cut_step) % record_period == 0:
            if rec_idx < n_record:
                for vi in range(n_voi):
                    out[rec_idx, vi] = state[voi_indices[vi]]
                rec_idx += 1

    return out


# ---------------------------------------------------------------------------
# Parallel deterministic sweep  (prange over parameter sets)
# ---------------------------------------------------------------------------

@njit(parallel=True, cache=False)
def nb_sweep_det(
    param_sets, base_params, state0, srcbuf0,
    weights, delay_steps, horizon, G_idx, a, b, dt, n_steps, n_record,
    has_delays, cvar_indices,
    lower_bounds, has_lower, upper_bounds, has_upper,
    record_period, t_cut_step, n_voi, voi_indices,
    use_heun, sweep_param_indices, dfun_fn, use_kuramoto, alpha, stim_data, has_stimulus,
):
    """
    Parallel sweep over param_sets rows — deterministic.

    param_sets         : (n_samples, n_sweep_params) float64
    sweep_param_indices: (n_sweep_params,) int64 — row index into params first axis
    Returns            : (n_samples, n_record, n_voi, n_nodes)
    """
    n_samples = param_sets.shape[0]
    n_nodes   = state0.shape[1]

    out = np.empty((n_samples, n_record, n_voi, n_nodes))

    for i in prange(n_samples):
        params_i = base_params.copy()
        for j in range(param_sets.shape[1]):
            pidx = sweep_param_indices[j]
            val  = param_sets[i, j]
            for node in range(n_nodes):
                params_i[pidx, node] = val

        out[i] = nb_simulate_det(
            state0.copy(), srcbuf0.copy(),
            weights, delay_steps, horizon,
            params_i, G_idx, a, b, dt, n_steps, n_record,
            has_delays, cvar_indices,
            lower_bounds, has_lower, upper_bounds, has_upper,
            record_period, t_cut_step, n_voi, voi_indices,
            use_heun, dfun_fn, use_kuramoto, alpha, stim_data, has_stimulus,
        )

    return out


# ---------------------------------------------------------------------------
# Parallel stochastic sweep
# ---------------------------------------------------------------------------

@njit(parallel=True, cache=False)
def nb_sweep_stoch(
    param_sets, base_params, state0, srcbuf0,
    weights, delay_steps, horizon, G_idx, a, b, dt, n_steps, n_record,
    has_delays, cvar_indices,
    lower_bounds, has_lower, upper_bounds, has_upper,
    eff_noise_amp, noise_mask,
    record_period, t_cut_step, n_voi, voi_indices,
    use_heun, sweep_param_indices, seeds, dfun_fn, use_kuramoto, alpha, stim_data, has_stimulus,
):
    """
    Parallel sweep over param_sets rows — stochastic.

    seeds : (n_samples,) int64 — unique seed per run for independent noise
    Returns: (n_samples, n_record, n_voi, n_nodes)
    """
    n_samples = param_sets.shape[0]
    n_nodes   = state0.shape[1]

    out = np.empty((n_samples, n_record, n_voi, n_nodes))

    for i in prange(n_samples):
        params_i = base_params.copy()
        for j in range(param_sets.shape[1]):
            pidx = sweep_param_indices[j]
            val  = param_sets[i, j]
            for node in range(n_nodes):
                params_i[pidx, node] = val

        out[i] = nb_simulate_stoch(
            state0.copy(), srcbuf0.copy(),
            weights, delay_steps, horizon,
            params_i, G_idx, a, b, dt, n_steps, n_record,
            has_delays, cvar_indices,
            lower_bounds, has_lower, upper_bounds, has_upper,
            eff_noise_amp, noise_mask,
            record_period, t_cut_step, n_voi, voi_indices,
            use_heun, seeds[i], dfun_fn, use_kuramoto,
        )

    return out


# ---------------------------------------------------------------------------
# Parallel deterministic sweep — inline feature extraction (Tier-2)
# ---------------------------------------------------------------------------

@njit(parallel=True, cache=False)
def nb_sweep_det_feat(
    param_sets, base_params, state0, srcbuf0,
    weights, delay_steps, horizon, G_idx, a, b, dt, n_steps, n_record,
    has_delays, cvar_indices,
    lower_bounds, has_lower, upper_bounds, has_upper,
    record_period, t_cut_step, n_voi, voi_indices,
    use_heun, sweep_param_indices,
    do_mean, do_std, do_fc, do_fcd, fcd_window, n_features,
    dfun_fn, use_kuramoto,
):
    """
    Parallel deterministic sweep with inline feature extraction.

    Features are computed inside prange from the thread-local time series;
    only the feature vector is written to the global output — the full
    time series is never materialised across all samples simultaneously.

    Returns
    -------
    out : (n_samples, n_features) float64
    """
    n_samples = param_sets.shape[0]
    n_nodes   = state0.shape[1]

    out = np.empty((n_samples, n_features))

    for i in prange(n_samples):
        params_i = base_params.copy()
        for j in range(param_sets.shape[1]):
            pidx = sweep_param_indices[j]
            val  = param_sets[i, j]
            for node in range(n_nodes):
                params_i[pidx, node] = val

        # Run simulation — ts_i is thread-local (n_record, n_voi, n_nodes)
        ts_i = nb_simulate_det(
            state0.copy(), srcbuf0.copy(),
            weights, delay_steps, horizon,
            params_i, G_idx, a, b, dt, n_steps, n_record,
            has_delays, cvar_indices,
            lower_bounds, has_lower, upper_bounds, has_upper,
            record_period, t_cut_step, n_voi, voi_indices,
            use_heun, dfun_fn, use_kuramoto, alpha, stim_data, has_stimulus,
        )

        # Extract features from voi=0  →  (n_record, n_nodes)
        ts_2d = ts_i[:, 0, :]
        feat_buf = np.empty(n_features)
        nb_extract(ts_2d, do_mean, do_std, do_fc, do_fcd, fcd_window, feat_buf)
        out[i] = feat_buf

    return out


# ---------------------------------------------------------------------------
# Parallel stochastic sweep — inline feature extraction (Tier-2)
# ---------------------------------------------------------------------------

@njit(parallel=True, cache=False)
def nb_sweep_stoch_feat(
    param_sets, base_params, state0, srcbuf0,
    weights, delay_steps, horizon, G_idx, a, b, dt, n_steps, n_record,
    has_delays, cvar_indices,
    lower_bounds, has_lower, upper_bounds, has_upper,
    eff_noise_amp, noise_mask,
    record_period, t_cut_step, n_voi, voi_indices,
    use_heun, sweep_param_indices, seeds,
    do_mean, do_std, do_fc, do_fcd, fcd_window, n_features,
    dfun_fn, use_kuramoto,
):
    """
    Parallel stochastic sweep with inline feature extraction.

    Returns
    -------
    out : (n_samples, n_features) float64
    """
    n_samples = param_sets.shape[0]
    n_nodes   = state0.shape[1]

    out = np.empty((n_samples, n_features))

    for i in prange(n_samples):
        params_i = base_params.copy()
        for j in range(param_sets.shape[1]):
            pidx = sweep_param_indices[j]
            val  = param_sets[i, j]
            for node in range(n_nodes):
                params_i[pidx, node] = val

        ts_i = nb_simulate_stoch(
            state0.copy(), srcbuf0.copy(),
            weights, delay_steps, horizon,
            params_i, G_idx, a, b, dt, n_steps, n_record,
            has_delays, cvar_indices,
            lower_bounds, has_lower, upper_bounds, has_upper,
            eff_noise_amp, noise_mask,
            record_period, t_cut_step, n_voi, voi_indices,
            use_heun, seeds[i], dfun_fn, use_kuramoto,
        )

        ts_2d = ts_i[:, 0, :]
        feat_buf = np.empty(n_features)
        nb_extract(ts_2d, do_mean, do_std, do_fc, do_fcd, fcd_window, feat_buf)
        out[i] = feat_buf

    return out
