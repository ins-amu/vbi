"""
Code generation for the Numba-CUDA backend.

In Numba CUDA, device functions must be in the same compilation unit as the
kernel that calls them.  Therefore this module generates a *complete* Python
source string containing:

    1. A @cuda.jit(device=True) dfun  — scalar per-node, float32
    2. The deterministic sweep kernel
    3. The stochastic sweep kernel

The source string is exec()‑ed as one module so the kernel sees the dfun.

This mirrors TVB's approach (Mako template → exec); VBI uses plain string
generation instead of Mako to avoid an extra dependency.

Thread / block layout (same as TVB nb_hybrid_cuda_sweep.py)
------------------------------------------------------------
    grid  = ((n_samples + TPB - 1) // TPB,)
    block = (TPB,)
    tid   = cuda.grid(1)   →   1 thread = 1 complete simulation

Each thread integrates ALL nodes of ONE parameter set sequentially.
This is simple, avoids inter-thread communication, and scales naturally
to thousands of sweep points.

Memory layout (float32 throughout)
------------------------------------
state  : (n_samples, n_sv,   n_nodes)
buf    : (n_samples, n_cvar, n_nodes, horizon)  — ring buffer (dense)
params : (n_samples, n_params)
ts_out : (n_samples, n_record, n_sv, n_nodes)   — time-series output
noise  : (n_samples, n_steps, n_sv, n_nodes)    — pre-generated dW (stoch)
"""
from __future__ import annotations

import hashlib
import importlib.util
import re
import sys
import types
from pathlib import Path

import numpy as np

from vbi.simulator.spec.model import ModelSpec
from vbi.simulator.spec.simulation import SimulationSpec

_CACHE_DIR = Path.home() / ".cache" / "vbi" / "cuda"
_MODULE_CACHE: dict[str, types.ModuleType] = {}

# Python expression → CUDA-compatible math.* equivalent
_MATH_MAP = [
    (r'\bexp\b',   'math.exp'),
    (r'\blog2\b',  'math.log2'),
    (r'\blog\b',   'math.log'),
    (r'\bsin\b',   'math.sin'),
    (r'\bcos\b',   'math.cos'),
    (r'\btanh\b',  'math.tanh'),
    (r'\bsinh\b',  'math.sinh'),
    (r'\bcosh\b',  'math.cosh'),
    (r'\bsqrt\b',  'math.sqrt'),
    (r'\bfabs\b',  'math.fabs'),
    (r'\babs\b',   'math.fabs'),
    (r'\batan2\b', 'math.atan2'),
    (r'\batan\b',  'math.atan'),
    (r'\bfloor\b', 'math.floor'),
    (r'\bceil\b',  'math.ceil'),
    (r'\bpi\b',    'math.pi'),
]


def _cuda_expr(expr: str) -> str:
    """Translate a dfun_str expression to CUDA math.* form."""
    for pattern, repl in _MATH_MAP:
        expr = re.sub(pattern, repl, expr)
    return expr


def _generate_source(spec: ModelSpec) -> str:
    """
    Generate a complete Python module source containing:
      - _dfun_cuda   : @cuda.jit(device=True) scalar dfun
      - cuda_sweep_det   : deterministic sweep kernel
      - cuda_sweep_stoch : stochastic sweep kernel
    """
    sv_names    = spec.sv_names
    cvar_names  = spec.cvar
    param_names = spec.param_names
    n_sv        = spec.n_sv
    n_cvar      = len(cvar_names)
    n_params    = len(param_names)

    # ----------------------------------------------------------------
    # dfun device function
    # ----------------------------------------------------------------
    c_args  = [f"c_{cn}" for cn in cvar_names]
    dfun_args = list(sv_names) + c_args + list(param_names)

    lines = [
        f"# Auto-generated CUDA module for model: {spec.name}",
        "import math",
        "import numpy as np",
        "from numba import cuda, float32",
        "from numba.cuda.random import (xoroshiro128p_normal_float32,",
        "                               create_xoroshiro128p_states)",
        "",
        "# ---- dfun device function ----",
        "@cuda.jit(device=True)",
        f"def _dfun_cuda({', '.join(dfun_args)}):",
    ]
    if cvar_names:
        lines.append(f"    c = c_{cvar_names[0]}")
    for sv in sv_names:
        lines.append(f"    d_{sv} = {_cuda_expr(spec.dfun_str[sv])}")
    returns = ", ".join(f"d_{sv}" for sv in sv_names)
    lines.append(f"    return {returns}")
    lines.append("")

    # ----------------------------------------------------------------
    # Shared helpers: coupling and bounds (inlined into kernel)
    # ----------------------------------------------------------------
    # These are inlined via string substitution; see kernel template below.

    # ----------------------------------------------------------------
    # Kernel template (shared structure for det + stoch)
    # ----------------------------------------------------------------
    # We generate the two kernels with a helper function to avoid duplication.

    def _kernel(stochastic: bool) -> list[str]:
        kname = "cuda_sweep_stoch" if stochastic else "cuda_sweep_det"
        extra_args = (
            ["    noise,        # (n_samples, n_steps, n_sv, n_nodes) float32"]
            if stochastic else []
        )
        klines = [
            "# ---- " + kname + " ----",
            "@cuda.jit",
            f"def {kname}(",
            "    state_in,      # (n_samples, n_sv, n_nodes) float32",
            "    buf,           # (n_samples, n_cvar, n_nodes, horizon) float32",
            "    weights,       # (n_nodes, n_nodes) float32",
            "    delay_steps,   # (n_nodes, n_nodes) int32",
            "    params,        # (n_samples, n_params) float32",
            "    cvar_indices,  # (n_cvar,) int32",
            "    lower_bounds,  # (n_sv,) float32",
            "    has_lower,     # (n_sv,) bool",
            "    upper_bounds,  # (n_sv,) float32",
            "    has_upper,     # (n_sv,) bool",
            "    horizon,       # int32",
            "    dt,            # float32",
            "    n_steps,       # int32",
            "    n_record,      # int32",
            "    t_cut_step,    # int32",
            "    record_period, # int32",
            "    coup_a,        # float32",
            "    coup_b,        # float32",
            "    has_delays,    # bool",
            "    use_heun,      # bool",
        ]
        klines.extend(extra_args)
        klines.append("    ts_out,        # (n_samples, n_record, n_sv, n_nodes) float32")
        klines.append("):")

        klines += [
            "    tid     = cuda.grid(1)",
            "    n_samp  = state_in.shape[0]",
            f"    n_sv    = {n_sv}",
            f"    n_nodes = state_in.shape[2]",
            f"    n_cvar  = {n_cvar}",
            "    if tid >= n_samp:",
            "        return",
            "",
            "    # Per-thread local arrays (registers/local memory)",
        ]

        # Local arrays for state, intermediates, coupling
        klines.append(f"    sv     = cuda.local.array(({n_sv},), dtype=float32)")
        klines.append(f"    k1     = cuda.local.array(({n_sv},), dtype=float32)")
        klines.append(f"    k2     = cuda.local.array(({n_sv},), dtype=float32)")
        klines.append(f"    pred   = cuda.local.array(({n_sv},), dtype=float32)")
        klines.append(f"    coup   = cuda.local.array(({n_cvar},), dtype=float32)")
        klines += ["    rec_idx = 0", ""]

        klines.append("    for step in range(n_steps):")
        klines.append("        for node in range(n_nodes):")
        klines.append("")
        klines.append("            # ---- Load state ----")
        for i, sv in enumerate(sv_names):
            klines.append(f"            {sv} = state_in[tid, {i}, node]")
        klines.append("")

        klines.append("            # ---- Coupling ----")
        for cv_i, cv_name in enumerate(cvar_names):
            ci = f"cvar_indices[{cv_i}]"
            klines.append(f"            _s{cv_i} = float32(0.0)")
            klines.append(f"            if has_delays:")
            klines.append(f"                _t_last = step - 1")
            klines.append(f"                for _src in range(n_nodes):")
            klines.append(f"                    _d   = delay_steps[_src, node]")
            klines.append(f"                    _idx = (_t_last - _d + horizon) % horizon")
            klines.append(f"                    _s{cv_i} += weights[node, _src] * buf[tid, {cv_i}, _src, _idx]")
            klines.append(f"            else:")
            klines.append(f"                for _src in range(n_nodes):")
            klines.append(f"                    _s{cv_i} += weights[node, _src] * state_in[tid, {ci}, _src]")
            klines.append(f"            coup[{cv_i}] = coup_a * _s{cv_i} + coup_b")
        klines.append("")

        # Unpack coupling for dfun call
        coup_vals = [f"coup[{i}]" for i in range(n_cvar)]
        param_vals = [f"params[tid, {i}]" for i in range(n_params)]
        dfun_call_args = list(sv_names) + coup_vals + param_vals

        klines.append("            # ---- k1 = dfun(state, coupling, params) ----")
        klines.append(f"            {returns} = _dfun_cuda({', '.join(dfun_call_args)})")
        for i, sv in enumerate(sv_names):
            klines.append(f"            k1[{i}] = d_{sv}")
        klines.append("")

        klines.append("            # ---- Predictor ----")
        for i, sv in enumerate(sv_names):
            klines.append(f"            pred[{i}] = {sv} + dt * k1[{i}]")
        klines.append("")

        klines.append("            # ---- k2 = dfun(pred, coupling, params) ----")
        pred_vals = [f"pred[{i}]" for i in range(n_sv)]
        dfun_pred_args = pred_vals + coup_vals + param_vals
        klines.append(f"            {returns} = _dfun_cuda({', '.join(dfun_pred_args)})")
        for i, sv in enumerate(sv_names):
            klines.append(f"            k2[{i}] = d_{sv}")
        klines.append("")

        klines.append("            # ---- Heun corrector (or Euler) ----")
        if stochastic:
            klines.append("            # stochastic: add noise to both predictor and corrector")
            for i, sv in enumerate(sv_names):
                klines.append(
                    f"            _dw{i} = noise[tid, step, {i}, node]"
                )
            klines.append("            if use_heun:")
            for i, sv in enumerate(sv_names):
                klines.append(
                    f"                _new_{sv} = {sv} + float32(0.5)*dt*(k1[{i}]+k2[{i}]) + _dw{i}"
                )
            klines.append("            else:")
            for i, sv in enumerate(sv_names):
                klines.append(
                    f"                _new_{sv} = {sv} + dt*k1[{i}] + _dw{i}"
                )
        else:
            klines.append("            if use_heun:")
            for i, sv in enumerate(sv_names):
                klines.append(
                    f"                _new_{sv} = {sv} + float32(0.5)*dt*(k1[{i}]+k2[{i}])"
                )
            klines.append("            else:")
            for i, sv in enumerate(sv_names):
                klines.append(
                    f"                _new_{sv} = {sv} + dt*k1[{i}]"
                )
        klines.append("")

        klines.append("            # ---- Post-corrector bounds ----")
        for i, sv in enumerate(sv_names):
            klines.append(f"            if has_lower[{i}]:")
            klines.append(f"                if _new_{sv} < lower_bounds[{i}]:")
            klines.append(f"                    _new_{sv} = lower_bounds[{i}]")
            klines.append(f"            if has_upper[{i}]:")
            klines.append(f"                if _new_{sv} > upper_bounds[{i}]:")
            klines.append(f"                    _new_{sv} = upper_bounds[{i}]")
        klines.append("")

        klines.append("            # ---- Write back ----")
        for i, sv in enumerate(sv_names):
            klines.append(f"            state_in[tid, {i}, node] = _new_{sv}")
        klines.append("")

        klines.append("        # ---- Update ring buffer ----")
        klines.append("        if has_delays:")
        klines.append("            _slot = step % horizon")
        klines.append("            for node in range(n_nodes):")
        for cv_i, cv_name in enumerate(cvar_names):
            ci = f"cvar_indices[{cv_i}]"
            klines.append(f"                buf[tid, {cv_i}, node, _slot] = state_in[tid, {ci}, node]")
        klines.append("")

        klines.append("        # ---- Record ----")
        klines.append("        if step >= t_cut_step:")
        klines.append("            if (step - t_cut_step) % record_period == 0:")
        klines.append("                if rec_idx < n_record:")
        klines.append("                    for node in range(n_nodes):")
        for i in range(n_sv):
            klines.append(f"                        ts_out[tid, rec_idx, {i}, node] = state_in[tid, {i}, node]")
        klines.append("                    rec_idx += 1")
        klines.append("")

        return klines

    lines += _kernel(stochastic=False)
    lines += _kernel(stochastic=True)

    return "\n".join(lines)


def _source_hash(src: str) -> str:
    return hashlib.sha256(src.encode()).hexdigest()[:24]


def build_cuda_module(spec: ModelSpec) -> types.ModuleType:
    """
    Generate, cache to disk, exec(), and return the compiled CUDA module.

    The returned module has attributes:
        .cuda_sweep_det   : deterministic sweep kernel
        .cuda_sweep_stoch : stochastic sweep kernel
    """
    src = _generate_source(spec)
    key = _source_hash(src)

    if key in _MODULE_CACHE:
        return _MODULE_CACHE[key]

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    mod_path = _CACHE_DIR / f"cuda_sweep_{key}.py"
    if not mod_path.exists():
        mod_path.write_text(src)

    mod_name = f"vbi_cuda_sweep_{key}"
    spec_obj = importlib.util.spec_from_file_location(mod_name, mod_path)
    mod = importlib.util.module_from_spec(spec_obj)
    sys.modules[mod_name] = mod
    spec_obj.loader.exec_module(mod)

    _MODULE_CACHE[key] = mod
    return mod


# ---------------------------------------------------------------------------
# Parameter / state helpers (reused by sweeper + simulator)
# ---------------------------------------------------------------------------

def build_params_matrix(spec: SimulationSpec,
                        n_samples: int,
                        sweep_names: list[str] | None = None,
                        sweep_sets: np.ndarray | None = None) -> np.ndarray:
    """
    (n_samples, n_params) float32 array.

    Base values from spec.model.default_params + spec.node_params (scalar mean).
    Sweep columns overwritten per-row when sweep_names / sweep_sets are provided.
    """
    model       = spec.model
    param_names = list(model.param_names)
    n_params    = len(param_names)
    defaults    = model.default_params
    overrides   = spec.node_params

    base = np.empty(n_params, dtype=np.float32)
    for i, p in enumerate(model.parameters):
        val = np.asarray(overrides.get(p.name, defaults[p.name]), dtype=np.float64)
        base[i] = float(val.mean())

    arr = np.tile(base, (n_samples, 1))   # (n_samples, n_params)

    if sweep_names and sweep_sets is not None:
        for j, name in enumerate(sweep_names):
            if name in param_names:
                col = param_names.index(name)
                arr[:, col] = sweep_sets[:, j].astype(np.float32)

    return arr


def build_initial_state(spec: SimulationSpec) -> np.ndarray:
    """(n_sv, n_nodes) float32."""
    st = np.zeros((spec.model.n_sv, spec.n_nodes), dtype=np.float32)
    for i, sv in enumerate(spec.model.state_variables):
        st[i, :] = sv.default_init
    return st


def build_ring_buffer(n_samples: int, n_cvar: int, n_nodes: int,
                      horizon: int, init_cvar: np.ndarray) -> np.ndarray:
    """(n_samples, n_cvar, n_nodes, horizon) float32, filled with init_cvar."""
    buf = np.empty((n_samples, n_cvar, n_nodes, horizon), dtype=np.float32)
    for h in range(horizon):
        buf[:, :, :, h] = init_cvar[np.newaxis, :, :]
    return buf


def get_bounds_arrays(spec: ModelSpec):
    """Returns (lo, has_lo, hi, has_hi) each float32/bool shape (n_sv,)."""
    n_sv = spec.n_sv
    lo, hi = np.zeros(n_sv, np.float32), np.zeros(n_sv, np.float32)
    hlo, hhi = np.zeros(n_sv, np.bool_), np.zeros(n_sv, np.bool_)
    for i, sv in enumerate(spec.state_variables):
        if sv.lower_bound is not None:
            lo[i], hlo[i] = np.float32(sv.lower_bound), True
        if sv.upper_bound is not None:
            hi[i], hhi[i] = np.float32(sv.upper_bound), True
    return lo, hlo, hi, hhi


def generate_noise(spec: SimulationSpec, n_steps: int,
                   n_samples: int, seed_base: int) -> np.ndarray:
    """
    Pre-generate noise increments (n_samples, n_steps, n_sv, n_nodes) float32.

    Uses numpy CPU RNG; each sample gets seed = seed_base + sample_idx.
    Shape matches the ts_out layout so indexing is consistent in the kernel.
    """
    n_sv    = spec.model.n_sv
    n_nodes = spec.n_nodes
    ni      = list(spec.model.noise_indices)
    nsig    = spec.integrator.noise_nsig
    if nsig is None:
        nsig = np.ones(len(ni), dtype=np.float64) * 1e-3
    nsig = np.asarray(nsig, dtype=np.float64)

    style   = getattr(spec.integrator, "noise_style", "amplitude")
    amp     = np.sqrt(2.0 * nsig) if style == "tvb" else nsig
    eff_amp = (amp * np.sqrt(spec.integrator.dt)).astype(np.float32)

    noise = np.zeros((n_samples, n_steps, n_sv, n_nodes), dtype=np.float32)
    for s in range(n_samples):
        rng = np.random.default_rng(seed_base + s)
        for k, sv_idx in enumerate(ni):
            noise[s, :, sv_idx, :] = (
                eff_amp[k] * rng.standard_normal((n_steps, n_nodes))
            ).astype(np.float32)
    return noise
