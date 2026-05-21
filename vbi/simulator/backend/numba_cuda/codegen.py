"""
Code generation for the Numba-CUDA backend.

Memory layout — coalesced (tid is always the LAST dimension)
-------------------------------------------------------------
Old layout (bad for GPU):
    state[n_samples, n_sv,   n_nodes]   thread tid reads state[tid, sv, node]
    buf  [n_samples, n_cvar, n_nodes, horizon]
    → adjacent threads are n_sv*n_nodes elements apart → NO coalescing

New layout (coalesced):
    state[n_sv,   n_nodes, n_samples]   thread tid reads state[sv, node, tid]
    buf  [n_cvar, n_nodes, horizon, n_samples]
    → adjacent threads access consecutive addresses → ONE memory transaction per warp

The Python wrappers (sweeper.py / simulator.py) transpose host arrays before
transferring to device and transpose back after copy_to_host.

Connectivity
------------
Two modes are generated depending on the ``sparse`` flag:

Dense (``sparse=False``):
    Coupling: weights[node, src] — full (n_nodes × n_nodes) float32 matrix.
    Best for small N or fully-connected topologies.

CSR sparse (``sparse=True``):
    Coupling: w_data[nnz], w_indices[nnz], w_indptr[n_nodes+1]
    Optional: idelays_csr[nnz]  — delay per non-zero edge (int32)
    Best for large N with real connectomes (typical density 20–40 %).

Thread layout (same for both):
    grid  = ((n_samples + TPB - 1) // TPB,)
    block = (TPB,)                              TPB = 256
    tid   = cuda.grid(1)   →  1 thread = 1 complete simulation
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
    for p, r in _MATH_MAP:
        expr = re.sub(p, r, expr)
    return expr


def _generate_source(spec: ModelSpec, sparse: bool = False,
                     use_kuramoto: bool = False, alpha: float = 0.0) -> str:
    """
    Generate a complete Python module with:
      - _dfun_cuda        : @cuda.jit(device=True) scalar dfun
      - cuda_sweep_det    : deterministic sweep kernel
      - cuda_sweep_stoch  : stochastic sweep kernel

    Both kernels use coalesced memory layout (tid last).
    When sparse=True, coupling uses CSR instead of the dense weight matrix.
    """
    sv      = spec.sv_names
    cvars   = spec.cvar
    params  = spec.param_names
    n_sv    = spec.n_sv
    n_cvar  = len(cvars)
    n_par   = len(params)
    returns = ", ".join(f"d_{s}" for s in sv)

    # ----------------------------------------------------------------
    # dfun device function
    # ----------------------------------------------------------------
    dfun_args = list(sv) + [f"c_{c}" for c in cvars] + list(params)
    L = [
        f"# Auto-generated CUDA module for: {spec.name}  sparse={sparse}",
        "import math",
        "import numpy as np",
        "from numba import cuda, float32",
        "from numba.cuda.random import xoroshiro128p_normal_float32, create_xoroshiro128p_states",
        "",
        "@cuda.jit(device=True)",
        f"def _dfun_cuda({', '.join(dfun_args)}):",
    ]
    if cvars:
        L.append(f"    c = c_{cvars[0]}")
    for s in sv:
        L.append(f"    d_{s} = {_cuda_expr(spec.dfun_str[s])}")
    L += [f"    return {returns}", ""]

    # ----------------------------------------------------------------
    # Kernel generator (shared structure for det / stoch)
    # ----------------------------------------------------------------
    def _kernel(stochastic: bool) -> list[str]:  # noqa: C901
        kn = "cuda_sweep_stoch" if stochastic else "cuda_sweep_det"
        conn_args = (
            [
                "    w_data,        # (nnz,) float32  CSR values",
                "    w_indices,     # (nnz,) int32    CSR col indices",
                "    w_indptr,      # (n_nodes+1,) int32  CSR row pointers",
                "    idelays_csr,   # (nnz,) int32  delay per edge (or empty)",
            ]
            if sparse else
            [
                "    weights,       # (n_nodes, n_nodes) float32  dense",
                "    delay_steps,   # (n_nodes, n_nodes) int32  dense",
            ]
        )
        noise_arg = (
            ["    noise,         # (n_steps, n_sv, n_nodes, n_samples) float32"]
            if stochastic else []
        )

        K = [
            f"@cuda.jit",
            f"def {kn}(",
            "    state,         # (n_sv,   n_nodes, n_samples) float32  coalesced",
            "    buf,           # (n_cvar, n_nodes, horizon, n_samples) float32",
        ] + conn_args + [
            "    params,        # (n_params, n_samples) float32",
            "    cvar_indices,  # (n_cvar,) int32",
            "    lower_bounds,  # (n_sv,)   float32",
            "    has_lower,     # (n_sv,)   bool",
            "    upper_bounds,  # (n_sv,)   float32",
            "    has_upper,     # (n_sv,)   bool",
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
            "    stim_data,     # (n_steps * n_cvar * n_nodes,) float32  flat pre-sampled",
            "    has_stimulus,  # bool",
        ] + noise_arg + [
            "    ts_out,        # (n_record, n_sv, n_nodes, n_samples) float32",
            "):",
            "    tid      = cuda.grid(1)",
            "    n_samp   = state.shape[2]",
            f"    n_nodes  = state.shape[1]",
            "    if tid >= n_samp:",
            "        return",
            "",
            f"    sv     = cuda.local.array(({n_sv},), dtype=float32)",
            f"    k1     = cuda.local.array(({n_sv},), dtype=float32)",
            f"    k2     = cuda.local.array(({n_sv},), dtype=float32)",
            f"    pred   = cuda.local.array(({n_sv},), dtype=float32)",
            f"    coup   = cuda.local.array(({n_cvar},), dtype=float32)",
            "    rec_idx = 0",
            "",
            "    for step in range(n_steps):",
            "        for node in range(n_nodes):",
        ]

        # Load state — coalesced: state[sv, node, tid]
        K.append("")
        K.append("            # -- load state (coalesced) --")
        for i, s in enumerate(sv):
            K.append(f"            {s} = state[{i}, node, tid]")

        # Coupling
        K.append("")
        K.append("            # -- coupling --")
        if use_kuramoto:
            # Kuramoto: c[node] = (G/N) Σ_src W[node,src] sin(θ_src − θ_node + alpha)
            # coup_a = G/N, coup_b = alpha
            cv0 = f"cvar_indices[0]"
            K.append(f"            _theta_node = state[{cv0}, node, tid]")
            K.append(f"            _s0 = float32(0.0)")
            if sparse:
                K.append(f"            _rstart = w_indptr[node]")
                K.append(f"            _rend   = w_indptr[node + 1]")
                K.append(f"            for _ptr in range(_rstart, _rend):")
                K.append(f"                _src = w_indices[_ptr]")
                K.append(f"                _w   = w_data[_ptr]")
                K.append(f"                if has_delays:")
                K.append(f"                    _d   = idelays_csr[_ptr]")
                K.append(f"                    _idx = (step - 1 - _d + horizon) % horizon")
                K.append(f"                    _s0 += _w * math.sin(buf[0, _src, _idx, tid] - _theta_node + coup_b)")
                K.append(f"                else:")
                K.append(f"                    _s0 += _w * math.sin(state[{cv0}, _src, tid] - _theta_node + coup_b)")
            else:
                K.append(f"            if has_delays:")
                K.append(f"                _tl = step - 1")
                K.append(f"                for _src in range(n_nodes):")
                K.append(f"                    _d   = delay_steps[_src, node]")
                K.append(f"                    _idx = (_tl - _d + horizon) % horizon")
                K.append(f"                    _s0 += weights[node, _src] * math.sin(buf[0, _src, _idx, tid] - _theta_node + coup_b)")
                K.append(f"            else:")
                K.append(f"                for _src in range(n_nodes):")
                K.append(f"                    _s0 += weights[node, _src] * math.sin(state[{cv0}, _src, tid] - _theta_node + coup_b)")
            K.append(f"            coup[0] = coup_a * _s0")
        else:
            for cv_i, cv_name in enumerate(cvars):
                ci = f"cvar_indices[{cv_i}]"
                K.append(f"            _s{cv_i} = float32(0.0)")
                if sparse:
                    K.append(f"            _rstart = w_indptr[node]")
                    K.append(f"            _rend   = w_indptr[node + 1]")
                    K.append(f"            for _ptr in range(_rstart, _rend):")
                    K.append(f"                _src = w_indices[_ptr]")
                    K.append(f"                _w   = w_data[_ptr]")
                    K.append(f"                if has_delays:")
                    K.append(f"                    _d   = idelays_csr[_ptr]")
                    K.append(f"                    _idx = (step - 1 - _d + horizon) % horizon")
                    K.append(f"                    _s{cv_i} += _w * buf[{cv_i}, _src, _idx, tid]")
                    K.append(f"                else:")
                    K.append(f"                    _s{cv_i} += _w * state[{ci}, _src, tid]")
                else:
                    K.append(f"            if has_delays:")
                    K.append(f"                _tl = step - 1")
                    K.append(f"                for _src in range(n_nodes):")
                    K.append(f"                    _d   = delay_steps[_src, node]")
                    K.append(f"                    _idx = (_tl - _d + horizon) % horizon")
                    K.append(f"                    _s{cv_i} += weights[node, _src] * buf[{cv_i}, _src, _idx, tid]")
                    K.append(f"            else:")
                    K.append(f"                for _src in range(n_nodes):")
                    K.append(f"                    _s{cv_i} += weights[node, _src] * state[{ci}, _src, tid]")
                K.append(f"            coup[{cv_i}] = coup_a * _s{cv_i} + coup_b")

        # Stimulus injection — same point as all other backends
        # stim_data layout: stim_data[step * n_cvar * n_nodes + cv * n_nodes + node]
        K.append("")
        K.append("            # -- stimulus injection --")
        K.append("            if has_stimulus:")
        for cv_i in range(n_cvar):
            K.append(f"                _sb{cv_i} = step * {n_cvar} * n_nodes + {cv_i} * n_nodes + node")
            K.append(f"                coup[{cv_i}] += stim_data[_sb{cv_i}]")

        # k1
        K.append("")
        K.append("            # -- k1 --")
        coup_v = [f"coup[{i}]" for i in range(n_cvar)]
        par_v  = [f"params[{i}, tid]" for i in range(n_par)]
        dcall  = ", ".join(list(sv) + coup_v + par_v)
        K.append(f"            {returns} = _dfun_cuda({dcall})")
        for i, s in enumerate(sv):
            K.append(f"            k1[{i}] = d_{s}")

        # predictor
        K.append("")
        K.append("            # -- predictor --")
        for i in range(n_sv):
            K.append(f"            pred[{i}] = sv[{i}] + dt * k1[{i}]" if False
                     else f"            pred[{i}] = {sv[i]} + dt * k1[{i}]")

        # k2
        K.append("")
        K.append("            # -- k2 --")
        pred_v = [f"pred[{i}]" for i in range(n_sv)]
        K.append(f"            {returns} = _dfun_cuda({', '.join(pred_v + coup_v + par_v)})")
        for i, s in enumerate(sv):
            K.append(f"            k2[{i}] = d_{s}")

        # corrector + noise
        K.append("")
        K.append("            # -- corrector --")
        if stochastic:
            for i in range(n_sv):
                K.append(f"            _dw{i} = noise[step, {i}, node, tid]")
            K.append("            if use_heun:")
            for i, s in enumerate(sv):
                K.append(f"                _new_{s} = {s} + float32(0.5)*dt*(k1[{i}]+k2[{i}]) + _dw{i}")
            K.append("            else:")
            for i, s in enumerate(sv):
                K.append(f"                _new_{s} = {s} + dt*k1[{i}] + _dw{i}")
        else:
            K.append("            if use_heun:")
            for i, s in enumerate(sv):
                K.append(f"                _new_{s} = {s} + float32(0.5)*dt*(k1[{i}]+k2[{i}])")
            K.append("            else:")
            for i, s in enumerate(sv):
                K.append(f"                _new_{s} = {s} + dt*k1[{i}]")

        # bounds
        K.append("")
        K.append("            # -- post-corrector bounds --")
        for i, s in enumerate(sv):
            K.append(f"            if has_lower[{i}]:")
            K.append(f"                if _new_{s} < lower_bounds[{i}]: _new_{s} = lower_bounds[{i}]")
            K.append(f"            if has_upper[{i}]:")
            K.append(f"                if _new_{s} > upper_bounds[{i}]: _new_{s} = upper_bounds[{i}]")

        # write back — coalesced
        K.append("")
        K.append("            # -- write back (coalesced) --")
        for i, s in enumerate(sv):
            K.append(f"            state[{i}, node, tid] = _new_{s}")

        # ring buffer — coalesced: buf[cvar, node, slot, tid]
        K += [
            "",
            "        # -- ring buffer update --",
            "        if has_delays:",
            "            _slot = step % horizon",
            "            for node in range(n_nodes):",
        ]
        for cv_i in range(n_cvar):
            ci = f"cvar_indices[{cv_i}]"
            K.append(f"                buf[{cv_i}, node, _slot, tid] = state[{ci}, node, tid]")

        # record — coalesced: ts_out[rec, sv, node, tid]
        K += [
            "",
            "        # -- record --",
            "        if step >= t_cut_step:",
            "            if (step - t_cut_step) % record_period == 0:",
            "                if rec_idx < n_record:",
            "                    for node in range(n_nodes):",
        ]
        for i in range(n_sv):
            K.append(f"                        ts_out[rec_idx, {i}, node, tid] = state[{i}, node, tid]")
        K += ["                    rec_idx += 1", ""]

        return K

    L += _kernel(False)
    L += _kernel(True)
    return "\n".join(L)


def _hash(src: str) -> str:
    return hashlib.sha256(src.encode()).hexdigest()[:24]


def build_cuda_module(spec: ModelSpec, sparse: bool = False,
                      use_kuramoto: bool = False,
                      alpha: float = 0.0) -> types.ModuleType:
    """Generate, cache and exec() the complete CUDA module."""
    src = _generate_source(spec, sparse=sparse, use_kuramoto=use_kuramoto, alpha=alpha)
    key = _hash(src)
    if key in _MODULE_CACHE:
        return _MODULE_CACHE[key]

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = _CACHE_DIR / f"cuda_sweep_{key}.py"
    if not p.exists():
        p.write_text(src)

    mod_name = f"vbi_cuda_{key}"
    spec_obj = importlib.util.spec_from_file_location(mod_name, p)
    mod = importlib.util.module_from_spec(spec_obj)
    sys.modules[mod_name] = mod
    spec_obj.loader.exec_module(mod)
    _MODULE_CACHE[key] = mod
    return mod


# ---------------------------------------------------------------------------
# Host array builders
# ---------------------------------------------------------------------------

def build_params_matrix(spec: SimulationSpec, n_samples: int,
                        sweep_names: list[str] | None = None,
                        sweep_sets: np.ndarray | None = None) -> np.ndarray:
    """
    Returns (n_params, n_samples) float32  — coalesced layout.
    Base values broadcast; sweep columns overwritten per sample.
    """
    pnames  = list(spec.model.param_names)
    n_par   = len(pnames)
    defs    = spec.model.default_params
    ovr     = spec.node_params

    base = np.empty(n_par, dtype=np.float32)
    for i, p in enumerate(spec.model.parameters):
        val = np.asarray(ovr.get(p.name, defs[p.name]), dtype=np.float64)
        base[i] = float(val.mean())

    arr = np.tile(base[:, np.newaxis], (1, n_samples))  # (n_params, n_samples)

    if sweep_names and sweep_sets is not None:
        for j, name in enumerate(sweep_names):
            if name in pnames:
                arr[pnames.index(name), :] = sweep_sets[:, j].astype(np.float32)

    return np.ascontiguousarray(arr)


def build_initial_state(spec: SimulationSpec, n_samples: int) -> np.ndarray:
    """(n_sv, n_nodes, n_samples) float32 — coalesced."""
    n_sv    = spec.model.n_sv
    n_nodes = spec.n_nodes
    st = np.zeros((n_sv, n_nodes), dtype=np.float32)
    for i, sv in enumerate(spec.model.state_variables):
        st[i, :] = sv.default_init
    return np.ascontiguousarray(np.broadcast_to(st[:, :, np.newaxis],
                                                (n_sv, n_nodes, n_samples))
                                 .copy())   # (n_sv, n_nodes, n_samples)


def build_ring_buffer(n_samples: int, n_cvar: int, n_nodes: int,
                      horizon: int, init_cvar: np.ndarray) -> np.ndarray:
    """(n_cvar, n_nodes, horizon, n_samples) float32 — coalesced."""
    buf = np.zeros((n_cvar, n_nodes, horizon, n_samples), dtype=np.float32)
    for h in range(horizon):
        buf[:, :, h, :] = init_cvar[:, :, np.newaxis]
    return np.ascontiguousarray(buf)


def get_bounds_arrays(spec: ModelSpec):
    n_sv = spec.n_sv
    lo, hi = np.zeros(n_sv, np.float32), np.zeros(n_sv, np.float32)
    hlo, hhi = np.zeros(n_sv, np.bool_), np.zeros(n_sv, np.bool_)
    for i, sv in enumerate(spec.state_variables):
        if sv.lower_bound is not None:
            lo[i], hlo[i] = np.float32(sv.lower_bound), True
        if sv.upper_bound is not None:
            hi[i], hhi[i] = np.float32(sv.upper_bound), True
    return lo, hlo, hi, hhi


def to_csr(weights: np.ndarray, delay_steps: np.ndarray | None = None,
           has_delays: bool = False):
    """
    Convert dense weight matrix to CSR float32 arrays.

    Returns
    -------
    w_data   : (nnz,)       float32
    w_indices: (nnz,)       int32
    w_indptr : (n_nodes+1,) int32
    idelays  : (nnz,)       int32  (zeros if has_delays=False)
    nnz      : int
    density  : float
    """
    import scipy.sparse as sp
    W  = sp.csr_matrix(weights.astype(np.float32))
    W.eliminate_zeros()
    nnz     = W.nnz
    n_nodes = weights.shape[0]
    density = nnz / (n_nodes * n_nodes) if n_nodes > 0 else 0.0

    if has_delays and delay_steps is not None:
        # Reorder idelays to match CSR non-zero positions
        ds_csr = np.zeros(nnz, dtype=np.int32)
        for i in range(n_nodes):
            start, end = W.indptr[i], W.indptr[i + 1]
            for k, j in enumerate(W.indices[start:end]):
                ds_csr[start + k] = int(delay_steps[j, i])
    else:
        ds_csr = np.zeros(nnz, dtype=np.int32)

    return (
        np.ascontiguousarray(W.data,    dtype=np.float32),
        np.ascontiguousarray(W.indices, dtype=np.int32),
        np.ascontiguousarray(W.indptr,  dtype=np.int32),
        ds_csr,
        nnz,
        density,
    )


def generate_noise(spec: SimulationSpec, n_steps: int,
                   n_samples: int, seed_base: int) -> np.ndarray:
    """
    (n_steps, n_sv, n_nodes, n_samples) float32 — coalesced layout.
    Each sample gets seed = seed_base + sample_idx.
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

    noise = np.zeros((n_steps, n_sv, n_nodes, n_samples), dtype=np.float32)
    for s in range(n_samples):
        rng = np.random.default_rng(seed_base + s)
        for k, sv_idx in enumerate(ni):
            noise[:, sv_idx, :, s] = (
                eff_amp[k] * rng.standard_normal((n_steps, n_nodes))
            ).astype(np.float32)
    return np.ascontiguousarray(noise)
