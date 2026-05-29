"""
Code generation for the Numba CPU backend.

The model-specific dfun is the only dynamically-generated piece.  Following the
TVB hybrid approach, we write the generated source to disk so Numba's on-disk
cache (cache=True) works correctly across Python sessions.  The rest of the
backend (_nb_sim.py) is static @njit code.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
import types
from pathlib import Path

import numpy as np

from vbi.simulator.spec.model import ModelSpec
from vbi.simulator.spec.simulation import SimulationSpec

# Cache directory (override via VBI_NB_CACHE env var)
_CACHE_DIR = Path(os.environ.get("VBI_NB_CACHE",
                                  Path.home() / ".cache" / "vbi" / "numba"))

# In-process module cache  {source_hash: compiled_module}
_MODULE_CACHE: dict[str, types.ModuleType] = {}


# ---------------------------------------------------------------------------
# dfun source generation
# ---------------------------------------------------------------------------

def _dfun_source(spec: ModelSpec) -> str:
    """Return the source string for a model-specific @njit dfun."""
    sv          = spec.sv_names
    param_names = spec.param_names
    cvar_names  = spec.cvar

    lines = [
        "# Auto-generated dfun for model: " + spec.name,
        "import numpy as np",
        "from numpy import pi, exp, log, sin, cos, tanh, sqrt",
        "from numba import njit",
        "",
        "@njit(cache=True)",
        "def _dfun_nb(state, coupling, params):",
        "    # state   : (n_sv, n_nodes)",
        "    # coupling: (n_cvar, n_nodes)",
        "    # params  : (n_params, n_nodes)",
    ]
    # Unpack state variables → (n_nodes,) arrays (vectorised over nodes)
    for i, name in enumerate(sv):
        lines.append(f"    {name} = state[{i}]")
    # Unpack params (each row is one parameter, columns are nodes)
    for i, name in enumerate(param_names):
        lines.append(f"    {name} = params[{i}]")
    # Coupling inputs by cvar name  (c_r, c_V, …)  + legacy alias c = coupling[0]
    for i, cname in enumerate(cvar_names):
        lines.append(f"    c_{cname} = coupling[{i}]")
    lines.append("    c = coupling[0]")
    # Output array
    lines.append("    out = np.empty_like(state)")
    for i, name in enumerate(sv):
        lines.append(f"    out[{i}] = {spec.dfun_str[name]}")
    lines.append("    return out")
    lines.append("")
    return "\n".join(lines)


def _source_hash(src: str) -> str:
    return hashlib.sha256(src.encode()).hexdigest()[:24]


def build_numba_dfun(spec: ModelSpec):
    """
    Generate, cache to disk, and return a @njit-compiled dfun for spec.

    Returned callable signature:
        dfun_fn(state, coupling, params) -> np.ndarray  shape (n_sv, n_nodes)

    The source file is written to ~/.cache/vbi/numba/ so Numba's disk cache
    persists the compiled bytecode across Python sessions (cache=True in the
    generated @njit decorator).
    """
    src  = _dfun_source(spec)
    key  = _source_hash(src)

    if key in _MODULE_CACHE:
        return _MODULE_CACHE[key]._dfun_nb

    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        mod_path = _CACHE_DIR / f"nb_dfun_{key}.py"
        if not mod_path.exists():
            mod_path.write_text(src)
    except OSError as exc:
        raise OSError(
            f"Cannot write Numba dfun cache to {_CACHE_DIR}. "
            f"Set the VBI_NB_CACHE environment variable to a writable directory. "
            f"Original error: {exc}"
        ) from exc

    mod_name = f"vbi_nb_dfun_{key}"
    spec_obj = importlib.util.spec_from_file_location(mod_name, mod_path)
    mod = importlib.util.module_from_spec(spec_obj)
    # Register in sys.modules BEFORE exec so Numba's cache can find it by name
    sys.modules[mod_name] = mod
    spec_obj.loader.exec_module(mod)

    _MODULE_CACHE[key] = mod
    return mod._dfun_nb


# ---------------------------------------------------------------------------
# Parameter packing
# ---------------------------------------------------------------------------

def build_params(spec: SimulationSpec) -> np.ndarray:
    """
    Pack all model parameters into a (n_params, n_nodes) float64 array.

    Scalar parameters are broadcast to all nodes.
    Per-node parameters (shape (n_nodes,)) are stored as-is.
    node_params overrides model defaults (same semantics as the NumPy backend).
    """
    n_nodes  = spec.n_nodes
    n_params = len(spec.model.parameters)
    result   = np.empty((n_params, n_nodes), dtype=np.float64)

    defaults = spec.model.default_params
    overrides = spec.node_params

    for i, p in enumerate(spec.model.parameters):
        val = overrides.get(p.name, defaults[p.name])
        val = np.asarray(val, dtype=np.float64)
        if val.ndim == 0:
            result[i, :] = val.item()
        elif val.shape == (1,):
            result[i, :] = val.item()
        elif val.shape == (n_nodes,):
            result[i, :] = val
        else:
            raise ValueError(
                f"Parameter {p.name!r} has shape {val.shape!r}; "
                f"expected scalar, (1,), or ({n_nodes},)."
            )

    return result


def get_G_idx(spec: ModelSpec) -> int:
    """Index of the 'G' parameter in model.param_names, or -1 if absent."""
    try:
        return list(spec.param_names).index("G")
    except ValueError:
        return -1


def build_srcbuf(n_cvar: int, n_nodes: int, horizon: int,
                 initial_cvar_state: np.ndarray) -> np.ndarray:
    """
    Initialise the ring buffer.

    Shape: (n_cvar, n_nodes, horizon)
    All horizon slots are set to initial_cvar_state (n_cvar, n_nodes).
    """
    buf = np.empty((n_cvar, n_nodes, horizon), dtype=np.float64)
    for h in range(horizon):
        buf[:, :, h] = initial_cvar_state
    return buf


def build_initial_state(spec: SimulationSpec) -> np.ndarray:
    """Initial state array (n_sv, n_nodes) from ModelSpec.state_variables defaults."""
    state = np.zeros((spec.model.n_sv, spec.n_nodes), dtype=np.float64)
    for i, sv in enumerate(spec.model.state_variables):
        state[i, :] = sv.default_init
    return state


def get_bounds(spec: ModelSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (lower_bounds, has_lower, upper_bounds, has_upper) each shape (n_sv,).
    has_lower / has_upper are bool arrays.
    """
    n_sv = spec.n_sv
    lower_bounds = np.zeros(n_sv, dtype=np.float64)
    upper_bounds = np.zeros(n_sv, dtype=np.float64)
    has_lower    = np.zeros(n_sv, dtype=np.bool_)
    has_upper    = np.zeros(n_sv, dtype=np.bool_)
    for i, sv in enumerate(spec.state_variables):
        if sv.lower_bound is not None:
            lower_bounds[i] = sv.lower_bound
            has_lower[i]    = True
        if sv.upper_bound is not None:
            upper_bounds[i] = sv.upper_bound
            has_upper[i]    = True
    return lower_bounds, has_lower, upper_bounds, has_upper


def get_noise_params(spec: SimulationSpec) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (eff_noise_amp, noise_mask).

    eff_noise_amp : (n_noise_vars,) float64 - amplitude * sqrt(dt), already
                    accounting for noise_style ("amplitude" or "tvb").
    noise_mask    : (n_sv,) bool - True for state vars that receive noise.
    """
    ni       = spec.model.noise_indices
    n_sv     = spec.model.n_sv
    nsig     = spec.integrator.noise_nsig
    if nsig is None:
        nsig = np.ones(len(ni)) * 1e-3
    nsig = np.atleast_1d(np.asarray(nsig, dtype=np.float64))
    if nsig.ndim != 1:
        raise ValueError(
            f"noise_nsig must be a 1-D array; got shape {nsig.shape}."
        )
    if nsig.shape[0] == 1 and len(ni) > 1:
        nsig = np.broadcast_to(nsig, (len(ni),)).copy()
    if nsig.shape[0] != len(ni):
        raise ValueError(
            f"noise_nsig length ({nsig.shape[0]}) does not match "
            f"the number of noise variables ({len(ni)}) in model "
            f"{spec.model.name!r}."
        )

    style = getattr(spec.integrator, "noise_style", "amplitude")
    if style == "amplitude":
        amp = nsig
    elif style == "tvb":
        amp = np.sqrt(2.0 * nsig)
    else:
        raise ValueError(f"Unknown noise_style: {style!r}")

    eff_amp   = amp * np.sqrt(spec.integrator.dt)
    mask      = np.zeros(n_sv, dtype=np.bool_)
    mask[list(ni)] = True
    return eff_amp, mask
