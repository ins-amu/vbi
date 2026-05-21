from __future__ import annotations
from typing import Callable

from vbi.simulator.spec.model import ModelSpec


def build_jax_dfun(spec: ModelSpec) -> Callable:
    """
    Compile dfun_str expressions into a JAX-traceable function.

    Generated signature:
        fn(state, coupling, params) -> jnp.ndarray  shape (n_sv, n_nodes)

    where:
        state    : (n_sv, n_nodes)
        coupling : (n_cvar, n_nodes)  — one row per coupling variable
        params   : dict[str, scalar | jnp.ndarray]

    Uses jnp.stack to build the output — avoids .at[].set() in the hot path.
    Compiled once at build() time via exec() on our own spec strings.
    """
    sv = spec.sv_names
    param_names = spec.param_names
    cvar_names = spec.cvar

    lines = [
        "import jax.numpy as _jnp",
        "from jax.numpy import pi, exp, log, sin, cos, tanh, sqrt",
        "def _dfun_jax(state, coupling, params):",
    ]
    # Unpack state variables
    for i, name in enumerate(sv):
        lines.append(f"    {name} = state[{i}]")
    # Unpack params (values may be traced JAX scalars when vmapped)
    for name in param_names:
        lines.append(f"    {name} = params['{name}']")
    # Inject coupling by cvar name: c_r, c_V, …
    for i, cname in enumerate(cvar_names):
        lines.append(f"    c_{cname} = coupling[{i}]")
    # c = first coupling term — backward compat for single-cvar models
    lines.append("    c = coupling[0]")
    # Stack derivative expressions into (n_sv, n_nodes) output
    sv_exprs = ", ".join(f"({spec.dfun_str[name]})" for name in sv)
    lines.append(f"    return _jnp.stack([{sv_exprs}])")

    src = "\n".join(lines)
    globs: dict = {}
    exec(compile(src, "<dfun_jax>", "exec"), globs)
    return globs["_dfun_jax"]
