"""
Code generation for the VBI C++ backend.

Translates ModelSpec.dfun_str expressions to C++ via an AST visitor,
then renders Mako templates and compiles with pybind11 + CMake.
"""
from __future__ import annotations

import ast as _ast
import importlib.machinery
import os
import shutil
import subprocess
import sys
import sysconfig
import types
from pathlib import Path

import numpy as np

from vbi.simulator.spec.model import ModelSpec
from vbi.simulator.spec.simulation import SimulationSpec

_TEMPLATES_DIR = Path(__file__).resolve().parent / "_src"

# ---------------------------------------------------------------------------
# Python → C++ expression translator (adapted from tvb-root-hybrid-cpp)
# ---------------------------------------------------------------------------

_BARE_TO_STD: dict[str, str] = {
    "exp":   "std::exp",
    "log":   "std::log",
    "log2":  "std::log2",
    "log10": "std::log10",
    "sin":   "std::sin",
    "cos":   "std::cos",
    "tanh":  "std::tanh",
    "sinh":  "std::sinh",
    "cosh":  "std::cosh",
    "sqrt":  "std::sqrt",
    "abs":   "std::abs",
    "fabs":  "std::fabs",
    "atan":  "std::atan",
    "atan2": "std::atan2",
    "floor": "std::floor",
    "ceil":  "std::ceil",
    "pow":   "std::pow",
}


class _CppExprGen(_ast.NodeVisitor):
    """Walk a Python math-expression AST and emit C++ source."""

    def __init__(
        self,
        sv_names: tuple[str, ...],
        param_names: tuple[str, ...],
        cvar_names: tuple[str, ...],    # e.g. ("r", "V")
        n_nodes_sym: str = "kNumNodes", # C++ constant for n_nodes
        node_var: str = "n",            # loop variable name
    ) -> None:
        self._sv    = {name: i for i, name in enumerate(sv_names)}
        self._params= {name: i for i, name in enumerate(param_names)}
        # coupling term name → slot index:  c_r → 0, c_V → 1
        self._cvar_terms = {f"c_{name}": i for i, name in enumerate(cvar_names)}
        self._N  = n_nodes_sym
        self._nd = node_var

    def visit_Constant(self, node: _ast.Constant) -> str:
        if isinstance(node.value, bool):
            return "true" if node.value else "false"
        if isinstance(node.value, float):
            return f"{node.value:.17g}"
        if isinstance(node.value, int):
            return f"{node.value}.0"
        return str(node.value)

    def visit_Name(self, node: _ast.Name) -> str:
        name = node.id
        if name == "pi":
            return "M_PI"
        if name == "c":
            # backward-compat alias for first coupling term
            return f"coupling[0 * {self._N} + {self._nd}]"
        if name in self._cvar_terms:
            idx = self._cvar_terms[name]
            return f"coupling[{idx} * {self._N} + {self._nd}]"
        if name in self._sv:
            idx = self._sv[name]
            return f"state[{idx} * {self._N} + {self._nd}]"
        if name in self._params:
            idx = self._params[name]
            return f"params[{idx} * {self._N} + {self._nd}]"
        if name in _BARE_TO_STD:
            return _BARE_TO_STD[name]
        raise ValueError(
            f"Unknown name {name!r} in model expression — not a state variable, "
            f"parameter, coupling term, or supported math function."
        )

    def visit_BinOp(self, node: _ast.BinOp) -> str:
        left  = self.visit(node.left)
        right = self.visit(node.right)
        if isinstance(node.op, _ast.Pow):
            return f"std::pow({left}, {right})"
        op = {
            _ast.Add:  "+",
            _ast.Sub:  "-",
            _ast.Mult: "*",
            _ast.Div:  "/",
        }.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported binary op: {type(node.op).__name__}")
        return f"({left} {op} {right})"

    def visit_UnaryOp(self, node: _ast.UnaryOp) -> str:
        operand = self.visit(node.operand)
        if isinstance(node.op, _ast.USub):
            return f"(-{operand})"
        if isinstance(node.op, _ast.UAdd):
            return operand
        raise ValueError(f"Unsupported unary op: {type(node.op).__name__}")

    def visit_Call(self, node: _ast.Call) -> str:
        args = [self.visit(a) for a in node.args]
        if isinstance(node.func, _ast.Attribute):
            obj = node.func.value
            if isinstance(obj, _ast.Name) and obj.id == "math":
                fn = _BARE_TO_STD.get(node.func.attr, f"std::{node.func.attr}")
                return f"{fn}({', '.join(args)})"
        if isinstance(node.func, _ast.Name):
            fn = node.func.id
            if fn in _BARE_TO_STD:
                return f"{_BARE_TO_STD[fn]}({', '.join(args)})"
            raise ValueError(
                f"Unknown function {fn!r} in model expression — "
                f"not a supported math function."
            )
        raise ValueError(f"Unsupported call: {_ast.dump(node.func)}")

    def visit_IfExp(self, node: _ast.IfExp) -> str:
        cond  = self.visit(node.test)
        then  = self.visit(node.body)
        else_ = self.visit(node.orelse)
        return f"({cond} ? {then} : {else_})"

    def visit_Compare(self, node: _ast.Compare) -> str:
        _op_map = {
            _ast.Lt:    "<",  _ast.LtE: "<=",
            _ast.Gt:    ">",  _ast.GtE: ">=",
            _ast.Eq:    "==", _ast.NotEq: "!=",
        }
        parts = [self.visit(node.left)]
        for op, comp in zip(node.ops, node.comparators):
            parts.append(_op_map[type(op)])
            parts.append(self.visit(comp))
        return " ".join(parts)

    def generic_visit(self, node: _ast.AST) -> str:
        raise ValueError(f"Unsupported AST node: {type(node).__name__}: {_ast.dump(node)}")


def py_expr_to_cpp(expr: str, spec: ModelSpec) -> str:
    gen = _CppExprGen(
        sv_names    = spec.sv_names,
        param_names = spec.param_names,
        cvar_names  = spec.cvar,
    )
    tree = _ast.parse(expr.strip(), mode="eval")
    return gen.visit(tree.body)


# ---------------------------------------------------------------------------
# Template context builders
# ---------------------------------------------------------------------------

def _build_dfun_cpp(spec: ModelSpec) -> dict[str, str]:
    return {sv: py_expr_to_cpp(expr, spec)
            for sv, expr in spec.dfun_str.items()}


def _build_params_info(spec: ModelSpec) -> list[dict]:
    return [{"name": p.name, "idx": i} for i, p in enumerate(spec.parameters)]


def _build_cvar_info(spec: ModelSpec) -> list[dict]:
    return [{"name": name, "idx": i} for i, name in enumerate(spec.cvar)]


def _build_bounds(spec: ModelSpec) -> list[dict]:
    out = []
    for sv in spec.state_variables:
        out.append({
            "has_lower": sv.lower_bound is not None,
            "lower":     float(sv.lower_bound) if sv.lower_bound is not None else 0.0,
            "has_upper": sv.upper_bound is not None,
            "upper":     float(sv.upper_bound) if sv.upper_bound is not None else 0.0,
        })
    return out


# ---------------------------------------------------------------------------
# Template rendering
# ---------------------------------------------------------------------------

def _render(template_path: Path, ctx: dict) -> str:
    try:
        from mako.lookup import TemplateLookup
        from mako.template import Template
    except ImportError as exc:
        raise ImportError(
            "VBI C++ simulator backend requires mako: pip install mako"
        ) from exc
    lookup = TemplateLookup(directories=[str(template_path.parent)])
    tmpl = Template(
        template_path.read_text(encoding="utf-8"),
        lookup=lookup,
        strict_undefined=True,
    )
    return tmpl.render(**ctx)


def render_sim_module(spec: SimulationSpec, cache_key: str) -> str:
    noise_indices = list(spec.model.noise_indices)
    return _render(
        _TEMPLATES_DIR / "sim_module.cpp.mako",
        {
            "spec":         spec,
            "dfun_cpp":     _build_dfun_cpp(spec.model),
            "params_info":  _build_params_info(spec.model),
            "cvar_info":    _build_cvar_info(spec.model),
            "bounds":       _build_bounds(spec.model),
            "noise_indices": noise_indices,
            "cache_key":    cache_key,
            "coup_kind":    spec.coupling.kind,
        },
    )


def render_bindings(spec: SimulationSpec, module_name: str, cpp_filename: str) -> str:
    return _render(
        _TEMPLATES_DIR / "bindings.cpp.mako",
        {
            "module_name":  module_name,
            "cpp_filename": cpp_filename,
            "n_sv":         spec.model.n_sv,
            "n_nodes":      spec.n_nodes,
            "n_params":     len(spec.model.parameters),
            "n_noise_sv":   len(spec.model.noise_indices),
        },
    )


def render_cmake(module_name: str, bindings_cpp_filename: str) -> str:
    _env = os.environ.get("VBI_CPP_CXXFLAGS", "")
    cxx_flags = _env if _env else "-O3 -march=native -ffast-math"
    return _render(
        _TEMPLATES_DIR / "cmake_template.mako",
        {
            "module_name":           module_name,
            "bindings_cpp_filename": bindings_cpp_filename,
            "python_executable":     sys.executable,
            "cxx_flags_release":     cxx_flags,
        },
    )


# ---------------------------------------------------------------------------
# Parameter packing  (matches Numba backend layout: (n_params, n_nodes))
# ---------------------------------------------------------------------------

def build_params_array(spec: SimulationSpec) -> np.ndarray:
    n_nodes  = spec.n_nodes
    n_params = len(spec.model.parameters)
    arr = np.empty((n_params, n_nodes), dtype=np.float64)
    defaults  = spec.model.default_params
    overrides = spec.node_params
    for i, p in enumerate(spec.model.parameters):
        val = np.asarray(overrides.get(p.name, defaults[p.name]), dtype=np.float64)
        if val.ndim == 0:
            arr[i, :] = val.item()
        elif val.shape == (1,):
            arr[i, :] = val.item()
        elif val.shape == (n_nodes,):
            arr[i, :] = val
        else:
            raise ValueError(
                f"Parameter {p.name!r} has shape {val.shape!r}; "
                f"expected scalar, (1,), or ({n_nodes},)."
            )
    return arr  # (n_params, n_nodes), C-contiguous


def get_G(spec: SimulationSpec) -> float:
    """Return scalar G from model params (default 1.0 if absent)."""
    defaults  = spec.model.default_params
    overrides = spec.node_params
    val = np.asarray(overrides.get("G", defaults.get("G", 1.0)), dtype=np.float64)
    if val.ndim == 0 or val.shape == (1,):
        return float(val.flat[0])
    raise ValueError(
        f"G must be a scalar; got shape {val.shape!r}."
    )


def get_noise_data(spec: SimulationSpec, n_steps: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Pre-generate all noise increments as a (n_steps, n_sv, n_nodes) array.
    Returns (noise_data, noise_sv_indices_int32).
    """
    ni = list(spec.model.noise_indices)
    n_sv = spec.model.n_sv
    n_nodes = spec.n_nodes
    nsig = spec.integrator.noise_nsig
    if nsig is None:
        nsig = np.ones(len(ni), dtype=np.float64) * 1e-3
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
    amp = np.sqrt(2.0 * nsig) if style == "tvb" else nsig
    eff_amp = amp * np.sqrt(spec.integrator.dt)   # (n_noise_vars,)

    rng = np.random.default_rng(spec.integrator.noise_seed)
    # Full (n_steps, n_sv, n_nodes) noise array — only noisy rows are non-zero
    noise = np.zeros((n_steps, n_sv, n_nodes), dtype=np.float64)
    for k, sv_idx in enumerate(ni):
        noise[:, sv_idx, :] = eff_amp[k] * rng.standard_normal((n_steps, n_nodes))

    return (
        np.ascontiguousarray(noise),
        np.array(ni, dtype=np.int32),
    )
