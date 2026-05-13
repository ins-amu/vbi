from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal
import numpy as np


@dataclass(frozen=True)
class StateVar:
    name: str
    default_init: float = 0.0
    noise: bool = False
    lower_bound: float | None = None
    upper_bound: float | None = None


@dataclass(frozen=True)
class Parameter:
    name: str
    default: float | np.ndarray
    description: str = ""


@dataclass(frozen=True)
class ModelSpec:
    """
    Backend-agnostic description of a neural mass model.

    dfun_str maps each state variable name to a bare math expression string.
    Only these symbols may appear: state variable names, parameter names, 'c'
    (coupling input, scalar per node), and the math functions:
        exp, log, sin, cos, tanh, sqrt, abs, pi
    No 'np.' prefix — the code generator injects the namespace.
    """
    name: str
    state_variables: tuple[StateVar, ...]
    parameters: tuple[Parameter, ...]
    cvar: tuple[str, ...]         # names of coupling-variable state vars
    dfun_str: dict[str, str]      # {sv_name: expression_string}
    noise_variables: tuple[str, ...] = ()
    reference: str = ""

    @property
    def sv_names(self) -> tuple[str, ...]:
        return tuple(sv.name for sv in self.state_variables)

    @property
    def n_sv(self) -> int:
        return len(self.state_variables)

    @property
    def cvar_indices(self) -> tuple[int, ...]:
        names = self.sv_names
        return tuple(names.index(c) for c in self.cvar)

    @property
    def noise_indices(self) -> tuple[int, ...]:
        names = self.sv_names
        return tuple(names.index(n) for n in self.noise_variables)

    @property
    def param_names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.parameters)

    @property
    def default_params(self) -> dict[str, float | np.ndarray]:
        return {p.name: p.default for p in self.parameters}
