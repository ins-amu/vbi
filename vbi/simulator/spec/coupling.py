from __future__ import annotations
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class CouplingSpec:
    kind: Literal["linear", "sigmoidal", "kuramoto"] = "linear"
    # linear: c_i = G * a * sum_j(w_ij * x_j(t-tau)) + b
    a: float = 1.0
    b: float = 0.0
    # sigmoidal extras
    midpoint: float = 0.0
    sigma: float = 1.0
    # kuramoto: frustration (phase offset) in sin(θ_j - θ_i + alpha)
    alpha: float = 0.0
