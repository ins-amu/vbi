from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import numpy as np


@dataclass(frozen=True)
class IntegratorSpec:
    method: Literal["euler", "heun"] = "heun"
    dt: float = 0.01                        # ms
    stochastic: bool = False
    noise_nsig: np.ndarray | None = None   # shape (n_noise_vars,), one per noise sv
    noise_seed: int = 42
