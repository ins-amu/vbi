from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import numpy as np


@dataclass(frozen=True)
class IntegratorSpec:
    method: Literal["euler", "heun"] = "heun"
    dt: float = 0.01                        # ms
    stochastic: bool = False
    # One value per noisy state variable.
    # "amplitude": noise_nsig * sqrt(dt) * N(0, 1)
    # "tvb": sqrt(2 * noise_nsig) * sqrt(dt) * N(0, 1)
    noise_nsig: np.ndarray | None = None
    noise_style: Literal["amplitude", "tvb"] = "amplitude"
    noise_seed: int = 42
