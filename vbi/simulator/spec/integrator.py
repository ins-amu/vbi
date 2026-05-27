from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import numpy as np


@dataclass(frozen=True)
class IntegratorSpec:
    method: Literal["euler", "heun"] = "heun"
    dt: float = 0.01                        # ms — must be > 0
    stochastic: bool = False

    def __post_init__(self):
        if self.dt <= 0:
            raise ValueError(f"IntegratorSpec.dt must be > 0; got dt={self.dt!r}.")
    # One value per noisy state variable.
    # "amplitude": noise_nsig * sqrt(dt) * N(0, 1)
    # "tvb": sqrt(2 * noise_nsig) * sqrt(dt) * N(0, 1)
    noise_nsig: np.ndarray | None = None
    noise_style: Literal["amplitude", "tvb"] = "amplitude"
    noise_seed: int = 42
    # JAX-backend precision: "float32" (default, GPU-optimised) or "float64"
    # (matches NumPy/Numba for validation; requires jax_enable_x64=True).
    jax_dtype: Literal["float32", "float64"] = "float32"
