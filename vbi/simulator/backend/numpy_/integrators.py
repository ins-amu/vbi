from __future__ import annotations
from typing import Callable
import numpy as np


def _additive_white_noise(
    state: np.ndarray,
    noise_amp: np.ndarray,
    noise_mask: np.ndarray,
    dt: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Additive white-noise increment: amp * sqrt(dt) * N(0, 1)."""
    n_noise_vars = int(noise_mask.sum())
    noise = np.zeros_like(state)
    diffusion = noise_amp[:, np.newaxis]
    white = rng.standard_normal((n_noise_vars, state.shape[1])) * np.sqrt(dt)
    noise[noise_mask] = diffusion * white
    return noise


class EulerDeterministic:
    def step(self, state: np.ndarray, dfun_fn: Callable,
             coupling: np.ndarray, dt: float) -> np.ndarray:
        return state + dt * dfun_fn(state, coupling)


class HeunDeterministic:
    def step(self, state: np.ndarray, dfun_fn: Callable,
             coupling: np.ndarray, dt: float) -> np.ndarray:
        k1 = dfun_fn(state, coupling)
        k2 = dfun_fn(state + dt * k1, coupling)
        return state + 0.5 * dt * (k1 + k2)


class EulerStochastic:
    """Euler-Maruyama with additive white noise."""
    def step(self, state: np.ndarray, dfun_fn: Callable,
             coupling: np.ndarray, dt: float,
             noise_amp: np.ndarray, noise_mask: np.ndarray,
             rng: np.random.Generator) -> np.ndarray:
        dW = _additive_white_noise(state, noise_amp, noise_mask, dt, rng)
        return state + dt * dfun_fn(state, coupling) + dW


class HeunStochastic:
    """Stochastic Heun with additive white noise."""
    def step(self, state: np.ndarray, dfun_fn: Callable,
             coupling: np.ndarray, dt: float,
             noise_amp: np.ndarray, noise_mask: np.ndarray,
             rng: np.random.Generator) -> np.ndarray:
        dW = _additive_white_noise(state, noise_amp, noise_mask, dt, rng)
        k1 = dfun_fn(state, coupling)
        x_pred = state + dt * k1 + dW
        k2 = dfun_fn(x_pred, coupling)
        return state + 0.5 * dt * (k1 + k2) + dW


def build_integrator(method: str, stochastic: bool):
    if method == "euler":
        return EulerStochastic() if stochastic else EulerDeterministic()
    if method == "heun":
        return HeunStochastic() if stochastic else HeunDeterministic()
    raise ValueError(f"Unknown integrator method: {method!r}")
