from __future__ import annotations
from typing import Callable
import numpy as np


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
    """Euler-Maruyama."""
    def step(self, state: np.ndarray, dfun_fn: Callable,
             coupling: np.ndarray, dt: float,
             noise_nsig: np.ndarray, noise_mask: np.ndarray,
             rng: np.random.Generator) -> np.ndarray:
        dW = np.zeros_like(state)
        dW[noise_mask] = rng.standard_normal(noise_mask.sum()) * np.sqrt(dt)
        dW[noise_mask] *= noise_nsig[:, np.newaxis]
        return state + dt * dfun_fn(state, coupling) + dW


class HeunStochastic:
    """Stochastic Heun (Stratonovich midpoint)."""
    def step(self, state: np.ndarray, dfun_fn: Callable,
             coupling: np.ndarray, dt: float,
             noise_nsig: np.ndarray, noise_mask: np.ndarray,
             rng: np.random.Generator) -> np.ndarray:
        dW = np.zeros_like(state)
        dW[noise_mask] = rng.standard_normal(noise_mask.sum()) * np.sqrt(dt)
        dW[noise_mask] *= noise_nsig[:, np.newaxis]
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
