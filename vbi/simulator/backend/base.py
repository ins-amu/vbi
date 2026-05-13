from __future__ import annotations
from typing import Protocol, runtime_checkable
import numpy as np
from vbi.simulator.spec.simulation import SimulationSpec


@runtime_checkable
class AbstractBackend(Protocol):
    def build(self, spec: SimulationSpec) -> None: ...
    def run(self, duration: float) -> dict[str, tuple[np.ndarray, np.ndarray]]: ...


def load_backend(name: str) -> type:
    if name == "numpy":
        from vbi.simulator.backend.numpy_.simulator import NumpySimulator
        return NumpySimulator
    raise ImportError(
        f"Backend {name!r} is not available. "
        "Available now: 'numpy'. Coming: 'numba', 'cpp', 'cuda', 'jax'."
    )


def load_sweep_backend(name: str) -> type:
    if name == "numpy":
        from vbi.simulator.backend.numpy_.sweeper import NumpySweeper
        return NumpySweeper
    raise ImportError(
        f"Sweep backend {name!r} is not available. "
        "Available now: 'numpy'. Coming: 'numba', 'cpp', 'cuda', 'jax'."
    )
