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
    if name == "numba":
        try:
            from vbi.simulator.backend.numba_.simulator import NumbaSimulator
            return NumbaSimulator
        except ImportError as exc:
            raise ImportError(
                "Numba backend requires numba: pip install numba"
            ) from exc
    if name == "cpp":
        try:
            from vbi.simulator.backend.cpp.simulator import CppSimulator
            return CppSimulator
        except ImportError as exc:
            raise ImportError(
                "C++ backend requires pybind11 and mako: "
                "pip install pybind11 mako"
            ) from exc
    raise ImportError(
        f"Backend {name!r} is not available. "
        "Available: 'numpy', 'numba', 'cpp'. Coming: 'cuda', 'jax'."
    )


def load_sweep_backend(name: str) -> type:
    if name == "numpy":
        from vbi.simulator.backend.numpy_.sweeper import NumpySweeper
        return NumpySweeper
    if name == "numba":
        try:
            from vbi.simulator.backend.numba_.sweeper import NumbaSweeperCPU
            return NumbaSweeperCPU
        except ImportError as exc:
            raise ImportError(
                "Numba backend requires numba: pip install numba"
            ) from exc
    if name == "cpp":
        try:
            from vbi.simulator.backend.cpp.sweeper import CppSweeper
            return CppSweeper
        except ImportError as exc:
            raise ImportError(
                "C++ backend requires pybind11 and mako: "
                "pip install pybind11 mako"
            ) from exc
    raise ImportError(
        f"Sweep backend {name!r} is not available. "
        "Available: 'numpy', 'numba', 'cpp'. Coming: 'cuda', 'jax'."
    )
