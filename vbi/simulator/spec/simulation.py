from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass, field
import numpy as np

from .model import ModelSpec
from .integrator import IntegratorSpec
from .coupling import CouplingSpec
from .monitor import MonitorSpec
from .stimulus import StimSpec


@dataclass
class SimulationSpec:
    """
    Complete description of one simulation.

    Parameters
    ----------
    model : ModelSpec
    integrator : IntegratorSpec
    coupling : CouplingSpec
    monitors : tuple[MonitorSpec, ...]
    weights : np.ndarray
        (n_nodes, n_nodes) structural connectivity; weights[tgt, src].
    tract_lengths : np.ndarray
        (n_nodes, n_nodes) delay distances in mm; tract_lengths[src, tgt].
    speed : float
        Conduction velocity in mm/ms (default 4.0 mm/ms = 4 m/s).
    node_params : dict[str, np.ndarray]
        Per-node parameter overrides, e.g. {"eta": np.full(80, -4.6)}.
        Overrides ModelSpec.parameters defaults for named entries.
    """
    model: ModelSpec
    integrator: IntegratorSpec
    coupling: CouplingSpec
    monitors: tuple[MonitorSpec, ...]
    weights: np.ndarray
    tract_lengths: np.ndarray | None = None  # None → zero delays (pure ODE/SDE)
    speed: float = 4.0
    node_params: dict[str, np.ndarray] = field(default_factory=dict)
    stimuli: tuple[StimSpec, ...] = field(default_factory=tuple)

    def __post_init__(self):
        # Coerce weights and tract_lengths to float64 ndarray (accepts lists etc.)
        object.__setattr__(self, "weights",
                           np.asarray(self.weights, dtype=np.float64))
        if self.tract_lengths is None:
            object.__setattr__(self, "tract_lengths",
                               np.zeros_like(self.weights))
        else:
            object.__setattr__(self, "tract_lengths",
                               np.asarray(self.tract_lengths, dtype=np.float64))
        # Shape / value validation
        w = self.weights
        tl = self.tract_lengths
        if w.ndim != 2 or w.shape[0] != w.shape[1]:
            raise ValueError(
                f"weights must be a square 2-D array; got shape {w.shape}."
            )
        if tl.shape != w.shape:
            raise ValueError(
                f"tract_lengths shape {tl.shape} must match weights shape {w.shape}."
            )
        if np.any(tl < 0):
            raise ValueError(
                "tract_lengths must be non-negative; found negative values."
            )
        if self.speed <= 0:
            raise ValueError(
                f"speed must be > 0 mm/ms; got speed={self.speed!r}."
            )

    @property
    def n_nodes(self) -> int:
        return self.weights.shape[0]

    @property
    def has_delays(self) -> bool:
        """True only when at least one tract length is non-zero."""
        return bool(self.tract_lengths.any())

    def delay_steps(self, dt: float | None = None) -> np.ndarray:
        """(n_nodes, n_nodes) int32 array of delay in integration steps."""
        _dt = dt if dt is not None else self.integrator.dt
        raw = self.tract_lengths / (self.speed * _dt)
        return np.round(raw).astype(np.int32)

    def horizon(self, dt: float | None = None) -> int:
        """Ring-buffer depth = max delay + 1."""
        d = self.delay_steps(dt)
        return int(d.max()) + 1 if d.size > 0 else 1

    def cache_key(self) -> str:
        """SHA-256 of a canonical payload — same spec → same compiled binary."""
        payload = {
            "model": self.model.name,
            "sv": list(self.model.sv_names),
            "params": {p.name: float(p.default) if np.isscalar(p.default) else list(p.default)
                       for p in self.model.parameters},
            "dfun_str": self.model.dfun_str,
            "integrator": {"method": self.integrator.method, "dt": self.integrator.dt,
                           "stochastic": self.integrator.stochastic},
            "coupling": {"kind": self.coupling.kind, "a": self.coupling.a, "b": self.coupling.b},
            "n_nodes": self.n_nodes,
            "speed": self.speed,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()
