from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
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

    @classmethod
    def from_dict(cls, d: dict) -> "SimulationSpec":
        """
        Build a SimulationSpec from a plain dictionary (e.g. parsed from YAML).

        Required keys
        -------------
        ``model`` : str  model name, e.g. ``"mpr"``
        ``connectivity`` : str | dict
            Path to a ``.npz`` file containing ``weights`` (and optionally
            ``tract_lengths``), **or** a dict with ``weights`` and optionally
            ``tract_lengths`` as lists/arrays.

        Optional keys
        -------------
        ``dt``, ``method``, ``stochastic``, ``noise_seed``,
        ``monitors``  (list of ``{kind, period}`` dicts),
        ``coupling``  (``{kind, a, b}`` dict),
        ``node_params``  (``{name: value}`` dict),
        ``speed``, ``noise_nsig``
        """
        from vbi.simulator.models import (
            mpr, jansen_rit, wilson_cowan, reduced_wong_wang,
            wong_wang_exc_inh, generic_2d_oscillator, kuramoto,
            sup_hopf, linear, larter_breakspear,
            coombes_byrne_2d, gast_sd, gast_sf, vep, ghb, sl,
            damped_oscillator,
        )
        _all_models = (
            mpr, jansen_rit, wilson_cowan, reduced_wong_wang,
            wong_wang_exc_inh, generic_2d_oscillator, kuramoto,
            sup_hopf, linear, larter_breakspear,
            coombes_byrne_2d, gast_sd, gast_sf, vep, ghb, sl,
            damped_oscillator,
        )
        # Register by long model name and by short Python variable alias
        _aliases = {
            "mpr": mpr, "jansen_rit": jansen_rit, "jansenrit": jansen_rit,
            "wilson_cowan": wilson_cowan, "wilsoncowan": wilson_cowan,
            "reduced_wong_wang": reduced_wong_wang,
            "wong_wang_exc_inh": wong_wang_exc_inh,
            "generic_2d_oscillator": generic_2d_oscillator,
            "generic2doscillator": generic_2d_oscillator,
            "sup_hopf": sup_hopf, "suphopf": sup_hopf,
            "larter_breakspear": larter_breakspear,
            "coombes_byrne_2d": coombes_byrne_2d,
            "gast_sd": gast_sd, "gast_sf": gast_sf,
            "ghb": ghb, "sl": sl,
            "damped_oscillator": damped_oscillator,
            "kuramoto": kuramoto, "linear": linear, "vep": vep,
        }
        _MODELS = {m.name.lower(): m for m in _all_models}
        _MODELS.update(_aliases)

        # Model
        model_key = str(d["model"]).lower()
        if model_key not in _MODELS:
            raise ValueError(
                f"Unknown model {d['model']!r}. "
                f"Available: {sorted(_MODELS)}"
            )
        model = _MODELS[model_key]

        # Connectivity — delegate to prepare_connectivity for format flexibility
        from .connectivity import prepare_connectivity
        conn = d["connectivity"]
        if isinstance(conn, (str, Path)):
            path = Path(conn)
            if path.suffix.lower() == ".npz":
                npz = np.load(path)
                weights       = npz["weights"]
                tract_lengths = npz.get("tract_lengths")
                weights, tract_lengths = prepare_connectivity(
                    weights, tract_lengths, normalize=False
                )
            else:
                # .txt / .csv / .npy
                weights, tract_lengths = prepare_connectivity(
                    path, normalize=False
                )
        elif isinstance(conn, dict):
            weights       = np.asarray(conn["weights"])
            tl_raw        = conn.get("tract_lengths")
            tract_lengths = np.asarray(tl_raw) if tl_raw is not None else None
            weights, tract_lengths = prepare_connectivity(
                weights, tract_lengths, normalize=False
            )
        else:
            weights, tract_lengths = prepare_connectivity(conn, normalize=False)

        # Integrator
        integrator = IntegratorSpec(
            method     = d.get("method", "heun"),
            dt         = float(d.get("dt", 0.01)),
            stochastic = bool(d.get("stochastic", False)),
            noise_nsig = (np.asarray(d["noise_nsig"]) if "noise_nsig" in d else None),
            noise_seed = int(d.get("noise_seed", 42)),
        )

        # Coupling
        c = d.get("coupling", {})
        coupling = CouplingSpec(
            kind  = c.get("kind", "linear"),
            a     = float(c.get("a", 1.0)),
            b     = float(c.get("b", 0.0)),
        )

        # Monitors
        raw_monitors = d.get("monitors", [{"kind": "tavg", "period": 1.0}])
        monitors = tuple(
            MonitorSpec(
                kind   = m["kind"],
                period = float(m["period"]) if "period" in m else None,
            )
            for m in raw_monitors
        )

        # node_params: scalar values broadcast to per-node arrays
        raw_np = d.get("node_params", {})
        n_nodes = weights.shape[0]
        node_params = {
            k: (np.full(n_nodes, float(v)) if np.isscalar(v) else np.asarray(v))
            for k, v in raw_np.items()
        }

        return cls(
            model        = model,
            integrator   = integrator,
            coupling     = coupling,
            monitors     = monitors,
            weights      = weights,
            tract_lengths = tract_lengths,
            speed        = float(d.get("speed", 4.0)),
            node_params  = node_params,
        )

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
