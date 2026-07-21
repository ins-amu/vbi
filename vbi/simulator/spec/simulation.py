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
from .connectivity import Connectivity


_MODEL_REGISTRY: "dict | None" = None


def _get_model_registry() -> dict:
    """Build (once) and return the name → model-module registry."""
    global _MODEL_REGISTRY
    if _MODEL_REGISTRY is not None:
        return _MODEL_REGISTRY
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
    reg = {m.name.lower(): m for m in _all_models}
    reg.update({
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
    })
    _MODEL_REGISTRY = reg
    return _MODEL_REGISTRY


def register_model(model, *aliases: str) -> None:
    """
    Add a custom model to the ``SimulationSpec.from_dict()`` registry.

    Call this before any ``from_dict()`` or ``from_config()`` call that uses
    the new model name.

    Parameters
    ----------
    model : ModelSpec
        The model object (must have a ``.name`` attribute).
    *aliases : str
        Extra lookup names, e.g. ``"my_model"``, ``"mymodel"``.

    Examples
    --------
    >>> from vbi.simulator.spec.simulation import register_model
    >>> register_model(my_model, "my_model", "mymodel")
    """
    reg = _get_model_registry()          # ensures base registry is built first
    reg[model.name.lower()] = model
    for alias in aliases:
        reg[alias.lower()] = model


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
    connectivity : Connectivity
        Structural connectivity (weights, tract_lengths, speed, optional metadata).
    node_params : dict[str, np.ndarray]
        Per-node parameter overrides, e.g. {"eta": np.full(80, -4.6)}.
        Overrides ModelSpec.parameters defaults for named entries.
    """
    model:        ModelSpec
    integrator:   IntegratorSpec
    coupling:     CouplingSpec
    monitors:     tuple[MonitorSpec, ...]
    connectivity: Connectivity
    node_params:  dict[str, np.ndarray] = field(default_factory=dict)
    stimuli:      tuple[StimSpec, ...] = field(default_factory=tuple)

    # ------------------------------------------------------------------
    # Delegation properties — backends use these without change
    # ------------------------------------------------------------------

    @property
    def weights(self) -> np.ndarray:
        return self.connectivity.weights

    @property
    def tract_lengths(self) -> np.ndarray:
        return self.connectivity.tract_lengths

    @property
    def speed(self) -> float:
        return self.connectivity.speed

    @property
    def n_nodes(self) -> int:
        return self.connectivity.n_nodes

    @property
    def has_delays(self) -> bool:
        """True only when at least one tract length is non-zero."""
        return self.connectivity.has_delays

    def delay_steps(self, dt: float | None = None) -> np.ndarray:
        """(n_nodes, n_nodes) int32 array of delay in integration steps."""
        _dt = dt if dt is not None else self.integrator.dt
        return self.connectivity.delay_steps(_dt)

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
        _MODELS = _get_model_registry()

        # Model
        model_key = str(d["model"]).lower()
        if model_key not in _MODELS:
            raise ValueError(
                f"Unknown model {d['model']!r}. "
                f"Available: {sorted(_MODELS)}"
            )
        model = _MODELS[model_key]

        # Connectivity
        raw_conn = d["connectivity"]
        speed = float(d.get("speed", 4.0))
        if isinstance(raw_conn, Connectivity):
            connectivity = raw_conn
        elif isinstance(raw_conn, (str, Path)):
            path = Path(raw_conn)
            if path.suffix.lower() == ".npz":
                connectivity = Connectivity.load(path)
            else:
                connectivity = Connectivity.from_file(path, normalize=False, speed=speed)
        elif isinstance(raw_conn, dict):
            weights = np.asarray(raw_conn["weights"])
            tl_raw  = raw_conn.get("tract_lengths")
            tl      = np.asarray(tl_raw) if tl_raw is not None else None
            connectivity = Connectivity(weights=weights, tract_lengths=tl, speed=speed)
        else:
            connectivity = Connectivity.from_file(raw_conn, normalize=False, speed=speed)

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
        n_nodes = connectivity.n_nodes
        node_params = {
            k: (np.full(n_nodes, float(v)) if np.isscalar(v) else np.asarray(v))
            for k, v in raw_np.items()
        }

        if "stimuli" in d:
            raise ValueError(
                "Stimuli configuration is not yet supported in SimulationSpec.from_dict(). "
                "Create the SimulationSpec manually and pass stimuli= directly."
            )

        return cls(
            model        = model,
            integrator   = integrator,
            coupling     = coupling,
            monitors     = monitors,
            connectivity = connectivity,
            node_params  = node_params,
        )

    def cache_key(self) -> str:
        """SHA-256 of a canonical payload - same spec → same compiled binary."""
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
