from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from vbi.simulator.spec.monitor import MonitorSpec
from vbi.simulator.spec.model import ModelSpec

# ---------------------------------------------------------------------------
# Balloon-Windkessel helpers (simplified 2-state model matching existing vbi)
# State per node: [s, f]  (vasodilatory signal, blood inflow)
# Based on vbi/models/cpp/_src/bold.hpp BOLD_2D
# ---------------------------------------------------------------------------

_BW_DEFAULTS = dict(
    rho=0.8, e=0.02, taus=0.8, tauf=0.4, k1=5.6, eps=0.5
)


def _bw_init(n_nodes: int) -> np.ndarray:
    """Initial BW state (2, n_nodes): s=0, f=1."""
    state = np.zeros((2, n_nodes))
    state[1] = 1.0   # f starts at rest = 1
    return state


def _bw_step(bw: np.ndarray, neural: np.ndarray, dt: float,
             p: dict) -> np.ndarray:
    """Euler step of the simplified Balloon-Windkessel ODE.

    ds/dt = neural * eps - s/taus - (f-1)/tauf
    df/dt = s
    """
    s, f = bw[0], bw[1]
    ds = neural * p["eps"] - s / p["taus"] - (f - 1.0) / p["tauf"]
    df = s
    bw = bw.copy()
    bw[0] = s + dt * ds
    bw[1] = f + dt * df
    return bw


def _bw_bold(bw: np.ndarray, p: dict) -> np.ndarray:
    """BOLD signal from BW state: coef * (f - 1)."""
    coef = (100.0 / p["rho"]) * p["e"] * p["k1"]
    return coef * (bw[1] - 1.0)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _resolve_voi(variables: tuple[str, ...], model: ModelSpec) -> np.ndarray:
    """Map variable names → state row indices. Empty = all sv indices."""
    if not variables:
        return np.arange(model.n_sv)
    return np.array([model.sv_names.index(v) for v in variables])


# ---------------------------------------------------------------------------
# Monitor classes
# ---------------------------------------------------------------------------

class Monitor(ABC):
    """Abstract monitor — mirrors TVB Monitor interface."""
    istep: int
    dt: float
    voi: np.ndarray

    @abstractmethod
    def configure(self, spec: MonitorSpec, model: ModelSpec, dt: float) -> None: ...

    @abstractmethod
    def sample(self, step: int, state: np.ndarray) -> None: ...

    @abstractmethod
    def result(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (times, data) after the run."""


class RawMonitor(Monitor):
    """Every integration step. TVB: Raw."""

    def configure(self, spec: MonitorSpec, model: ModelSpec, dt: float) -> None:
        self.dt = dt
        self.istep = 1
        self.voi = _resolve_voi(spec.variables, model)
        self._times: list[float] = []
        self._data: list[np.ndarray] = []

    def sample(self, step: int, state: np.ndarray) -> None:
        # Time label is step*dt — state AFTER the step-th integration step.
        # Step 0 therefore labels the state after the first integration step,
        # not the initial condition.  This matches the TVB Raw monitor convention.
        self._times.append(step * self.dt)
        self._data.append(state[self.voi].copy())

    def result(self) -> tuple[np.ndarray, np.ndarray]:
        if not self._data:
            raise ValueError(
                "RawMonitor collected no samples. "
                "duration must be > 0 to produce output."
            )
        return np.array(self._times), np.stack(self._data)


class SubSampleMonitor(Monitor):
    """Decimation: record every `period` ms. TVB: SubSample."""

    def configure(self, spec: MonitorSpec, model: ModelSpec, dt: float) -> None:
        self.dt = dt
        self.istep = max(1, round(spec.period / dt))
        self.voi = _resolve_voi(spec.variables, model)
        self._times: list[float] = []
        self._data: list[np.ndarray] = []

    def sample(self, step: int, state: np.ndarray) -> None:
        if step % self.istep == 0:
            self._times.append(step * self.dt)
            self._data.append(state[self.voi].copy())

    def result(self) -> tuple[np.ndarray, np.ndarray]:
        if not self._data:
            raise ValueError(
                f"SubSampleMonitor collected no samples. "
                f"duration must be >= period ({self.istep * self.dt} ms)."
            )
        return np.array(self._times), np.stack(self._data)


class TemporalAverageMonitor(Monitor):
    """Time-average over `period` ms windows. TVB: TemporalAverage.
    Preferred for SBI — acts as a low-pass filter."""

    def configure(self, spec: MonitorSpec, model: ModelSpec, dt: float) -> None:
        self.dt = dt
        self.istep = max(1, round(spec.period / dt))
        self.voi = _resolve_voi(spec.variables, model)
        self._stock: np.ndarray | None = None
        self._stock_idx = 0
        self._times: list[float] = []
        self._data: list[np.ndarray] = []

    def sample(self, step: int, state: np.ndarray) -> None:
        observed = state[self.voi]   # (n_voi, n_nodes)
        if self._stock is None:
            self._stock = np.zeros((self.istep, *observed.shape))
        self._stock[self._stock_idx] = observed
        self._stock_idx += 1
        if self._stock_idx == self.istep:
            avg = self._stock.mean(axis=0)
            t = (step - self.istep / 2.0) * self.dt
            self._times.append(t)
            self._data.append(avg.copy())
            self._stock_idx = 0

    def result(self) -> tuple[np.ndarray, np.ndarray]:
        if not self._data:
            raise ValueError(
                f"TemporalAverageMonitor collected no samples. "
                f"duration must be >= period ({self.istep * self.dt} ms)."
            )
        return np.array(self._times), np.stack(self._data)


class GlobalAverageMonitor(Monitor):
    """Spatial mean of VOIs, every `period` ms. TVB: GlobalAverage."""

    def configure(self, spec: MonitorSpec, model: ModelSpec, dt: float) -> None:
        self.dt = dt
        self.istep = max(1, round(spec.period / dt))
        self.voi = _resolve_voi(spec.variables, model)
        self._times: list[float] = []
        self._data: list[np.ndarray] = []

    def sample(self, step: int, state: np.ndarray) -> None:
        if step % self.istep == 0:
            data = state[self.voi].mean(axis=1, keepdims=True)  # (n_voi, 1)
            self._times.append(step * self.dt)
            self._data.append(data.copy())

    def result(self) -> tuple[np.ndarray, np.ndarray]:
        if not self._data:
            raise ValueError(
                f"GlobalAverageMonitor collected no samples. "
                f"duration must be >= period ({self.istep * self.dt} ms)."
            )
        return np.array(self._times), np.stack(self._data)


class BoldMonitor(Monitor):
    """Balloon-Windkessel BOLD signal, output at `tr` ms. TVB: Bold."""

    def configure(self, spec: MonitorSpec, model: ModelSpec, dt: float) -> None:
        self.dt = dt
        self.voi = _resolve_voi(spec.variables, model)
        self.tr_steps = max(1, round(spec.tr / dt))
        self._bw_p = dict(_BW_DEFAULTS)
        self._bw: np.ndarray | None = None
        self._times: list[float] = []
        self._data: list[np.ndarray] = []

    def sample(self, step: int, state: np.ndarray) -> None:
        neural = state[self.voi[0]]    # (n_nodes,) — BOLD driving variable
        if self._bw is None:
            self._bw = _bw_init(neural.shape[0])

        # dt in ms; BW uses seconds internally (divide by 1000)
        self._bw = _bw_step(self._bw, neural, self.dt * 1e-3, self._bw_p)

        if step > 0 and step % self.tr_steps == 0:
            bold = _bw_bold(self._bw, self._bw_p)
            self._times.append(step * self.dt)
            self._data.append(bold[np.newaxis, :].copy())   # (1, n_nodes)

    def result(self) -> tuple[np.ndarray, np.ndarray]:
        if not self._data:
            raise ValueError(
                f"BoldMonitor collected no samples. "
                f"duration must be >= tr ({self.tr_steps * self.dt} ms)."
            )
        return np.array(self._times), np.stack(self._data)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_KIND_MAP = {
    "raw":       RawMonitor,
    "subsample": SubSampleMonitor,
    "tavg":      TemporalAverageMonitor,
    "gavg":      GlobalAverageMonitor,
    "bold":      BoldMonitor,
}


def build_monitor(spec: MonitorSpec, model: ModelSpec, dt: float) -> Monitor:
    cls = _KIND_MAP.get(spec.kind)
    if cls is None:
        raise ValueError(f"Unknown monitor kind: {spec.kind!r}")
    # Validate period / tr before constructing (finding 6)
    if spec.kind in ("subsample", "tavg", "gavg"):
        if spec.period is None or spec.period <= 0:
            raise ValueError(
                f"MonitorSpec(kind={spec.kind!r}) requires period > 0; "
                f"got period={spec.period!r}."
            )
    if spec.kind == "bold":
        if spec.tr is None or spec.tr <= 0:
            raise ValueError(
                f"MonitorSpec(kind='bold') requires tr > 0; "
                f"got tr={spec.tr!r}."
            )
    mon = cls()
    mon.configure(spec, model, dt)
    return mon
