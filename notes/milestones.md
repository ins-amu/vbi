# Detailed Milestone Plan — `vbi/simulator/`

> **Scope note:** Unlike TVB hybrid there is no subnetwork mixing here. Every
> simulation runs one neural-mass model type across N homogeneous (or
> node-heterogeneous) nodes. The TVB hybrid ideas that *do* carry over are:
> expression-string dfuns, a separate coupling class, a ring-buffer history,
> pluggable integrators, and monitor objects.

---

## Primary Use Case: Simulation-Based Inference (SBI)

The simulator is a **generative model for SBI**. The dominant workload is not
single runs but **parameter sweeps** that produce training data:

1. Sample thousands of parameter sets `θ` (G, eta, J, noise_amp, …).
2. For each `θ`, simulate a brain network and extract summary statistics
   (FC matrix, FCD, power spectrum).
3. Feed `(θ, features)` pairs into an inference engine (SNPE, SNLE, ABC, …).

### Consequences for design

| Consequence | Design response |
|---|---|
| Single simulation is never the bottleneck; **the sweep is** | `Sweeper` is a first-class public class, not an afterthought |
| Storing raw time series for 10 000 simulations is prohibitive | Feature monitors compute statistics **inline** inside the sim loop; raw time series is optional |
| CPU and GPU parallelism must be automatic | Each backend implements its own batch strategy (prange / vmap / CUDA grid) |
| JAX backend enables gradient-based inference (SNPE with reparameterization) | `jax.vmap` + `jax.grad` through the full simulation loop |
| Feature extraction already exists in `vbi/feature_extraction/` | Feature monitors **reuse** those functions where possible; provide Numba/JAX/CUDA variants where needed |
| Parameter sets can be huge grids or Latin hypercube samples | `SweepSpec` accepts arbitrary arrays of parameter values, not just grids |

---

## API Contract — enforced across every milestone

**The goal is to minimize user learning time.** A user should be able to go
from `import` to a running simulation in ≤ 10 lines. Every design decision
should be measured against this.

### Canonical usage — single run (exploration, debugging)

```python
from vbi.simulator import Simulator
from vbi.simulator.models.mpr import mpr
from vbi.simulator.spec import SimulationSpec, IntegratorSpec, CouplingSpec, MonitorSpec

spec = SimulationSpec(
    model        = mpr,
    integrator   = IntegratorSpec(method="heun", dt=0.01),
    coupling     = CouplingSpec(kind="linear", a=1.0),
    monitors     = (MonitorSpec(kind="subsample", period=1.0),),
    weights      = W,          # (N, N) connectivity matrix
    tract_lengths= D,          # (N, N) delay distances in mm
)

sim = Simulator(spec, backend="numba")   # swap to "numpy", "cpp", "jax", "cuda"
result = sim.run(duration=5000.0)        # ms
t, state = result["subsample"]
```

### Canonical usage — parameter sweep (primary SBI use case)

```python
from vbi.simulator import Sweeper
from vbi.simulator.spec import SweepSpec
from vbi.feature_extraction.pipeline import FeaturePipeline

# 1. Define what to extract from each simulation
pipeline = FeaturePipeline(
    features=["fc", "fcd_ks"],   # registered feature names
    signal="tavg",               # monitor to consume
    t_cut=500.0,                 # ms burn-in to discard
    fcd_window_ms=1000.0,
    fcd_overlap=0.5,
)

# 2. Define the parameter sweep
sweep_spec = SweepSpec(
    params = {
        "G":   np.linspace(0.5, 5.0, 50),
        "eta": np.linspace(-5.5, -3.0, 50),
    },                     # 2500 runs (outer product); or (N,k) array for LHS
    pipeline=pipeline,
)

# 3. Run -- returns DataFrame ready for SBI
sweeper = Sweeper(spec, sweep_spec, backend="cuda")   # or "numba", "cpp", "jax"
df = sweeper.run_df(duration=5000.0)
# df.columns: ["G", "eta", "fc_0_1", "fc_0_2", ..., "fcd_ks"]
# df.shape:   (2500, 2 + n_fc_entries + 1)

# Or as (labels, values) arrays:
labels, values = sweeper.run(duration=5000.0)
# labels: list[str] -- parameter names + feature names
# values: shape (n_samples, len(labels))
```

### Rules derived from this contract

| Rule | Rationale |
|------|-----------|
| `Simulator` and `Sweeper` are the only two public entry points | Users never import backend-specific classes |
| `SimulationSpec` is the only place to set base parameters | No hidden config, env vars, or kwargs scattered across classes |
| `SweepSpec` holds the parameter grid / sample array | Decoupled from the model; can be reused across models |
| Changing backend = change one string in either class | Backend internals are invisible |
| All monitors use `result[kind]` → `(t, state)` (single) or `features[kind]` → array (sweep) | Consistent output convention |
| Feature monitors exist for FC, FCD, bold — raw is opt-in for sweeps | Sweeps don't store full time series by default |
| Optional/advanced parameters are keyword-only with sensible defaults | New users see a flat, minimal interface |
| Backend-specific imports fail with a clear `ImportError` + install hint | No silent fallbacks that change behaviour |
| `ModelSpec` can be written in 5–10 lines of Python | Adding a new model is self-contained |

These rules apply at every milestone. Any PR that breaks either canonical
usage example must be rejected or must update the example to remain ≤ 10 lines.

---

## How correctness is verified at each milestone

Every milestone ends with a reference comparison test. The hierarchy is:

```
TVB reference (gold)
    ↓  (M0)  NumPy backend   ← first gold standard in vbi
                ↓  (M1)  Numba CPU
                ↓  (M2)  C++
                ↓  (M3)  Numba-CUDA
                ↓  (M4)  JAX
```

- **Against TVB:** use `tvb-library` (dev dep only). Run same model/params in
  TVB simulator; compare time-series with `np.allclose(rtol=1e-3)` after a
  short warmup (discard first 500 ms).
- **Against NumPy baseline:** once NumPy is validated, all other backends must
  reproduce it to `rtol=1e-4` (deterministic) or match first two statistical
  moments (stochastic, 1 000 realizations).
- **TVB-only models (GHB, SL, RWW):** compare against current vbi numba
  implementation (frozen snapshot test).

---

## M0 — Spec Layer + NumPy Baseline  ✅ COMPLETE

**Goal:** Define the data model, implement a pure-NumPy simulator, validate
against TVB for MPR. Everything that follows builds on this layer.

### Status (implemented)

| Component | File | State |
|-----------|------|-------|
| Spec dataclasses | `vbi/simulator/spec/` | ✅ done |
| MPR model spec | `vbi/simulator/models/mpr.py` | ✅ done — TVB-consistent `cvar=(r,V)`, `cr`/`cv` params |
| Ring buffer | `vbi/simulator/backend/numpy_/history.py` | ✅ done |
| Coupling (linear, sigmoidal) | `vbi/simulator/backend/numpy_/coupling.py` | ✅ done |
| Integrators (Euler/Heun det+stoch) | `vbi/simulator/backend/numpy_/integrators.py` | ✅ done |
| Monitors (Raw/SubSample/TAvg/GAvg/Bold) | `vbi/simulator/backend/numpy_/monitors.py` | ✅ done |
| dfun codegen (exec-based) | `vbi/simulator/backend/numpy_/simulator.py` | ✅ done |
| NumPy simulator loop | `vbi/simulator/backend/numpy_/simulator.py` | ✅ done |
| NumPy sweeper | `vbi/simulator/backend/numpy_/sweeper.py` | ✅ done |
| Public API (`Simulator`, `Sweeper`) | `vbi/simulator/api.py` | ✅ done |
| FeaturePipeline (NumPy) | `vbi/feature_extraction/pipeline.py` | ✅ done |
| Validation tests | `vbi/tests/validation/` | ✅ 23 tests passing |
| TVB trajectory comparison | — | ⏳ deferred (needs `tvb-library` as dev dep) |

**Note:** Backend-specific feature extraction variants (`get_fc_nb`, `get_fc_jax`, etc.)
are deferred to their respective backend milestones (M1, M3, M4).

### Files to create

```
vbi/simulator/__init__.py
vbi/simulator/spec/__init__.py
vbi/simulator/spec/model.py
vbi/simulator/spec/integrator.py
vbi/simulator/spec/coupling.py
vbi/simulator/spec/monitor.py        # raw, subsample, tavg, gavg, bold
vbi/simulator/spec/simulation.py
vbi/simulator/spec/sweep.py          # SweepSpec — NEW

vbi/simulator/models/__init__.py
vbi/simulator/models/mpr.py

vbi/simulator/backend/__init__.py
vbi/simulator/backend/base.py
vbi/simulator/backend/numpy_/__init__.py
vbi/simulator/backend/numpy_/history.py
vbi/simulator/backend/numpy_/coupling.py
vbi/simulator/backend/numpy_/integrators.py
vbi/simulator/backend/numpy_/monitors.py
vbi/simulator/backend/numpy_/simulator.py
vbi/simulator/backend/numpy_/sweeper.py  # NEW — loop over param sets

vbi/simulator/api.py                 # Simulator + Sweeper public classes

vbi/tests/validation/__init__.py
vbi/tests/validation/conftest.py
vbi/tests/validation/test_mpr_numpy.py
vbi/tests/validation/test_sweep_numpy.py  # NEW
```

---

### M0.1 — Spec dataclasses (including sweep and feature monitors)

**`vbi/simulator/spec/model.py`**

```python
@dataclass(frozen=True)
class StateVar:
    name: str
    default_init: float = 0.0
    noise: bool = False
    lower_bound: float | None = None
    upper_bound: float | None = None

@dataclass(frozen=True)
class Parameter:
    name: str
    default: float | np.ndarray
    description: str = ""

@dataclass(frozen=True)
class ModelSpec:
    name: str
    state_variables: tuple[StateVar, ...]
    parameters: tuple[Parameter, ...]
    cvar: tuple[str, ...]          # names of coupling-variable state vars
    dfun_str: dict[str, str]       # {"dr": "expr", "dV": "expr"} — bare math
    noise_variables: tuple[str, ...] = ()
    reference: str = ""

    # Derived convenience
    @property
    def sv_names(self) -> tuple[str, ...]: ...

    @property
    def n_sv(self) -> int: ...

    @property
    def cvar_indices(self) -> tuple[int, ...]:
        # index of each cvar name in sv_names
```

**`vbi/simulator/spec/integrator.py`**

```python
@dataclass(frozen=True)
class IntegratorSpec:
    method: Literal["euler", "heun"]  # default "heun"
    dt: float
    stochastic: bool = False
    noise_nsig: np.ndarray | None = None  # shape (n_noise_vars,)
```

**`vbi/simulator/spec/coupling.py`**

```python
@dataclass(frozen=True)
class CouplingSpec:
    kind: Literal["linear", "sigmoidal", "kuramoto"]  # default "linear"
    a: float = 1.0
    b: float = 0.0
    # sigmoidal extras
    midpoint: float = 0.0
    sigma: float = 1.0
```

**`vbi/simulator/spec/monitor.py`**

Monitors follow the TVB pattern — they record simulator state during the run.
Feature extraction is a **separate post-processing step** applied to monitor
output (see M0.10).

```python
@dataclass(frozen=True)
class MonitorSpec:
    kind: Literal["raw", "subsample", "tavg", "gavg", "bold"]
    period: float | None = None      # ms — ignored for "raw" (uses dt)
    variables: tuple[str, ...] = ()  # state-var names to record; empty → all VOIs
    # bold-only
    tr: float = 2000.0               # BOLD repetition time in ms
```

| Monitor | TVB equivalent | What it records | Output shape |
|---------|---------------|----------------|--------------|
| `raw` | `Raw` | Every integration step, all VOIs | `(n_steps, n_voi, n_nodes)` |
| `subsample` | `SubSample` | Every `period` ms (decimation) | `(n_steps//k, n_voi, n_nodes)` |
| `tavg` | `TemporalAverage` | Time-average of VOIs over `period` ms windows | `(n_windows, n_voi, n_nodes)` |
| `gavg` | `GlobalAverage` | Spatial mean of VOIs, every `period` ms | `(n_steps//k, n_voi, 1)` |
| `bold` | `Bold` | Balloon-Windkessel haemodynamics driven by first VOI | `(n_bold_steps, n_nodes)` |

- **`tavg`** accumulates states into a rolling stock array and flushes the
  time-average every `istep = round(period/dt)` steps — identical to TVB
  `TemporalAverage`. This is the most useful monitor for SBI because it acts
  as a low-pass filter before feature extraction.
- **`bold`** integrates its own Balloon-Windkessel ODE at every neural step
  and outputs at `tr` (repetition time). The neural input is the `tavg` of
  the BOLD-driving variable (typically `r` or `E`) over one TR window.
- **`subsample`** and **`raw`** are interchangeable for SBI; prefer `subsample`
  or `tavg` to reduce memory.

**`vbi/simulator/spec/sweep.py`** — NEW

```python
@dataclass
class SweepSpec:
    """
    Describes a parameter sweep for SBI training data generation.

    params:
        Either a dict of {param_name: 1-D array of values} for grid/LHS sweeps,
        or a 2-D array of shape (n_samples, n_params) with param_names provided.

    Examples
    --------
    # Grid (outer product — 50x50 = 2500 runs):
    SweepSpec(params={"G": np.linspace(0.5,5,50), "eta": np.linspace(-5.5,-3,50)})

    # Latin hypercube / arbitrary samples (5000 runs, 3 params):
    SweepSpec(params=theta_array, param_names=["G","eta","noise_amp"])
    """
    params: dict[str, np.ndarray] | np.ndarray
    param_names: tuple[str, ...] | None = None  # required if params is ndarray
    t_cut: float = 500.0    # ms to discard as burn-in before recording

    @property
    def param_sets(self) -> np.ndarray:
        # Returns (n_samples, n_params) float64 array regardless of input form
        if isinstance(self.params, np.ndarray):
            return self.params
        names = list(self.params.keys())
        grids = np.meshgrid(*[self.params[n] for n in names], indexing="ij")
        return np.stack([g.ravel() for g in grids], axis=1)

    @property
    def n_samples(self) -> int:
        return self.param_sets.shape[0]
```

**`vbi/simulator/spec/simulation.py`**

```python
@dataclass(frozen=True)
class SimulationSpec:
    model: ModelSpec
    integrator: IntegratorSpec
    coupling: CouplingSpec
    monitors: tuple[MonitorSpec, ...]
    weights: np.ndarray           # (n_nodes, n_nodes) connectivity matrix
    tract_lengths: np.ndarray     # (n_nodes, n_nodes) delay distances in mm
    speed: float = 4.0            # conduction velocity mm/ms
    node_params: dict[str, np.ndarray] = field(default_factory=dict)
        # per-node parameter overrides, e.g. {"eta": np.array([...])}

    def delay_steps(self, dt: float) -> np.ndarray:
        # returns (n_nodes, n_nodes) int array of delay in steps
        return np.round(self.tract_lengths / (self.speed * dt)).astype(int)

    def horizon(self, dt: float) -> int:
        return int(self.delay_steps(dt).max()) + 1

    def cache_key(self) -> str:
        # SHA256 of payload dict (same pattern as tvb hybrid spec.py)
```

---

### M0.2 — MPR model spec

**`vbi/simulator/models/mpr.py`**

```python
from vbi.simulator.spec.model import ModelSpec, StateVar, Parameter

mpr = ModelSpec(
    name="MontbrioPopulationRate",
    state_variables=(
        StateVar("r", default_init=0.0, noise=True, lower_bound=0.0),
        StateVar("V", default_init=-2.0, noise=True),
    ),
    parameters=(
        Parameter("tau",    1.0,    "time constant"),
        Parameter("I",      0.0,    "external input"),
        Parameter("Delta",  0.7,    "heterogeneity"),
        Parameter("J",      14.5,   "synaptic weight"),
        Parameter("eta",   -4.6,    "excitability"),
        Parameter("G",      2.0,    "global coupling"),
    ),
    cvar=("r",),
    dfun_str={
        "r": "(Delta / (pi * tau) + 2 * r * V) / tau",
        "V": "(V**2 + eta + J * tau * r + I - (pi * tau * r)**2 + c) / tau",
    },
    noise_variables=("r", "V"),
    reference="Montbrio et al. 2015 PLoS Comput Biol",
)
```

Notes on `dfun_str`:
- `c` is the **pre-computed coupling input** passed in from outside the dfun.
  It is the only external variable besides state vars and parameters.
- Uses only: `+`, `-`, `*`, `/`, `**`, `exp`, `log`, `sin`, `cos`, `tanh`,
  `sqrt`, `abs`, `pi`.
- No `np.` prefix — the code generator injects the math namespace.
- Node-heterogeneous parameters (e.g. `eta` as a per-node array) are supported;
  the generator emits `eta[node]` for C++/Numba and lets NumPy broadcast.

---

### M0.3 — Ring buffer / history

**`vbi/simulator/backend/numpy_/history.py`**

The ring buffer is the hardest shared abstraction. Get it right here; all other
backends will port or replace this.

```python
class History:
    """
    Circular delay buffer.
    Shape: (horizon, n_cvar, n_nodes)
    Indexed by: buf[step % horizon, cvar_idx, node]
    """
    def __init__(self, horizon: int, n_cvar: int, n_nodes: int, dtype=np.float64):
        self.horizon = horizon
        self.buf = np.zeros((horizon, n_cvar, n_nodes), dtype=dtype)
        self._step = 0

    def write(self, cvar_state: np.ndarray) -> None:
        # cvar_state: shape (n_cvar, n_nodes)
        self.buf[self._step % self.horizon] = cvar_state
        self._step += 1

    def read_delayed(self, delay_steps: np.ndarray) -> np.ndarray:
        """
        delay_steps: (n_nodes, n_nodes) int array
        Returns: (n_cvar, n_nodes) coupling input for each target node
        Uses: buf[(step - 1 - delay[j, i]) % horizon, cvar, j]
        """
        step = self._step - 1
        out = np.zeros((self.buf.shape[1], self.buf.shape[2]))
        for cvar in range(self.buf.shape[1]):
            for target in range(self.buf.shape[2]):
                for src in range(self.buf.shape[2]):
                    d = delay_steps[src, target]
                    idx = (step - d) % self.horizon
                    out[cvar, target] += self.buf[idx, cvar, src]
                    # weighted sum applied in coupling layer, not here
        return out

    def initialize(self, init_state: np.ndarray) -> None:
        # Fill entire buffer with init_state (n_cvar, n_nodes)
        self.buf[:] = init_state[np.newaxis, :, :]
```

> **Note:** The vectorized version replaces the triple loop above. The loop
> form is written first for clarity; replace with advanced indexing before M1.
> See TVB `history.py` `BaseHistory` for the index-table trick.

---

### M0.4 — Coupling

**`vbi/simulator/backend/numpy_/coupling.py`**

```python
class LinearCoupling:
    """
    c_i = G * a * sum_j(w_ij * x_j(t - tau_ij)) + b
    Returned shape: (n_cvar, n_nodes)
    """
    def __init__(self, spec: CouplingSpec, weights: np.ndarray, G: float):
        self.a = spec.a
        self.b = spec.b
        self.weights = weights   # (n_nodes, n_nodes)
        self.G = G

    def compute(self,
                delayed_state: np.ndarray,  # (n_cvar, n_nodes) from History
                ) -> np.ndarray:
        # delayed_state already has delay applied; this applies weights
        # c_i = G * a * weights @ delayed_state + b
        return self.G * self.a * (self.weights @ delayed_state.T).T + self.b


class SigmoidalCoupling:
    def compute(self, delayed_state, weights, G): ...
```

> `G` lives in `SimulationSpec.node_params` or as a top-level parameter if
> scalar. For MPR, `G` enters the dfun as the pre-multiplier; it is baked into
> `LinearCoupling` so the dfun only sees scalar `c`.

---

### M0.5 — Integrators

**`vbi/simulator/backend/numpy_/integrators.py`**

```python
class EulerDeterministic:
    def step(self,
             state: np.ndarray,    # (n_sv, n_nodes)
             dfun_fn,              # callable(state, coupling) → (n_sv, n_nodes)
             coupling: np.ndarray, # (n_cvar, n_nodes)
             dt: float,
             ) -> np.ndarray:
        return state + dt * dfun_fn(state, coupling)


class HeunDeterministic:
    def step(self, state, dfun_fn, coupling, dt) -> np.ndarray:
        k1 = dfun_fn(state, coupling)
        k2 = dfun_fn(state + dt * k1, coupling)
        return state + 0.5 * dt * (k1 + k2)


class EulerStochastic:
    # Euler-Maruyama
    def step(self, state, dfun_fn, coupling, dt, noise_nsig, rng) -> np.ndarray:
        dW = rng.normal(0, np.sqrt(dt), state.shape) * noise_nsig[:, np.newaxis]
        return state + dt * dfun_fn(state, coupling) + dW


class HeunStochastic:
    # Stochastic Heun (Stratonovich midpoint)
    def step(self, state, dfun_fn, coupling, dt, noise_nsig, rng) -> np.ndarray:
        dW = rng.normal(0, np.sqrt(dt), state.shape) * noise_nsig[:, np.newaxis]
        k1 = dfun_fn(state, coupling)
        x_pred = state + dt * k1 + dW
        k2 = dfun_fn(x_pred, coupling)
        return state + 0.5 * dt * (k1 + k2) + dW
```

> Same 4 integrators are exposed in every backend. `noise_nsig` shape is
> `(n_noise_vars,)` — one entry per noise-enabled state variable.

---

### M0.6 — dfun executor (NumPy layer)

**`vbi/simulator/backend/numpy_/simulator.py`** — dfun eval function

The NumPy backend builds a `dfun_fn` from `ModelSpec.dfun_str` at
`Simulator.build()` time via Python's `compile()` + `eval()`. This is safe
(not user-supplied code) and gives debuggable tracebacks:

```python
def _build_numpy_dfun(spec: ModelSpec) -> Callable:
    """
    Compiles dfun_str expressions into a vectorized numpy function.
    Returns: fn(state, coupling_vec, params_dict) -> np.ndarray (n_sv, n_nodes)
    """
    sv = spec.sv_names
    param_names = tuple(p.name for p in spec.parameters)
    cvar_indices = spec.cvar_indices   # which rows of state are coupling vars

    src_lines = [
        "import numpy as np",
        "from numpy import pi, exp, log, sin, cos, tanh, sqrt, abs",
        "def _dfun(state, c, params):",
    ]
    # Unpack state rows
    for i, name in enumerate(sv):
        src_lines.append(f"    {name} = state[{i}]")
    # Unpack params
    for name in param_names:
        src_lines.append(f"    {name} = params['{name}']")
    # Build output
    src_lines.append(f"    out = np.empty_like(state)")
    for i, name in enumerate(sv):
        src_lines.append(f"    out[{i}] = {spec.dfun_str[name]}")
    src_lines.append("    return out")

    src = "\n".join(src_lines)
    globs = {}
    exec(compile(src, "<dfun>", "exec"), globs)
    return globs["_dfun"]
```

> `exec` is acceptable here because the source comes from our own spec files,
> not from user input. The function is compiled once at `build()`, not per
> step.

---

### M0.7 — Monitors

**`vbi/simulator/backend/numpy_/monitors.py`**

All monitors share the same abstract interface. The simulator calls `sample()`
at every integration step; each monitor decides internally whether this step
should be recorded.

```python
class Monitor(ABC):
    """Abstract base — mirrors TVB Monitor interface."""
    istep: int      # record every istep integration steps
    dt: float
    voi: np.ndarray # indices into state array to record

    def configure(self, spec: MonitorSpec, sv_names: tuple, dt: float) -> None:
        self.dt = dt
        self.voi = _resolve_var_indices(spec.variables, sv_names)
        self.istep = max(1, round(spec.period / dt)) if spec.period else 1

    @abstractmethod
    def sample(self, step: int, state: np.ndarray) -> tuple[float, np.ndarray] | None:
        """Called every integration step. Returns (t, data) or None."""

    def result(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (times, data) after simulation."""
        return np.array(self._times), np.stack(self._data)


class RawMonitor(Monitor):
    """Records every integration step. TVB: Raw."""
    def configure(self, spec, sv_names, dt):
        super().configure(spec, sv_names, dt)
        self.istep = 1
        self._times, self._data = [], []

    def sample(self, step, state):
        t = step * self.dt
        self._times.append(t)
        self._data.append(state[self.voi].copy())
        return t, state[self.voi]


class SubSampleMonitor(Monitor):
    """Decimates: records every period ms. TVB: SubSample."""
    def configure(self, spec, sv_names, dt):
        super().configure(spec, sv_names, dt)
        self._times, self._data = [], []

    def sample(self, step, state):
        if step % self.istep == 0:
            t = step * self.dt
            self._times.append(t)
            self._data.append(state[self.voi].copy())
            return t, state[self.voi]
        return None


class TemporalAverageMonitor(Monitor):
    """
    Accumulates states into a rolling stock; flushes the time-average
    every istep steps. TVB: TemporalAverage.

    Useful for SBI: acts as a low-pass filter before feature extraction.
    """
    def configure(self, spec, sv_names, dt):
        super().configure(spec, sv_names, dt)
        n_nodes = None  # set in first sample call
        self._stock = None
        self._stock_idx = 0
        self._times, self._data = [], []

    def sample(self, step, state):
        if self._stock is None:
            n_nodes = state.shape[1]
            self._stock = np.zeros((self.istep, len(self.voi), n_nodes))

        self._stock[self._stock_idx] = state[self.voi]
        self._stock_idx += 1

        if self._stock_idx == self.istep:
            avg = self._stock.mean(axis=0)
            t = (step - self.istep / 2.0) * self.dt
            self._times.append(t)
            self._data.append(avg.copy())
            self._stock_idx = 0
            return t, avg
        return None


class GlobalAverageMonitor(Monitor):
    """
    Spatial mean of VOIs across all nodes, every period ms.
    TVB: GlobalAverage. Useful for quick sanity-check of mean activity.
    """
    def configure(self, spec, sv_names, dt):
        super().configure(spec, sv_names, dt)
        self._times, self._data = [], []

    def sample(self, step, state):
        if step % self.istep == 0:
            t = step * self.dt
            data = state[self.voi].mean(axis=1, keepdims=True)  # (n_voi, 1)
            self._times.append(t)
            self._data.append(data.copy())
            return t, data
        return None


class BoldMonitor(Monitor):
    """
    Balloon-Windkessel haemodynamic response. TVB: Bold.

    Integrates the BW ODE at every neural step using the BOLD-driving variable
    (first VOI). Outputs at tr ms (repetition time).
    The neural input is the mean of the driving variable over one TR window.
    """
    def configure(self, spec, sv_names, dt):
        self.dt = dt
        self.voi = _resolve_var_indices(spec.variables, sv_names)
        self.tr = spec.tr                          # ms
        self.tr_steps = round(spec.tr / dt)
        self._bw_state = None    # Balloon-Windkessel state (4, n_nodes)
        self._neural_avg = None  # running average of driving variable
        self._avg_count = 0
        self._times, self._data = [], []

    def sample(self, step, state):
        neural = state[self.voi[0]]   # (n_nodes,) — BOLD-driving variable

        # Accumulate for TR-window average (input to BW model)
        if self._neural_avg is None:
            self._neural_avg = np.zeros_like(neural)
            self._bw_state = _init_bw_state(neural.shape[0])

        self._neural_avg += neural
        self._avg_count += 1

        # BW ODE step every neural step
        self._bw_state = _bw_euler_step(self._bw_state, neural, self.dt)

        if step % self.tr_steps == 0 and step > 0:
            t = step * self.dt
            bold = _bw_bold_signal(self._bw_state)   # (n_nodes,)
            self._times.append(t)
            self._data.append(bold.copy())
            self._neural_avg[:] = 0.0
            self._avg_count = 0
            return t, bold
        return None
```

The Balloon-Windkessel ODEs (`_bw_euler_step`, `_bw_bold_signal`,
`_init_bw_state`) are ported from `vbi/models/numba/bold.py` and
`vbi/models/cpp/_src/bold.hpp` — no reimplementation needed.

---

### M0.8 — Simulator loop

**`vbi/simulator/backend/numpy_/simulator.py`** — main class

```python
class NumpySimulator:
    def build(self, spec: SimulationSpec) -> None:
        self.spec = spec
        self.dt = spec.integrator.dt
        self.n_nodes = spec.weights.shape[0]
        self.n_sv = spec.model.n_sv
        self.delay_steps = spec.delay_steps(self.dt)
        self.horizon = spec.horizon(self.dt)

        # Build components
        self._dfun = _build_numpy_dfun(spec.model)
        self._history = History(self.horizon, len(spec.model.cvar), self.n_nodes)
        self._coupling = _build_coupling(spec.coupling, spec.weights, spec)
        self._integrator = _build_integrator(spec.integrator)
        self._monitors = [_build_monitor(m, spec.model.sv_names, self.dt)
                          for m in spec.monitors]

        # Resolve per-node params
        self._params = _build_params(spec.model, spec.node_params)

        # Initial state: (n_sv, n_nodes)
        self._state = _build_initial_state(spec.model, self.n_nodes)
        self._history.initialize(self._state[list(spec.model.cvar_indices)])

    def run(self, duration: float) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        n_steps = round(duration / self.dt)
        rng = np.random.default_rng(self.spec.integrator.noise_seed
                                    if hasattr(self.spec.integrator, "noise_seed")
                                    else 42)

        for step in range(n_steps):
            t = step * self.dt

            # 1. Read delayed coupling input
            delayed = self._history.read_delayed(self.delay_steps)
            coupling = self._coupling.compute(delayed)  # (n_cvar, n_nodes)

            # 2. dfun wrapper: passes coupling as scalar c (first cvar only)
            def dfun_fn(state, coup):
                return self._dfun(state, coup[0], self._params)

            # 3. Integrate one step
            if spec.integrator.stochastic:
                self._state = self._integrator.step(
                    self._state, dfun_fn, coupling, self.dt,
                    self.spec.integrator.noise_nsig, rng)
            else:
                self._state = self._integrator.step(
                    self._state, dfun_fn, coupling, self.dt)

            # 4. Apply state bounds (clamp)
            self._apply_bounds()

            # 5. Write new cvar state to history
            self._history.write(self._state[list(self.spec.model.cvar_indices)])

            # 6. Record monitors
            for mon in self._monitors:
                mon.record(t, self._state, step)

        return {mon.spec.kind: mon.result() for mon in self._monitors}
```

---

### M0.9 — Public API

**`vbi/simulator/api.py`**

```python
class Simulator:
    """Single-run interface — exploration, debugging, TVB validation."""
    def __init__(self, spec: SimulationSpec, backend: str = "numpy"):
        self._backend = _load_backend(backend)(spec)

    def run(self, duration: float) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        return self._backend.run(duration)


class Sweeper:
    """
    Primary SBI interface — generates training data via parameter sweeps.

    Parameters
    ----------
    spec : SimulationSpec
        Base simulation (model, integrator, coupling, monitors, connectivity).
    sweep_spec : SweepSpec
        Which parameters to vary and over what values.
    backend : str
        "numpy" (reference), "numba" (CPU parallel), "cpp", "cuda", "jax".

    Returns
    -------
    dict
        Keys match monitor kinds. Values are arrays of shape
        (n_param_sets, ...) — one result per parameter combination.

        "fc"     → (n_param_sets, n_nodes, n_nodes)
        "fcd_ks" → (n_param_sets,)
        "bold"   → (n_param_sets, n_nodes, n_bold_steps)
        "raw"    → only returned if explicitly requested; large!
    """
    def __init__(self, spec: SimulationSpec, sweep_spec: SweepSpec,
                 backend: str = "numba"):
        self._impl = _load_sweep_backend(backend)(spec, sweep_spec)

    def run(self, duration: float) -> dict[str, np.ndarray]:
        return self._impl.run(duration)
```

### M0.10 — Feature extraction pipeline (separate from monitors)

**Data flow:**

```
Simulator / Sweeper
    |__ Monitors (TVB-style)
           tavg  -> (t, state)   neural signal (low-pass averaged)
           bold  -> (t, bold)    haemodynamic signal (BOLD fMRI)
           raw   -> (t, state)   full time series (large)
                    |
                    v  FeaturePipeline (post-processing)
         vbi.feature_extraction
                    |
                    |-- fc             (n_nodes, n_nodes)  Pearson correlation
                    |-- fcd            (n_win, n_win)       sliding-window FC
                    |-- fcd_ks         scalar               KS-stat vs empirical FCD
                    |-- power_spectrum (n_freqs, n_nodes)
                    |-- band_power     (n_bands, n_nodes)
                    |-- mean           (n_nodes,)
                    |-- std            (n_nodes,)
                    |__ ... (anything in vbi/feature_extraction/)
```

Monitors are **not** feature extractors. `FeaturePipeline` consumes monitor
output and produces labelled feature vectors — the direct input to SBI.

**`vbi/feature_extraction/pipeline.py`** — NEW

```python
class FeaturePipeline:
    """
    Applies a sequence of named feature functions to monitor output.
    Returns (labels, values) lists or a pandas DataFrame.

    Parameters
    ----------
    features : list[str]
        Names of features to compute. Keys from the feature registry.
    t_cut : float
        Milliseconds of burn-in to discard before extracting features.
    signal : str
        Which monitor to consume: "tavg", "bold", "subsample", "raw".
    **kwargs
        Per-feature configuration passed through to each feature function:
        fcd_window_ms, fcd_overlap, freq_band, fc_function, ...

    Examples
    --------
    pipeline = FeaturePipeline(
        features=["fc", "fcd_ks", "mean"],
        t_cut=500.0,
        signal="tavg",
        fcd_window_ms=1000.0,
        fcd_overlap=0.5,
    )
    labels, values = pipeline.extract(monitor_result)
    df = pipeline.extract_df(monitor_result)
    """

    def __init__(self, features: list[str], t_cut: float = 500.0,
                 signal: str = "tavg", **kwargs):
        self.features = features
        self.t_cut = t_cut
        self.signal = signal
        self.kwargs = kwargs

    def extract(self, monitor_result: dict) -> tuple[list[str], np.ndarray]:
        """
        Parameters
        ----------
        monitor_result : dict
            Output of Simulator.run() -- keys are monitor kinds.

        Returns
        -------
        labels : list[str]
            One label per feature value, e.g. "fc_0_1", "fc_0_2", "fcd_ks".
        values : np.ndarray
            1-D float64 array, same order as labels.
        """
        t, ts = monitor_result[self.signal]
        t_cut_idx = np.searchsorted(t, self.t_cut)
        ts_cut = ts[t_cut_idx:]       # (n_steps, n_voi, n_nodes)
        signal = ts_cut[:, 0, :]      # first VOI by default; (n_steps, n_nodes)

        labels, values = [], []
        for feat in self.features:
            fn = _FEATURE_REGISTRY[feat]
            lab, val = fn(signal, **self.kwargs)
            labels.extend(lab)
            values.extend(val)

        return labels, np.array(values, dtype=np.float64)

    def extract_df(self, monitor_result: dict) -> "pd.DataFrame":
        labels, values = self.extract(monitor_result)
        return pd.DataFrame([values], columns=labels)
```

**Feature registry** in `vbi/feature_extraction/pipeline.py`:

Each entry wraps an existing `features_utils.py` function with signature
`fn(ts, **kwargs) -> (list[str], list[float])`:

```python
_FEATURE_REGISTRY = {
    "fc":             _wrap_fc,          # get_fc  -> upper-triangle values
    "fcd":            _wrap_fcd_matrix,  # get_fcd -> upper-triangle of FCD matrix
    "fcd_ks":         _wrap_fcd_ks,      # get_fcd -> KS scalar
    "mean":           _wrap_mean,        # per-node temporal mean
    "std":            _wrap_std,         # per-node temporal std
    "power_spectrum": _wrap_psd,         # calc_fft -> band-averaged
    "band_power":     _wrap_band_power,  # band-pass filter + RMS per node
    # extend by adding entries here; no simulator changes needed
}
```

Adding a new feature = add one entry to `_FEATURE_REGISTRY`. No changes to
`Simulator`, `Sweeper`, or any monitor class.

**Backend variants** — *deferred to each backend milestone.*

The `FeaturePipeline` always works with NumPy functions from `features_utils.py`
at the Python level. JIT-compiled variants (`get_fc_nb`, `get_fcd_nb`, etc.) that
run inside the sweep inner loop are added when each backend is implemented:

- M1 (Numba CPU): `features_utils_nb.py` — `@njit` variants
- M2 (C++): FC/FCD computed in C++ sweep loop, result returned as ndarray
- M3 (CUDA): `features_utils_cuda.py` — `@cuda.jit` device functions
- M4 (JAX): extend `features_utils_jax.py` — already partially done

Until then, the NumPy sweeper calls the existing `features_utils.py` functions
directly at Python level — correct but not maximally fast.

**`SweepSpec` -- updated with pipeline:**

```python
@dataclass
class SweepSpec:
    params: dict[str, np.ndarray] | np.ndarray
    param_names: tuple[str, ...] | None = None
    t_cut: float = 500.0            # ms burn-in (passed to pipeline)
    pipeline: FeaturePipeline | None = None
    # If pipeline is None, sweeper returns raw monitor output per run.
    # If pipeline is set, sweeper.run() returns (labels, values_array)
    # and sweeper.run_df() returns a DataFrame.

    @property
    def n_samples(self) -> int: ...
    @property
    def param_sets(self) -> np.ndarray: ...
```

**Sweeper output format when pipeline is set:**

```python
pipeline = FeaturePipeline(features=["fc", "fcd_ks"], signal="tavg", t_cut=500.0)
sweep_spec = SweepSpec(params={"G": np.linspace(1,4,50)}, pipeline=pipeline)

sweeper = Sweeper(spec, sweep_spec, backend="numba")

# Option A: (labels, values) lists
labels, values = sweeper.run(duration=5000.0)
# labels: ["G", "fc_0_1", "fc_0_2", ..., "fcd_ks"]
# values: shape (n_samples, n_labels)

# Option B: DataFrame -- direct input to SBI
df = sweeper.run_df(duration=5000.0)
# df.columns = ["G", "fc_0_1", ..., "fcd_ks"]
# df.shape   = (n_samples, 1_param + n_features)
```

The DataFrame is the direct input to `sbi.utils.simulate_for_sbi` or any
inference engine that expects a table of (theta, x) pairs.

### M0.11 — NumPy Sweeper (reference implementation)

**`vbi/simulator/backend/numpy_/sweeper.py`**

```python
class NumpySweeper:
    """Reference sweep — sequential Python loop. Not fast; used for validation."""

    def __init__(self, spec: SimulationSpec, sweep_spec: SweepSpec):
        self.spec = spec
        self.sweep = sweep_spec

    def run(self, duration: float) -> dict[str, np.ndarray]:
        from vbi.feature_extraction.features_utils import get_fc, get_fcd

        param_sets  = self.sweep.param_sets    # (n_samples, n_params)
        param_names = self.sweep.param_names or list(self.sweep.params.keys())
        n           = param_sets.shape[0]

        # Pre-allocate output arrays — no appending in hot path
        outputs = _preallocate_outputs(self.sweep.features, n, self.spec)

        for i, theta in enumerate(param_sets):
            patched = _patch_params(self.spec, param_names, theta)
            sim = NumpySimulator()
            sim.build(patched)
            result = sim.run(duration)

            # Feature extraction applied per-run (not stored in bulk)
            t_cut_idx = int(self.sweep.t_cut / patched.integrator.dt)
            if "subsample" in result:
                _, ts = result["subsample"]
                ts_cut = ts[t_cut_idx:]
                if "fc" in self.sweep.features:
                    outputs["fc"][i] = get_fc(ts_cut[:, 0, :])
                if "fcd" in self.sweep.features or "fcd_ks" in self.sweep.features:
                    fcd_mat, ks = get_fcd(ts_cut[:, 0, :], ...)
                    if "fcd" in self.sweep.features:
                        outputs["fcd"][i] = fcd_mat
                    if "fcd_ks" in self.sweep.features:
                        outputs["fcd_ks"][i] = ks
            if "bold" in result and "bold" in self.sweep.features:
                _, bold = result["bold"]
                outputs["bold"][i] = bold[t_cut_idx:]

        return outputs
```

`_patch_params` returns a modified `SimulationSpec` with swept params
overridden (no array copies — only scalar fields differ).

`_preallocate_outputs` allocates result arrays before the loop based on
`sweep_spec.features` — no Python list appending in the hot path.

---

### M0 validation test

**`vbi/tests/validation/test_mpr_numpy.py`**

```python
@pytest.mark.skipif(not TVB_AVAILABLE, reason="tvb-library not installed")
def test_mpr_numpy_vs_tvb(n_nodes=2, duration=2000.0, dt=0.01, rtol=1e-3):
    # 1. Build TVB simulator with MPR, linear coupling, Heun, no noise
    tvb_sim = _build_tvb_mpr(n_nodes, dt)
    tvb_t, tvb_r, tvb_v = _run_tvb(tvb_sim, duration)

    # 2. Build vbi simulator with same params
    spec = SimulationSpec(model=mpr, integrator=IntegratorSpec("heun", dt),
                          coupling=CouplingSpec("linear", a=1.0),
                          monitors=(MonitorSpec("raw"),),
                          weights=tvb_sim.connectivity.weights,
                          tract_lengths=tvb_sim.connectivity.tract_lengths)
    sim = Simulator(spec, backend="numpy")
    result = sim.run(duration)
    vbi_t, vbi_state = result["raw"]

    # 3. Compare after 500 ms burn-in
    burn = int(500.0 / dt)
    assert np.allclose(vbi_state[burn:, 0], tvb_r[burn:], rtol=rtol), \
        "r (firing rate) diverges from TVB reference"
    assert np.allclose(vbi_state[burn:, 1], tvb_v[burn:], rtol=rtol), \
        "V (membrane potential) diverges from TVB reference"
```

**`vbi/tests/validation/test_sweep_numpy.py`**

```python
def test_sweep_fc_shape(n_nodes=10, n_G=5, n_eta=5):
    sweep_spec = SweepSpec(params={"G": np.linspace(1,4,n_G),
                                   "eta": np.linspace(-5,-3,n_eta)})
    sweeper = Sweeper(spec, sweep_spec, backend="numpy")
    features = sweeper.run(duration=3000.0)
    assert features["fc"].shape == (n_G * n_eta, n_nodes, n_nodes)

def test_sweep_fc_matches_single_run(n_nodes=10):
    # Run sweep with one param set; result must match single Simulator run
    theta = {"G": np.array([2.0]), "eta": np.array([-4.6])}
    sweep_spec = SweepSpec(params=theta)
    sweep_fc = Sweeper(spec, sweep_spec, backend="numpy").run(3000.0)["fc"][0]
    single_result = Simulator(spec, backend="numpy").run(3000.0)
    single_fc = _compute_fc(single_result["subsample"][1], t_cut=500.0)
    np.testing.assert_allclose(sweep_fc, single_fc, rtol=1e-6)
```

**M0 done when:** single-run TVB validation passes for MPR deterministic +
stochastic (2 and 80 nodes, with delays), and sweep FC shape + consistency
tests pass.

---

## M1 — Numba CPU Backend  ← NEXT

**Goal:** JIT-compile the dfun and the inner simulation loop using Numba.
The NumPy baseline is the reference. No new model specs needed — reuse M0 specs.

**Prerequisite:** M0 complete. ✅

**Feature extraction note:** Add `vbi/feature_extraction/features_utils_nb.py`
with `@njit` variants of `get_fc` and `get_fcd` as part of M1.5 (Numba sweeper).

### Files to create

```
vbi/simulator/backend/numba_/__init__.py
vbi/simulator/backend/numba_/codegen.py
vbi/simulator/backend/numba_/integrators.py
vbi/simulator/backend/numba_/history.py
vbi/simulator/backend/numba_/simulator.py

vbi/tests/validation/test_mpr_numba.py
```

---

### M1.1 — dfun codegen for Numba

**`vbi/simulator/backend/numba_/codegen.py`**

```python
def build_numba_dfun(spec: ModelSpec) -> Callable:
    """
    Generates and @njit-compiles a dfun from ModelSpec.dfun_str.
    Returns a compiled function: fn(state, c, params_array) -> state_array
    Params are passed as a flat float64 array (order matches spec.parameters).
    """
    sv = spec.sv_names
    param_names = tuple(p.name for p in spec.parameters)

    src_lines = [
        "import numpy as np",
        "from numba import njit",
        "from math import pi, exp, log, sin, cos, tanh, sqrt, fabs as abs",
        "@njit(cache=True)",
        "def _dfun_nb(state, c, params):",
    ]
    for i, name in enumerate(sv):
        src_lines.append(f"    {name} = state[{i}]")
    for i, name in enumerate(param_names):
        src_lines.append(f"    {name} = params[{i}]")
    src_lines.append("    out = np.empty_like(state)")
    for i, name in enumerate(sv):
        src_lines.append(f"    out[{i}] = {spec.dfun_str[name]}")
    src_lines.append("    return out")

    # Write to a temp file for debuggability; import from there
    return _compile_from_source(src_lines, "_dfun_nb")
```

Key difference from NumPy: parameters passed as `float64[:]` array (not dict),
because Numba jitclasses cannot hold Python dicts. The parameter order matches
`spec.parameters` tuple order.

For **node-heterogeneous parameters** (shape `(n_nodes,)`), the dfun signature
extends to `fn(state, c, scalar_params, vector_params, node_idx)` where
`node_idx` is the current node. The loop is over nodes, not vectorized.

---

### M1.2 — Numba ring buffer

**`vbi/simulator/backend/numba_/history.py`**

```python
@njit(cache=True)
def nb_write(buf, step, horizon, cvar_state):
    # buf: (horizon, n_cvar, n_nodes)
    buf[step % horizon] = cvar_state

@njit(cache=True)
def nb_read_delayed(buf, step, horizon, weights, delay_steps, G, a, b):
    """
    Vectorized ring-buffer read + coupling in one pass.
    Returns coupling: (n_cvar, n_nodes)
    delay_steps: (n_nodes, n_nodes) int32
    weights:     (n_nodes, n_nodes) float64
    """
    n_cvar, n_nodes = buf.shape[1], buf.shape[2]
    coupling = np.zeros((n_cvar, n_nodes))
    for cvar in range(n_cvar):
        for tgt in range(n_nodes):
            s = 0.0
            for src in range(n_nodes):
                d = delay_steps[src, tgt]
                idx = (step - d) % horizon
                s += weights[tgt, src] * buf[idx, cvar, src]
            coupling[cvar, tgt] = G * a * s + b
    return coupling
```

Fusing ring-buffer read and coupling into a single `@njit` function avoids
Python-level loop overhead and allows Numba to optimize across the two
operations.

---

### M1.3 — Numba integrators

**`vbi/simulator/backend/numba_/integrators.py`**

```python
@njit(cache=True)
def heun_det(state, dfun_fn, coupling, dt, params):
    k1 = dfun_fn(state, coupling, params)
    k2 = dfun_fn(state + dt * k1, coupling, params)
    return state + 0.5 * dt * (k1 + k2)

@njit(cache=True)
def heun_stoch(state, dfun_fn, coupling, dt, params, noise_nsig, dW):
    # dW pre-generated outside (n_sv, n_nodes) normal(0, sqrt(dt))
    k1 = dfun_fn(state, coupling, params)
    x_pred = state + dt * k1 + dW * noise_nsig[:, np.newaxis]
    k2 = dfun_fn(x_pred, coupling, params)
    return state + 0.5 * dt * (k1 + k2) + dW * noise_nsig[:, np.newaxis]
```

---

### M1.4 — Numba simulator loop

**`vbi/simulator/backend/numba_/simulator.py`**

The inner loop is a single `@njit` function; the Python wrapper handles setup
and monitor output:

```python
@njit(cache=True)
def _nb_run_loop(state, buf, weights, delay_steps, horizon, params,
                 scalar_params, dt, n_steps, G, a, b,
                 noise_nsig, noise_seed, stochastic,
                 record_period, n_sv, n_cvar):
    """
    Returns (times, recorded) where recorded shape is
    (n_record_steps, n_sv, n_nodes).
    """
    ...

class NumbaSimulator:
    def build(self, spec: SimulationSpec) -> None:
        # Compile dfun once
        self._dfun = build_numba_dfun(spec.model)
        # Pre-pack params as float64 array
        self._params = _pack_params(spec.model, spec.node_params)
        ...

    def run(self, duration: float) -> dict:
        ...
        result = _nb_run_loop(...)
        return _unpack_result(result, self.spec)
```

---

### M1.5 — Numba CPU Sweeper (primary SBI workhorse on CPU)

**`vbi/simulator/backend/numba_/sweeper.py`**

The Numba sweeper parallelizes over parameter sets using `numba.prange`.
Each iteration is an independent simulation — embarrassingly parallel.

```python
@njit(parallel=True, cache=True)
def _nb_sweep_loop(param_sets, base_params, weights, delay_steps, horizon,
                   state0, dt, n_steps, noise_nsig, seeds,
                   record_period, t_cut_step,
                   out_fc, out_fcd_ks):
    """
    param_sets: (n_samples, n_sweep_params) float64
    out_fc:     (n_samples, n_nodes, n_nodes)  pre-allocated
    out_fcd_ks: (n_samples,)                   pre-allocated
    seeds:      (n_samples,) int64 — different seed per run for stochastic
    """
    n_samples = param_sets.shape[0]
    for i in prange(n_samples):
        # Copy base params; overwrite swept params for this run
        params_i = base_params.copy()
        _apply_sweep_params(params_i, param_sets[i], ...)

        # Independent ring buffer per simulation (stack-allocated inside prange)
        buf_i = np.zeros((horizon, n_cvar, n_nodes))
        buf_i[:] = state0[cvar_indices]

        state_i = state0.copy()
        ts_buf  = np.empty((n_record, n_sv, n_nodes))  # temp storage

        for step in range(n_steps):
            coup = nb_read_delayed(buf_i, step, horizon, weights, delay_steps, ...)
            state_i = heun_stoch(state_i, _dfun_nb, coup, dt, params_i,
                                 noise_nsig, seeds[i], step)
            nb_write(buf_i, step, horizon, state_i[cvar_indices])
            if step >= t_cut_step and (step % record_period) == 0:
                ts_buf[_record_idx(step, t_cut_step, record_period)] = state_i

        # Feature extraction — @njit functions from vbi.feature_extraction
        # get_fc_nb, get_fcd_nb are Numba-compatible variants of get_fc / get_fcd
        out_fc[i]     = get_fc_nb(ts_buf[:, voi_idx, :])
        out_fcd_ks[i] = get_fcd_nb(ts_buf[:, voi_idx, :], window, overlap)
```

Key implementation notes:
- Each `prange` iteration owns its own `buf_i` (no shared state → no race
  conditions).
- `ts_buf` is a temporary time-series buffer **per thread**; only the
  extracted feature is written to the global output. This keeps memory O(n_threads)
  not O(n_samples).
- `_nb_fc` is a `@njit` Pearson correlation (reuse or adapt from
  `vbi/feature_extraction/features_utils.py:get_fc`).
- `_nb_fcd_ks` is a `@njit` sliding-window FC + KS distance.
- `seeds[i]` ensures statistically independent noise per run when stochastic.

```python
class NumbaSweeperCPU:
    def __init__(self, spec, sweep_spec): ...

    def run(self, duration: float) -> dict[str, np.ndarray]:
        out_fc     = np.empty((self.sweep.n_samples, n_nodes, n_nodes))
        out_fcd_ks = np.empty(self.sweep.n_samples)
        _nb_sweep_loop(..., out_fc, out_fcd_ks)
        return {"fc": out_fc, "fcd_ks": out_fcd_ks}
```

### M1 validation test

**`vbi/tests/validation/test_mpr_numba.py`**

```python
def test_numba_matches_numpy(n_nodes=80, duration=5000.0, dt=0.01, rtol=1e-4):
    spec = _build_mpr_spec(n_nodes, dt, stochastic=False)
    np_result  = Simulator(spec, backend="numpy").run(duration)
    nb_result  = Simulator(spec, backend="numba").run(duration)
    t_np, s_np = np_result["raw"]
    t_nb, s_nb = nb_result["raw"]
    assert np.allclose(s_np, s_nb, rtol=rtol), "Numba diverges from NumPy baseline"

def test_numba_sweep_fc_matches_numpy_sweep(n_nodes=10, n_samples=20):
    sweep_spec = SweepSpec(params={"G": np.linspace(1,4,n_samples)})
    np_fc  = Sweeper(spec, sweep_spec, backend="numpy").run(3000.0)["fc"]
    nb_fc  = Sweeper(spec, sweep_spec, backend="numba").run(3000.0)["fc"]
    np.testing.assert_allclose(np_fc, nb_fc, rtol=1e-4)

def test_numba_sweep_throughput(n_nodes=80, n_samples=500, duration=5000.0):
    # Benchmark — not a pass/fail test, but records samples/s
    sweep_spec = SweepSpec(params={"G": np.linspace(0.5,5,n_samples)})
    t0 = time.perf_counter()
    Sweeper(spec, sweep_spec, backend="numba").run(duration)
    rate = n_samples / (time.perf_counter() - t0)
    print(f"Numba sweep: {rate:.1f} samples/s on {n_nodes} nodes")
```

Also test: JR, WilsonCowan, stochastic Heun (moment comparison).

**M1 done when:** single-run and sweep tests pass for deterministic + stochastic,
2 and 80 nodes, with and without delays. Throughput benchmark recorded.

---

## M2 — C++ Backend

**Goal:** Generate a `.hpp` model file + CMake target from `ModelSpec`. Reuse
the SWIG wrapper pattern already established in `vbi/models/cpp/`.

### Files to create

```
vbi/simulator/backend/cpp/__init__.py
vbi/simulator/backend/cpp/codegen.py      # ModelSpec → .hpp
vbi/simulator/backend/cpp/build.py        # cmake invocation + cache
vbi/simulator/backend/cpp/simulator.py    # Python wrapper
vbi/simulator/backend/cpp/_src/
    template_model.hpp.j2       # Jinja2 template
    template_main.cpp.j2
    template_bindings.i.j2      # SWIG interface
    CMakeLists.txt.j2
    utility.hpp                 # shared helpers (copy from existing)

vbi/tests/validation/test_mpr_cpp.py
```

---

### M2.1 — dfun expression → C++ (AST translator)

**`vbi/simulator/backend/cpp/codegen.py`**

Implement `_CppExprGen` following the TVB `backend_cpp/codegen.py` pattern.

```python
import ast

_MATH_MAP = {
    "exp": "std::exp",   "log": "std::log",
    "sin": "std::sin",   "cos": "std::cos",
    "tanh": "std::tanh", "sqrt": "std::sqrt",
    "abs": "std::abs",   "pi":  "M_PI",
}

class _CppExprGen(ast.NodeVisitor):
    """
    Translates a Python math expression AST node to C++ source.
    State variables are mapped to state(sv_index, node_idx).
    Scalar params to P.param_name or params[i].
    Node-heterogeneous params to params_vec[i][node_idx].
    Coupling input 'c' → coupling[0][node_idx] (first cvar).
    """
    def __init__(self, spec: ModelSpec, node_idx_var: str = "n"):
        self.spec = spec
        self.nvar = node_idx_var
        self._sv_map = {sv: i for i, sv in enumerate(spec.sv_names)}
        self._param_map = {p.name: i for i, p in enumerate(spec.parameters)}

    def visit_Name(self, node):
        name = node.id
        if name in self._sv_map:
            return f"state({self._sv_map[name]}, {self.nvar})"
        if name == "c":
            return f"coupling(0, {self.nvar})"
        if name in self._param_map:
            return f"P.{name}"   # scalar; vector case handled separately
        if name in _MATH_MAP:
            return _MATH_MAP[name]
        return name

    ... # visit_BinOp, visit_Call, visit_UnaryOp, visit_Constant

def py_expr_to_cpp(expr_str: str, spec: ModelSpec) -> str:
    tree = ast.parse(expr_str, mode="eval")
    return _CppExprGen(spec).visit(tree.body)
```

---

### M2.2 — C++ model template (`.hpp`)

**`_src/template_model.hpp.j2`** (Jinja2)

```cpp
// AUTO-GENERATED — do not edit by hand
// Model: {{ spec.name }}
#pragma once
#include <cmath>
#include <array>

struct {{ spec.name }}Params {
    {% for p in spec.parameters %}
    double {{ p.name }} = {{ p.default }};
    {% endfor %}
};

// Heun deterministic integrator step (per node, per sv)
inline void {{ spec.name }}_dfun(
        const double* __restrict__ state,   // [n_sv * n_nodes]
        const double* __restrict__ coupling, // [n_cvar * n_nodes]
        double* __restrict__ deriv,          // [n_sv * n_nodes] out
        const {{ spec.name }}Params& P,
        int n_nodes) {
    for (int n = 0; n < n_nodes; ++n) {
        {% for i, sv in enumerate(spec.sv_names) %}
        const double {{ sv }} = state[{{ i }} * n_nodes + n];
        {% endfor %}
        const double c = coupling[0 * n_nodes + n];
        {% for i, sv in enumerate(spec.sv_names) %}
        deriv[{{ i }} * n_nodes + n] = {{ dfun_cpp[sv] }};
        {% endfor %}
    }
}
```

The main `.cpp` template wraps the ring buffer, Heun loop, and monitor logic.
The SWIG `.i` template exposes `integrate()`, `get_r_d()`, etc. — same
pattern as `vbi/models/cpp/_src/mpr_sde.i`.

---

### M2.3 — Build caching

**`vbi/simulator/backend/cpp/build.py`**

```python
CACHE_DIR = Path(os.environ.get("VBI_CPP_CACHE", Path.home() / ".cache/vbi/cpp"))

def build_or_load(spec: SimulationSpec) -> types.ModuleType:
    key = spec.cache_key()            # SHA256 from spec.payload()
    so_path = CACHE_DIR / f"{key}.so"
    if so_path.exists():
        return _load_so(so_path)      # reuse without recompile

    src_dir = CACHE_DIR / key
    _write_sources(spec, src_dir)     # render Jinja2 templates
    _cmake_build(src_dir, so_path)    # cmake + make
    return _load_so(so_path)
```

This is the same caching principle as TVB `spec.cache_key()`. Parameter sweeps
that change only `G` or `eta` produce the same `.so` — no recompile.

---

### M2.4 — C++ Sweeper (OpenMP parallel loop)

The C++ sweep adds an outer OpenMP loop over parameter sets. The generated
`.hpp` template gains a `sweep_integrate()` entry point:

```cpp
// template_main.cpp.j2 (sweep section)
void sweep_integrate(
    const double* param_sets,   // (n_samples, n_sweep_params)
    int n_samples, int n_sweep_params,
    const char** sweep_param_names,
    double duration,
    double* out_fc,             // (n_samples, n_nodes, n_nodes)
    double* out_fcd_ks          // (n_samples,)
) {
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n_samples; ++i) {
        {{ spec.name }}Params P = base_params;
        // override swept params for run i
        apply_sweep_params(P, param_sets + i * n_sweep_params,
                           sweep_param_names, n_sweep_params);
        // independent ring buffer on stack/thread-local heap
        auto buf = RingBuffer(horizon, n_cvar, n_nodes);
        auto state = initial_state.clone();
        double* ts_local = new double[n_record * n_sv * n_nodes];
        heun_loop(state, buf, P, dt, n_steps, t_cut_step,
                  record_period, ts_local);
        compute_fc(ts_local, n_record, n_nodes, out_fc + i*n_nodes*n_nodes);
        compute_fcd_ks(ts_local, n_record, n_nodes, window, overlap,
                       out_fcd_ks + i);
        delete[] ts_local;
    }
}
```

The Python `CppSweeper` calls `sweep_integrate()` via SWIG and returns numpy
arrays directly — no Python loop.

### M2 validation test

**`vbi/tests/validation/test_mpr_cpp.py`**

```python
def test_cpp_matches_numpy(n_nodes=80, duration=5000.0, dt=0.01, rtol=1e-4):
    spec = _build_mpr_spec(n_nodes, dt, stochastic=False)
    np_result  = Simulator(spec, backend="numpy").run(duration)
    cpp_result = Simulator(spec, backend="cpp").run(duration)
    ...

def test_cpp_sweep_fc_matches_numba_sweep(n_nodes=10, n_samples=20):
    sweep_spec = SweepSpec(params={"G": np.linspace(1,4,n_samples)})
    nb_fc  = Sweeper(spec, sweep_spec, backend="numba").run(3000.0)["fc"]
    cpp_fc = Sweeper(spec, sweep_spec, backend="cpp").run(3000.0)["fc"]
    np.testing.assert_allclose(nb_fc, cpp_fc, rtol=1e-4)

def test_cpp_cache_reuse():
    # Build same spec twice; assert second call returns in < 100 ms (cache hit)
    build_or_load(spec)
    t0 = time.perf_counter()
    build_or_load(spec)
    assert time.perf_counter() - t0 < 0.1
```

**M2 done when:** MPR, JR, WilsonCowan pass numeric comparison against NumPy
baseline; C++ sweep FC matches Numba sweep; build cache works.

---

## M3 — Numba-CUDA Backend

**Feature extraction note:** Add `vbi/feature_extraction/features_utils_cuda.py` with `@cuda.jit` device functions `get_fc_cuda`, `get_fcd_cuda` as part of M3.

**Goal:** Run the parameter sweep on GPU. The CUDA design is batch-first — the
primary kernel simulates N parameter sets simultaneously. A single-run mode is
derived from the sweep (batch size 1). This is the highest-throughput backend
for SBI training data generation.

### Files to create

```
vbi/simulator/backend/numba_cuda/__init__.py
vbi/simulator/backend/numba_cuda/codegen.py
vbi/simulator/backend/numba_cuda/sweeper.py   # primary class
vbi/simulator/backend/numba_cuda/simulator.py # thin wrapper (batch=1)

vbi/tests/validation/test_mpr_cuda.py
```

---

### M3.1 — CUDA kernel design (batch-first)

**Thread/block layout:**

```
grid  = (n_samples,)          # one block per parameter set
block = (n_nodes_rounded_up,) # one thread per node within a simulation
```

Each block simulates one `θ_i` independently. Threads within a block share:
- `__shared__` ring buffer slice (if small enough) for fast delay reads
- `__shared__` coupling accumulation buffer

```python
@cuda.jit
def _cuda_sweep_kernel(
    param_sets,      # (n_samples, n_sweep_params) on device
    base_params,     # scalar params array on device
    weights_d,       # (n_nodes, n_nodes)
    delay_steps_d,   # (n_nodes, n_nodes) int32
    state0_d,        # (n_sv, n_nodes) initial state
    buf_d,           # (n_samples, horizon, n_cvar, n_nodes) ring buffers
    ts_d,            # (n_samples, n_record, n_sv, n_nodes) temp or None
    out_fc_d,        # (n_samples, n_nodes, n_nodes)
    out_fcd_ks_d,    # (n_samples,)
    dt, n_steps, horizon, t_cut_step, record_period, noise_seed_base
):
    sim_idx = cuda.blockIdx.x    # which parameter set
    node    = cuda.threadIdx.x   # which node within this simulation

    if sim_idx >= n_samples or node >= n_nodes:
        return

    # Each block has its own slice of buf_d — no cross-simulation interference
    buf_sim = buf_d[sim_idx]        # (horizon, n_cvar, n_nodes)
    state   = cuda.local.array(n_sv, dtype=float64)   # registers per thread

    # Load initial state for this node
    for sv in range(n_sv):
        state[sv] = state0_d[sv, node]

    rng = cuda.random.create_xoroshiro128p_states(1, seed=noise_seed_base + sim_idx)

    for step in range(n_steps):
        c = _read_coupling_cuda(buf_sim, step, horizon, weights_d,
                                delay_steps_d, node)
        # Heun step — all state vars for this node
        _heun_cuda(state, c, base_params, param_sets[sim_idx], dt, rng, node)
        # Write cvar to ring buffer (atomic not needed — only this thread owns node)
        for cv in range(n_cvar):
            buf_sim[step % horizon, cv, node] = state[cvar_indices[cv]]
        # Record if past burn-in
        if step >= t_cut_step and (step % record_period) == 0:
            ridx = _record_idx(step, t_cut_step, record_period)
            for sv in range(n_sv):
                ts_d[sim_idx, ridx, sv, node] = state[sv]

    # After time loop: apply feature extraction (separate from monitors)
    # get_fc_cuda / get_fcd_cuda from vbi.feature_extraction — CUDA variants
    get_fc_cuda(ts_d[sim_idx], out_fc_d[sim_idx], n_record, n_nodes)
```

**Memory layout rationale:**
- `buf_d` shape `(n_samples, horizon, n_cvar, n_nodes)` — each simulation owns
  a contiguous slice; blocks never touch each other's memory.
- `ts_d` is optional: if only FC/FCD monitors are requested, `ts_d` can be a
  rolling window buffer rather than the full time series.
- FC computation is done on GPU via a second reduction kernel (avoid
  D→H transfer of full time series).

---

### M3.2 — CUDA dfun codegen

**`codegen.py`**

```python
from numba import cuda

def build_cuda_dfun(spec: ModelSpec):
    """
    Generates a @cuda.jit device function (not a kernel — called from kernel).
    Signature: _dfun_cuda(state_local, c, base_params, sweep_params) -> nothing
    Writes derivatives back into a local array.
    Uses math.* (no numpy) — compatible with CUDA device functions.
    """
    _MATH_MAP_CUDA = {
        "exp": "math.exp", "log": "math.log",
        "tanh": "math.tanh", "sqrt": "math.sqrt",
        "pi": "math.pi", "abs": "math.fabs",
        ...
    }
```

---

### M3.3 — Memory management

For large sweeps (>10 000 samples, 80 nodes), `ts_d` can exceed GPU memory.
Two modes:

- **Feature-only mode (default):** `ts_d` holds only a sliding window of
  `window_ms / dt` steps, reused in place. FC accumulates as a running
  Pearson sum `(Σxy, Σx, Σy, Σx², Σy², n)` in registers/shared memory.
  Memory: O(n_samples × window × n_sv × n_nodes).

- **Full time-series mode (opt-in via `MonitorSpec(kind="raw")`):** `ts_d` is
  the full `(n_samples, n_steps, n_sv, n_nodes)` array. Transferred to host
  at end. Warning printed if size > 1 GB.

---

### M3 validation test

```python
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="no CUDA device")
def test_cuda_single_matches_numba(n_nodes=80, duration=5000.0, dt=0.01):
    spec = _build_mpr_spec(n_nodes, dt, stochastic=False)
    nb_result   = Simulator(spec, backend="numba").run(duration)
    cuda_result = Simulator(spec, backend="cuda").run(duration)
    # Float32 GPU vs float64 CPU — accept rtol=1e-3
    np.testing.assert_allclose(
        nb_result["raw"][1].mean(0), cuda_result["raw"][1].mean(0), rtol=1e-3)

@pytest.mark.skipif(not CUDA_AVAILABLE, reason="no CUDA device")
def test_cuda_sweep_fc_matches_numba_sweep(n_nodes=10, n_samples=50):
    sweep_spec = SweepSpec(params={"G": np.linspace(1,4,n_samples)})
    nb_fc   = Sweeper(spec, sweep_spec, backend="numba").run(3000.0)["fc"]
    cuda_fc = Sweeper(spec, sweep_spec, backend="cuda").run(3000.0)["fc"]
    np.testing.assert_allclose(nb_fc.mean(), cuda_fc.mean(), rtol=1e-2)

@pytest.mark.skipif(not CUDA_AVAILABLE, reason="no CUDA device")
def test_cuda_sweep_throughput(n_nodes=80, n_samples=2000, duration=5000.0):
    sweep_spec = SweepSpec(params={"G": np.linspace(0.5,5,n_samples)})
    t0 = time.perf_counter()
    Sweeper(spec, sweep_spec, backend="cuda").run(duration)
    rate = n_samples / (time.perf_counter() - t0)
    print(f"CUDA sweep: {rate:.1f} samples/s on {n_nodes} nodes")
```

---

## M4 — JAX Backend

**Feature extraction note:** Extend `vbi/feature_extraction/features_utils_jax.py` (already partially exists) with `get_fc_jax`, `get_fcd_jax` compatible with `jax.vmap` as part of M4.3.

**Goal:** JIT + grad-able simulator for gradient-based inference. The JAX
backend uses `jax.lax.scan` for the time loop, enabling `jax.grad` through
the entire simulation.

### Files to create

```
vbi/simulator/backend/jax_/__init__.py
vbi/simulator/backend/jax_/codegen.py
vbi/simulator/backend/jax_/simulator.py

vbi/tests/validation/test_mpr_jax.py
```

---

### M4.1 — JAX dfun codegen

```python
import jax.numpy as jnp

def build_jax_dfun(spec: ModelSpec) -> Callable:
    """
    Same string-substitution approach as NumPy backend.
    Substitutes math namespace with jax.numpy equivalents.
    Returns a pure function: fn(state, c, params_dict) -> deriv
    where state is a JAX array.
    """
    _MATH_MAP_JAX = {
        "exp": "jnp.exp", "log": "jnp.log",
        "sin": "jnp.sin", "cos": "jnp.cos",
        "tanh": "jnp.tanh", "sqrt": "jnp.sqrt",
        "abs": "jnp.abs", "pi": "jnp.pi",
    }
```

---

### M4.2 — JAX simulator with `lax.scan`

```python
import jax
import jax.numpy as jnp

def _make_scan_fn(dfun, coupling_fn, integrator, dt, delay_steps, spec):
    def scan_body(carry, _):
        state, buf, step = carry
        delayed = _read_ring_jax(buf, step, delay_steps)
        c = coupling_fn(delayed)
        new_state = integrator(state, dfun, c, dt)
        new_buf = _write_ring_jax(buf, step, new_state, spec.model.cvar_indices)
        return (new_state, new_buf, step + 1), new_state[voi_indices]

    return scan_body

class JaxSimulator:
    def build(self, spec): ...

    def run(self, duration):
        n_steps = round(duration / self.spec.integrator.dt)
        init_carry = (self._state, self._buf, 0)
        _, outputs = jax.lax.scan(self._scan_fn, init_carry, None, length=n_steps)
        return {"raw": (self._times, outputs)}
```

`jax.lax.scan` makes the entire time loop differentiable, enabling
`jax.grad(loss_fn)(params)` through the simulation for gradient-based
inference in VBI.

---

### M4.3 — JAX Sweeper via `vmap`

The JAX backend's sweep is the most elegant: `jax.vmap` vectorizes the
single-run function over a batch of parameter sets, and `jax.jit` compiles
the entire batch:

```python
def _simulate_one(theta, static):
    """
    theta: dict of swept param values for one run (JAX arrays, differentiable)
    static: non-differentiable state (weights, connectivity, spec)
    Returns: dict of feature arrays for this run
    """
    spec_i = _patch_params_jax(static["spec"], theta)
    carry_init = (static["state0"], static["buf0"], jnp.int32(0))
    _, ts = jax.lax.scan(_scan_body(spec_i), carry_init, None, length=static["n_steps"])
    # ts: (n_steps_recorded, n_sv, n_nodes)
    ts_cut = ts[static["t_cut_idx"]:]
    return {
        "raw": ts_cut,   # or compute features here
    }

# Vectorize over batch of theta values
_simulate_batch = jax.jit(jax.vmap(_simulate_one, in_axes=(0, None)))

class JaxSweeper:
    def run(self, duration: float) -> dict:
        theta_batch = _pack_theta(self.sweep.param_sets, self.sweep.param_names)
        raw_batch = _simulate_batch(theta_batch, self._static)
        # Apply feature extraction to batch output
        # (FC, FCD computed with jax.vmap over the batch axis)
        return self._extract_features(raw_batch)
```

**Why JAX is special for SBI:**
- `jax.grad(loss)(theta)` backpropagates through the entire simulation,
  enabling score-based and gradient-based SBI methods (e.g. SNPE-C,
  gradient matching, adjoint-based inference).
- `jax.vmap` batches the sweep with zero Python overhead — all N simulations
  run in a single compiled XLA program.
- Works on CPU, GPU (CUDA), and TPU without code changes.

**Gradient test:**
```python
def test_jax_grad_through_simulation():
    from vbi.simulator import Simulator
    from vbi.simulator.models.mpr import mpr

    def loss(G):
        spec_i = _build_spec_with_G(G)
        sim = Simulator(spec_i, backend="jax")
        result = sim.run(2000.0)
        ts = result["raw"][1]
        return jnp.mean(ts ** 2)  # dummy loss

    grad_fn = jax.grad(loss)
    g = grad_fn(jnp.array(2.0))
    assert jnp.isfinite(g), "gradient is NaN/Inf"
```

---

### M4 validation test

```python
@pytest.mark.skipif(not JAX_AVAILABLE, reason="jax not installed")
def test_jax_matches_numpy(n_nodes=10, duration=2000.0, dt=0.01):
    spec = _build_mpr_spec(n_nodes, dt, stochastic=False)
    np_result  = Simulator(spec, backend="numpy").run(duration)
    jax_result = Simulator(spec, backend="jax").run(duration)
    np.testing.assert_allclose(
        np_result["raw"][1], np.array(jax_result["raw"][1]), rtol=1e-3)

@pytest.mark.skipif(not JAX_AVAILABLE, reason="jax not installed")
def test_jax_sweep_matches_numpy_sweep(n_nodes=10, n_samples=20):
    sweep_spec = SweepSpec(params={"G": np.linspace(1,4,n_samples)})
    np_raw  = Sweeper(spec, sweep_spec, backend="numpy").run(3000.0)["raw"]
    jax_raw = Sweeper(spec, sweep_spec, backend="jax").run(3000.0)["raw"]
    np.testing.assert_allclose(np_raw.mean(), np.array(jax_raw).mean(), rtol=1e-3)

def test_jax_grad_runs(): ...   # as above
```

Also test: `jax.jit` reuse (second call faster than first), `vmap` batch
size scaling.

---

## M5 — Model Coverage

Once all backends are verified on MPR, add remaining models as specs.
Each model spec requires:
1. `vbi/simulator/models/<name>.py` — `ModelSpec` instance
2. Validation test against TVB (if available) or existing vbi model

| Model | TVB reference | vbi reference |
|-------|--------------|---------------|
| MontbrioPopulationRate (MPR) | ✓ TVB | — |
| JansenRit | ✓ TVB | `vbi/models/numba/jansen_rit.py` |
| WilsonCowan | ✓ TVB | `vbi/models/numba/wilson_cowan.py` |
| Epileptor (VEP) | ✓ TVB | `vbi/models/numba/vep.py` |
| Generic2dOscillator | ✓ TVB | — |
| ReducedWongWang (RWW) | ✓ TVB | `vbi/models/numba/rww.py` |
| GHB | ✗ no TVB | `vbi/models/numba/ghb.py` |
| Stuart-Landau (SL) | ✗ no TVB | `vbi/models/numba/sl.py` |
| DampedOscillator | ✗ no TVB | `vbi/models/numba/damp_oscillator.py` |
| BOLD (Balloon-Windkessel) | ✓ TVB | `vbi/models/numba/bold.py` |

BOLD is added as a `BoldMonitor` (driven by the `r` state variable), not as a
standalone model.

---

## M6 — Deprecation & Release

| Sub-package | Action | Version |
|---|---|---|
| `vbi/models/pytorch/` | `DeprecationWarning` in `__init__` | current |
| `vbi/models/cupy/` | `DeprecationWarning` in `__init__` | current |
| `vbi/models/tvbk/` | `DeprecationWarning` in `__init__` | current |
| `vbi/simulator/` reaches M1 parity | migration guide added | next minor |
| `vbi/models/numba/` | `DeprecationWarning` | after M1 parity |
| `vbi/models/cpp/` | `DeprecationWarning` | after M2 parity |
| pytorch, cupy, tvbk | hard removal | +1 minor |
| numba, cpp stubs (re-export) | stubs redirect to `vbi/simulator/` | +1 minor after parity |
| stubs removed | clean removal | +2 minor after parity |

---

## Shared Utilities (across milestones)

### State bounds clamping
```python
def apply_bounds(state: np.ndarray, spec: ModelSpec) -> np.ndarray:
    for i, sv in enumerate(spec.state_variables):
        if sv.lower_bound is not None:
            state[i] = np.maximum(state[i], sv.lower_bound)
        if sv.upper_bound is not None:
            state[i] = np.minimum(state[i], sv.upper_bound)
    return state
```

### Node-heterogeneous parameter broadcasting
```python
def _build_params(spec: ModelSpec, node_params: dict, n_nodes: int) -> dict:
    params = {p.name: p.default for p in spec.parameters}
    for name, arr in node_params.items():
        params[name] = np.broadcast_to(arr, (n_nodes,)).copy()
    return params
```

### Delay step computation
```python
def compute_delay_steps(tract_lengths, speed, dt) -> np.ndarray:
    raw = tract_lengths / (speed * dt)
    return np.round(raw).astype(np.int32)
```

---

---

## MF — Feature Extraction Pipeline (cross-cutting, all backends)

**Goal:** Define a single user-facing `FeaturePipeline` API that works identically
across all backends while dispatching to the right JIT-compiled implementation
internally. The existing `vbi/feature_extraction/` machinery (cfg dicts,
`get_features_by_domain`, `extract_features`, `n_workers`) is **kept and reused**
as the NumPy tier; compiled variants are added per-backend milestone.

---

### MF.0 — Two-tier architecture

Feature extraction splits into two tiers depending on where it runs:

```
Tier 1 — Python level (NumPy backend + post-processing)
    vbi/feature_extraction/calc_features.py
        extract_features(ts, fs, cfg, n_workers=N)
        get_features_by_domain(domain="statistical")
        get_features_by_given_names(cfg, names=[...])
        report_cfg(cfg)

    Used for:
    - Single Simulator runs (any backend — result already on CPU)
    - NumpySweeper (sequential; Python overhead acceptable)
    - Post-processing stored BOLD/tavg batches
    - Rich feature sets (100+ features, catch22, spectral, HMM, …)

Tier 2 — JIT level (inside Numba prange / CUDA kernel / JAX scan)
    vbi/feature_extraction/features_utils_nb.py    (M1)
    vbi/feature_extraction/features_utils_cuda.py  (M3)
    vbi/feature_extraction/features_utils_jax.py   (M4, partially exists)

    Used for:
    - NumbaSweeperCPU — @njit functions called inside prange
    - CUDA sweeper    — @cuda.jit device functions inside kernel
    - JAX sweeper     — jit+vmap compatible pure functions

    Supported core set: fc, fcd_ks, mean, std, band_power
    (anything expressible as array ops on a (n_steps, n_nodes) buffer)
```

**Why two tiers?** `extract_features()` is a Python-level function with dict
dispatch, list appending, and pandas output — none of which can enter a Numba
`@njit` or CUDA kernel. The JIT tier supports only the subset of features that
can be expressed as pure numeric kernels. The `FeaturePipeline` hides this split.

---

### MF.1 — FeaturePipeline: cfg-first API

`FeaturePipeline` wraps the existing cfg-dict machinery. Per-feature parameters
live in the cfg dict (exactly where they already are); the pipeline only adds
the three sim-level concerns it uniquely knows: which monitor signal to consume,
how much burn-in to discard, and (later) which JIT backend to dispatch to.

```python
from vbi.feature_extraction.pipeline import FeaturePipeline
from vbi.feature_extraction.features_settings import (
    get_features_by_domain, get_features_by_given_names, update_cfg
)

# Build and tune cfg — per-feature params stay in the cfg
cfg = get_features_by_domain(domain="connectivity")
cfg = get_features_by_given_names(cfg, names=["calc_fc", "calc_fcd"])
cfg = update_cfg(cfg, "calc_fcd", {"window_size": 30})   # per-feature param

# Wrap in FeaturePipeline — only sim-level concerns here
pipeline = FeaturePipeline(cfg, signal="tavg", t_cut=500.0)

# Single run
result = Simulator(spec, backend="numpy").run(5000.0)
labels, values = pipeline.extract(result)          # (list[str], ndarray)
df = pipeline.extract_df(result)                   # one-row DataFrame

# Parameter sweep
sweep_spec = SweepSpec(params={"G": np.linspace(1, 4, 50)}, pipeline=pipeline)
df = Sweeper(spec, sweep_spec, backend="numba").run_df(5000.0)
# df.columns = ["G", "fc_0_1", "fc_0_2", ..., "fcd_ks"]
```

`FeaturePipeline.__init__` signature:
```python
class FeaturePipeline:
    def __init__(self, cfg: dict, signal: str = "tavg", t_cut: float = 500.0):
        ...
    def extract(self, monitor_result: dict) -> tuple[list[str], np.ndarray]:
        ...
    def extract_df(self, monitor_result: dict) -> pd.DataFrame:
        ...
```

The existing workflow without `FeaturePipeline` (models outside `vbi.simulator`)
is unchanged:
```python
cfg = get_features_by_domain(domain="statistical")
cfg = get_features_by_given_names(cfg, names=["calc_std", "calc_mean"])
stat_vec = extract_features(ts=[sol["x"].T], cfg=cfg, fs=fs, n_workers=4).values
```

---

### MF.2 — Internal dispatch: Tier 1 (Python) only for now

---

### MF.3 — Backend-specific implementations (schedule)

| Backend | File | Features | Added in |
|---------|------|----------|----------|
| NumPy (Tier 1) | `features_utils.py` (exists) | All 100+ | M0 (already done) |
| Numba CPU | `features_utils_nb.py` | fc, fcd_ks, mean, std, band_power | M1.5 |
| C++ | computed in C++ sweep loop; returned as ndarray | fc, fcd_ks | M2.4 |
| CUDA | `features_utils_cuda.py` | fc, fcd_ks, mean | M3 |
| JAX | `features_utils_jax.py` (partially exists) | fc, fcd_ks, mean, std | M4.3 |

**Core JIT feature set** (must be in every JIT backend):

| Feature key | Input | Output | Notes |
|-------------|-------|--------|-------|
| `fc` | `(n_steps, n_nodes)` | `(n_nodes, n_nodes)` | Pearson correlation matrix |
| `fcd_ks` | `(n_steps, n_nodes)` | scalar | KS-distance of FCD vs. empirical |
| `mean` | `(n_steps, n_nodes)` | `(n_nodes,)` | Temporal mean per node |
| `std` | `(n_steps, n_nodes)` | `(n_nodes,)` | Temporal std per node |
| `band_power` | `(n_steps, n_nodes)` | `(n_bands, n_nodes)` | Band-pass RMS |

These five are sufficient for SBI training data. The full `extract_features`
suite is available whenever results are back on the CPU (Tier 1 path).

---

### MF.4 — Memory strategy for sweep feature extraction

For large sweeps, the time-series buffer is the bottleneck:

```
NumPy sweeper:    store full ts per run (ok — sequential; only 1 run in memory)
Numba prange:     ts_buf is thread-local (n_record × n_sv × n_nodes per thread)
                  feature written to pre-allocated out array; ts_buf discarded
CUDA:             ts_d is (n_samples, n_record, n_sv, n_nodes) on GPU
                  → use rolling window buffer in "feature-only" mode (M3.3)
JAX:              lax.scan accumulates ts; vmap over batch axis; fc via jax.vmap
```

Rule: **never store full time series for all sweep samples simultaneously** unless
the user explicitly requests `MonitorSpec(kind="raw")`.

---

### MF.5 — BOLD + FCD workflow (common SBI pattern)

When the model is BOLD-based (RWW, MPR with BOLD monitor):

```python
# Option A: compute features inline (all backends)
pipeline = FeaturePipeline(
    features=["fc", "fcd_ks"],
    signal="bold",
    t_cut=500.0,
    fcd_window_ms=30_000.0,   # 30 s window for BOLD (TR=2s)
    fcd_overlap=0.5,
)

# Option B: store BOLD time series first, then extract in bulk (NumPy only)
# Useful when you have a large existing dataset of BOLD signals
from vbi.feature_extraction.calc_features import extract_features
cfg = get_features_by_domain(domain="connectivity")
df = extract_features(ts=bold_signals, fs=0.5, cfg=cfg, n_workers=8)
```

Option A is the preferred path for new code — it works with all backends.
Option B is the legacy path and remains fully supported for compatibility.

---

## Open questions (decide before M0)

1. **`c` as scalar vs vector:** Current dfun_str uses a single `c` (first
   cvar). If a model has multiple coupling variables (e.g. E and I in
   WilsonCowan), `c` becomes a vector. Decision: use `c[0]`, `c[1]` in
   dfun_str for multi-cvar models.

2. **Node-heterogeneous params in dfun_str:** Using `eta` as-is in dfun_str
   works for NumPy (broadcasts) but requires special-casing in Numba/C++
   (need per-node indexing). Decision: add a `heterogeneous` flag to
   `Parameter` so the codegen knows to emit `eta[n]` instead of `eta`.

3. **BOLD monitor:** Does BOLD run at every integration step (internally) and
   output at a lower rate, or does it see sub-sampled `r`? Decision: follow TVB
   — BOLD balloon model integrates at every step using its own Euler step
   driven by `r`.
