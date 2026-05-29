# Multi-Backend Simulator - Implementation Plan

## Vision

Build a spec-driven, multi-backend neural mass simulator alongside the existing
`vbi/models/` code. The model equations are written **once** in a Python spec
(later also YAML); a code generator emits backend-specific implementations for
NumPy, Numba (CPU), Numba-CUDA (GPU), JAX, and C++. The public API stays
identical across backends.

---

## Migration Strategy: Keeping Existing Users Safe

### Guiding principle
The existing `vbi/models/{numba,cpp,cupy,pytorch,tvbk}/` tree is **not
touched**. New work lives exclusively under `vbi/simulator/`. Both trees are
importable simultaneously.

### Phase 0 - Soft-deprecation notice (now, one release)
- Add `DeprecationWarning` to `vbi/models/pytorch/__init__.py` and
  `vbi/models/cupy/__init__.py` (heaviest maintenance burden).
- Do the same for `vbi/models/tvbk/` (hard TVB dependency).
- CHANGELOG entry: these sub-packages are deprecated, kept through vX.Y, then
  removed in vX.(Y+1).
- `vbi/models/numba/` and `vbi/models/cpp/` stay **stable** until the new
  backends reach feature parity (tracked by a milestone below).

### Phase 1 - New backend lives alongside (this plan)
All new code goes into `vbi/simulator/`. Existing tests still pass; no breakage
for users of `vbi/models/numba/` or `vbi/models/cpp/`.

### Phase 2 - Parity declared
When the new Numba and C++ backends reproduce all existing models, add a
migration guide and bump the minor version.

### Phase 3 - Hard removal
Remove `pytorch`, `cupy`, `tvbk` sub-packages. Keep `numba` and `cpp` stubs
that import from `vbi/simulator/` for one more release, then remove.

---

## Development Order

```
Python (NumPy) → Numba CPU → C++ → Numba-CUDA → JAX
```

**Why this order:**
- NumPy baseline is easiest to validate (compare vs TVB reference).
- Numba shares Python code structure; the dfun can be the same string, just
  `@njit`-compiled. Catches spec design issues early.
- C++ is most critical for large-scale sweeps; doing it third means the spec is
  already stable.
- Numba-CUDA and JAX are high-value but more specialist; added last so they
  don't block neuroimaging users.

---

## Architecture

### Inspiration from `tvb-root-hybrid-cpp`
The reference project (`tvb/simulator/backend_cpp/`) separates concerns cleanly:
- **Spec** (`spec.py`) - frozen dataclasses describing what to simulate, not
  how. `SimulationSpec`, `SubnetworkSpec`, `IntegratorSpec`, `ProjectionSpec`,
  `MonitorSpec`. Hashed for build-cache keying.
- **Backend** - reads the spec, generates/compiles code, returns callable.
- **Hybrid** (`hybrid/`) - Python-level network/coupling/monitor logic that
  wraps the compiled inner loop.

This project follows the same pattern but owns all layers (no TVB simulator
dependency at runtime).

---

## Directory Layout

```
vbi/
├── models/             # EXISTING - do not change
│   ├── numba/          # stable
│   ├── cpp/            # stable
│   ├── cupy/           # deprecated → removal in vX.(Y+1)
│   ├── pytorch/        # deprecated → removal in vX.(Y+1)
│   └── tvbk/           # deprecated → removal in vX.(Y+1)
│
└── simulator/          # NEW - all new backend work lives here
    ├── __init__.py
    ├── spec/
    │   ├── model.py        # ModelSpec, StateVar, Parameter dataclasses
    │   ├── integrator.py   # IntegratorSpec (Euler/Heun, det/stoch)
    │   ├── coupling.py     # CouplingSpec (Linear, Sigmoidal, Kuramoto)
    │   ├── monitor.py      # MonitorSpec (Raw, SubSample, Bold)
    │   └── simulation.py   # SimulationSpec (top-level, hashable)
    ├── models/             # model specs as Python files
    │   ├── mpr.py
    │   ├── jansen_rit.py
    │   ├── wilson_cowan.py
    │   ├── epileptor.py
    │   ├── generic2doscillator.py
    │   └── ...
    ├── backend/
    │   ├── base.py         # AbstractBackend protocol
    │   ├── numpy_/         # pure NumPy (reference, no compilation)
    │   │   ├── simulator.py
    │   │   ├── integrators.py
    │   │   ├── coupling.py
    │   │   └── monitors.py
    │   ├── numba_/         # Numba CPU JIT
    │   │   ├── codegen.py
    │   │   ├── simulator.py
    │   │   └── integrators.py
    │   ├── numba_cuda/     # Numba CUDA GPU
    │   │   ├── codegen.py
    │   │   └── simulator.py
    │   ├── cpp/            # C++ via SWIG or pybind11
    │   │   ├── codegen.py  # emits .hpp from spec
    │   │   ├── build.py    # cmake/swig build automation
    │   │   └── simulator.py
    │   └── jax_/           # JAX JIT + grad
    │       ├── codegen.py
    │       └── simulator.py
    ├── history.py          # Ring buffer / delay buffer (shared)
    └── api.py              # Public entry point: Simulator(spec, backend="numba")
```

---

## Spec Format

### Python spec file (`vbi/simulator/models/mpr.py`)

```python
from vbi.simulator.spec.model import ModelSpec, StateVar, Parameter

spec = ModelSpec(
    name="MontbrioPopulationRate",
    state_variables=[
        StateVar("r", default_init=0.0, noise=True),
        StateVar("V", default_init=0.0, noise=True),
    ],
    parameters=[
        Parameter("tau",   1.0,   description="time constant"),
        Parameter("I",     0.0,   description="external input"),
        Parameter("Delta", 0.7,   description="heterogeneity"),
        Parameter("J",     14.5,  description="coupling weight"),
        Parameter("eta",  -4.6,   description="excitability"),
    ],
    cvar=["r"],          # variable used for coupling
    # dfun as a string so codegen can emit it into any backend
    dfun_str={
        "dr": "(Delta / (pi * tau) + 2 * r * V) / tau",
        "dV": "(V**2 + eta + J * tau * r + I - (pi * tau * r)**2 + c) / tau",
    },
    noise_variables=["r", "V"],
    reference="Montbrio et al. 2015, PLoS Comput Biol",
)
```

The `dfun_str` dict contains symbolic expression strings (no `np.` prefix).
The code generator substitutes the correct math namespace per backend
(`np`, `jnp`, `nb.float64`, C++ `std::` calls, etc.).

### YAML spec (Phase 2, optional)

```yaml
name: MontbrioPopulationRate
state_variables:
  - name: r
    default_init: 0.0
    noise: true
  - name: V
    default_init: 0.0
    noise: true
parameters:
  tau: {value: 1.0, description: "time constant"}
  ...
cvar: [r]
dfun:
  dr: "(Delta / (pi * tau) + 2 * r * V) / tau"
  dV: "(V**2 + eta + J * tau * r + I - (pi * tau * r)**2 + c) / tau"
```

---

## Core Components

### 1. Model Spec (`spec/model.py`)

Frozen dataclasses: `StateVar`, `Parameter`, `ModelSpec`. Hash-able for build
caching (same idea as `tvb-root-hybrid-cpp/spec.py`).

### 2. Simulation Spec (`spec/simulation.py`)

`SimulationSpec` composes:
- `ModelSpec`
- `IntegratorSpec` - method (euler/heun), dt, stochastic flag, noise_nsig
- `CouplingSpec` - type, parameters (a, b for linear; threshold, sigma for
  sigmoidal)
- `MonitorSpec[]` - list of monitors with periods
- `weights`, `tract_lengths` arrays
- `speed` (conduction velocity → delays)

`SimulationSpec.cache_key()` returns a SHA-256 of the payload (identical to
the reference project pattern), enabling compiled C++ binaries to be reused
across parameter sweeps without recompilation.

### 3. Integrators

Both deterministic and stochastic variants. Same interface for all backends:

```
integrate(state, dfun, coupling, dt, noise) -> new_state
```

| Method | Deterministic | Stochastic (Itô) |
|--------|--------------|-------------------|
| Euler  | ✓            | ✓ (Euler-Maruyama)|
| Heun   | ✓            | ✓ (stochastic Heun)|

### 4. Coupling + Delay / Ring Buffer

The ring buffer stores history for delay-coupled networks:

```
buffer[node, time_step % horizon] = state[cvar, node]
```

Coupling functions operate on the delayed state:
- `Linear(a, b)` - `c_i = a * Σ_j w_ij * x_j(t - τ_ij) + b`
- `Sigmoidal(midpoint, slope, amp)` - wrapped sigmoid
- `Kuramoto` - phase-difference coupling

The same Python implementation of the ring buffer is used in the NumPy
backend. Numba and C++ backends get code-generated or specialized versions.

### 5. Monitors

| Monitor    | What it records                            |
|------------|--------------------------------------------|
| `Raw`      | Every time step                            |
| `SubSample`| Every N steps (period in ms)               |
| `Bold`     | Balloon-Windkessel haemodynamic response   |

API: `monitor.record(t, state)` - all backends call the same Python-level
monitor for the NumPy backend. Numba/C++ backends accumulate into pre-allocated
arrays and hand off to Python monitors only at monitor.period boundaries.

### 6. Backend Protocol (`backend/base.py`)

```python
class AbstractBackend(Protocol):
    def build(self, spec: SimulationSpec) -> None: ...
    def run(self, duration: float, initial_state: np.ndarray) -> dict[str, np.ndarray]: ...
```

`Simulator(spec, backend="numba")` in `api.py` selects and builds the
appropriate backend.

---

## TVB as Dev Dependency

TVB (`tvb-library`) is added to `pyproject.toml` as an **optional dev
dependency**:

```toml
[project.optional-dependencies]
dev = ["tvb-library", "pytest", "hypothesis", ...]
```

It is never imported in production code paths. Tests under
`vbi/tests/validation/` import TVB to generate reference trajectories for
comparison:
- Models available in both: check trajectory L2 error < tolerance.
- Models only in vbi: check conserved quantities or published reference data.

---

## Testing Strategy

```
vbi/tests/
├── ...existing tests...
└── validation/
    ├── conftest.py          # TVB fixtures, skip if tvb-library absent
    ├── test_mpr_numpy.py    # NumPy backend vs TVB MPR
    ├── test_mpr_numba.py    # Numba backend vs NumPy backend
    ├── test_jr_cpp.py       # C++ backend vs NumPy baseline
    └── test_delay_coupling.py
```

Validation tolerance: `rtol=1e-4` for deterministic, trajectory-level
comparison after burn-in.

---

## Milestones

### M0 - Spec & NumPy baseline (start here)
- [ ] Define `StateVar`, `Parameter`, `ModelSpec` dataclasses
- [ ] Write `IntegratorSpec`, `CouplingSpec`, `MonitorSpec`, `SimulationSpec`
- [ ] Implement NumPy Euler integrator
- [ ] Implement ring-buffer delay + Linear coupling (NumPy)
- [ ] Implement Raw and SubSample monitors
- [ ] Port MPR spec, reproduce trajectory vs TVB reference

### M1 - Numba CPU backend
- [ ] Code generator: `ModelSpec.dfun_str` → `@njit` function
- [ ] Numba Heun integrator (deterministic + stochastic)
- [ ] Numba ring-buffer (typed List or np array)
- [ ] Validate Numba vs NumPy baseline on MPR, JansenRit, WilsonCowan

### M2 - C++ backend
- [ ] Code generator: `ModelSpec` → `.hpp` + `SimulationSpec` → CMake target
- [ ] SWIG or pybind11 wrapper (reuse pattern from existing `vbi/models/cpp/`)
- [ ] C++ Euler and Heun (det/stoch)
- [ ] C++ ring buffer with idelays
- [ ] Validate C++ vs NumPy baseline

### M3 - Numba-CUDA backend
- [ ] CUDA kernel codegen from `ModelSpec.dfun_str`
- [ ] GPU ring-buffer, coupling kernel
- [ ] Parameter sweep parallelized over nodes or realizations

### M4 - JAX backend
- [ ] `jax.jit`-compiled dfun via `exec`-free string substitution
- [ ] `jax.lax.scan`-based integration loop (supports `jit` + `grad`)
- [ ] Validate JAX vs NumPy baseline

### M5 - Model coverage
- [ ] MPR, JansenRit, WilsonCowan, Epileptor, Generic2DOscillator
- [ ] RWW (reduced Wong-Wang), GHB (if no TVB equivalent, use vbi reference)
- [ ] Bold monitor (Balloon-Windkessel)

### M6 - Deprecation milestones
- [ ] Numba parity → soft-deprecate `vbi/models/numba/`
- [ ] C++ parity → soft-deprecate `vbi/models/cpp/`
- [ ] Remove pytorch, cupy, tvbk in next minor release

---

## Notes

- **No `exec`/`eval` at import time.** Code generation writes `.py` files to
  disk (or uses `type()`) for Numba; for C++ it writes `.hpp` and compiles.
  The generated files are debuggable and diffable.
- **dfun_str uses bare names.** No `np.` or `jnp.` prefix in spec expressions.
  The code generator injects the correct math namespace. Use only: `+`, `-`,
  `*`, `/`, `**`, `exp`, `log`, `sin`, `cos`, `pi`, `sqrt`, `tanh`, `abs`.
- **Ring buffer is the critical shared abstraction.** Get it right in NumPy
  first; test delay correctness before any JIT backend.
- **TVB is reference, not runtime.** Production imports never touch TVB.
- **Backends are opt-in.** Missing optional deps (JAX, CUDA toolkit) raise
  `ImportError` with an install hint, not a hard crash at package import.
