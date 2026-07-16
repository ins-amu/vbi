# Milestone: TVB Hybrid Simulator Backend Integration

## Status

Planning only. No implementation started. This note captures the design
so the work can be picked up later without re-deriving context.

## Goal

Let `vbi.simulator.Simulator` / `Sweeper` dispatch to the TVB **hybrid**
simulator (from the local fork at
`/home/ziaee/git/tvb-root-hybrid-cpp/tvb_library`, package `tvb-library`,
importable as `tvb`) as an additional backend, instead of only vbi's own
`numpy` / `numba` / `cpp` / `cuda` / `jax` backends.

The user-facing result should stay consistent with the existing pattern:

```python
sim = Simulator(spec, backend="tvb-numba")
result = sim.run(duration=1000.0)
```

Changing backend should still mean changing one string, matching the
project's [[feedback_api_simplicity]] principle — the TVB path should not
require the caller to learn TVB's own object model (`NetworkSet`,
`Subnetwork`, `Stim`, …).

## Why

TVB's hybrid fork already re-implements several models/cfuns with
multiple execution backends of its own:

- `ReferenceBackend` — plain Python/NumPy loop (`tvb/simulator/backend/ref.py`)
- `NbHybridBackend` — Numba-compiled (`tvb/simulator/backend/nb_hybrid.py`)
- `CppHybridBackend` — generated C++ extension (`tvb/simulator/backend_cpp/backend.py`)
- a CUDA sweep backend (`tvb/simulator/backend/nb_hybrid_cuda_sweep_backend.py`)

This overlaps with vbi's own backend set, but is a separate implementation
with its own validated numerics (coupling functions, connectivity/delay
conventions, monitors). Wiring it in as a backend gives:

- a way to cross-validate vbi models against TVB's reference implementation
  without hand-rolled comparison scripts (see `notes/milestones.md` TVB
  parity section, currently deferred),
- a path for users who already have TVB `Connectivity`/model definitions to
  reuse them directly inside vbi inference workflows.

## Current State

- `vbi.simulator.backend.base.load_backend(name)` / `load_sweep_backend(name)`
  are simple string-keyed registries returning a class that satisfies
  `AbstractBackend` (`build(spec)`, `run(duration)`). See
  `vbi/simulator/backend/base.py:13`. Adding a backend is a matter of adding
  another branch plus a new implementation module.
- `SimulationSpec` (`vbi/simulator/spec/simulation.py`) bundles model,
  integrator, coupling, monitors, weights, node params — this is vbi's own
  spec object, distinct from TVB's `Connectivity` / `Model` / `Coupling`
  classes.
- `vbi/models/tvbk/` already wraps a *different* TVB-adjacent project
  (`tvbk`, a compiled MPR kernel) behind a `DeprecationWarning`-guarded
  optional import — it is not the same code path as the hybrid fork and
  should not be conflated with it.
- `tvb-library` is currently installed as an editable dev dependency
  pointing at the fork, not a pinned/released version. The fork is a
  personal branch (`tvb-root-hybrid-cpp`), not upstream TVB.

## Design Options

1. **Adapter backend module** (recommended): add
   `vbi/simulator/backend/tvb_/` with `build(spec)` translating
   `SimulationSpec` → TVB's `Subnetwork` + `NetworkSet`, and `run(duration)`
   dispatching to the requested TVB execution backend. Register under
   backend names `"tvb-reference"`, `"tvb-numba"`, `"tvb-cpp"` (mirrors TVB's
   own `ReferenceBackend` / `NbHybridBackend` / `CppHybridBackend`), or a
   single `"tvb"` name with a secondary `tvb_backend=` kwarg on `Simulator`.
   Keeps `load_backend` as the single dispatch point and requires no changes
   to `SimulationSpec` itself.
2. **Separate top-level class** (`vbi.simulator.TVBSimulator`): avoids
   stretching `AbstractBackend` to cover TVB's richer run/compile/snapshot
   API (`backend.compile(ns)` → `compiled.run(...)`), but breaks the
   "change one string" UX and duplicates the `Simulator`/`Sweeper` surface.

Leaning toward option 1 for user-facing consistency, falling back to
option 2 only if TVB's compile/chunked-run model turns out to not fit
`build`/`run` cleanly (needs a spike to confirm).

## Implementation Tasks (draft — not started)

1. **Spec translation layer**
   - Map `SimulationSpec.model` (vbi `ModelSpec`) to a TVB `Model` instance.
     Only feasible for models that exist on both sides (MPR first, since
     it already has a partial TVB comparison harness per `notes/milestones.md`).
   - Map `weights` / `tract_lengths` / `dt` to a TVB `Connectivity`.
   - Map `CouplingSpec` to a TVB `Coupling` (`Linear`, `Sigmoidal`,
     `SigmoidalJansenRit`, …) — note TVB cfun parameter names differ
     (`_CFUN_PARAM_ATTRS` in `backend_cpp/backend.py`), needs an explicit
     name-mapping table per model.
   - Map `MonitorSpec` to TVB monitors (`TemporalAverage`, raw, …).
   - Decide behavior when a vbi spec feature has no TVB equivalent (raise
     `NotImplementedError` with a clear message, not a silent fallback).

2. **Backend dispatch**
   - Add `tvb_` package under `vbi/simulator/backend/` with `simulator.py`
     (and `sweeper.py` if sweeps are in scope) per execution backend.
   - Register in `load_backend` / `load_sweep_backend`
     (`vbi/simulator/backend/base.py`), following the existing
     `try/except ImportError` pattern used for `numba`/`jax` so a missing
     `tvb-library` install fails with an actionable message rather than an
     import traceback.
   - `tvb-library` should be an optional extra (e.g. `pip install vbi[tvb]`),
     not a hard dependency — it currently only exists as a local editable
     fork, so packaging/versioning needs a decision before this ships
     (vendor a pinned fork commit? wait for it to be published?).

3. **Model coverage**
   - Start with MPR only (has partial validation precedent).
   - Jansen-Rit next, once [[milestone_jansen_rit_tvb_coupling]] coupling
     work lands — the two milestones should agree on the same TVB-style
     equation/parameter convention so comparisons are apples-to-apples.
   - Explicitly out of scope initially: models with no TVB equivalent.

4. **Sweeper support**
   - Investigate whether TVB's own CUDA sweep backend
     (`nb_hybrid_cuda_sweep_backend.py`) can back `Sweeper(..., backend="tvb-cuda")`
     directly, or whether it needs its own adapter layer distinct from the
     single-run backend.

## Validation Plan

- Parity tests analogous to the deferred `test_mpr_numpy_vs_tvb` sketch in
  `notes/milestones.md` (~line 1097), but promoted to a real test using the
  new backend instead of a hand-built TVB simulator in the test file:
  ```
  sim_vbi = Simulator(spec, backend="numpy")
  sim_tvb = Simulator(spec, backend="tvb-reference")
  ```
  and assert trajectories match within tolerance for a small network.
- Cross-backend parity within TVB itself (reference vs numba vs cpp) is
  already covered by the fork's own tests
  (`tvb/tests/library/simulator/backend*/`) — vbi's tests only need to
  check the *translation layer*, not re-validate TVB's internal backends.
- Skip cleanly (`pytest.mark.skipif`) when `tvb-library` / the fork path is
  not installed, matching the existing optional-dependency test pattern
  (see `vbi/tests/test_optional_imports.py`).

## Open Questions

- Is the target dependency the fork (`tvb-root-hybrid-cpp`) specifically,
  or should this wait until the hybrid backend work merges upstream into
  `tvb-library` proper? Affects packaging/versioning above.
- Single `"tvb"` backend name with a `tvb_backend=` kwarg, vs multiple
  explicit names (`"tvb-reference"`, `"tvb-numba"`, `"tvb-cpp"`)? Multiple
  names are more consistent with vbi's existing flat `backend=` string
  convention; a kwarg avoids polluting the top-level name registry with
  TVB-internal naming. Needs a decision before implementation.
- How much of TVB's stimulus (`StimuliRegion`) and delay/connectivity
  convention needs to be exposed vs. hidden behind vbi's own
  `Connectivity`/`stimulus.py` specs?

## Acceptance Criteria (once implemented)

- `Simulator(spec, backend="tvb-<x>")` runs at least the MPR model and
  returns results in the same `{monitor_kind: (times, data)}` shape as
  every other backend.
- Missing `tvb-library` produces a clear `ImportError`, not an unrelated
  traceback.
- A parity test demonstrates vbi numpy backend and TVB reference backend
  agree on MPR trajectories within tolerance.
- `graphify update .` run after edits.

---

## Downstream: Inference Workflow API (plan only, not started)

Separate from the TVB backend work above. The SBI/inference workflows
(`docs/examples/workflows/jr_vbi.py`, `mpr_sbi.py`) currently call
`Simulator`/`Sweeper` with a `backend=` string directly inside
`VBIInference`/`simulate_for_vbi_sweep_cached`. Before or alongside adding
the `"tvb-*"` backends, the inference workflow API likely needs a look:

- Whether backend selection should stay a plain string passed straight
  through to `Simulator`/`Sweeper`, or needs structure once TVB backends
  add a second axis (execution backend *and*, potentially, TVB-specific
  run options like compile/chunk size).
- Whether `VBIInference` needs to know about backend-specific caveats
  (e.g. a TVB backend that can't cleanly support `Sweeper` yet, per the
  open question above) rather than surfacing a raw import error mid-run.
- General cleanup of the inference workflow API is an open task on its
  own, independent of TVB — no concrete design yet, revisit when picking
  this up.

No action items here yet — just a flag that touching the inference
workflow API is expected work adjacent to this milestone, not a
prerequisite for it.
