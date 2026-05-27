# Review: VBI Simulator NumPy Backend

## Scope

Reviewed the pure NumPy simulator backend and its immediate API/spec/test surface:

- `vbi/simulator/backend/numpy_/simulator.py`
- `vbi/simulator/backend/numpy_/integrators.py`
- `vbi/simulator/backend/numpy_/monitors.py`
- `vbi/simulator/backend/numpy_/coupling.py`
- `vbi/simulator/backend/numpy_/history.py`
- `vbi/simulator/backend/numpy_/sweeper.py`
- `vbi/simulator/api.py`
- `vbi/simulator/spec/*`
- validation tests under `vbi/tests/validation`

This backend is described as the reference implementation for correctness validation, so the review prioritizes deterministic semantics, edge-case behavior, reproducibility, and contract clarity.

## Summary

The NumPy backend is compact and readable. The main simulator loop is easy to follow, and the tests already cover important TVB alignment cases for Wilson-Cowan, Jansen-Rit, MPR, monitors, delays, coupling, stochastic reproducibility, and sweep consistency. The strongest parts are:

- clear separation of model specs, coupling, integration, monitoring, and sweep orchestration;
- direct TVB comparison tests for key deterministic models;
- explicit delay-buffer test documenting the TVB history convention;
- reproducibility tests for direct stochastic simulator runs;
- good coverage of core monitor shapes and basic multi-node behavior.

The remaining issues are mostly around edge cases and API consistency. The highest-impact issue is that `NumpySweeper` drops `stimuli` when patching specs, so sweep runs can silently differ from direct simulator runs for stimulated simulations.

## Findings

### 1. `NumpySweeper` drops stimuli from the base simulation spec

**Severity:** High

**File:** `vbi/simulator/backend/numpy_/sweeper.py:21`

`_patch_spec()` reconstructs a `SimulationSpec` but does not pass `stimuli=base.stimuli`:

```python
patched = SimulationSpec(
    model=base.model,
    integrator=new_integrator,
    coupling=base.coupling,
    monitors=base.monitors,
    weights=base.weights,
    tract_lengths=base.tract_lengths,
    speed=base.speed,
    node_params=new_node_params,
)
```

As a result, any sweep using a stimulated base spec silently runs without stimulation. This breaks consistency between:

- `Simulator(stimulated_spec).run(...)`
- `Sweeper(stimulated_spec, sweep_spec, backend="numpy").run(...)`

It is especially risky because stimulation tests exist for direct simulation, but not for sweeps.

**Suggested correction:**

```python
patched = SimulationSpec(
    model=base.model,
    integrator=new_integrator,
    coupling=base.coupling,
    monitors=base.monitors,
    weights=base.weights,
    tract_lengths=base.tract_lengths,
    speed=base.speed,
    node_params=new_node_params,
    stimuli=base.stimuli,
)
```

**Suggested tests:**

- Add a sweep test with a non-zero `StimSpec` and one parameter set.
- Compare sweep output/features against a direct `Simulator` run with the same patched parameter.
- Add a zero-stimulus sweep test to verify no-stim and zero-stim remain identical.

### 2. Monitor `result()` crashes when no samples were collected

**Severity:** Medium

**Files:**
- `vbi/simulator/backend/numpy_/monitors.py:93`
- `vbi/simulator/backend/numpy_/monitors.py:112`
- `vbi/simulator/backend/numpy_/monitors.py:142`
- `vbi/simulator/backend/numpy_/monitors.py:162`
- `vbi/simulator/backend/numpy_/monitors.py:191`

All monitors call `np.stack(self._data)` unconditionally. This raises a low-level `ValueError` if no samples were collected. Cases that can trigger this:

- `duration == 0`;
- `duration < period` for `TemporalAverageMonitor`;
- `duration < tr` for `BoldMonitor`;
- malformed monitor periods that round in surprising ways.

This behavior is awkward for pipelines and sweeps because a short run fails at result assembly rather than returning a well-defined empty output or raising a clear validation error earlier.

**Suggested correction:**

- Decide on one contract:
  - return empty arrays with the correct trailing shape; or
  - raise a clear `ValueError` with monitor kind, duration, period/tr, and dt.
- Prefer explicit validation in `NumpySimulator.run()` or monitor configuration for configurations guaranteed to produce no samples.

**Suggested tests:**

- `MonitorSpec("tavg", period=10.0)` with `duration=1.0`.
- `MonitorSpec("bold", tr=2000.0)` with `duration=100.0`.
- `duration=0.0` for `raw`.

### 3. Repeated `Simulator.run()` calls are stateful but time resets to zero

**Severity:** Medium

**Files:**
- `vbi/simulator/backend/numpy_/simulator.py:160`
- `vbi/simulator/backend/numpy_/simulator.py:176`
- `vbi/simulator/backend/numpy_/monitors.py:89`

`NumpySimulator.run()` mutates `_state`, `_history`, and the monitor buffers. If a caller invokes `run()` twice on the same `Simulator` instance, the second run continues from the previous final state, but monitor times start again at `step * dt` from zero and old monitor data remains in the monitor buffers.

That gives ambiguous behavior:

- it is not a clean rerun from the initial state;
- it is not a clean continuation because time stamps restart at zero;
- returned monitor data includes previous samples plus new samples.

**Suggested correction:**

- Choose and document one behavior:
  - `run()` is single-use and raises if called twice;
  - `run()` resets simulator state and monitors before each run;
  - `run()` supports continuation and maintains an absolute simulation clock.
- For a reference backend, explicit single-use or reset semantics are usually less surprising than implicit continuation.

**Suggested tests:**

- Call `sim.run(10.0)` twice and assert the chosen behavior.
- If continuation is intended, assert strictly increasing times across calls.
- If reset is intended, assert two runs with the same spec produce identical data.

### 4. `SweepSpec.same_noise` is ignored by the NumPy sweeper

**Severity:** Medium

**Files:**
- `vbi/simulator/backend/numpy_/sweeper.py:10`
- `vbi/simulator/spec/sweep.py:24`

`SweepSpec` exposes `same_noise`, but `NumpySweeper` always reuses the base `IntegratorSpec` unchanged for every parameter set. For stochastic simulations, that means every sweep run gets the same `noise_seed`, regardless of `same_noise`.

This may be intentional for variance reduction, but the API exposes a switch that NumPy does not honor. The docstring currently says `same_noise` is JAX-backend only, but the examples also imply independent noise can be requested with `same_noise=False`.

**Suggested correction:**

- Either explicitly document that NumPy always uses the base `noise_seed`, and reject `same_noise=False` for NumPy with a clear error.
- Or implement NumPy support:
  - `same_noise=True`: keep the base seed for all runs;
  - `same_noise=False`: derive per-run seeds deterministically from `base_seed + i` or a master RNG.

**Suggested tests:**

- Stochastic sweep with duplicate parameters and `same_noise=True` should produce identical feature rows.
- Stochastic sweep with duplicate parameters and `same_noise=False` should produce different feature rows.

### 5. Noise amplitude shape is not validated

**Severity:** Medium

**Files:**
- `vbi/simulator/backend/numpy_/simulator.py:151`
- `vbi/simulator/backend/numpy_/integrators.py:16`

When stochastic integration is enabled, `noise_nsig` is converted to an array and passed through as `self._noise_amp`. `_additive_white_noise()` then assumes:

```python
noise_amp.shape == (n_noise_vars,)
```

Problems:

- scalar `noise_nsig` becomes a 0-D array, so `noise_amp[:, np.newaxis]` fails;
- a length mismatch gives a broadcasting error inside the integrator;
- a model with no noise variables and stochastic integration can fail obscurely depending on `noise_nsig`.

**Suggested correction:**

- Normalize scalar `noise_nsig` to all noisy variables if scalar support is desired.
- Validate that `noise_nsig.shape == (len(model.noise_indices),)`.
- Raise a clear `ValueError` during `build()`, before the integration loop.

**Suggested tests:**

- scalar `noise_nsig` behavior, either accepted and broadcast or rejected clearly;
- wrong-length `noise_nsig`;
- stochastic integrator with a model that has no `noise_variables`.

### 6. Monitor period/tr validation is deferred to runtime errors

**Severity:** Medium

**Files:**
- `vbi/simulator/backend/numpy_/monitors.py:100`
- `vbi/simulator/backend/numpy_/monitors.py:120`
- `vbi/simulator/backend/numpy_/monitors.py:149`
- `vbi/simulator/backend/numpy_/monitors.py:169`

For `subsample`, `tavg`, and `gavg`, `spec.period` is used directly:

```python
self.istep = max(1, round(spec.period / dt))
```

If `period` is `None`, zero, negative, or smaller than intended relative to `dt`, the backend either crashes with a Python type error or silently coerces to `istep=1`. `tr` has the same issue for BOLD.

**Suggested correction:**

- Validate in `MonitorSpec.__post_init__()` or `build_monitor()`:
  - `period` is required and positive for `subsample`, `tavg`, and `gavg`;
  - `tr` is positive for `bold`;
  - optionally warn or error if `period / dt` is non-integer beyond tolerance.

**Suggested tests:**

- `MonitorSpec("tavg")` should fail with a clear message.
- `MonitorSpec("subsample", period=0.0)` should fail.
- `MonitorSpec("bold", tr=0.0)` should fail.

### 7. Public `ModelSpec.dfun_str` is executed as Python code

**Severity:** Medium

**File:** `vbi/simulator/backend/numpy_/simulator.py:18`

`build_dfun()` uses `exec()` to compile model derivative expressions. The comment says the strings are not user-supplied, but `ModelSpec` is public and users can create custom models.

That is acceptable for a trusted local modeling framework, but it should be treated as an explicit trust boundary. Without validation, arbitrary Python code can be embedded in `dfun_str`.

**Suggested correction:**

- Document that `ModelSpec.dfun_str` is trusted code.
- Or add AST validation to allow only safe expression nodes, known symbols, indexing, arithmetic, and approved math functions.
- Keep `exec()` for speed if desired, but validate before compiling.

**Suggested tests:**

- Invalid symbol should fail with a clear error.
- Disallowed expression should be rejected before `exec()`.

### 8. Time labels may be misleading for post-step samples

**Severity:** Low

**Files:**
- `vbi/simulator/backend/numpy_/simulator.py:192`
- `vbi/simulator/backend/numpy_/monitors.py:89`
- `vbi/simulator/backend/numpy_/monitors.py:107`

The simulator integrates first, then monitors sample the updated state using `step * dt` as the time label. Therefore the sample labeled `t=0` is the state after the first integration step, not the initial state.

The TVB comparison tests suggest this may match the intended convention. If so, it should be documented clearly because downstream users commonly interpret raw monitor time zero as the initial condition.

**Suggested correction:**

- Document the convention in `Simulator.run()` and monitor docstrings.
- Consider labeling post-step samples as `(step + 1) * dt` if TVB compatibility does not require current labels.
- Add a regression test that locks the chosen convention.

### 9. BOLD monitor output shape differs from other monitor outputs

**Severity:** Low

**File:** `vbi/simulator/backend/numpy_/monitors.py:186`

Most monitors return data shaped like:

```text
(n_samples, n_voi, n_nodes)
```

`BoldMonitor` appends `bold.copy()` with shape `(n_nodes,)`, so its result is:

```text
(n_samples, n_nodes)
```

This is tested today, so it may be intentional. The inconsistency should still be documented because feature pipelines and user code often assume monitor outputs are rank-3.

**Suggested correction:**

- Document the BOLD shape explicitly.
- Or return `bold[np.newaxis, :]` so BOLD follows `(n_samples, 1, n_nodes)`.

### 10. Specification shape validation is light

**Severity:** Low

**Files:**
- `vbi/simulator/spec/simulation.py:36`
- `vbi/simulator/backend/numpy_/simulator.py:111`
- `vbi/simulator/backend/numpy_/coupling.py:17`

The backend assumes:

- `weights` is square;
- `tract_lengths.shape == weights.shape`;
- `node_params` values are scalars or `(n_nodes,)`;
- `speed > 0`;
- delay steps are non-negative;
- state variable defaults broadcast cleanly to `(n_nodes,)`.

Invalid inputs will usually fail later through NumPy broadcasting or indexing errors.

**Suggested correction:**

- Add validation in `SimulationSpec.__post_init__()` or `NumpySimulator.build()`.
- Raise clear errors with parameter names and expected shapes.

## Test Results

Focused tests run locally:

```bash
python3 -m pytest \
  vbi/tests/validation/test_mpr_numpy.py \
  vbi/tests/validation/test_sweep_numpy.py -q
# 27 passed, 1 warning in 38.79s

python3 -m pytest \
  vbi/tests/validation/test_wilson_cowan_numpy.py \
  vbi/tests/validation/test_jansen_rit_numpy.py -q
# 6 passed, 1 warning in 2.74s
```

## Suggested Follow-Up Plan

1. Fix `NumpySweeper._patch_spec()` to preserve `stimuli`.
2. Add clear monitor edge-case handling for empty outputs.
3. Decide and test `Simulator.run()` repeat-call semantics.
4. Validate stochastic noise shapes during `build()`.
5. Clarify or implement `same_noise` behavior for NumPy sweeps.
6. Add spec-level validation for common shape and value errors.
7. Document the trusted-code nature of `dfun_str` or add AST validation.

## Notes For Future Review

- The NumPy backend is a reference backend, so correctness and clarity should be prioritized over aggressive optimization.
- TVB compatibility is already tested for several deterministic models; keep those tests as the anchor when changing time labels, coupling timing, or history semantics.
- If NumPy remains the behavioral source of truth for faster backends, every edge-case contract fixed here should have equivalent backend consistency tests for Numba/JAX/C++ where applicable.

## Follow-Up Review After Corrections

Reviewed commit:

```text
d2a278a feat: Enhance monitors with sample validation and reshape output; update simulation spec validation
```

The main issues from the first review are mostly addressed:

- `NumpySweeper._patch_spec()` now preserves `stimuli`.
- `NumpySweeper` now implements `same_noise=False` by deriving per-run seeds.
- monitor `result()` methods now raise clearer errors when no samples are collected.
- BOLD output is now rank-3: `(n_samples, 1, n_nodes)`.
- `NumpySimulator.run()` now resets state and monitor buffers for repeatable reruns.
- stochastic `noise_nsig` shape is validated and scalar values broadcast to noise variables.
- `SimulationSpec` now validates square `weights`, matching `tract_lengths`, and positive `speed`.
- `build_dfun()` now documents the `dfun_str` trust boundary.

### 11. New correction paths are not yet covered by tests

**Severity:** Medium

**Files:**
- `vbi/simulator/backend/numpy_/sweeper.py`
- `vbi/simulator/backend/numpy_/simulator.py`
- `vbi/simulator/backend/numpy_/monitors.py`
- `vbi/simulator/spec/simulation.py`

The commit fixes several important contracts, but the only visible test update is the BOLD shape assertion in `test_mpr_numpy.py`. The new behavior should be locked down so it does not regress.

**Suggested tests:**

- Sweep with `StimSpec` preserves stimulation and matches a direct `Simulator` run.
- Stochastic sweep with duplicate parameters:
  - `same_noise=True` gives identical features;
  - `same_noise=False` gives different features.
- Calling `Simulator(spec).run(duration)` twice returns identical outputs and does not append old monitor samples.
- Empty-output monitor cases raise the intended `ValueError`.
- Scalar `noise_nsig` broadcasts; wrong-length `noise_nsig` raises.
- Invalid `SimulationSpec` inputs raise clear errors:
  - non-square weights;
  - mismatched tract lengths;
  - non-positive speed.

### 12. History step counter is not reset during `run()` reset

**Severity:** Low

**Files:**
- `vbi/simulator/backend/numpy_/simulator.py:197`
- `vbi/simulator/backend/numpy_/history.py`

`NumpySimulator.run()` resets `_state`, refills the history buffer, and reconfigures monitors, but `History.initialize()` does not reset `History._step`. Because the buffer is filled with the initial state, repeated delayed runs are likely still numerically equivalent, but the internal reset is incomplete and makes the reset contract harder to reason about.

**Suggested correction:**

- Add a `History.reset(cvar_state)` method that both fills the buffer and sets `_step = 0`.
- Or update `History.initialize()` to reset `_step`.

**Suggested tests:**

- Run the same delayed simulation twice on one `Simulator` instance and assert identical outputs.
- Add an assertion that the first delayed coupling read after reset uses the initialized state.

### 13. Monitor reset uses `list.index()` inside the monitor loop

**Severity:** Low

**File:** `vbi/simulator/backend/numpy_/simulator.py:198`

The monitor reset loop uses:

```python
for mon in self._monitors:
    mon.configure(spec.monitors[self._monitors.index(mon)], spec.model, spec.integrator.dt)
```

This works today because monitor objects compare by identity, but it is brittle and unnecessarily scans the list on every monitor. If monitor equality is ever implemented, duplicate monitor classes/specs could be reset with the wrong spec.

**Suggested correction:**

```python
for mon, mon_spec in zip(self._monitors, spec.monitors):
    mon.configure(mon_spec, spec.model, spec.integrator.dt)
```

### 14. `SimulationSpec` validation should coerce or reject non-array inputs clearly

**Severity:** Low

**File:** `vbi/simulator/spec/simulation.py:50`

The new validation assumes `weights` and `tract_lengths` already have `.ndim` and `.shape`. If a caller passes Python lists, the error will be an `AttributeError` rather than a clear validation message.

**Suggested correction:**

- Convert `weights` and `tract_lengths` with `np.asarray(..., dtype=np.float64)` in `__post_init__()`.
- Or explicitly reject non-NumPy inputs with a clear `TypeError`.

### 15. Negative tract lengths are still accepted

**Severity:** Medium

**File:** `vbi/simulator/spec/simulation.py:74`

`tract_lengths` shape is validated, but values are not. Negative tract lengths produce negative delay steps, which are non-physical and can make ring-buffer indexing semantics surprising.

**Suggested correction:**

- Add `np.any(self.tract_lengths < 0)` validation in `SimulationSpec.__post_init__()`.
- Raise `ValueError("tract_lengths must be non-negative")`.

### 16. BOLD empty-output message is off by one under current sampling convention

**Severity:** Low

**File:** `vbi/simulator/backend/numpy_/monitors.py:215`

BOLD samples only when:

```python
step > 0 and step % self.tr_steps == 0
```

For `duration == tr`, the loop runs steps `0..tr_steps-1`, so no BOLD sample is collected. The error message currently says:

```text
duration must be >= tr
```

Under the current post-step sampling convention, it should be `duration > tr`, or the sampling condition should be adjusted if `duration == tr` is intended to emit one sample.

**Suggested correction:**

- Update the message to `duration must be > tr`.
- Or revisit the monitor time convention and sampling boundary.

### 17. Duration and integrator `dt` validation are still implicit

**Severity:** Low

**Files:**
- `vbi/simulator/backend/numpy_/simulator.py:203`
- `vbi/simulator/spec/integrator.py`

`run()` computes:

```python
n_steps = round(duration / dt)
```

There is no explicit validation that `duration >= 0` or `dt > 0`. Invalid values will fail indirectly or produce empty monitor errors.

**Suggested correction:**

- Validate `IntegratorSpec.dt > 0`.
- Validate `duration > 0` or document that zero duration intentionally raises monitor-specific no-sample errors.

## Follow-Up Verification Run

```bash
python3 -m pytest \
  vbi/tests/validation/test_mpr_numpy.py \
  vbi/tests/validation/test_sweep_numpy.py -q
# 27 passed, 1 warning in 38.31s

python3 -m pytest \
  vbi/tests/validation/test_stimulation.py -q
# 13 passed, 3 warnings in 24.00s

python3 -m pytest \
  vbi/tests/validation/test_wilson_cowan_numpy.py \
  vbi/tests/validation/test_jansen_rit_numpy.py -q
# 6 passed, 1 warning in 2.64s
```

## Updated Follow-Up Plan

1. Add tests for the newly fixed contracts, especially stimuli preservation in sweeps and `same_noise`.
2. Replace monitor reset `list.index()` with `zip(self._monitors, spec.monitors)`.
3. Reset `History._step` as part of simulator reset.
4. Add non-negative tract-length validation.
5. Clarify BOLD no-sample boundary and duration/dt validation.
