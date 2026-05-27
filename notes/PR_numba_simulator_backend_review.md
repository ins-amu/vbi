# PR Review: `vbi.simulator` Numba Backend

Reviewed: 2026-05-27

Scope:
- `vbi/simulator/backend/numba_/simulator.py`
- `vbi/simulator/backend/numba_/sweeper.py`
- `vbi/simulator/backend/numba_/codegen.py`
- `vbi/simulator/backend/numba_/_nb_sim.py`
- Numba validation tests under `vbi/tests/validation/`

## Summary

The Numba backend has a solid core for deterministic/stochastic single-run simulation and parallel sweeps. The generated `dfun`, static JIT kernels, delay ring buffer, bounds handling, stimulation injection, and raw/tavg single-run parity are covered by useful validation tests. The existing Numba validation suites pass.

The remaining concerns are mostly API and semantic parity issues with the NumPy backend. The highest-risk items are BOLD monitor mismatch, sweep monitor post-processing being skipped, and stochastic sweep seed semantics ignoring `same_noise`. These can produce silently different scientific outputs while still returning plausible shapes.

## Verification Run

Commands run:

```bash
python3 -m pytest vbi/tests/validation/test_mpr_numba.py -q
python3 -m pytest vbi/tests/validation/test_new_models_numba.py -q
python3 -m pytest vbi/tests/validation/test_wilson_cowan_numba.py vbi/tests/validation/test_jansen_rit_numba.py -q
```

Results:
- `test_mpr_numba.py`: 19 passed, 1 warning
- `test_new_models_numba.py`: 54 passed, 1 warning
- `test_wilson_cowan_numba.py` + `test_jansen_rit_numba.py`: 24 passed, 1 warning

These tests confirm broad execution coverage, but they do not currently cover the monitor and sweep parity cases listed below.

## Findings

### 1. High: Numba BOLD monitor advances Balloon-Windkessel dynamics in milliseconds instead of seconds

Location: `vbi/simulator/backend/numba_/simulator.py:74`

`_apply_monitor("bold", ...)` computes `dt_raw` from `raw_times`, where times are in milliseconds, then calls:

```python
bw_state = _bw_step(bw_state, neural, dt_raw, _BW_DEFAULTS)
```

The NumPy `BoldMonitor` explicitly passes seconds:

```python
self._bw = _bw_step(self._bw, neural, self.dt * 1e-3, self._bw_p)
```

Because `_bw_step()` models BW ODE time constants in seconds, the Numba BOLD path currently integrates about 1000x too fast. This will distort BOLD amplitude and timing while still returning arrays of the expected shape.

Recommendation:
- Change the Numba BOLD call to pass `dt_raw * 1e-3`.
- Add a Numba-vs-NumPy BOLD parity test with the same deterministic simulation, same `tr`, and a duration long enough to produce at least one BOLD sample.

Suggested test:
- Build the same model/spec with `backend="numpy"` and `backend="numba"`.
- Use `MonitorSpec(kind="bold", tr=...)`.
- Compare time arrays exactly or with `np.testing.assert_allclose`.
- Compare data with a tolerance appropriate for numerical ordering differences.

### 2. High: Numba BOLD output shape differs from the NumPy backend

Location: `vbi/simulator/backend/numba_/simulator.py:88`

The Numba BOLD path returns:

```python
np.stack(data_bold)
```

Each `data_bold` item is shape `(n_nodes,)`, so the final output is `(n_samples, n_nodes)`. The NumPy `BoldMonitor` appends `bold[np.newaxis, :]`, so its output is `(n_samples, 1, n_nodes)`.

This breaks the monitor contract used by other monitor types, where data is consistently `(n_time, n_voi, n_nodes)`. Downstream feature extraction or user code that handles NumPy and Numba interchangeably will need backend-specific branches.

Recommendation:
- Append `bold[np.newaxis, :].copy()` in the Numba path.
- Return an empty BOLD array with shape `(0, 1, n_nodes)` if keeping empty-output behavior.
- Add an explicit shape assertion in the BOLD parity test.

### 3. High: Numba sweeps do not apply monitor post-processing when `pipeline is None`

Location: `vbi/simulator/backend/numba_/sweeper.py:273`

When no feature pipeline is configured, `NumbaSweeperCPU.run()` wraps raw recorded arrays under the first monitor kind:

```python
mon_kind = self.spec.monitors[0].kind if self.spec.monitors else "raw"
return [{mon_kind: (t, raw[i])} for i in range(n_samples)]
```

This is only correct for `raw` in a narrow case. For `tavg`, `gavg`, `subsample`, or `bold`, the returned data has not gone through the corresponding monitor logic. For example, a `tavg` monitor receives decimated raw states labeled as `"tavg"` rather than true temporal averages.

This is risky because the shape can look plausible, so downstream code may not notice that the values have different semantics.

Recommendation:
- Reuse `_apply_monitor()` from the Numba single-run path for each monitor when returning no-pipeline sweep results.
- For memory/performance, a first fix can be correctness-first: record enough raw data and post-process in Python per sample.
- Later optimization can add JIT monitor-specific sweep paths.
- Add tests comparing no-pipeline Numba sweeps against a loop of single-run Numba simulations for `raw`, `tavg`, `gavg`, and `bold`.

### 4. High: Tier-1 pipeline extraction can receive raw data under a processed monitor name

Location: `vbi/simulator/backend/numba_/sweeper.py:285`

The Tier-1 pipeline path constructs:

```python
monitor_result = {pipeline.signal: (t_i, raw[i])}
feat_labels_i, feat_vals_i = pipeline.extract(monitor_result)
```

If `pipeline.signal` is `"tavg"` or `"gavg"`, the extractor receives raw or decimated raw data, not the processed monitor output that the signal name promises. That can change mean/std/FC/FCD features without an obvious error.

Recommendation:
- Before calling `pipeline.extract()`, construct a real monitor result matching `pipeline.signal`.
- If the Numba backend intentionally supports only raw-signal Tier-1 pipelines, validate that `pipeline.signal == "raw"` and raise a clear `NotImplementedError` otherwise.
- Add a test where the same sweep/pipeline is run through NumPy and Numba for `pipeline.signal="tavg"` and compares feature values for a small deterministic model.

### 5. Medium: Stochastic Numba sweeps ignore `SweepSpec.same_noise`

Location: `vbi/simulator/backend/numba_/sweeper.py:145` and `vbi/simulator/backend/numba_/sweeper.py:238`

Stochastic sweep seeds are always generated as:

```python
seeds = np.arange(n_samples, dtype=np.int64) + base_seed
```

That forces different noise streams for every sample. If `SweepSpec.same_noise=True`, the expected behavior is to reuse the same noise stream across parameter sets so parameter effects can be compared under common random numbers.

Recommendation:
- Respect `self.sweep.same_noise`.
- For `same_noise=True`, fill all seeds with `base_seed`.
- For `same_noise=False`, use `base_seed + np.arange(n_samples)`.
- Add two tests:
  - With identical parameter sets and `same_noise=True`, stochastic outputs should match.
  - With identical parameter sets and `same_noise=False`, stochastic outputs should differ.

### 6. Medium: Sweeping injected `G` can fail when `G` is not a model parameter

Location: `vbi/simulator/backend/numba_/sweeper.py:102`

The Numba backend can inject `G` into the packed parameter matrix when the model does not define a `G` parameter:

```python
if self._G_idx < 0:
    ...
    self._G_idx = self._params.shape[0] - 1
```

But sweep parameter indices are resolved only against `spec.model.param_names`:

```python
[model_param_names.index(n) for n in sweep_names]
```

If a sweep includes `"G"` for a model where `G` was injected rather than declared, initialization raises `ValueError` even though single-run simulation supports the injected `G`.

Recommendation:
- Resolve sweep names against the actual packed parameter layout, including the injected `G` row.
- If `G` sweeping is intentionally unsupported for these models, raise a clear backend-specific error before constructing the index list.
- Add a regression test for sweeping `G` on a model whose `ModelSpec.param_names` does not include `G`.

### 7. Medium: Parameter shape validation silently takes the first element for invalid arrays

Locations:
- `vbi/simulator/backend/numba_/codegen.py:129`
- `vbi/simulator/backend/numba_/simulator.py:122`
- `vbi/simulator/backend/numba_/sweeper.py:66`

`build_params()` accepts scalars and per-node vectors, but for any other shape it silently broadcasts `val.flat[0]`:

```python
else:
    result[i, :] = float(val.flat[0])
```

The same pattern is used when injecting `G`. This can hide user mistakes such as shape `(n_nodes, 1)`, wrong-length arrays, or multi-dimensional parameter grids accidentally passed as node parameters.

Recommendation:
- Accept only scalars, shape `(1,)`, or shape `(n_nodes,)`.
- Raise `ValueError` for any other shape, with the parameter name and expected shapes in the message.
- Apply the same validation to injected `G`.
- Add negative tests for invalid parameter shapes.

### 8. Medium: Noise amplitude shape validation is weaker than the current NumPy backend contract

Location: `vbi/simulator/backend/numba_/codegen.py:193`

`get_noise_params()` converts `noise_nsig` to an array but does not validate that it matches the number of noise-enabled state variables. The Numba loop later indexes `eff_noise_amp[noise_j]` for each noisy variable:

```python
dW[sv_i] = eff_noise_amp[noise_j] * np.random.randn(n_nodes)
```

Wrong shapes will either fail inside JIT code with an obscure error or silently produce unintended broadcasting if the shape happens to be accepted by Numba.

Recommendation:
- Normalize `noise_nsig` to a 1D array.
- Accept scalar by broadcasting to `len(model.noise_indices)`.
- Accept shape `(len(noise_indices),)`.
- Raise `ValueError` otherwise.
- Add Numba-specific tests for scalar, correct vector, wrong vector length, and multidimensional noise amplitudes.

### 9. Medium: Empty-output behavior differs from NumPy monitors

Locations:
- `vbi/simulator/backend/numba_/simulator.py:56`
- `vbi/simulator/backend/numba_/simulator.py:88`

For short durations, the Numba monitor post-processing returns empty arrays for `tavg` and BOLD. The NumPy monitors now raise descriptive `ValueError`s when no samples are collected.

Recommendation:
- Match NumPy behavior and raise `ValueError` for no-sample monitor outputs.
- Include the monitor kind, requested duration/period/tr, and backend in the error message.
- Add short-duration parity tests for `tavg`, `subsample`, `gavg`, and `bold`.

### 10. Low: Single-run Numba records every state variable at every step before monitor processing

Location: `vbi/simulator/backend/numba_/simulator.py:163`

For single-run simulation, Numba records every step and every state variable, then applies monitors in Python. This is simple and helps monitor parity, but it can be memory-heavy for long simulations or large connectomes, especially when only a low-rate monitor is requested.

Recommendation:
- Keep the correctness-first implementation until monitor parity is fixed.
- After that, consider monitor-aware recording for `raw`/`subsample`/`gavg`/`tavg`, or document the memory tradeoff.
- Add a note in backend docs if single-run Numba is not intended for very long raw-memory workloads.

### 11. Low: Supported coupling limitations need explicit parity tests or documentation

Locations:
- `vbi/simulator/backend/numba_/simulator.py:109`
- `vbi/simulator/backend/numba_/sweeper.py:54`

The Numba backend supports `linear` and `kuramoto` coupling, and raises `NotImplementedError` for other coupling kinds. That is acceptable if intentional, but the public simulator API should make the limitation obvious.

Recommendation:
- Document this in backend docs or simulator backend selection docs.
- Add a negative test that verifies unsupported coupling kinds raise a clear `NotImplementedError`.
- If `sigmoidal` coupling is a supported NumPy backend feature expected by users, add a roadmap item or explicit skipped parity test.

### 12. Low: Generated code cache location can fail in locked-down environments

Location: `vbi/simulator/backend/numba_/codegen.py:23`

The generated `dfun` module is written to `~/.cache/vbi/numba` unless `VBI_NB_CACHE` is set. This is a reasonable default for local use, but it can fail in read-only home directories, restricted containers, or test environments with unusual permissions.

Recommendation:
- Catch cache directory write failures and raise a clear error mentioning `VBI_NB_CACHE`.
- Consider falling back to a temporary directory when persistent cache is unavailable.
- Add a small test around `VBI_NB_CACHE` override if practical.

## Suggested Test Additions

Priority tests to add next:

1. BOLD Numba-vs-NumPy parity for time labels, shape `(n_time, 1, n_nodes)`, and values.
2. No-pipeline Numba sweep with `tavg`, verifying it equals single-run Numba monitor output per parameter set.
3. Tier-1 pipeline with `pipeline.signal="tavg"` comparing Numba and NumPy feature values.
4. Stochastic sweep `same_noise=True` and `same_noise=False` behavior.
5. Invalid node parameter shape and invalid injected `G` shape errors.
6. Invalid `noise_nsig` shape errors.
7. Unsupported coupling negative path.

## Recommended Fix Order

1. Fix BOLD `dt` units and output shape.
2. Make Numba sweep monitor outputs truthful by applying monitor post-processing before returning or extracting features.
3. Respect `SweepSpec.same_noise` in both raw and inline-feature stochastic sweep paths.
4. Harden parameter and noise validation.
5. Align empty-output errors with NumPy monitors.
6. Expand parity and negative-path tests.

## Overall Assessment

The backend is close to being a useful accelerated path, and the computational core is already covered well enough to catch many basic regressions. The remaining work should focus on preventing silent semantic drift from the NumPy backend. In particular, monitors and sweeps should either produce the same contract as NumPy or reject unsupported combinations clearly.
