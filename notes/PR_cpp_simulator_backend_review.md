# PR Review: `vbi.simulator` C++ Backend

Reviewed: 2026-05-27

Scope:
- `vbi/simulator/backend/cpp/simulator.py`
- `vbi/simulator/backend/cpp/sweeper.py`
- `vbi/simulator/backend/cpp/codegen.py`
- `vbi/simulator/backend/cpp/build.py`
- `vbi/simulator/backend/cpp/_src/*.mako`
- `vbi/simulator/backend/cpp/_src/runtime.hpp`
- C++ validation tests under `vbi/tests/validation/`

## Current Status

Follow-up review after the committed corrections is included at the end of this file. In short:

- Most original high/medium findings have been addressed.
- Focused C++ validation passes:
  - `test_cpp_sweep.py`: 21 passed
  - `test_mpr_cpp.py` + `test_cpp_models.py`: 19 passed
- Remaining items are narrower:
  - `raw` can still be decimated in mixed-monitor sweeps when a decimated monitor comes first.
  - Pipeline `t_cut` is still applied before monitor post-processing.
  - Default cache writes can fail in read-only home directories.
  - CMake still hardcodes native/fast-math flags.
  - Build cache writes are still not concurrency-safe.

## Summary

The C++ backend has a strong deterministic core. It compiles model-specific code, validates against NumPy across several models, releases the GIL during simulation, and has useful sweep coverage for serial, parallel, batching, and pipeline paths. The focused C++ validation suites pass locally.

The main risks are not in the generated C++ arithmetic itself; they are in the Python wrapper/sweeper contract around monitors, stochastic sweep seed semantics, parameter validation, and build/cache robustness. The highest-priority item is sweep monitor correctness: in several C++ sweep paths raw or decimated raw arrays are labeled as processed monitor outputs.

## Verification Run

Commands run:

```bash
python3 -m pytest vbi/tests/validation/test_mpr_cpp.py -q
python3 -m pytest vbi/tests/validation/test_cpp_models.py -q
python3 -m pytest vbi/tests/validation/test_cpp_sweep.py -q
```

Results:
- `test_mpr_cpp.py`: 3 passed, 1 warning
- `test_cpp_models.py`: 16 passed, 1 warning
- `test_cpp_sweep.py`: 21 passed, 1 warning

These tests give good coverage for deterministic C++ vs NumPy raw trajectories, stochastic smoke tests, delayed MPR, serial/parallel sweep consistency, and some pipeline feature checks. They do not fully cover monitor semantic parity in C++ sweeps.

## What Looks Good

- The generated deterministic integration loop is simple and easy to audit: coupling, stimulus injection, integration, bounds, history update, and recording happen in a clear order.
- The C++ extension releases the GIL while running the simulation, so `ThreadPoolExecutor` can provide real parallelism for sweep workloads.
- The cache key includes a template hash, which avoids reusing stale binaries after template changes.
- The backend has helpful prerequisite checks and error messages for missing `mako`, `pybind11`, or a compiler.
- Existing validation covers several equation structures: MPR, Jansen-Rit, Wilson-Cowan, Reduced Wong-Wang, and Wong-Wang Exc/Inh.
- Delayed coupling is covered against NumPy for MPR.

## Findings

### 1. High: C++ sweeps can return processed monitor names backed by unprocessed raw data

Locations:
- `vbi/simulator/backend/cpp/sweeper.py:137`
- `vbi/simulator/backend/cpp/sweeper.py:191`
- `vbi/simulator/backend/cpp/sweeper.py:270`

`CppSweeper` chooses `record_every` from the first monitor period before calling C++:

```python
record_every = max(1, round(spec.monitors[0].period / dt))
```

Then no-pipeline mode calls `_apply_monitor()` on the already-decimated raw output. For `tavg`, this means the temporal average is computed over the decimated samples, not over the original integration steps. If `period=1.0` and `dt=0.01`, C++ records every 100 steps, then `_apply_monitor("tavg")` sees `dt_raw=1.0` and `istep=1`, so the output is effectively a subsample, not a temporal average.

This is a silent semantic bug because the returned key is still `"tavg"` and the shape can look correct.

Recommendation:
- For no-pipeline sweep mode, record every integration step when any requested monitor requires post-processing over raw steps (`tavg`, `bold`, possibly multi-monitor combinations).
- Or implement monitor-aware C++ recording paths that compute `tavg`/`gavg`/`bold` inside C++.
- Add C++ sweep parity tests comparing `CppSweeper(...).run_serial()` to individual `CppSimulator.run()` outputs for `tavg`, `gavg`, `bold`, and multi-monitor specs.

### 2. High: C++ pipeline mode passes raw data under `pipeline.signal`

Location: `vbi/simulator/backend/cpp/sweeper.py:284`

Pipeline mode constructs:

```python
monitor_result = {pipeline.signal: (raw_times, raw)}
feat_labels, feat_vals = pipeline.extract(monitor_result)
```

If `pipeline.signal == "tavg"`, the feature pipeline receives raw or decimated raw state arrays, not the true `tavg` monitor output. The current pipeline tests only use statistical features on one model and do not prove general monitor parity.

Recommendation:
- Build the real monitor result before `pipeline.extract()`, using the same path as no-pipeline mode after fixing recording resolution.
- If C++ pipeline mode is intended to support only raw-like signals, reject other signals clearly with `NotImplementedError`.
- Add tests for `pipeline.signal="tavg"` and `pipeline.signal="bold"` comparing C++ feature values against NumPy for deterministic runs.

### 3. High: Multi-monitor sweeps are controlled by only the first monitor

Locations:
- `vbi/simulator/backend/cpp/sweeper.py:137`
- `vbi/simulator/backend/cpp/sweeper.py:267`

`record_every` is derived from `spec.monitors[0]`, but post-processing loops over all monitors. This creates order-dependent behavior:

- If the first monitor is `raw`, later `tavg` and BOLD have enough raw data.
- If the first monitor is `tavg`, a later `raw` monitor receives decimated raw data.
- If monitors have different periods, later monitors are computed from the first monitor's sampling grid.

Recommendation:
- For correctness-first behavior, record every integration step in sweep mode whenever multiple monitors are requested.
- Add a multi-monitor test with `("raw", "tavg")` and the reverse order to prove order independence.
- Add a test with two different periods to verify each monitor respects its own period.

### 4. Medium: `SweepSpec.same_noise` is ignored by C++ sweeps

Location: `vbi/simulator/backend/cpp/sweeper.py:151`

The C++ sweeper always derives per-run seeds using:

```python
noise_seed=spec.integrator.noise_seed + seed_offset
```

That forces independent stochastic streams even when `SweepSpec.same_noise=True`. The NumPy backend now documents and implements:

- `same_noise=True`: all runs share the base seed.
- `same_noise=False`: each run uses `base_seed + run_index`.

Recommendation:
- Respect `self.sweep.same_noise` in `_build_run_args()`.
- Use `seed_offset=0` when `same_noise=True`.
- Keep `seed_offset=i` when `same_noise=False`.
- Update `test_stochastic_sweep_unique_trajectories()` to explicitly set `same_noise=False`.
- Add a matching `same_noise=True` test with identical parameter sets that produce identical trajectories.

### 5. Medium: Sweeping `G` fails when `G` is not a model parameter

Locations:
- `vbi/simulator/backend/cpp/sweeper.py:31`
- `vbi/simulator/backend/cpp/sweeper.py:126`

Single-run C++ supports `G` through `get_G(spec)`, including `node_params["G"]` even when the model does not declare a `G` parameter. Sweep mode, however, patches only rows in `model.param_names`:

```python
idx = model_param_names.index(name)
```

For models where `G` is an external coupling scalar rather than a model parameter, `SweepSpec(params={"G": ...})` raises `ValueError`.

Recommendation:
- Treat `"G"` as a special sweepable coupling scalar when absent from `model.param_names`.
- This likely requires passing per-run `coup_a` into `_run_one()` rather than only patching `params_flat`.
- Add a regression test for sweeping `G` on a model that does not define `G` in `model.param_names`.

### 6. Medium: Node parameter shape validation silently takes the first element

Location: `vbi/simulator/backend/cpp/codegen.py:263`

`build_params_array()` silently broadcasts `float(val.flat[0])` for any invalid shape:

```python
else:
    arr[i, :] = float(val.flat[0])
```

This hides errors such as wrong-length arrays, `(n_nodes, 1)` arrays, or accidental parameter grids passed as node parameters. `get_G()` also takes `flat[0]`, so invalid `G` shapes are silently accepted.

Recommendation:
- Accept only scalar, shape `(1,)`, or shape `(n_nodes,)`.
- Raise `ValueError` for all other shapes, including the parameter name and expected shapes.
- Apply the same validation to `get_G()`.
- Add negative tests for wrong-length and multidimensional parameter arrays.

### 7. Medium: Noise amplitude shape validation is incomplete

Location: `vbi/simulator/backend/cpp/codegen.py:288`

`get_noise_data()` converts `noise_nsig` to an array and indexes `eff_amp[k]` for each noise-enabled state variable. A scalar `noise_nsig` or wrong-length vector can fail with an unhelpful `IndexError`, or worse, be misinterpreted before reaching C++.

Recommendation:
- Normalize scalar `noise_nsig` to a vector of length `len(model.noise_indices)`.
- Accept only shape `(len(noise_indices),)`.
- Raise `ValueError` with a clear message otherwise.
- Add scalar, correct vector, wrong-length vector, and multidimensional noise tests for the C++ backend.

### 8. Medium: C++ generated bindings do not validate input array sizes

Location: `vbi/simulator/backend/cpp/_src/bindings.cpp.mako:21`

The pybind wrapper accepts typed arrays but does not check sizes before passing pointers into `run_simulation()`. The normal Python wrappers pass correct arrays, but a user or internal caller can call the generated module directly with wrong sizes and cause out-of-bounds reads.

Recommendation:
- Add explicit size checks in `bindings.cpp.mako` for:
  - `initial_state.size() == n_sv * n_nodes`
  - `weights.size() == n_nodes * n_nodes`
  - `idelays.size() == n_nodes * n_nodes`
  - `params.size() == n_params * n_nodes`
  - `noise_data.size() == 0 or n_steps * n_sv * n_nodes`
  - `stim_data.size() == 0 or n_steps * n_cvar * n_nodes`
- Raise `py::value_error` with the expected and actual size.

### 9. Medium: C++ cache and build directories are not concurrency-safe

Locations:
- `vbi/simulator/backend/cpp/build.py:111`
- `vbi/simulator/backend/cpp/build.py:127`
- `vbi/simulator/backend/cpp/build.py:155`

Two processes building the same spec can write to the same cache directory and compile the same extension simultaneously. This can corrupt intermediate build files or load a partially built `.so`.

Recommendation:
- Add a file lock per cache key around source generation and compilation.
- Write sources to a temporary build directory and atomically move/copy the final `.so` into place.
- Add a stress test or small multiprocessing test that calls `build_or_load()` concurrently for the same spec.

### 10. Low: Direct compiler fallback uses `-march=native` and `-ffast-math`

Location: `vbi/simulator/backend/cpp/build.py:232`

The direct build path uses:

```python
"-O3", "-march=native", "-ffast-math"
```

This can reduce portability of cached binaries and may change floating-point semantics relative to NumPy. It is acceptable for a performance backend, but should be explicit and configurable.

Recommendation:
- Make optimization flags configurable through an environment variable such as `VBI_CPP_CXXFLAGS`.
- Consider defaulting direct builds to `-O3` only, with `-march=native` and `-ffast-math` opt-in.
- Add the effective compiler flags to verbose build output and possibly generated metadata.

### 11. Low: CMake template is not included in the template hash

Location: `vbi/simulator/backend/cpp/build.py:99`

The cache key hashes `sim_module.cpp.mako` and `bindings.cpp.mako`, but not `cmake_template.mako` or `runtime.hpp`. Changes to compiler configuration or runtime helper code may not invalidate cached binaries.

Recommendation:
- Include all build-affecting templates and headers in the template hash:
  - `sim_module.cpp.mako`
  - `bindings.cpp.mako`
  - `cmake_template.mako`
  - `runtime.hpp`
- Add a small unit test or helper test around the hash inputs if the build module grows.

### 12. Low: Expression translation leaves unknown names for C++ to catch

Location: `vbi/simulator/backend/cpp/codegen.py:78`

Unknown names are returned unchanged:

```python
return name  # unknown - let C++ catch it
```

That defers model expression mistakes to a compiler error, which is slower and less user-friendly than a Python-side validation error.

Recommendation:
- Raise `ValueError` for unknown names unless they are explicitly allowlisted.
- Include the model name, state variable, expression, and unknown symbol.
- Add negative tests with unsupported expressions or unknown identifiers.

## Suggested Test Additions

Priority tests:

1. `CppSweeper` no-pipeline `tavg` equals individual `CppSimulator` `tavg` for each parameter set.
2. `CppSweeper` pipeline with `pipeline.signal="tavg"` equals NumPy feature values on a signal where temporal averaging changes values measurably.
3. Multi-monitor order-independence: `("raw", "tavg")` and `("tavg", "raw")`.
4. `same_noise=True` and `same_noise=False` stochastic sweep behavior.
5. Sweep `"G"` for a model where `G` is not a model parameter.
6. Invalid node parameter shape and invalid `G` shape.
7. Invalid `noise_nsig` shapes.
8. Direct generated module input-size validation.
9. Concurrent `build_or_load()` for the same spec.

## Recommended Fix Order

1. Fix C++ sweep monitor semantics by recording enough raw data or computing monitors inside C++.
2. Fix pipeline mode to pass true monitor outputs to `FeaturePipeline`.
3. Respect `SweepSpec.same_noise`.
4. Harden parameter and noise shape validation.
5. Add generated binding size checks.
6. Improve cache invalidation and concurrent build safety.
7. Make direct compiler optimization flags configurable.

## Overall Assessment

The generated C++ simulation core is in good shape and already has stronger validation coverage than many accelerated backends. The remaining work should focus on making C++ sweeps contract-compatible with NumPy for monitors and stochastic settings. Once sweep monitor semantics are fixed, this backend will be much safer to use for SBI feature-generation workloads where silent signal differences can invalidate training data.

## Follow-Up Review After Corrections

Reviewed after committed corrections: 2026-05-27

### Verification Run

Commands run:

```bash
python3 -m pytest vbi/tests/validation/test_cpp_sweep.py -q
python3 -m pytest vbi/tests/validation/test_mpr_cpp.py vbi/tests/validation/test_cpp_models.py -q
```

Results:
- `test_cpp_sweep.py`: 21 passed, 1 warning
- `test_mpr_cpp.py` + `test_cpp_models.py`: 19 passed, 1 warning

Additional targeted check:
- `CppSweeper` with monitors ordered as `(subsample(period=1.0), raw)` returns `raw` shape `(10, 2, 4)` for a 10 ms run at `dt=0.01`.
- Individual `CppSimulator` raw output for the same run returns `(1000, 2, 4)`.
- This confirms the remaining multi-monitor order issue below.

### Resolved Items

The following review findings appear addressed:

1. `tavg`/BOLD sweep monitor semantics are improved by forcing `record_every=1` whenever a monitor kind is in `{"tavg", "bold"}`.
2. Pipeline mode now resolves a matching monitor spec and calls `_apply_monitor()` before feature extraction.
3. `SweepSpec.same_noise` is now respected: `same_noise=True` uses seed offset `0`, `same_noise=False` uses per-run offsets.
4. Sweeping external `"G"` is supported when `"G"` is not in `model.param_names`, by recalculating per-run `coup_a`.
5. Node parameter shapes now raise `ValueError` unless scalar, `(1,)`, or `(n_nodes,)`.
6. `noise_nsig` shape handling now supports scalar broadcast and raises clear errors for invalid shapes.
7. Generated bindings now validate array sizes before passing pointers to C++.
8. Cache invalidation now includes `cmake_template.mako` and `runtime.hpp`.
9. Unknown model-expression names/functions now fail on the Python side with `ValueError`.
10. Direct compiler fallback can read `VBI_CPP_CXXFLAGS`.

### Remaining Findings

#### 1. Medium: Multi-monitor sweeps still make `raw` order-dependent

Location: `vbi/simulator/backend/cpp/sweeper.py:148`

`_record_every()` records every step only when any monitor is `tavg` or `bold`. That fixes stateful/windowed monitors, but it does not protect `raw` when the first monitor is a decimated monitor:

```python
monitors=(MonitorSpec("subsample", period=1.0), MonitorSpec("raw"))
```

In this case, `record_every` becomes `period / dt`, and the `"raw"` result is decimated. The monitor key says `"raw"`, but the data is not raw.

Recommendation:
- If any requested monitor has `kind == "raw"`, use `record_every=1`.
- Add a regression test where monitor orders `(raw, subsample)` and `(subsample, raw)` both produce full raw output.

#### 2. Medium: Pipeline `t_cut` is applied before monitor post-processing

Location: `vbi/simulator/backend/cpp/sweeper.py:165`

Pipeline mode passes `t_cut_steps` into C++ and then applies monitor post-processing to the already-cut raw output. This is fine for raw-like signals, but it can diverge from NumPy for monitors whose state or windows should be computed from the full trajectory before pipeline burn-in is discarded:

- `bold`: Balloon-Windkessel state starts at `t_cut` instead of integrating from simulation start.
- `tavg`: windows are realigned to `t_cut`; if `t_cut` is not an exact multiple of the averaging period, values can differ from NumPy.

Recommendation:
- In pipeline mode, run C++ with `t_cut_steps=0`, apply the monitor normally, then let `FeaturePipeline.extract()` discard burn-in using the monitor time array.
- Add parity tests for `pipeline.signal="bold"` and for `tavg` with a non-period-aligned `t_cut`.

#### 3. Low: Default C++ cache path can fail in restricted environments

Location: `vbi/simulator/backend/cpp/build.py:25`

New C++ builds still default to `~/.cache/vbi/cpp`. In this sandbox, a new spec failed with:

```text
OSError: [Errno 30] Read-only file system: '/home/ziaee/.cache/vbi/cpp/.../runtime.hpp'
```

The existing validation passed because already-built cache entries were available. Users can work around this with `VBI_CPP_CACHE=/tmp/...`, but the backend could provide a clearer failure or fallback.

Recommendation:
- Catch cache write failures and raise `CppBackendUnavailable` mentioning `VBI_CPP_CACHE`.
- Optionally fall back to a writable temp directory.

#### 4. Low: CMake path still hardcodes native/fast-math flags

Location: `vbi/simulator/backend/cpp/_src/cmake_template.mako:10`

`VBI_CPP_CXXFLAGS` affects the direct compiler fallback, but the CMake path still uses:

```cmake
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -ffast-math")
```

Recommendation:
- Thread configurable flags into the CMake template as well, or document that `VBI_CPP_CXXFLAGS` applies only to direct builds.

#### 5. Low: Build cache is still not concurrency-safe

Location: `vbi/simulator/backend/cpp/build.py:111`

The cache hash is improved, but there is still no per-key build lock. Two processes building the same uncached spec can write and compile in the same directory concurrently.

Recommendation:
- Add a file lock around `_write_sources()` and `_compile()`.
- Prefer building into a temporary directory and atomically publishing the final `.so`.

### Updated Priority

The high-risk original issues are mostly resolved. The remaining priority is:

1. Fix `raw` in mixed-monitor sweeps by recording full resolution whenever `raw` is requested.
2. Move pipeline `t_cut` handling after monitor post-processing for parity with NumPy.
3. Improve cache write errors/fallback.
4. Make CMake flags configurable or document the direct-build-only behavior.
5. Add a build lock for concurrent cache writes.
