# Review: Numba-CUDA Simulator Backend

## Scope

Reviewed the Numba-CUDA simulator backend under:

- `vbi/simulator/backend/numba_cuda/simulator.py`
- `vbi/simulator/backend/numba_cuda/sweeper.py`
- `vbi/simulator/backend/numba_cuda/codegen.py`
- `vbi/simulator/backend/numba_cuda/__init__.py`
- `vbi/tests/validation/test_mpr_cuda.py`

This review focuses on backend correctness, API parity with the NumPy/Numba sweepers, pipeline behavior, stochastic behavior, GPU memory behavior, and validation coverage.

## Verification

Ran:

```bash
python3 -m pytest vbi/tests/validation/test_mpr_cuda.py -q -m cuda_fast
```

Result:

```text
43 passed, 9 deselected, 16 warnings in 132.14s
```

The current fast CUDA validation suite passes. The remaining concerns below are mostly semantic/API gaps and missing edge-case coverage rather than failures in the existing CUDA smoke/parity tests.

Warnings observed:

- `pytest.mark.cuda_fast` and `pytest.mark.cuda_slow` are not registered.
- Several CUDA tests launch with grid size 1, which is expected for small validation cases but produces Numba occupancy warnings.

## What Looks Good

- The backend has a clear separation between single-run `CudaSimulator`, sweep-oriented `CudaSweeperGPU`, and generated kernel code in `codegen.py`.
- The CUDA path validates CUDA availability early through `_require_cuda()`.
- Coupling support is explicitly restricted to `"linear"` and `"kuramoto"`, which is better than silently compiling unsupported coupling semantics.
- Dense and CSR sparse connectivity paths are both implemented.
- Delay handling is present in both dense and CSR paths, and the tests include delay parity against the Numba backend.
- Bounds are transferred to the GPU and enforced inside the generated kernel.
- Fast CUDA validation covers deterministic parity against Numba for multiple models, stochastic finite checks, delays, sweeps, bounds, and pipeline shape/finite checks.

## Findings

### 1. High: CUDA sweep pipeline bypasses monitor post-processing

In `CudaSweeperGPU.run()`, pipeline mode passes:

```python
{pipeline.signal: (t_arr, ts[i])}
```

directly into `pipeline.extract()` (`vbi/simulator/backend/numba_cuda/sweeper.py:309-312`).

That means the data is always the recorded state tensor, not the monitor output named by `pipeline.signal`. If `pipeline.signal == "tavg"`, the pipeline receives raw or decimated raw state labeled as `"tavg"`. If `pipeline.signal == "bold"`, the pipeline receives state variables rather than BOLD output.

This is a correctness issue for the main SBI use case because pipeline mode is the primary sweep feature path.

Suggested fix:

- In pipeline mode, build the monitor dictionary the same way as non-pipeline mode before calling `pipeline.extract()`.
- If memory is the concern, add an optimized monitor-specific path later, but first make semantics match the CPU backends.

Suggested regression test:

- Create a pipeline with `signal="tavg"` and compare CUDA sweep features against NumPy or Numba sweep features for the same deterministic spec.
- Add a test where the selected monitor changes output shape relative to raw state, so this cannot pass accidentally.

### 2. High: Sweep monitor decimation is based only on the first monitor

`CudaSweeperGPU.run()` chooses one `record_period` from `spec.monitors[0].period` (`sweeper.py:186-189`) and records the GPU output at that period before applying monitors on the host (`sweeper.py:295-303`).

This breaks several valid monitor configurations:

- If the first monitor is `raw` and a later monitor is `tavg`, memory usage is full raw resolution, but the result may still work.
- If the first monitor has a long period and a later monitor needs raw resolution, the later monitor receives already-decimated data.
- If the monitor is `tavg`, `_apply_monitor()` receives decimated samples with `dt_raw == monitor.period`; its internal averaging no longer sees the original time steps.
- Multi-monitor output becomes order-dependent.

The single-run CUDA simulator avoids this by recording all steps and then applying monitors (`simulator.py:104-116`, `simulator.py:164-166`). The sweep backend should preserve the same semantics.

Suggested fix:

- Record full resolution whenever any requested monitor requires full resolution post-processing.
- Or implement monitor computation in-kernel/on-device per monitor, but ensure each monitor gets the correct sampling semantics.
- At minimum, treat `tavg`, `bold`, and multi-monitor specs conservatively by recording raw resolution before host monitor processing.

Suggested regression tests:

- `monitors=[tavg]`: CUDA sweep result should match Numba/NumPy sweep result.
- `monitors=[raw, tavg]` and `monitors=[tavg, raw]`: monitor outputs should not depend on ordering.
- `monitors=[bold]`: CUDA sweep BOLD features should match CPU backend within tolerance.

### 3. High: Pipeline burn-in is applied before monitor computation

In pipeline mode, `t_cut_step` is sent into the CUDA kernel (`sweeper.py:191-193`, `sweeper.py:267-268`), so the recorded output starts after burn-in. Since monitor post-processing is bypassed in pipeline mode, this is currently part of the same bug as Finding 1.

Even after fixing Finding 1, applying burn-in before monitor computation can be wrong for stateful monitors:

- `bold` needs state history before the extracted time interval.
- `tavg` window alignment can change depending on whether the cut happens before or after averaging.

Suggested fix:

- Match the CPU backend contract: run/record enough data for monitor state, compute monitor outputs, then apply feature extraction on the intended post-cut signal.
- If pipeline-level `t_cut` is intentionally before monitor extraction, document that clearly and align all backends.

Suggested regression test:

- Compare CUDA and Numba pipeline outputs with a nonzero `SweepSpec.t_cut` and `pipeline.signal="tavg"` or `"bold"`.

### 4. Medium: Per-node model parameters are collapsed to scalar means

`build_params_matrix()` converts every model parameter override to a scalar mean:

```python
base[i] = float(val.mean())
```

(`vbi/simulator/backend/numba_cuda/codegen.py:396-400`)

Likewise, coupling scale `G` is read as a mean in both `CudaSimulator` and `CudaSweeperGPU` (`simulator.py:78-86`, `sweeper.py:133-142`).

This means node-specific parameter arrays are silently flattened to one scalar value. That is a parity gap if NumPy/Numba/JAX support per-node `node_params`.

Suggested fix:

- Either explicitly reject non-scalar node parameters in the CUDA backend with a clear `NotImplementedError`, or extend the generated kernel to accept `(n_params, n_nodes, n_samples)` or a separate node-parameter matrix.
- Do not silently average user-provided heterogeneity.

Suggested regression test:

- Use a deterministic model with a per-node parameter array and compare CUDA against NumPy/Numba.
- Add a negative test if CUDA intentionally does not support this yet.

### 5. Medium: Sweeping external coupling parameter `G` is silently ignored

`CudaSweeperGPU` computes `_coup_a` once from the base spec during initialization (`sweeper.py:133-142`). `build_params_matrix()` only overwrites sweep names that are present in `spec.model.param_names` (`codegen.py:403-407`).

For common sweep specs like:

```python
SweepSpec(params={"G": values})
```

the sweep has no effect unless `G` is also a model parameter. If `G` is intended as a coupling/global gain parameter, the CUDA backend silently runs identical coupling for every sample.

Suggested fix:

- Decide whether `G` belongs to `node_params`, model parameters, or coupling parameters in the simulator API.
- If `G` is supported as a sweep parameter in CPU backends, CUDA should pass per-sample coupling scale to the kernel.
- If unsupported, raise `ValueError` for sweep names that the CUDA backend cannot apply.

Suggested regression test:

- Sweep `G` with deterministic dynamics and assert outputs differ across values and match the CPU backend.

### 6. Medium: Unknown sweep parameters are silently ignored

In `build_params_matrix()`, unknown `sweep_names` are skipped:

```python
if name in pnames:
    arr[pnames.index(name), :] = sweep_sets[:, j]
```

(`codegen.py:403-407`)

This makes typo failures difficult to detect. A sweep over `"eta"` works, but a sweep over `"etta"` produces repeated base simulations without error.

Suggested fix:

- Validate `sweep_spec._param_names_list` in `CudaSweeperGPU.__init__()`.
- Accept only supported model parameters and explicitly supported backend-level parameters such as `G`.
- Raise a `ValueError` listing unsupported names.

Suggested regression test:

- `SweepSpec(params={"not_a_param": [1.0, 2.0]})` should raise before launching CUDA.

### 7. Medium: `SweepSpec.same_noise` is ignored by the CUDA backend

`SweepSpec.same_noise` documents shared noise as the default behavior for JAX and notes independent noise for non-JAX examples. The CUDA backend never reads `same_noise`; it always calls:

```python
generate_noise(spec, n_steps, n_samples, self._seed_base)
```

(`sweeper.py:275-277`)

`generate_noise()` then uses `seed_base + sample_idx` for each sample (`codegen.py:503-509`), so each sweep sample receives independent noise.

This may be acceptable as a documented backend limitation, but it should not be silent if the user explicitly requests `same_noise=True`.

Suggested fix:

- Either implement shared-noise generation when `self.sweep.same_noise` is true, or document and warn/raise that CUDA currently uses independent noise.
- Align the `SweepSpec` documentation with actual backend behavior.

Suggested regression tests:

- `same_noise=True`: two identical parameter sets should produce identical stochastic trajectories.
- `same_noise=False`: two identical parameter sets should produce different stochastic trajectories.

### 8. Medium: GPU output memory can still OOM after warning

`CudaSweeperGPU.run()` estimates `ts_out` size and warns if it exceeds 80% of free GPU memory (`sweeper.py:214-227`), but then always allocates:

```python
ts_out_h = np.zeros((n_record, n_sv, n_nodes, n_samples), dtype=np.float32)
ts_out_d = cuda.to_device(ts_out_h)
```

(`sweeper.py:230-241`)

For long sweeps with raw output or many samples, this can still fail hard with CUDA allocation errors. The single-run simulator also always records every step (`simulator.py:104-116`), which is correct but can be memory-heavy for long raw runs.

Suggested fix:

- Convert the warning into a configurable hard error before allocation, or support chunked execution across sweep samples.
- For pipeline mode, avoid storing all raw states when the pipeline only needs summary features, once monitor semantics are fixed.
- Consider allocating the device array directly instead of zero-initializing a full host array and copying it to device.

Suggested regression test:

- Add a host-side sizing test for `_count_records` and estimated memory behavior.
- Add a test that chunked execution produces the same features as one-shot execution once chunking exists.

### 9. Medium: Generated CUDA module cache is hardcoded to `~/.cache/vbi/cuda`

`codegen.py` defines:

```python
_CACHE_DIR = Path.home() / ".cache" / "vbi" / "cuda"
```

(`vbi/simulator/backend/numba_cuda/codegen.py:51`)

There is no environment override and no fallback to a temporary directory. This can fail in read-only home directories, restricted containers, CI jobs with unusual home permissions, or shared clusters.

Suggested fix:

- Add `VBI_CUDA_CACHE_DIR` or reuse a project-wide cache configuration.
- Fall back to `tempfile.gettempdir()` if the preferred cache is not writable.
- Consider including Python, Numba, CUDA, and model metadata in cache invalidation if compatibility issues appear.

Suggested regression test:

- Unit test `build_cuda_module()` cache path selection with a temporary cache directory.

### 10. Medium: Expression translation is regex-based and fails late

`_cuda_expr()` performs simple regex substitutions for math functions (`codegen.py:54-77`). Unsupported names or unsupported expression constructs are only caught when generated code is imported or when Numba compiles the kernel.

This is manageable for internal model specs, but it is fragile for a codegen boundary.

Suggested fix:

- Validate model expressions before generating CUDA code.
- Prefer AST-based validation/translation if the model expression language grows.
- Raise a clear error naming the model, state variable, expression, and unsupported symbol.

Suggested regression test:

- A model spec with an unsupported function should fail with a helpful `ValueError`, not a deep CUDA/Numba compile traceback.

### 11. Low: Invalid `connectivity` values silently select dense mode

`CudaSweeperGPU.__init__()` treats any `connectivity` value other than `"auto"` or `"sparse"` as dense (`sweeper.py:104-110`). `CudaSimulator.build()` similarly maps non-`"auto"` values to `connectivity == "sparse"` (`simulator.py:54-58`).

Suggested fix:

- Validate `connectivity in {"auto", "dense", "sparse"}` and raise `ValueError` otherwise.

Suggested regression test:

- `connectivity="spares"` should raise `ValueError`.

### 12. Low: CUDA pytest marks are not registered

The test run reports:

```text
PytestUnknownMarkWarning: Unknown pytest.mark.cuda_fast
PytestUnknownMarkWarning: Unknown pytest.mark.cuda_slow
```

Suggested fix:

- Add these markers to `pytest.ini` or `pyproject.toml`.

Example:

```ini
[pytest]
markers =
    cuda_fast: fast CUDA validation tests
    cuda_slow: slow CUDA validation and throughput tests
```

## Missing Test Coverage

Recommended additions:

- CUDA sweep `tavg` parity against Numba/NumPy.
- CUDA sweep `bold` parity or at least shape/value sanity against Numba/NumPy.
- Pipeline value parity, not only shape/finite checks.
- Pipeline with nonzero `t_cut`.
- Multi-monitor order independence.
- `same_noise=True` and `same_noise=False` semantics.
- Unknown sweep parameter raises.
- Sweep over global coupling `G`.
- Per-node parameter behavior: parity if supported, explicit error if not.
- Dense versus sparse parity for the same deterministic spec.
- Invalid `connectivity` argument raises.
- CUDA codegen cache directory override.
- Host-only unit tests for `build_params_matrix()`, `generate_noise()`, `to_csr()`, `_count_records()`, and cache path behavior. These would protect much of the CUDA backend even on machines without a GPU.

## Recommended Fix Order

1. Fix pipeline semantics so `pipeline.extract()` receives real monitor output.
2. Fix sweep monitor recording/post-processing so `tavg`, `bold`, and multi-monitor configurations match CPU backends.
3. Validate sweep parameter names and decide explicit support for global `G`.
4. Decide whether CUDA supports per-node parameter heterogeneity; reject clearly if not.
5. Implement or explicitly reject `SweepSpec.same_noise=True`.
6. Add CUDA parity tests for pipeline values, monitor semantics, `G`, and unknown parameters.
7. Improve memory behavior through chunked sweeps or a hard preallocation guard.
8. Add cache configurability and connectivity validation.

## Overall Assessment

The Numba-CUDA backend is in good shape for the covered deterministic raw-output and selected sweep cases. The fast CUDA suite passing is a good sign.

The main remaining risk is that the sweeper does not yet preserve the same monitor and pipeline semantics as the CPU backends. Since sweeps and pipelines are the expected SBI data-generation path, I would treat Findings 1 and 2 as release blockers before advertising the CUDA backend as a drop-in replacement for NumPy/Numba sweeps.
