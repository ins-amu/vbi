# Feature Extraction Milestones - `vbi/feature_extraction/`

> **Context:** The simulator backends (M0–M4) are complete including
> stimulation support across all backends (numpy, numba, C++, JAX, CUDA).
> This document tracks what remains for feature extraction to reach full
> parity across backends and what new capabilities are needed for the
> SBI pipeline.

---

## The workflow bottleneck (why this milestone matters)

**Observed in practice** (building the SBI/inference workflows in
`docs/examples/workflows/`): the simulator backends themselves are fast and
parallel — the sweeper batches/parallelizes the simulation loop (numba
`prange`, C++ threads, CUDA kernel, JAX `vmap`). But once the workflow needs
to go from "raw time series" to "feature vector + parameter row in a
dataframe/file", it falls back to calling `FeaturePipeline.extract()` /
`calc_features()` **once per sample, serially, in plain Python** (Tier-1).
For a sweep of thousands of parameter sets this Python-loop feature step
dominates wall time and erases the speedup the backend gave on the
simulation side — the workflow ends up bottlenecked on serial Python feature
extraction, not on simulation.

Root cause: only the Numba backend has a Tier-2 *inline* path (`nb_extract`
running feature accumulation inside the same `@njit`/`prange` loop as the
simulation, see `features_utils_nb.py`). NumPy, C++, CUDA, and JAX all only
have Tier-1 (simulate everything → materialize full time series → hand off
to Python `FeaturePipeline` in a serial loop). This is exactly what MF2
(more numba inline features), MF3 (JAX vmap inline), MF4 (CUDA inline), and
MF5 (C++ inline) are meant to fix, and MF6 is meant to unify so workflow
code doesn't need to know which tier is active. Until then, any workflow
that needs more than a handful of samples on non-numba backends should
expect feature extraction, not simulation, to be the bottleneck.

---

## Current state (May 2026)

### What exists

| File | Status | Notes |
|------|--------|-------|
| `features_utils.py` | ✅ Complete | 68 functions, full NumPy, covers FC, FCD, power, coherence, etc. |
| `features_utils_nb.py` | ⚠️ Partial | 7 `@njit` functions: `nb_mean`, `nb_std`, `nb_fc_flat`, `nb_fcd_mean`, `nb_extract`, `NbExtractorSpec` - used by Tier-2 numba sweep inline |
| `features_utils_jax.py` | ⚠️ Partial | `get_fc`, `get_fcd` only - not vmap-compatible, not connected to sweeper |
| `features_utils_cuda.py` | ❌ Missing | Not yet created |
| `pipeline.py` | ⚠️ Partial | `FeaturePipeline` works for NumPy only; no backend dispatch; no inline sweep integration for JAX/C++/CUDA |
| `calc_features.py` | ✅ Exists | Legacy wrapper, kept for backward compat |

### What works end-to-end today

- **NumPy sweeper Tier-1**: raw time series → `FeaturePipeline` in Python ✅
- **Numba sweeper Tier-2 (inline)**: `nb_extract` inside `@njit` loop → `(mean, std, fc, fcd_ks)` only ✅
- **All other backends**: Tier-1 only via Python post-processing ✅

### What is missing

- **Serial Python feature extraction is the workflow bottleneck** on every
  backend except Numba Tier-2 — see "The workflow bottleneck" above
- JAX sweeper does not yet feed into `FeaturePipeline` inline (vmap-compatible)
- C++ sweeper has Tier-1 Python post-processing but no inline C++ feature loop
- CUDA sweeper has no inline feature extraction
- `FeaturePipeline` has no backend awareness - always uses NumPy
- Stimulus is not yet supported inside feature pipelines (burn-in with stimulus)
- No `features_utils_cuda.py`
- `features_utils_jax.py` is not vmap-compatible and missing most features
- G, alpha sweeping not yet unified in `CouplingSpec` (separate issue)

---

## MF1 - `FeaturePipeline` backend dispatch  *(numpy, numba)*

**Goal:** `FeaturePipeline` dispatches to the right implementation based on
the backend that produced the data, without changing user code.

```python
# Same call regardless of where data came from
labels, values = pipeline.extract(monitor_result)
```

**Tasks:**

- [ ] Add `backend: str = "numpy"` parameter to `FeaturePipeline`
- [ ] Dispatch `fc`, `fcd_ks`, `mean`, `std` to `features_utils_nb.py` when
      `backend="numba"` (already implemented, just needs wiring)
- [ ] Validate that Numba Tier-1 matches NumPy to rtol=1e-4
- [ ] Tests: `test_pipeline_backend_dispatch.py`

**Effort:** Small - the numba functions already exist.

---

## MF2 - Extend `features_utils_nb.py`  *(numba inline coverage)*

**Goal:** Numba inline sweep (Tier-2) covers more features beyond the current
`(mean, std, fc, fcd_ks)`.

**Tasks:**

- [ ] `nb_band_power(ts, dt, band_lo, band_hi)` - `@njit`, Welch or AR
- [ ] `nb_psd_peak(ts, dt)` - dominant frequency
- [ ] `nb_fcd_variance(fcd_matrix)` - variance of upper triangle
- [ ] `nb_synchrony(ts)` - Kuramoto order parameter R from phase series
- [ ] `nb_autocorr_decay(ts, max_lag)` - first zero-crossing of autocorrelation
- [ ] Extend `NbExtractorSpec` with flags for all new features
- [ ] Extend `nb_extract` to include new features
- [ ] Tests: add to `test_new_models_numba.py` Tier-2 sweep tests

**Effort:** Medium - each function is self-contained `@njit`.

---

## MF3 - `features_utils_jax.py` - vmap-compatible  *(JAX sweeper inline)*

**Goal:** Full set of SBI-relevant features computable inside `jax.vmap` over
the sweep batch.

**Design:** All functions must be pure JAX with no Python loops or side effects.
Input shape: `(n_steps, n_nodes)` per simulation.

**Tasks:**

- [ ] Refactor existing `get_fc`, `get_fcd` to accept batched input
      `(n_samples, n_steps, n_nodes)` with `jax.vmap`
- [ ] `jax_mean(ts)` - spatial/temporal mean
- [ ] `jax_std(ts)` - standard deviation
- [ ] `jax_fc(ts)` → `(n_nodes, n_nodes)` Pearson correlation
- [ ] `jax_fcd_ks(ts, window)` → scalar KS statistic (differentiable approx)
- [ ] `jax_band_power(ts, dt, lo, hi)` - FFT-based, differentiable
- [ ] `jax_synchrony(phases)` - `|mean(exp(i*theta))|`
- [ ] `JaxExtractorSpec` - mirrors `NbExtractorSpec`, flags for vmap batch
- [ ] Wire into `JaxSweeper._build_batch_runner`: inline `vmap` over features
- [ ] Gradient compatibility: all functions must pass `jax.grad` through them
- [ ] Tests: `test_new_models_jax.py` - add inline JAX feature tests

**Effort:** Large. FCD KS requires a differentiable surrogate.
The FFT path is straightforward. The vmap wiring needs care.

---

## MF4 - `features_utils_cuda.py` - CUDA inline  *(CUDA sweeper inline)*

**Goal:** Feature extraction inside the CUDA kernel, avoiding copy-back of
full time series to host memory (the main CUDA bottleneck for large sweeps).

**Design:** Device functions `@cuda.jit(device=True)` accumulate running
statistics as the simulation steps. Output written to a per-sample feature
buffer at end of simulation.

**Architecture:**

```
cuda_sweep_det kernel:
  for step:
    → simulate one step (existing)
    → if step >= t_cut: accumulate running stats
  at end of step loop:
    → compute FC from accumulated sums
    → write feature vector to feat_out[tid]
```

**Tasks:**

- [ ] Create `vbi/feature_extraction/features_utils_cuda.py`
- [ ] `cuda_running_mean(acc, count, x)` - online mean update `@cuda.jit(device=True)`
- [ ] `cuda_running_cov(acc_xx, acc_x, count, x)` - online covariance for FC
- [ ] `cuda_fc_from_cov(cov, mean, n)` - normalize to Pearson r
- [ ] `cuda_fcd_ks(ts_window, ...)` - sliding-window FCD, KS approx
- [ ] `CudaExtractorSpec` - dataclass with flags and window sizes
- [ ] Update `numba_cuda/codegen.py` to generate feature accumulation code
      inside the simulation kernel (alongside the existing ts_out path)
- [ ] Update `CudaSweeper.run()` to accept `CudaExtractorSpec`, skip ts_out,
      return feature matrix directly
- [ ] Tests: `test_mpr_cuda.py` - add inline feature extraction tests
- [ ] Benchmark: compare with Tier-1 (copy-back) path

**Effort:** Large. Requires generating additional CUDA device code and
rethinking the ts_out vs feat_out dual-path in the kernel.

---

## MF5 - C++ inline feature extraction  *(C++ sweeper)*

**Goal:** FC and FCD computed inside the C++ sweep loop, never materializing
the full `(n_samples, n_record, n_sv, n_nodes)` array.

**Current state:** C++ sweeper already has Tier-1 Python post-processing.
The `sim_module.cpp.mako` template already generates the time-series loop.

**Tasks:**

- [ ] Add `features.hpp` runtime header with `compute_fc`, `compute_fcd_ks`
- [ ] Add Mako template section for inline feature accumulation loop
      (activated when `feature_mode=True` in `render_sim_module`)
- [ ] Add `CppExtractorSpec` (mirrors `NbExtractorSpec`)
- [ ] Wire into `CppSweeper.run()` - return `(labels, values)` directly
- [ ] Tests: `test_cpp_sweep.py` - add inline feature tests

**Effort:** Medium. The C++ template already generates loops; adding
running statistics is straightforward.

---

## MF6 - `FeaturePipeline` unified API across all backends

**Goal:** One pipeline object works correctly regardless of backend.

```python
pipeline = FeaturePipeline(features=["fc", "fcd_ks", "mean"], t_cut=500.0)

# Same call - backend detected from spec or passed explicitly
labels, values = sweeper.run(duration, pipeline=pipeline)
```

**Tasks:**

- [ ] `FeaturePipeline.backend` property - auto-detected from sweeper context
- [ ] Backend dispatch table:
  - `numpy` → `features_utils.py`
  - `numba` → `features_utils_nb.py` (Tier-2) or `features_utils.py` (Tier-1)
  - `jax` → `features_utils_jax.py`
  - `cuda` → `features_utils_cuda.py` / CUDA kernel inline
  - `cpp` → C++ inline or Python post-processing
- [ ] Stimulation-aware burn-in: `t_cut` applies after stimulus offset by default
- [ ] `FeaturePipeline.from_config(dict)` - load from YAML/JSON for reproducibility
- [ ] Tests: `test_pipeline.py` - extend with cross-backend consistency tests

**Effort:** Medium design work; small implementation once MF1–MF5 done.

---

## MF7 - SBI integration

**Goal:** End-to-end SBI workflow where `vbi` generates training data and
passes directly to `sbi` or `lampe` without intermediate files.

**Tasks:**

- [ ] `VBISimulator` wrapper for `sbi.utils.simulate_for_sbi` interface
- [ ] `VBISimulator` wrapper for `lampe` / `zuko` interface
- [ ] Batched simulation + feature extraction in one call returning
      `(theta_batch, x_batch)` as tensors
- [ ] Support for PyTorch tensors as direct output (bypass NumPy)
- [ ] Example notebook: SBI with MPR + FC features, 10 000 samples, numba backend
- [ ] Example notebook: gradient-based inference with JAX backend + `jax.grad`

**Effort:** Medium-large. Mostly API glue.

---

## Priority order

```
MF1 (small, unblocks MF6)
  → MF2 (medium, extends numba inline coverage)
  → MF3 (large, JAX vmap + grad path - high value for gradient-based SBI)
  → MF5 (medium, C++ inline)
  → MF4 (large, CUDA inline - highest throughput path)
  → MF6 (medium, unifies everything)
  → MF7 (medium-large, SBI integration)
```

---

## Open design questions

1. **G and alpha in `CouplingSpec`**: currently G lives in `node_params` for
   sweepability. Moving G and alpha to `CouplingSpec` with sweeper support
   would clean up the API (see session discussion). Blocks MF6 API design.

2. **FCD differentiability**: for JAX gradient-based SBI, the KS statistic is
   non-differentiable. Options: soft KS, Wasserstein, or moment-matching of
   the FCD distribution. Blocks MF3 fully differentiable path.

3. **Feature standardization**: should `FeaturePipeline` z-score features
   automatically, or leave it to the inference engine? Affects MF6 and MF7.

4. **Heterogeneous alpha (per-node frustration)**: `StimSpec` supports
   per-node spatial patterns; similarly `alpha` could be per-node if moved to
   `node_params`. Design consistency with stimulus injection.

5. **Multi-model sweeps**: can a single sweep mix models? Currently no - one
   `SimulationSpec` has one model. Relevant for hierarchical brain models.
