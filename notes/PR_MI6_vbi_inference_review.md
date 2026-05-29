# MI6 VBIInference Code Review

Reviewed commits:

- `418e56b` through `18d12e8`
- Main files: `vbi/inference/_vbi_inference.py`, `vbi/inference/_utils.py`,
  `vbi/inference/_prior.py`, `vbi/simulator/spec/simulation.py`,
  `vbi/simulator/spec/connectivity.py`, and MI6 validation tests.

Review stance: API correctness, end-to-end workflow reliability, checkpoint
recoverability, and gaps against the MI6 milestone.

## Summary

The implementation establishes the intended end-to-end shape: `VBIInference`
owns simulation, feature extraction, SNPE training, posterior construction,
checkpointing, config loading, cache extraction, and diagnostic helpers. The
NumPy path with `maf` is covered by useful smoke/integration tests.

The largest risks are around persistence and backend coverage. Checkpointing is
currently only safe for a narrow estimator shape, the optional `sbi` backend has
several broken or misleading paths, and sequential/proposal metadata is lost
outside the immediate in-memory SNPE object. These should be fixed before
calling MI6 complete.

## Follow-up Status: 2026-05-29

Most original findings have been partially or fully addressed. The focused MI6
validation subset now passes:

```bash
python3 -m pytest \
  vbi/tests/validation/test_vbi_inference.py \
  vbi/tests/validation/test_vbi_inference_config.py \
  vbi/tests/validation/test_vbi_inference_diag.py \
  vbi/tests/validation/test_sim_cache.py -q
```

Result:

```text
51 passed, 1 warning in 926.19s
```

`pytest-xdist` is available (`xdist 3.8.0`), so future broad runs can use:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python3 -m pytest vbi -n 8 --lf
```

Current remaining actionable items:

- High: MDN checkpoint load is still broken after unpack because
  `_offdiag_basis` is not rebuilt. A direct pack/unpack reproduction raises
  `AttributeError: 'MDNEstimator' object has no attribute '_offdiag_basis'`.
- High: sequential/proposal metadata is still lost in `VBIInference`;
  `_sim_rounds` stores only `(theta, x)`, `get_simulations()` returns an empty
  proposals list, and save/load flattens all rounds into one.
- Medium: non-default MAF/NSF checkpoint config is still incomplete; load
  preserves `n_flows` but not the rest of the architecture/config fields.
- Medium: `from_config()` still does not pass through
  `inference_backend="sbi"`.
- Medium: cache extraction now validates labels, but still materializes all
  rows in Python lists before stacking.

Resolved or materially improved since the original review:

- `Path` import/config connectivity failure is fixed.
- `sbi` save/load no longer crashes on the old `_snpe.n_simulations` log path,
  and trained `sbi` estimators are saved to a companion `<stem>_sbi.pt` file.
- `sbi` simulation append now converts `theta`/`x` to `torch.float32` and passes
  `proposal` when provided.
- `stimuli` in config is explicitly rejected instead of silently ignored.
- cache re-extraction validates feature-label consistency.
- `pairplot()` now tells users to call `train()` first.

## Findings

### Resolved High: `VBIInference.from_config()` was broken for path-based connectivity

Location: `vbi/simulator/spec/simulation.py`, `SimulationSpec.from_dict()`

`from_dict()` checks `isinstance(conn, (str, Path))`, but `Path` is not
imported in `simulation.py`. Any config that uses the documented string path
form for `sim.connectivity` fails before prior/pipeline parsing:

```text
NameError: name 'Path' is not defined
```

This caused 13 failures in the targeted MI6 validation run, all in
`test_vbi_inference_config.py`.

Recommended fix:

- Add `from pathlib import Path` to `vbi/simulator/spec/simulation.py`.
- Keep the existing config tests; they are correctly catching this.

Follow-up status: fixed. `Path` is imported, and the focused config validation
tests now pass.

### High: Checkpoint load is still broken for `density_estimator="mdn"`

Location: `vbi/inference/_vbi_inference.py`, `_unpack_estimator()`

Original issue: `_unpack_estimator()` always reconstructed an estimator with
`EstCls(n_flows=n_flows)`, which did not fit `MDNEstimator`.

Follow-up status: partially fixed. `_unpack_estimator()` now branches for
MDN-like estimators and restores `n_components` / `hidden_sizes`, but it does
not rebuild MDN internal derived state. A direct pack/unpack reproduction still
fails when the restored estimator is used:

```text
AttributeError: 'MDNEstimator' object has no attribute '_offdiag_basis'
```

Recommended fix:

- Rebuild MDN derived state during unpack, e.g. call
  `_create_offdiag_basis()` after restoring `param_dim`, or add a proper
  estimator-native load hook.
- Save estimator class/config explicitly, not only selected fields.
- Reconstruct by estimator type:
  - MDN: `n_components`, `hidden_sizes`, `param_dim`, `feature_dim`
  - MAF: `n_flows`, `hidden_units`, `num_blocks`, `activation`,
    `z_score_theta`, `z_score_x`, `use_actnorm`, `embedding_dim`, etc.
  - NSF: MAF config plus `num_bins`, `tail_bound`
- Add save/load tests for all supported density estimators: `maf`, `mdn`, `nsf`.

### High: Checkpointing still does not preserve enough estimator configuration

Location: `vbi/inference/_vbi_inference.py`, `_pack_estimator()` /
`_unpack_estimator()`

Follow-up status: partially fixed. The checkpoint now stores enough state for
the default MAF happy path and some MDN fields, but MAF/NSF reconstruction still
uses default constructor values for most architecture and behavior settings. If
the user trains with non-default architecture or behavior, the loaded estimator
can silently differ from the trained estimator. Examples:

- MAF: `hidden_units`, `num_blocks`, `activation`, `use_actnorm`,
  `embedding_dim`, `actnorm_eps`
- NSF: all MAF fields plus `num_bins`, `tail_bound`
- MDN: `n_components`, `hidden_sizes`
- Training/inference metadata: loss history, validation history, best epoch,
  collapse monitor state
- Embedding network structure if `embedding_net` was used

Because weights are restored into an object whose config may no longer match,
this can become either a runtime shape error or a worse silent behavior change.

Recommended fix:

- Store a JSON-like `est__config` payload in the `.npz` checkpoint.
- Prefer estimator-native `save()` / `load()` helpers if they already exist and
  extend them to cover all estimator types.
- Add a regression test with a non-default MAF/NSF architecture and verify
  identical seeded posterior samples after load.

### Improved High: `sbi` backend save/load path

Location: `vbi/inference/_vbi_inference.py`, `save()` / `load()`

Original issue: `save()` did not save `sbi` estimator weights but still recorded
`meta__inference_backend="sbi"`, and `load()` had an unconditional `_snpe` log
path.

Follow-up status: improved. The old log crash path is gone, and a trained
`sbi` estimator is saved to a companion `<stem>_sbi.pt` file and loaded when
present. Remaining risk: the current validation only checks no-crash behavior
and simulation restoration; it does not yet verify posterior equivalence after
loading a trained `sbi` estimator.

Recommended fix:

- Add a trained `sbi` save/load test that builds a posterior after load and
  verifies sampling/log-prob behavior.
- Decide whether companion-file persistence is an acceptable public checkpoint
  contract and document it.

### High: Sequential/proposal information is lost in `VBIInference`

Location: `vbi/inference/_vbi_inference.py`, `simulate()`, `get_simulations()`,
`save()`, `load()`

`simulate(proposal=...)` passes the proposal into the internal VBI SNPE object,
but `_sim_rounds` stores only `(theta, x)`. `get_simulations()` always returns
an empty proposals list, save/load flattens all simulations into a single round,
and proposal objects are never persisted.

This weakens MI3/MI6 integration: the immediate in-memory training path can
still see proposal metadata through `_snpe`, but `VBIInference` itself cannot
report, checkpoint, or resume multi-round sequential inference faithfully.

Recommended fix:

- Store `_sim_rounds` as `(theta, x, proposal, metadata)` or a small dataclass.
- Preserve round boundaries in checkpoints.
- Return proposals from `get_simulations()` for VBI backend.
- Document what proposal persistence means; if proposal objects cannot be
  serialized, store enough metadata and fail clearly on resume-training when a
  proposal is required.

### Resolved/Needs Coverage Medium: `sbi` backend simulation append ignored proposal and types

Location: `vbi/inference/_vbi_inference.py`, `simulate()`

The `sbi` branch creates `torch_prior_proposal = proposal` but never uses it:

```python
torch_prior_proposal = proposal  # sbi handles proposal natively
self._sbi_engine.append_simulations(theta, x)
```

For multi-round `sbi`, `append_simulations(..., proposal=proposal)` is the part
that lets sbi track proposal correction. Also, `theta` and `x` are passed as
NumPy arrays; depending on installed sbi version, this may not be accepted or
may trigger implicit conversion in an uncontrolled way.

Recommended fix:

- Convert `theta` and `x` to `torch.float32` explicitly for the `sbi` branch.
- Convert/wrap `proposal` to an sbi-compatible posterior/proposal or reject
  proposal-based rounds for `inference_backend="sbi"` until supported.
- Add an `sbi` backend smoke test for one-round and two-round append behavior.

Follow-up status: implementation is fixed for explicit torch conversion and
passing `proposal`. A two-round/proposal `sbi` smoke test is still worthwhile.

### Resolved/Deferred Medium: Config loader omits `stimuli` even though `SimulationSpec` supports it

Location: `vbi/simulator/spec/simulation.py`, `SimulationSpec.from_dict()`

`SimulationSpec` has a `stimuli` field and imports `StimSpec`, but
`from_dict()` does not parse a `stimuli` section. This means a config-driven
MI6 workflow cannot reproduce stimulation-based simulations, despite the
simulator spec supporting them.

Recommended fix:

- Add a minimal `stimuli` schema to `SimulationSpec.from_dict()`.
- Include at least one config test with a stimulus entry.
- If stimulus config is intentionally deferred, reject a `stimuli` key with a
  clear error instead of silently ignoring it.

Follow-up status: deferred with explicit rejection. `SimulationSpec.from_dict()`
now raises a clear `ValueError` when `stimuli` is supplied.

### Medium: Cache extraction materializes all rows in Python lists

Location: `vbi/inference/_utils.py`, `_extract_from_cache_impl()`

The cache feature exists to make expensive and large simulations reusable, but
re-extraction appends every theta/features row into Python lists and stacks at
the end. This removes much of the scalability benefit for large cached sweeps.

Recommended fix:

- Process chunks into per-chunk feature arrays and either yield chunks, write a
  feature-cache file, or preallocate output arrays using metadata.
- Add a chunked API such as `iter_extract_from_cache()` for large workflows.

### Resolved Medium: Cache mode did not validate that cached labels match re-extracted labels

Location: `vbi/inference/_utils.py`, `_extract_from_cache_impl()`

`feat_labels` are taken from the first row, but subsequent rows are not checked.
If a feature extractor changes labels due to data-dependent behavior or config
mutation, the output can silently stack values under stale labels.

Recommended fix:

- Compare `labels_i` to `feat_labels` for each row and raise on mismatch.
- Persist `feature_labels` into cache metadata after first extraction.

Follow-up status: label comparison and mismatch error are implemented.
Persisting feature labels into cache metadata remains optional future polish.

### Resolved Medium: `VBIInference.pairplot()` error text is misleading

Location: `vbi/inference/_vbi_inference.py`, `pairplot()`

The method says "call train() and build_posterior() first", but it does not
require a prior `build_posterior()` call; it builds a posterior internally from
`_last_estimator`. The behavior is fine, but the error message and docstring
will confuse users.

Recommended fix:

- Change the error to "call train() first".
- Optionally cache the built posterior if repeated plotting is expected.

Follow-up status: fixed. The error now says `call train() first`.

### Low: `SimulationSpec.from_dict()` rebuilds the model registry on every call

Location: `vbi/simulator/spec/simulation.py`, `from_dict()`

The registry is constructed inside the method each time. This is not a
correctness issue, but config loading is cleaner and cheaper if the registry is
module-level or exposed through a central simulator model registry.

Recommended fix:

- Move aliases/model lookup into a helper or module-level constant.

### Low: Unused imports and variables

Locations:

- `vbi/inference/_vbi_inference.py`, `_make_simulator_fn()` imports
  `Simulator` but does not use it.
- `vbi/inference/_vbi_inference.py`, `simulate()` assigns
  `torch_prior_proposal` but never uses it.
- `vbi/inference/_vbi_inference.py` imports `_extract_from_cache_impl` into the
  module namespace but only uses it indirectly through the imported public
  wrapper.

Recommended fix:

- Remove unused imports/variables or wire them into the intended behavior.

## Test Coverage Gaps

Current MI6 tests cover the happy path well for:

- `VBIInference` with NumPy simulator backend
- `density_estimator="maf"`
- config loading for a minimal MPR config
- cache write/re-extract smoke tests
- plot helpers and SBC smoke tests

Missing high-value tests:

- Save/load for `mdn` and `nsf`, including MDN restored `log_prob()` /
  posterior use
- Save/load for non-default MAF/NSF architecture
- `inference_backend="sbi"` trained save/load posterior behavior
- `inference_backend="sbi"` proposal/two-round behavior
- `simulate(proposal=...)` with `get_simulations()` preserving proposals
- Round boundary preservation after save/load
- `SimulationSpec.from_dict()` with stimuli if config-driven stimuli is
  accepted later
- Config loading with `.txt`, `.csv`, `.npy`, and connectivity dict formats

## Verification Notes

Original targeted MI6 validation run:

```bash
python3 -m pytest \
  vbi/tests/validation/test_vbi_inference.py \
  vbi/tests/validation/test_vbi_inference_config.py \
  vbi/tests/validation/test_vbi_inference_diag.py \
  vbi/tests/validation/test_simulate_for_vbi_sweep.py \
  vbi/tests/validation/test_sim_cache.py -q
```

Result:

```text
43 passed, 13 failed, 1 warning in 890.69s
```

All 13 failures are from `test_vbi_inference_config.py` and share the same root
cause: `SimulationSpec.from_dict()` references `Path` without importing it.

Follow-up validation on 2026-05-29:

```bash
python3 -m pytest \
  vbi/tests/validation/test_vbi_inference.py \
  vbi/tests/validation/test_vbi_inference_config.py \
  vbi/tests/validation/test_vbi_inference_diag.py \
  vbi/tests/validation/test_sim_cache.py -q
```

Result:

```text
51 passed, 1 warning in 926.19s
```

A separate focused reproduction still confirms an MDN checkpoint load/use
failure after unpack:

```text
AttributeError: 'MDNEstimator' object has no attribute '_offdiag_basis'
```

The local environment also emits NumPy 2.x compatibility warnings from optional
compiled pandas/pyarrow/bottleneck imports, which are worth cleaning up
separately if they appear in CI.

## Recommended Next Steps

1. Fix MDN checkpoint restore by rebuilding `_offdiag_basis` and add a
   regression test.
2. Persist full estimator configs for MAF/NSF/MDN and test non-default
   architectures.
3. Preserve round/proposal metadata in `VBIInference`.
4. Add trained `sbi` save/load posterior tests and two-round/proposal tests.
5. Thread `inference_backend` through `from_config()`.
6. Add a chunked cache extraction API if large cached sweeps are expected.
