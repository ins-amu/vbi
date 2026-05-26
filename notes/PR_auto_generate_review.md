# Review: merge `auto_generate` into `main`

## Findings

- **High:** This is a major simulator pipeline change and should be merged with clear migration expectations. Existing public workflows should remain functional while the new `vbi.simulator` stack becomes the main path and gradually replaces older implementations.

- **Medium:** Backend parity is the main merge risk. The PR adds NumPy, Numba, C++, JAX, and CUDA paths, so signature drift between backends can break validation even when one backend works. Recent fixes around `coup_alpha`, stimulation arguments, and Kuramoto `G` handling should stay covered by targeted tests.

- **Medium:** Optional compiled dependencies can affect CI and user environments. Feature extraction should avoid eager imports of optional packages where possible, especially for simple statistical features that do not require them.

- **Low:** The PR is large, with many models, demos, benchmark scripts, and validation tests added at once. Follow-up PRs should continue reducing sharp edges rather than expanding the surface area before the core simulator API stabilizes.

## Summary

This PR introduces the new main simulator pipeline based on backend-agnostic model specifications, generated backend implementations, monitor/sweep abstractions, and validation tests. It keeps the existing VBI workflows in place while providing a more unified path for future simulator development.

The change is appropriate to merge as a major update if CI is green for the supported CPU paths and GPU/C++ tests are either passing in the intended environment or clearly marked/optional.

## What Looks Good

- The new `SimulationSpec`, `ModelSpec`, `CouplingSpec`, `IntegratorSpec`, `MonitorSpec`, `SweepSpec`, and `StimSpec` structure is a solid foundation for a maintainable simulator API.
- Cross-backend validation coverage is substantially improved.
- The examples and benchmark scripts make the new pipeline easier to evaluate.
- Stimulation support now has dedicated validation coverage across backends.

## Recommended Follow-Ups

- Document which backends are required in CI and which are optional due to toolchain/GPU availability.
- Keep model metadata and backend parameter handling aligned, especially for global parameters such as `G`.
- Add a short migration note for users explaining when to prefer the new `vbi.simulator` API over older simulator/model paths.
- Continue adding small, targeted regression tests whenever backend signatures or generated code paths change.

## Testing Notes

Recommended before merge:

```bash
python3 -m pytest vbi/tests/validation/test_pipeline.py -q
python3 -m pytest vbi/tests/validation/test_stimulation.py -q
python3 -m pytest vbi/tests/validation/test_mpr_jax.py::TestSweep -q
python3 -m pytest vbi/tests/validation/test_new_models_jax.py -q
python3 -m pytest vbi/tests/validation/test_mpr_numba.py::TestSweepDeterministic -q
```
