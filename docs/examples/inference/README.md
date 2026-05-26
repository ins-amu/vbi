# Inference Examples

Fast accuracy demos for `vbi.inference`.  Each runs in under 20 seconds
and uses problems with known analytical solutions so accuracy is measurable
without visual inspection.

## Run all

```bash
# From the project root, activate your environment first
source /path/to/vbienv/bin/activate
# or: uv run python ...

for f in docs/examples/inference/0*.py; do
    echo "=== $f ===" && python "$f" 2>/dev/null | grep -E "✓|✗|error|coverage"
done
```

## Demos

| File | Problem | What is checked |
|------|---------|----------------|
| `01_gaussian_1d.py` | 1-D linear-Gaussian, exact posterior known | Posterior mean error, std error, 90% CI coverage |
| `02_gaussian_2d.py` | 2-D Gaussian, identity + coupled simulators | Posterior mean, independence vs correlation structure |
| `03_two_moons.py` | Classic 2-D SBI benchmark (no analytical posterior) | Samples inside prior, more concentrated than prior, log_prob finite |
| `04_mdn_vs_maf.py` | Same 1-D Gaussian, MDN vs MAF | Side-by-side error + training time comparison |
| `05_sequential_rounds.py` | 2-round sequential inference | Round-2 focused sims improve or match round-1 accuracy |
| `07_jax_vs_numpy.py` | Same 1-D Gaussian, NumPy vs JAX backends | Backend/estimator side-by-side accuracy + training time |

## Pass/fail thresholds

All thresholds are intentionally **loose** — these demos use short training
(300 epochs max, early stopping at 20) for speed.  The numbers will improve
with longer training (`max_num_epochs=2000`, `stop_after_epochs=50`).

To run heavier accuracy benchmarks (longer training, more simulations),
increase those parameters and tighten the `assert` thresholds accordingly.

## What to add as new features land

| Feature | Demo to add |
|---------|------------|
| NSF density estimator | Covered in `07_jax_vs_numpy.py`; optionally add NSF to `04_mdn_vs_maf.py` |
| Rejection sampling | `01_gaussian_1d.py` — add `reject_outside_prior=True` check |
| SBC | `06_sbc_coverage.py` — rank uniformity test |
| TARP | `07_tarp_coverage.py` — expected coverage plot |
| Embedding network | `08_embedding_net.py` — raw time series as x |
| Sequential APT | `05_sequential_rounds.py` — add APT-weighted round comparison |
