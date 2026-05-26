"""
Demo 8 — Backend speed benchmark: sbi (torch) vs vbi numpy vs vbi jax.

Measures three things separately for each backend:
  1. Compilation / first-run overhead (JAX JIT; torch tracing for sbi).
  2. Steady-state training throughput (gradient steps / second, after warmup).
  3. Sampling throughput (posterior samples / second).
  4. Posterior accuracy (mean error vs analytical solution).

Problem: 1-D Gaussian, same as Demo 1.

Expected runtimes (CPU):
  sbi (torch) – varies by torch version / CPU
  vbi numpy   – baseline
  vbi jax     – first run slower (JIT); subsequent steps faster

Run with:
    python docs/examples/inference/08_backend_benchmark.py
"""

import os
import time
import warnings

import numpy as np

# Force JAX to CPU to avoid GPU cuDNN issues; override with JAX_PLATFORMS=gpu
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-vbi")

# ── Problem ────────────────────────────────────────────────────────────────────
SIGMA_LIK   = 0.3
SIGMA_PRIOR = 2.0
N_TRAIN     = 1000
N_STEPS     = 300      # fixed gradient steps, no early stopping
BATCH_SIZE  = 256
SEED        = 0
N_SAMPLES   = 1000     # posterior samples per timing call
N_SAMPLE_REPS = 5      # repeat sampling to get stable timing

X_OBS_VAL   = 1.0
# Analytical posterior for x_obs=1.0
_prec_p = 1.0 / SIGMA_PRIOR ** 2
_prec_l = 1.0 / SIGMA_LIK   ** 2
TRUE_STD  = float(np.sqrt(1.0 / (_prec_p + _prec_l)))
TRUE_MEAN = float(TRUE_STD ** 2 * _prec_l * X_OBS_VAL)

# ── Shared data ────────────────────────────────────────────────────────────────
rng_np = np.random.default_rng(SEED)

from vbi.inference import SNPE as vbiSNPE, Gaussian as vbiGaussian  # noqa: E402

prior_vbi = vbiGaussian(mean=np.array([0.0]), std=np.array([SIGMA_PRIOR]))
theta_tr  = prior_vbi.sample((N_TRAIN,), seed=SEED).astype("f")
x_tr      = (theta_tr + rng_np.normal(0, SIGMA_LIK, theta_tr.shape)).astype("f")
x_obs_vbi = np.array([[X_OBS_VAL]], dtype="f")


# ── MAF architecture (kept small for CPU speed) ────────────────────────────────
MAF_KWARGS = dict(n_flows=4, hidden_units=32, num_blocks=2)
# Training uses fixed N_STEPS with no early stopping
TRAIN_KWARGS = dict(
    training_batch_size=BATCH_SIZE,
    learning_rate=5e-4,
    stop_after_epochs=N_STEPS,   # patience = total steps → no early stop
    max_num_epochs=N_STEPS,
    verbose=False,
)


def _posterior_accuracy(samples):
    s = samples[:, 0] if samples.ndim == 2 else samples
    return abs(s.mean() - TRUE_MEAN), abs(s.std() - TRUE_STD)


def _print_result(name, compile_s, train_s, sample_s, mean_err, std_err, steps):
    throughput = steps / train_s if train_s > 0 else float("nan")
    print(f"  {name:<22}  compile={compile_s:6.2f}s  train={train_s:6.2f}s  "
          f"({throughput:6.0f} steps/s)  "
          f"sample={sample_s*1000:6.1f}ms  "
          f"mean_err={mean_err:.3f}  std_err={std_err:.3f}")


results = []


# ── vbi numpy ─────────────────────────────────────────────────────────────────
def bench_vbi(backend: str):
    name = f"vbi/{backend}"
    inf  = vbiSNPE(prior=prior_vbi, density_estimator="maf",
                   backend=backend, show_progress_bars=False)
    inf.append_simulations(theta_tr, x_tr)

    # Compile / first-run timing (relevant for jax)
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est = inf.train(seed=SEED, **TRAIN_KWARGS)
    compile_and_train = time.perf_counter() - t0

    # Build posterior
    post = inf.build_posterior(est)

    # Sample timing (average over reps)
    t0 = time.perf_counter()
    for _ in range(N_SAMPLE_REPS):
        samples = post.sample((N_SAMPLES,), x=x_obs_vbi, seed=SEED)
    sample_s = (time.perf_counter() - t0) / N_SAMPLE_REPS

    # For JAX: isolate compilation by doing a second identical training run.
    # The JIT cache is warm after the first run, so the second run is pure training.
    if backend == "jax":
        inf2 = vbiSNPE(prior=prior_vbi, density_estimator="maf",
                       backend="jax", show_progress_bars=False)
        inf2.append_simulations(theta_tr, x_tr)
        t1 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            inf2.train(seed=SEED + 1, **TRAIN_KWARGS)
        train_s   = time.perf_counter() - t1
        compile_s = max(0.0, compile_and_train - train_s)
    else:
        compile_s = 0.0
        train_s   = compile_and_train

    # Actual gradient steps = epochs × batches_per_epoch
    n_train_vbi = int(N_TRAIN * 0.9)
    batches_vbi = max(1, -(-n_train_vbi // BATCH_SIZE))
    actual_steps = N_STEPS * batches_vbi

    mean_err, std_err = _posterior_accuracy(samples)
    return dict(name=name, compile_s=compile_s, train_s=train_s,
                sample_s=sample_s, mean_err=mean_err, std_err=std_err,
                steps=actual_steps)


# ── sbi (torch) ───────────────────────────────────────────────────────────────
def _run_sbi(inf_obj, theta_t, x_t):
    """Train sbi SNPE, fully suppressing stdout/stderr progress output."""
    import io, contextlib
    buf = io.StringIO()
    with warnings.catch_warnings(), contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        warnings.simplefilter("ignore")
        return inf_obj.train(
            training_batch_size=BATCH_SIZE,
            learning_rate=5e-4,
            stop_after_epochs=N_STEPS,
            max_num_epochs=N_STEPS,
            show_train_summary=False,
        )


def bench_sbi():
    try:
        import torch
        from sbi.inference import SNPE as sbiSNPE
        from sbi import utils as sbi_utils
    except ImportError as e:
        print(f"  sbi/torch skipped: {e}")
        return None

    prior_sbi = sbi_utils.BoxUniform(
        low=torch.tensor([-6.0]),
        high=torch.tensor([6.0]),
    )
    theta_t = torch.tensor(theta_tr)
    x_t     = torch.tensor(x_tr)
    x_obs_t = torch.tensor(x_obs_vbi)

    # First run (includes torch tracing / first-step overhead)
    inf_sbi = sbiSNPE(prior=prior_sbi, density_estimator="maf")
    inf_sbi.append_simulations(theta_t, x_t)
    t0 = time.perf_counter()
    est_sbi = _run_sbi(inf_sbi, theta_t, x_t)
    compile_and_train = time.perf_counter() - t0
    post_sbi = inf_sbi.build_posterior(est_sbi)

    # Sample timing — suppress sbi sampler tqdm (writes to stderr)
    import io, contextlib
    x_obs_cond = x_obs_t.squeeze(0)
    t0 = time.perf_counter()
    for _ in range(N_SAMPLE_REPS):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            samples_t = post_sbi.sample((N_SAMPLES,), x=x_obs_cond)
    sample_s = (time.perf_counter() - t0) / N_SAMPLE_REPS

    # Second run (warm)
    inf2 = sbiSNPE(prior=prior_sbi, density_estimator="maf")
    inf2.append_simulations(theta_t, x_t)
    t1 = time.perf_counter()
    _run_sbi(inf2, theta_t, x_t)
    train_s   = time.perf_counter() - t1
    compile_s = max(0.0, compile_and_train - train_s)

    # sbi uses epochs; compute actual gradient steps for fair comparison
    n_train_sbi = int(N_TRAIN * 0.9)   # sbi uses 10% val by default
    batches_sbi = max(1, -(-n_train_sbi // BATCH_SIZE))   # ceiling div
    actual_steps = N_STEPS * batches_sbi

    samples_np = samples_t.numpy()
    mean_err, std_err = _posterior_accuracy(samples_np)
    return dict(name="sbi/torch", compile_s=compile_s, train_s=train_s,
                sample_s=sample_s, mean_err=mean_err, std_err=std_err,
                steps=actual_steps)


# ── Run ────────────────────────────────────────────────────────────────────────
print("=" * 76)
print("Demo 8 — Backend benchmark: sbi/torch vs vbi/numpy vs vbi/jax")
print("=" * 76)
print(f"  Problem: 1-D Gaussian  N_train={N_TRAIN}  steps={N_STEPS}  "
      f"batch={BATCH_SIZE}  MAF(flows=4,h=32)")
print(f"  True posterior @ x_obs={X_OBS_VAL}: mean={TRUE_MEAN:.3f}  std={TRUE_STD:.3f}")
print()
print(f"  {'Backend':<22}  {'Compile':>8}  {'Train':>7}  {'Steps/s':>8}  "
      f"{'Sample':>9}  {'Mean err':>8}  {'Std err':>7}")
print("  " + "-" * 74)

for backend in ("numpy", "jax"):
    print(f"  running vbi/{backend} ...", flush=True)
    try:
        r = bench_vbi(backend)
        results.append(r)
        _print_result(**{k: r[k] for k in
                         ("name","compile_s","train_s","sample_s","mean_err","std_err","steps")})
    except Exception as e:
        print(f"  vbi/{backend} failed: {e}")

print(f"  running sbi/torch ...", flush=True)
r_sbi = bench_sbi()
if r_sbi:
    results.append(r_sbi)
    _print_result(**{k: r_sbi[k] for k in
                     ("name","compile_s","train_s","sample_s","mean_err","std_err","steps")})

print()
print("  Notes:")
print("  • compile: one-time JIT/trace cost (amortised over repeated runs)")
print("  • train:   steady-state wall-clock for all gradient steps (warmup excluded)")
print("  • steps/s: actual gradient steps / second (epochs × batches_per_epoch)")
print("  • sample:  time for 1000 posterior samples, averaged over 5 repetitions")
print("  • JAX advantage grows with model size and N; on GPU, JAX typically beats torch.")
print("    On CPU with tiny models the per-step overhead dominates.")
print()

# ── Speedup summary ────────────────────────────────────────────────────────────
if len(results) >= 2:
    baseline = next((r for r in results if r["name"] == "vbi/numpy"), None)
    if baseline:
        print("  Speedup vs vbi/numpy (training, excluding compilation):")
        for r in results:
            if r["name"] == baseline["name"]:
                continue
            ratio = baseline["train_s"] / r["train_s"] if r["train_s"] > 0 else float("nan")
            print(f"    {r['name']:<22}  {ratio:+.2f}x")
        print()
