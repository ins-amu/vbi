"""
Demo 8 - Backend speed benchmark: sbi (torch) vs vbi numpy vs vbi jax.

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

import contextlib
import io
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

_stderr = io.StringIO()
with contextlib.redirect_stderr(_stderr):
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


def _result_row(name, compile_s, train_s, sample_s, mean_err, std_err, steps):
    throughput = steps / train_s if train_s > 0 else float("nan")
    return (
        f"  {name:<14}"
        f"{compile_s:>11.2f}"
        f"{train_s:>10.2f}"
        f"{throughput:>10.1f}"
        f"{sample_s * 1000:>12.1f}"
        f"{mean_err:>10.3f}"
        f"{std_err:>9.3f}"
    )


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

    # Sample timing - suppress sbi sampler tqdm (writes to stderr)
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
print("Demo 8 - Backend benchmark: sbi/torch vs vbi/numpy vs vbi/jax")
print("=" * 76)
print("  Problem")
print(f"    simulator    : 1-D Gaussian")
print(f"    N_train      : {N_TRAIN}")
print(f"    train budget : {N_STEPS} epochs, batch={BATCH_SIZE}")
print(f"    estimator    : MAF(flows=4, hidden_units=32)")
print(f"    x_obs={X_OBS_VAL}: true mean={TRUE_MEAN:.3f}, true std={TRUE_STD:.3f}")
print()

print("  Running")
failures = []
jobs = [("vbi/numpy", lambda: bench_vbi("numpy")),
        ("vbi/jax",   lambda: bench_vbi("jax")),
        ("sbi/torch", bench_sbi)]

for i, (name, fn) in enumerate(jobs, start=1):
    print(f"    [{i}/{len(jobs)}] {name:<9} ... ", end="", flush=True)
    try:
        r = fn()
        if r is None:
            print("skipped")
        else:
            results.append(r)
            print("done")
    except Exception as e:
        failures.append((name, str(e).splitlines()[0]))
        print("failed")

print()
if failures:
    print("  Failures")
    for name, message in failures:
        print(f"    {name:<9}: {message}")
    print()

print("  Results")
print(f"  {'Backend':<14}{'Compile s':>11}{'Train s':>10}{'Steps/s':>10}"
      f"{'Sample ms':>12}{'Mean err':>10}{'Std err':>9}")
print("  " + "-" * 76)
for r in results:
    print(_result_row(**{k: r[k] for k in
                         ("name", "compile_s", "train_s", "sample_s",
                          "mean_err", "std_err", "steps")}))
print()

# ── Speedup summary ────────────────────────────────────────────────────────────
if len(results) >= 2:
    baseline = next((r for r in results if r["name"] == "vbi/numpy"), None)
    if baseline:
        print("  Training speed vs vbi/numpy")
        for r in results:
            if r["name"] == baseline["name"]:
                continue
            ratio = baseline["train_s"] / r["train_s"] if r["train_s"] > 0 else float("nan")
            label = "faster" if ratio >= 1.0 else "slower"
            print(f"    {r['name']:<9}: {ratio:.2f}x ({label})")
        print()

print("  Timing definitions")
print("    compile   : first-run JIT/trace overhead, separated by a warm rerun")
print("    train     : warm training time for all gradient steps")
print("    steps/s   : actual optimizer steps per second")
print(f"    sample    : milliseconds for {N_SAMPLES} posterior samples, "
      f"averaged over {N_SAMPLE_REPS} runs")
