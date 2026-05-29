"""
Demo 0 - Same inference API, different backend.

This is the minimal backend-selection example.  The workflow is identical:

    SNPE(...) → append_simulations(...) → train(...) → build_posterior(...)

Only the ``backend`` argument changes between NumPy/autograd and JAX.

Expected runtime: < 15 seconds.
"""

import contextlib
import io
import os

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-vbi")
os.environ.setdefault("JAX_PLATFORMS", "cpu")


def import_inference_api():
    """Keep optional dependency diagnostics out of this small demo."""
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr):
        from vbi.inference import SNPE, Gaussian
    return SNPE, Gaussian


SNPE, Gaussian = import_inference_api()

# ── Problem definition ────────────────────────────────────────────────────────
SIGMA_LIK   = 0.3
SIGMA_PRIOR = 2.0
N_TRAIN     = 1000
N_POST      = 1500
SEED        = 42

rng = np.random.default_rng(SEED)


def analytical_posterior(x_obs_val: float):
    """theta | x ~ N(mu_post, sigma_post²)."""
    prec_p = 1.0 / SIGMA_PRIOR ** 2
    prec_l = 1.0 / SIGMA_LIK ** 2
    s_post = np.sqrt(1.0 / (prec_p + prec_l))
    m_post = s_post ** 2 * prec_l * x_obs_val
    return m_post, s_post


# ── Simulate once; reuse the same data for every backend ──────────────────────
prior    = Gaussian(mean=np.array([0.0]), std=np.array([SIGMA_PRIOR]))
theta_tr = prior.sample((N_TRAIN,), seed=SEED)
x_tr     = theta_tr + rng.normal(0, SIGMA_LIK, theta_tr.shape)
x_obs    = np.array([[1.0]])
mu_true, std_true = analytical_posterior(float(x_obs[0, 0]))


def run_backend(backend: str):
    # Same high-level API for every backend; only backend= changes.
    inference = SNPE(
        prior=prior,
        density_estimator="maf",
        backend=backend,
        show_progress_bars=True,
    )
    inference.append_simulations(theta_tr, x_tr)
    estimator = inference.train(
        training_batch_size=256,
        learning_rate=5e-4,
        stop_after_epochs=12,
        max_num_epochs=80,
        # Leave verbose unset so show_progress_bars=True controls tqdm output.
        seed=SEED,
    )
    posterior = inference.build_posterior(estimator)

    samples = posterior.sample((N_POST,), x=x_obs, seed=0)
    log_p   = posterior.log_prob(samples[:20], x=x_obs)
    return {
        "backend": backend,
        "sample_shape": samples.shape,
        "samples": samples,
        "mean": float(samples[:, 0].mean()),
        "std": float(samples[:, 0].std()),
        "finite_lp": float(np.isfinite(log_p).mean()),
    }


print("=" * 62)
print("Demo 0 - Same SNPE API, different backend")
print("=" * 62)
print(f"  density_estimator=mdn  N_train={N_TRAIN}  x_obs={x_obs[0, 0]:.1f}")
print(f"  true posterior: mean={mu_true:.3f}  std={std_true:.3f}")
print()
print("  The only code change is backend='numpy' vs backend='jax'.")
print("  Short training keeps this an API smoke test, not an accuracy benchmark.")
print()

results = []
for backend in ("numpy", "jax"):
    print(f"  running backend={backend!r} ...", flush=True)
    try:
        result = run_backend(backend)
        results.append(result)
        print(f"  done    backend={backend!r}")
    except Exception as exc:
        msg = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
        print(f"  skipped backend={backend!r}: {type(exc).__name__}: {msg}")

print()
print(f"  {'Backend':>8}  {'Sample shape':>14}  {'Mean':>8}  "
      f"{'Std':>8}  {'Mean err':>9}  {'Finite lp':>9}")
print("  " + "-" * 66)
for r in results:
    mean_err = abs(r["mean"] - mu_true)
    print(f"  {r['backend']:>8}  {str(r['sample_shape']):>14}  "
          f"{r['mean']:>8.3f}  {r['std']:>8.3f}  "
          f"{mean_err:>9.3f}  {r['finite_lp']:>9.2f}")

print()
assert results, "No backend ran successfully."
for r in results:
    assert r["sample_shape"] == (N_POST, 1)
    assert abs(r["mean"] - mu_true) < 0.75
    assert r["finite_lp"] > 0.8
    print(f"  ✓ backend={r['backend']!r} API check passed.")

# ── Minimal plot ──────────────────────────────────────────────────────────────
try:
    import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from helpers import posterior_1d
    from pathlib import Path
    out = Path(__file__).parent / "outputs"
    xlim = (mu_true - 5 * std_true, mu_true + 5 * std_true)

    for r in results:
        posterior_1d(
            r["samples"][:, 0],
            true_mean=mu_true,
            true_std=std_true,
            x_obs_val=float(x_obs[0, 0]),
            label_est=f"{r['backend']} posterior",
            title=f"Demo 0 - backend={r['backend']}",
            xlim=xlim,
            out_path=out / f"00_backend_{r['backend']}_posterior.png",
        )
except Exception as e:
    print(f"  (plots skipped: {e})")
