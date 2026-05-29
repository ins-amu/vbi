"""
Demo 7 - JAX vs NumPy density estimators.

Uses the same 1-D Gaussian problem as Demo 1, where the posterior is known
analytically, to compare each density estimator across backends:
  - NumPy/autograd backend
  - JAX backend

The goal is a smoke-test style comparison, not a strict speed benchmark.  JAX
includes compilation overhead on the first train/sample calls, so wall-clock
times are most useful for checking that both backends run end-to-end.

By default this runs MAF and MDN.  NumPy NSF is much slower because its spline
transform is differentiated by autograd; include it explicitly with:

    VBI_DEMO7_ESTIMATORS=maf,mdn,nsf python docs/examples/inference/07_jax_vs_numpy.py

Expected runtime: < 30 seconds.
"""

import contextlib
import io
import os
import time
import warnings

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-vbi")
# JAX_PLATFORMS defaults to "cpu" inside vbi.inference._backends - set it here
# before any vbi import so the demo works even when this file is run first.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

# ── Problem definition ────────────────────────────────────────────────────────
SIGMA_LIK   = 0.3
SIGMA_PRIOR = 2.0
N_TRAIN     = 600
N_POST      = 500
SEED        = 42

DENSITY_ESTIMATORS = tuple(
    item.strip()
    for item in os.environ.get("VBI_DEMO7_ESTIMATORS", "maf,mdn").split(",")
    if item.strip()
)
BACKENDS           = ("numpy", "jax")

rng = np.random.default_rng(SEED)


def import_inference_api():
    """Import through vbi while keeping optional dependency diagnostics quiet."""
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr):
        from vbi.inference import SNPE, Gaussian
    return SNPE, Gaussian


SNPE, Gaussian = import_inference_api()


def analytical_posterior(x_obs_val: float):
    """theta | x ~ N(mu_post, sigma_post²)."""
    prec_p = 1.0 / SIGMA_PRIOR ** 2
    prec_l = 1.0 / SIGMA_LIK ** 2
    s_post = np.sqrt(1.0 / (prec_p + prec_l))
    m_post = s_post ** 2 * prec_l * x_obs_val
    return m_post, s_post


# ── Simulate one shared training set ──────────────────────────────────────────
prior    = Gaussian(mean=np.array([0.0]), std=np.array([SIGMA_PRIOR]))
theta_tr = prior.sample((N_TRAIN,), seed=SEED)
x_tr     = theta_tr + rng.normal(0, SIGMA_LIK, theta_tr.shape)
x_test   = np.array([-1.0, 0.0, 1.0])


def evaluate(density_estimator: str, backend: str):
    if density_estimator == "mdn":
        max_epochs = 120
        patience   = 15
    else:
        max_epochs = 80
        patience   = 12

    inference = SNPE(
        prior=prior,
        density_estimator=density_estimator,
        backend=backend,
        show_progress_bars=False,
    )
    inference.append_simulations(theta_tr, x_tr)

    t0 = time.perf_counter()
    estimator = inference.train(
        training_batch_size=512,
        learning_rate=5e-4,
        stop_after_epochs=patience,
        max_num_epochs=max_epochs,
        verbose=False,
        seed=SEED,
    )
    train_sec = time.perf_counter() - t0
    posterior = inference.build_posterior(estimator)

    mean_errs, std_errs, finite_lps = [], [], []
    samples_at_one = None
    for xv in x_test:
        x_obs = np.array([[xv]])
        samples = posterior.sample((N_POST,), x=x_obs, seed=0)
        mu_t, s_t = analytical_posterior(xv)

        mean_errs.append(abs(samples[:, 0].mean() - mu_t))
        std_errs.append(abs(samples[:, 0].std() - s_t))
        finite_lps.append(np.isfinite(posterior.log_prob(samples[:50], x=x_obs)).mean())

        if xv == 1.0:
            samples_at_one = samples

    return {
        "backend": backend,
        "de": density_estimator,
        "train_sec": train_sec,
        "mean_err": float(np.mean(mean_errs)),
        "std_err": float(np.mean(std_errs)),
        "finite_lp": float(np.mean(finite_lps)),
        "loss_history": getattr(estimator, "loss_history", []),
        "val_loss_history": getattr(estimator, "val_loss_history", None),
        "samples_at_one": samples_at_one,
    }


def skip_message(exc: Exception) -> str:
    msg = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
    if len(msg) > 90:
        msg = msg[:87] + "..."
    return f"{type(exc).__name__}: {msg}"


print("=" * 68)
print("Demo 7 - JAX vs NumPy density estimators")
print("=" * 68)
print(f"  N_train={N_TRAIN}  sigma_lik={SIGMA_LIK}  sigma_prior={SIGMA_PRIOR}")
print(f"  estimators={','.join(DENSITY_ESTIMATORS)}")
print()

results = []
for de in DENSITY_ESTIMATORS:
    for backend in BACKENDS:
        print(f"  running {backend:>5} {de:>3} ...", flush=True)
        try:
            stderr = io.StringIO()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with contextlib.redirect_stderr(stderr):
                    result = evaluate(de, backend)
            results.append(result)
            print(f"  done    {backend:>5} {de:>3} ({result['train_sec']:.2f}s)")
        except Exception as exc:
            print(f"  skipped {backend:>5} {de:>3} ({skip_message(exc)})")

print()
print(f"  {'Backend':>8}  {'Estimator':>9}  {'Train (s)':>10}  "
      f"{'Mean err':>10}  {'Std err':>10}  {'Finite lp':>9}")
print("  " + "-" * 66)
for r in results:
    print(f"  {r['backend']:>8}  {r['de']:>9}  {r['train_sec']:>10.2f}  "
          f"{r['mean_err']:>10.4f}  {r['std_err']:>10.4f}  {r['finite_lp']:>9.2f}")

print()
for r in results:
    name = f"{r['backend']} {r['de']}"
    assert r["mean_err"] < 0.75, f"{name}: mean error too large: {r['mean_err']:.4f}"
    assert r["std_err"]  < 0.75, f"{name}: std error too large: {r['std_err']:.4f}"
    assert r["finite_lp"] > 0.70, f"{name}: too many non-finite log_prob values"
    print(f"  ✓ {name} checks passed.")

assert results, "No backend/estimator combinations ran."

# ── Plots ──────────────────────────────────────────────────────────────────────
try:
    import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from helpers import posterior_1d, loss_plot
    from pathlib import Path
    out = Path(__file__).parent / "outputs"

    mu_t, s_t = analytical_posterior(1.0)
    for r in results:
        label = f"{r['backend']}-{r['de']}"
        posterior_1d(
            r["samples_at_one"][:, 0],
            true_mean=mu_t,
            true_std=s_t,
            x_obs_val=1.0,
            label_est=label,
            title=f"Demo 7 - {label} posterior at x_obs=1.0",
            out_path=out / f"07_{r['backend']}_{r['de']}_posterior.png",
        )
        if r["loss_history"]:
            loss_plot(
                r["loss_history"],
                r["val_loss_history"],
                title=f"Demo 7 - {label} loss curve",
                out_path=out / f"07_{r['backend']}_{r['de']}_loss.png",
            )
except Exception as e:
    print(f"  (plots skipped: {e})")
