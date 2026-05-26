"""
Demo 4 — MDN vs MAF on the same problem.

Uses the 1-D Gaussian with known analytical posterior (same as Demo 1)
to directly compare MDNEstimator and MAFEstimator on:
  - posterior mean error
  - posterior std error
  - training time

Useful for deciding which estimator to use for a given problem size.

Expected runtime: < 20 seconds.
"""

import time
import numpy as np
from vbi.inference import SNPE, Gaussian

SIGMA_LIK   = 0.3
SIGMA_PRIOR = 2.0
N_TRAIN     = 2000
N_POST      = 1000
SEED        = 42

rng = np.random.default_rng(SEED)

prior    = Gaussian(mean=np.array([0.0]), std=np.array([SIGMA_PRIOR]))
theta_tr = prior.sample((N_TRAIN,), seed=SEED)
x_tr     = theta_tr + rng.normal(0, SIGMA_LIK, theta_tr.shape)


def analytical_posterior(x_obs_val):
    prec_p = 1 / SIGMA_PRIOR ** 2
    prec_l = 1 / SIGMA_LIK   ** 2
    s_post = np.sqrt(1 / (prec_p + prec_l))
    m_post = s_post ** 2 * prec_l * x_obs_val
    return m_post, s_post


x_test = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])


def evaluate(density_estimator: str):
    # MDN converges faster per epoch but may need more; MAF is slower but accurate
    epochs = 500 if density_estimator == "mdn" else 300
    inf = SNPE(prior=prior, density_estimator=density_estimator)
    inf.append_simulations(theta_tr, x_tr)
    t0  = time.perf_counter()
    est = inf.train(
        training_batch_size=256,
        learning_rate=5e-4,
        stop_after_epochs=30,
        max_num_epochs=epochs,
        verbose=False,
    )
    train_sec = time.perf_counter() - t0
    post = inf.build_posterior(est)

    mean_errs, std_errs = [], []
    for xv in x_test:
        x_obs = np.array([[xv]])
        samp  = post.sample((N_POST,), x=x_obs, seed=0)
        mu_t, s_t = analytical_posterior(xv)
        mean_errs.append(abs(samp[:, 0].mean() - mu_t))
        std_errs.append(abs(samp[:, 0].std()  - s_t))

    return {
        "de":        density_estimator,
        "train_sec": train_sec,
        "mean_err":  np.mean(mean_errs),
        "std_err":   np.mean(std_errs),
    }


print("=" * 60)
print("Demo 4 — MDN vs MAF on 1-D Gaussian")
print("=" * 60)
print(f"  N_train={N_TRAIN}  sigma_lik={SIGMA_LIK}  sigma_prior={SIGMA_PRIOR}")
print()

results = [evaluate("maf"), evaluate("mdn")]

print(f"  {'Estimator':>10}  {'Train (s)':>10}  {'Mean err':>10}  {'Std err':>10}")
print("  " + "-" * 45)
for r in results:
    print(f"  {r['de']:>10}  {r['train_sec']:>10.2f}  "
          f"{r['mean_err']:>10.4f}  {r['std_err']:>10.4f}")

print()
for r in results:
    assert r["mean_err"] < 0.20, \
        f"{r['de']}: mean error too large: {r['mean_err']:.4f}"
    assert r["std_err"]  < 0.15, \
        f"{r['de']}: std  error too large: {r['std_err']:.4f}"
    print(f"  ✓ {r['de'].upper()} accuracy checks passed.")

# ── Plots ──────────────────────────────────────────────────────────────────────
try:
    import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from helpers import posterior_1d, loss_plot
    from pathlib import Path
    out = Path(__file__).parent / "outputs"

    # Re-run both to get samples + loss history for plotting
    x_plot = np.array([[1.0]])
    for de, r in [("maf", results[0]), ("mdn", results[1])]:
        inf_p = SNPE(prior=prior, density_estimator=de)
        inf_p.append_simulations(theta_tr, x_tr)
        epochs = 500 if de == "mdn" else 300
        est_p = inf_p.train(
            training_batch_size=256, learning_rate=5e-4,
            stop_after_epochs=30, max_num_epochs=epochs, verbose=False,
        )
        post_p = inf_p.build_posterior(est_p)
        samp_p = post_p.sample((1000,), x=x_plot, seed=0)
        mu_t, s_t = analytical_posterior(1.0)
        posterior_1d(samp_p[:, 0], true_mean=mu_t, true_std=s_t,
                     x_obs_val=1.0, label_est=de.upper(),
                     title=f"Demo 4 — {de.upper()} posterior at x_obs=1.0",
                     out_path=out / f"04_{de}_posterior.png")
        loss_plot(est_p.loss_history,
                  getattr(est_p, "val_loss_history", None),
                  title=f"Demo 4 — {de.upper()} loss curve",
                  out_path=out / f"04_{de}_loss.png")
except Exception as e:
    print(f"  (plots skipped: {e})")
