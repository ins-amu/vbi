"""
Diagnostic tools for posterior validation in simulation-based inference.

Functions
---------
run_sbc          : Simulation-Based Calibration (rank statistics)
check_sbc        : Uniformity tests on SBC ranks
sbc_rank_plot    : Histogram plot of SBC ranks
run_tarp         : Test of Accuracy with Random Points
check_tarp       : Expected coverage probability error
plot_tarp        : Coverage plot (ECP vs alpha)
c2st             : Classifier Two-Sample Test accuracy
pairplot         : Triangle scatter/KDE plot of samples
conditional_pairplot : Pairplot conditioned on observed data
plot_loss        : Training (and optional validation) loss curve
"""
import logging
import warnings

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SBC
# ---------------------------------------------------------------------------

def run_sbc(posterior, simulator, prior, num_sbc_runs: int = 500,
            num_posterior_samples: int = 1000, seed: int = 0) -> dict:
    """
    Run Simulation-Based Calibration.

    For each of ``num_sbc_runs`` ground-truth parameters θ* drawn from the
    prior, simulate x* = simulator(θ*), draw ``num_posterior_samples`` from
    the posterior p(θ | x*), and record the rank of θ* among those samples
    for each parameter dimension.

    Parameters
    ----------
    posterior : Posterior
        A trained posterior object with a ``sample((N,), x=...)`` method.
    simulator : callable
        ``simulator(theta_1d) -> x_1d``
    prior : prior object
        Must expose ``.sample((n,))`` and ``.log_prob(theta)``.
    num_sbc_runs : int
    num_posterior_samples : int
    seed : int

    Returns
    -------
    dict with keys
        ``ranks``  : ndarray (num_sbc_runs, param_dim) — integer ranks in
                     [0, num_posterior_samples]
        ``thetas`` : ndarray (num_sbc_runs, param_dim) — ground-truth θ*
    """
    rng = np.random.RandomState(seed)
    thetas_star = prior.sample((num_sbc_runs,), seed=int(rng.randint(1 << 31)))
    thetas_star = np.asarray(thetas_star)
    param_dim   = thetas_star.shape[1] if thetas_star.ndim > 1 else 1
    if thetas_star.ndim == 1:
        thetas_star = thetas_star[:, None]

    ranks = np.empty((num_sbc_runs, param_dim), dtype=int)

    for i in range(num_sbc_runs):
        theta_i = thetas_star[i]
        try:
            x_i = np.asarray(simulator(theta_i))
        except Exception as exc:
            log.warning("SBC run %d: simulator failed (%s) — skipping.", i, exc)
            ranks[i] = -1
            continue

        post_samples = np.asarray(
            posterior.sample((num_posterior_samples,), x=x_i,
                             seed=int(rng.randint(1 << 31)))
        )
        if post_samples.ndim == 1:
            post_samples = post_samples[:, None]

        for d in range(param_dim):
            ranks[i, d] = int(np.sum(post_samples[:, d] < theta_i[d]))

    return {"ranks": ranks, "thetas": thetas_star}


def check_sbc(ranks: np.ndarray, num_posterior_samples: int) -> dict:
    """
    Test whether SBC ranks are uniform via Kolmogorov-Smirnov test.

    Parameters
    ----------
    ranks : ndarray (num_sbc_runs, param_dim)
    num_posterior_samples : int

    Returns
    -------
    dict with key ``uniformity_pvalues`` : ndarray (param_dim,)
        KS p-value per parameter.  Values close to 1 indicate good
        calibration; values << 0.05 signal miscalibration.
    """
    from scipy.stats import ks_1samp, uniform

    ranks = np.asarray(ranks)
    if ranks.ndim == 1:
        ranks = ranks[:, None]

    # filter failed runs
    valid = np.all(ranks >= 0, axis=1)
    ranks = ranks[valid]

    param_dim = ranks.shape[1]
    pvalues = np.empty(param_dim)
    for d in range(param_dim):
        # Normalise to [0, 1] then KS test against Uniform(0,1)
        normalised = ranks[:, d] / num_posterior_samples
        _, pvalues[d] = ks_1samp(normalised, uniform(0, 1).cdf)

    return {"uniformity_pvalues": pvalues}


def sbc_rank_plot(ranks: np.ndarray, num_posterior_samples: int,
                  labels=None, fig=None):
    """
    Plot histograms of SBC ranks for each parameter dimension.

    Parameters
    ----------
    ranks : ndarray (num_sbc_runs, param_dim)
    num_posterior_samples : int
    labels : list[str] | None
    fig : matplotlib Figure | None

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    ranks = np.asarray(ranks)
    if ranks.ndim == 1:
        ranks = ranks[:, None]

    valid = np.all(ranks >= 0, axis=1)
    ranks = ranks[valid]

    param_dim = ranks.shape[1]
    num_runs  = ranks.shape[0]

    if labels is None:
        labels = [f"θ[{d}]" for d in range(param_dim)]

    if fig is None:
        fig, axes = plt.subplots(1, param_dim,
                                 figsize=(4 * param_dim, 3),
                                 squeeze=False)
    else:
        axes = np.array(fig.axes).reshape(1, -1)

    n_bins = min(20, num_posterior_samples + 1)
    expected = num_runs / n_bins

    for d in range(param_dim):
        ax = axes[0, d]
        ax.hist(ranks[:, d], bins=n_bins,
                range=(0, num_posterior_samples),
                color="steelblue", alpha=0.7, edgecolor="white")
        ax.axhline(expected, color="red", linestyle="--", linewidth=1.2,
                   label="uniform")
        ax.set_xlabel(labels[d])
        ax.set_ylabel("count" if d == 0 else "")
        ax.set_title(f"SBC ranks: {labels[d]}")
        ax.legend(fontsize=8)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# TARP
# ---------------------------------------------------------------------------

def run_tarp(posterior, simulator, prior, num_runs: int = 500,
             num_posterior_samples: int = 1000, seed: int = 0) -> dict:
    """
    Test of Accuracy with Random Points (TARP).

    For each run, draw a reference point θ_ref from the prior and a
    ground-truth θ* from the prior, simulate x* = simulator(θ*), draw
    posterior samples, and compute the fraction of posterior samples closer
    to θ_ref than θ* is.  The empirical coverage probability (ECP) should
    equal α for a well-calibrated posterior.

    Parameters
    ----------
    posterior : Posterior
    simulator : callable  ``simulator(theta_1d) -> x_1d``
    prior : prior object
    num_runs : int
    num_posterior_samples : int
    seed : int

    Returns
    -------
    dict with keys
        ``alphas`` : ndarray (num_runs,) — expected coverage levels
        ``ecp``    : ndarray (num_runs,) — empirical coverage probabilities
        ``ranks``  : ndarray (num_runs,) — raw rank fraction
    """
    rng = np.random.RandomState(seed)
    thetas_star = prior.sample((num_runs,), seed=int(rng.randint(1 << 31)))
    thetas_ref  = prior.sample((num_runs,), seed=int(rng.randint(1 << 31)))
    thetas_star = np.asarray(thetas_star)
    thetas_ref  = np.asarray(thetas_ref)

    raw_ranks = np.empty(num_runs)

    for i in range(num_runs):
        theta_i = thetas_star[i]
        ref_i   = thetas_ref[i]
        try:
            x_i = np.asarray(simulator(theta_i))
        except Exception as exc:
            log.warning("TARP run %d: simulator failed (%s).", i, exc)
            raw_ranks[i] = np.nan
            continue

        samples = np.asarray(
            posterior.sample((num_posterior_samples,), x=x_i,
                             seed=int(rng.randint(1 << 31)))
        )

        d_samples = np.linalg.norm(samples - ref_i, axis=-1)
        d_star    = np.linalg.norm(theta_i - ref_i)
        raw_ranks[i] = float(np.mean(d_samples < d_star))

    valid      = ~np.isnan(raw_ranks)
    raw_ranks  = raw_ranks[valid]
    alphas     = np.linspace(0, 1, num_runs)[: len(raw_ranks)]
    ecp        = np.array([np.mean(raw_ranks <= a) for a in alphas])

    return {"alphas": alphas, "ecp": ecp, "ranks": raw_ranks}


def check_tarp(alphas: np.ndarray, ecp: np.ndarray) -> dict:
    """
    Compute the Expected Coverage Error (ECE) = mean |ECP(α) − α|.

    Parameters
    ----------
    alphas : ndarray (N,)
    ecp    : ndarray (N,)

    Returns
    -------
    dict with key ``ece`` : float  (0 = perfect, higher = worse)
    """
    alphas = np.asarray(alphas)
    ecp    = np.asarray(ecp)
    ece    = float(np.mean(np.abs(ecp - alphas)))
    return {"ece": ece}


def plot_tarp(alphas: np.ndarray, ecp: np.ndarray, fig=None):
    """
    Plot ECP vs α for TARP results.

    Parameters
    ----------
    alphas : ndarray (N,)
    ecp    : ndarray (N,)
    fig    : matplotlib Figure | None

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    alphas = np.asarray(alphas)
    ecp    = np.asarray(ecp)

    if fig is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    else:
        ax = fig.axes[0] if fig.axes else fig.add_subplot(111)

    ax.plot(alphas, ecp, color="steelblue", linewidth=2, label="ECP")
    ax.plot([0, 1], [0, 1], color="red", linestyle="--",
            linewidth=1.2, label="ideal")
    ax.fill_between(alphas, alphas, ecp, alpha=0.2, color="steelblue")
    ax.set_xlabel("Expected coverage α")
    ax.set_ylabel("Empirical coverage probability")
    ax.set_title("TARP coverage")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# C2ST
# ---------------------------------------------------------------------------

def c2st(samples_p: np.ndarray, samples_q: np.ndarray,
         seed: int = 0, n_folds: int = 5) -> float:
    """
    Classifier Two-Sample Test (C2ST).

    Train a logistic-regression classifier to distinguish samples from
    distribution P vs Q.  Accuracy = 0.5 → distributions identical;
    accuracy → 1.0 → easy to distinguish.

    Parameters
    ----------
    samples_p : ndarray (N, D)  — samples from P (e.g. posterior)
    samples_q : ndarray (N, D)  — samples from Q (e.g. prior or reference)
    seed      : int
    n_folds   : int  — cross-validation folds

    Returns
    -------
    float  accuracy in [0.5, 1.0]
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler

    samples_p = np.asarray(samples_p)
    samples_q = np.asarray(samples_q)
    if samples_p.ndim == 1:
        samples_p = samples_p[:, None]
    if samples_q.ndim == 1:
        samples_q = samples_q[:, None]

    n = min(len(samples_p), len(samples_q))
    rng = np.random.RandomState(seed)
    idx_p = rng.choice(len(samples_p), n, replace=False)
    idx_q = rng.choice(len(samples_q), n, replace=False)

    X = np.vstack([samples_p[idx_p], samples_q[idx_q]])
    y = np.concatenate([np.ones(n), np.zeros(n)])

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    clf = LogisticRegression(max_iter=1000, random_state=seed)
    scores = cross_val_score(clf, X, y, cv=n_folds, scoring="accuracy")
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Visualisation — pairplot
# ---------------------------------------------------------------------------

def pairplot(samples: np.ndarray, points=None, limits=None, labels=None,
             fig=None, kde: bool = False, alpha: float = 0.4,
             point_color: str = "red"):
    """
    Triangle (lower-triangle) scatter / KDE plot of posterior samples.

    Parameters
    ----------
    samples : ndarray (N, D)
    points  : ndarray (M, D) | None  — highlight specific parameter values
    limits  : list[(lo, hi)] | None  — axis limits per dimension
    labels  : list[str] | None
    fig     : matplotlib Figure | None
    kde     : bool  — overlay kernel-density estimate on diagonal
    alpha   : float — scatter point transparency
    point_color : str

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    samples = np.asarray(samples)
    if samples.ndim == 1:
        samples = samples[:, None]

    D = samples.shape[1]
    if labels is None:
        labels = [f"θ[{d}]" for d in range(D)]

    if fig is None:
        fig, axes = plt.subplots(D, D, figsize=(2.5 * D, 2.5 * D))
    else:
        axes = np.array(fig.axes).reshape(D, D)

    if D == 1:
        axes = np.array([[axes]])

    for row in range(D):
        for col in range(D):
            ax = axes[row, col]
            if col > row:
                ax.set_visible(False)
                continue

            if row == col:
                # Diagonal: 1D histogram (+ optional KDE)
                ax.hist(samples[:, row], bins=30, density=True,
                        color="steelblue", alpha=0.6, edgecolor="white")
                if kde:
                    from scipy.stats import gaussian_kde
                    kde_fn = gaussian_kde(samples[:, row])
                    xs = np.linspace(samples[:, row].min(),
                                     samples[:, row].max(), 200)
                    ax.plot(xs, kde_fn(xs), color="navy", linewidth=1.5)
                if points is not None:
                    pts = np.atleast_2d(points)
                    for pt in pts:
                        ax.axvline(pt[row], color=point_color,
                                   linewidth=1.5, linestyle="--")
            else:
                # Lower triangle: 2D scatter
                ax.scatter(samples[:, col], samples[:, row],
                           s=3, alpha=alpha, color="steelblue",
                           rasterized=True)
                if points is not None:
                    pts = np.atleast_2d(points)
                    ax.scatter(pts[:, col], pts[:, row],
                               color=point_color, s=40, zorder=5,
                               marker="x", linewidths=2)

            if limits is not None:
                ax.set_xlim(limits[col])
                if row != col:
                    ax.set_ylim(limits[row])

            # Labels only on outer edges
            if row == D - 1:
                ax.set_xlabel(labels[col], fontsize=9)
            else:
                ax.set_xticklabels([])
            if col == 0 and row != col:
                ax.set_ylabel(labels[row], fontsize=9)
            elif col == 0 and row == 0:
                ax.set_ylabel("density", fontsize=9)
            else:
                ax.set_yticklabels([])

    fig.tight_layout()
    return fig


def conditional_pairplot(posterior, x_obs, n_samples: int = 1000,
                         points=None, labels=None, seed: int = 0,
                         **pairplot_kwargs):
    """
    Draw samples from the posterior conditioned on ``x_obs`` and call
    :func:`pairplot`.

    Parameters
    ----------
    posterior : Posterior
    x_obs     : array_like  — observed data
    n_samples : int
    points    : array_like | None  — highlight (e.g. true parameters)
    labels    : list[str] | None
    seed      : int
    **pairplot_kwargs : forwarded to :func:`pairplot`

    Returns
    -------
    matplotlib.figure.Figure
    """
    samples = posterior.sample((n_samples,), x=x_obs, seed=seed)
    return pairplot(np.asarray(samples), points=points,
                    labels=labels, **pairplot_kwargs)


# ---------------------------------------------------------------------------
# Loss plot
# ---------------------------------------------------------------------------

def plot_loss(loss_history, val_loss_history=None, ax=None,
              log_scale: bool = False):
    """
    Plot training (and optional validation) loss vs iteration.

    Parameters
    ----------
    loss_history     : sequence of float — training loss per iteration
    val_loss_history : sequence of float | None — validation loss
    ax               : matplotlib Axes | None
    log_scale        : bool — use log y-axis

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 3))
    else:
        fig = ax.get_figure()

    iters = np.arange(len(loss_history))
    ax.plot(iters, loss_history, label="train", color="steelblue", linewidth=1.5)

    if val_loss_history is not None:
        val_iters = np.linspace(0, len(loss_history) - 1, len(val_loss_history))
        ax.plot(val_iters, val_loss_history,
                label="val", color="darkorange", linewidth=1.5, linestyle="--")

    if log_scale:
        ax.set_yscale("log")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss (NLL)")
    ax.set_title("Training loss")
    ax.legend()
    fig.tight_layout()
    return fig
