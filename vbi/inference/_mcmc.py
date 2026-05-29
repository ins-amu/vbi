"""
MCMC posterior samplers for simulation-based inference (MI4).

Classes
-------
MetropolisHastings
    Random-walk MH using the trained neural posterior as a log-likelihood
    surrogate.  Works with any backend (numpy / JAX).
HMC
    Hamiltonian Monte Carlo using JAX gradients with fixed-length leapfrog
    trajectories and dual-averaging step-size adaptation.
    Requires a JAX-backend estimator.

Aliases
-------
NUTS = HMC  (backward-compatible name; the sampler is HMC, not NUTS)

Diagnostics
-----------
r_hat(chains)            Gelman-Rubin convergence statistic.
effective_sample_size    Per-parameter ESS.
"""
from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def r_hat(chains: np.ndarray) -> np.ndarray:
    """
    Gelman-Rubin R-hat convergence diagnostic.

    Parameters
    ----------
    chains : ndarray (n_chains, n_samples, param_dim)

    Returns
    -------
    ndarray (param_dim,)  - values near 1.0 indicate convergence.
    """
    chains = np.asarray(chains)
    if chains.ndim == 2:
        chains = chains[None]   # treat as single chain
    M, N, D = chains.shape
    if M <= 1 or N < 2:
        return np.full(D, np.nan)

    chain_means = chains.mean(axis=1)          # (M, D)
    grand_mean  = chain_means.mean(axis=0)     # (D,)

    # Between-chain variance
    B = N / (M - 1) * np.sum((chain_means - grand_mean) ** 2, axis=0)

    # Within-chain variance
    W = np.mean(np.var(chains, axis=1, ddof=1), axis=0)

    var_hat = (N - 1) / N * W + B / N
    return np.sqrt(var_hat / (W + 1e-30))


def effective_sample_size(chain: np.ndarray) -> np.ndarray:
    """
    Bulk effective sample size via autocorrelation sum.

    Parameters
    ----------
    chain : ndarray (n_samples, param_dim)

    Returns
    -------
    ndarray (param_dim,)
    """
    chain = np.asarray(chain)
    if chain.ndim == 1:
        chain = chain[:, None]
    N, D = chain.shape
    ess = np.empty(D)
    for d in range(D):
        x = chain[:, d]
        x = x - x.mean()
        # Autocorrelation via FFT
        f  = np.fft.fft(x, n=2 * N)
        ac = np.real(np.fft.ifft(f * np.conj(f)))[:N] / (N - np.arange(N))
        ac = ac / (ac[0] + 1e-30)
        # Sum pairs until first non-positive pair
        rho_sum = 1.0
        for t in range(1, N // 2):
            pair = ac[2 * t - 1] + ac[2 * t]
            if pair <= 0:
                break
            rho_sum += 2 * pair
        ess[d] = N / max(rho_sum, 1.0)
    return ess


# ---------------------------------------------------------------------------
# MetropolisHastings
# ---------------------------------------------------------------------------

class MetropolisHastings:
    """
    Random-walk Metropolis-Hastings sampler.

    Uses ``estimator.log_prob(x_obs, theta)`` as the MCMC target, treating the
    trained SNPE network as an approximation to ``log p(theta | x)``.

    If a prior is supplied it is used only as a **support mask**: proposals
    outside the prior support (where ``prior.log_prob`` returns ``-inf``) are
    rejected immediately.  The prior density is *not* added to the target, so
    MCMC and direct sampling both target the same distribution.

    Parameters
    ----------
    estimator : ConditionalDensityEstimator
        Trained density estimator (any backend).
    prior : prior object | None
        Prior with ``.log_prob(theta)`` method.  Used for support masking only.
    step_size : float | ndarray
        Gaussian proposal std.  Scalar or per-parameter vector.
    """

    def __init__(self, estimator, prior=None, step_size: float | np.ndarray = 0.1):
        self._est  = estimator
        self._prior = prior
        self._step = np.asarray(step_size, dtype="f")

    def _log_target(self, theta: np.ndarray, x_obs: np.ndarray) -> float:
        """Log posterior (unnormalized) at theta given x_obs."""
        theta_2d = np.atleast_2d(theta).astype("f")
        x_2d     = np.atleast_2d(x_obs).astype("f")
        if x_2d.shape[0] == 1 and theta_2d.shape[0] > 1:
            x_2d = np.repeat(x_2d, theta_2d.shape[0], axis=0)
        # Support mask: reject immediately if outside prior support
        if self._prior is not None:
            lp_prior = float(np.array(self._prior.log_prob(theta_2d))[0])
            if not np.isfinite(lp_prior):
                return -np.inf
        return float(np.array(self._est.log_prob(x_2d, theta_2d))[0])

    def run(
        self,
        x_obs,
        n_samples: int,
        n_warmup: int = 500,
        seed: int | None = None,
        init_theta: np.ndarray | None = None,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Run the chain and return posterior samples.

        Parameters
        ----------
        x_obs : array  (1, feature_dim) or (feature_dim,)
        n_samples : int
            Number of samples to return (after warmup is discarded).
        n_warmup : int
            Warmup/burn-in steps (discarded).
        seed : int | None
        init_theta : ndarray (param_dim,) | None
            Starting point.  Defaults to a draw from the prior when available,
            otherwise zeros.
        show_progress : bool

        Returns
        -------
        ndarray  (n_samples, param_dim)
        """
        from tqdm.auto import trange

        x_obs = np.atleast_2d(np.asarray(x_obs, dtype="f"))
        rng   = np.random.default_rng(seed)

        param_dim = self._est.param_dim
        if init_theta is not None:
            theta_cur = np.asarray(init_theta, dtype="f")
        elif self._prior is not None:
            try:
                theta_cur = np.array(self._prior.sample((1,), seed=seed), dtype="f").flatten()
            except Exception:
                theta_cur = np.zeros(param_dim, dtype="f")
        else:
            theta_cur = np.zeros(param_dim, dtype="f")

        log_p_cur = self._log_target(theta_cur, x_obs)

        samples          = np.empty((n_samples, param_dim), dtype="f")
        n_total          = n_warmup + n_samples
        warmup_accepted  = 0
        sample_accepted  = 0

        it = trange(n_total, desc="MH sampling", disable=not show_progress)
        for i in it:
            step    = self._step if self._step.ndim > 0 else float(self._step)
            theta_p = theta_cur + rng.normal(0, step, size=param_dim).astype("f")
            log_p_p = self._log_target(theta_p, x_obs)

            log_alpha = log_p_p - log_p_cur
            if math.log(rng.uniform() + 1e-300) < log_alpha:
                theta_cur, log_p_cur = theta_p, log_p_p
                if i < n_warmup:
                    warmup_accepted += 1
                else:
                    sample_accepted += 1

            if i >= n_warmup:
                samples[i - n_warmup] = theta_cur

        warmup_rate = warmup_accepted / max(n_warmup, 1)
        sample_rate = sample_accepted / max(n_samples, 1)
        log.info("MH warmup acceptance: %.3f  sampling acceptance: %.3f",
                 warmup_rate, sample_rate)
        if sample_rate < 0.05:
            log.warning(
                "Very low MH sampling acceptance rate (%.3f). "
                "Consider reducing step_size (current: %s).",
                sample_rate, self._step,
            )
        return samples


# ---------------------------------------------------------------------------
# HMC (JAX backend only)
# ---------------------------------------------------------------------------

class HMC:
    """
    Hamiltonian Monte Carlo sampler using JAX automatic differentiation.

    Uses fixed-length leapfrog trajectories with dual-averaging step-size
    adaptation during warmup.

    Requires a JAX-backend estimator (JaxMAFEstimator / JaxNSFEstimator /
    JaxMDNEstimator) so that ``jax.grad`` can flow through ``log_prob``.

    The MCMC target is ``estimator.log_prob(x_obs, theta)`` - the trained SNPE
    posterior - so direct sampling and HMC target the same distribution.  If a
    prior is supplied it is used only as a **support mask**: samples outside the
    prior support are rejected; no density is added.

    Parameters
    ----------
    estimator : JaxConditionalDensityEstimator
        Trained JAX-backend estimator.
    prior : prior object | None
        Prior with ``.log_prob(theta)`` method.  Used for support masking only.
        ``RestrictedPrior`` is not supported (constraint function is not
        JAX-traceable); pass it to ``MetropolisHastings`` instead.
    step_size : float
        Initial leapfrog step size.  Adapted during warmup.
    num_leapfrog_steps : int | None
        Number of leapfrog steps per sample.  When ``None``, falls back to
        ``2 ** min(max_tree_depth, 4)`` for backward compatibility.
    max_tree_depth : int
        Deprecated legacy parameter.  Use ``num_leapfrog_steps`` instead.
    """

    def __init__(
        self,
        estimator,
        prior=None,
        step_size: float = 0.1,
        num_leapfrog_steps: int | None = None,
        max_tree_depth: int = 10,
    ):
        try:
            import jax
            import jax.numpy as jnp
        except ImportError as e:
            raise ImportError("HMC requires JAX: pip install jax jaxlib") from e

        if not hasattr(estimator, "_to_jax_key"):
            raise TypeError(
                "HMC requires a JAX-backend estimator "
                "(JaxMAFEstimator / JaxNSFEstimator / JaxMDNEstimator). "
                "Use MetropolisHastings for non-JAX estimators."
            )

        from ._prior import RestrictedPrior
        if prior is not None and isinstance(prior, RestrictedPrior):
            raise TypeError(
                "HMC does not support RestrictedPrior: the constraint function "
                "is not JAX-traceable. Use MetropolisHastings instead."
            )

        self._est               = estimator
        self._prior             = prior
        self._step_size         = step_size
        self._num_leapfrog_steps = (
            num_leapfrog_steps if num_leapfrog_steps is not None
            else 2 ** min(max_tree_depth, 4)
        )
        self._max_tree_depth    = max_tree_depth

    @staticmethod
    def _jax_prior_support_mask(prior, theta):
        """
        Return 0.0 inside prior support, -1e38 outside.

        Handles BoxUniform, Beta, Gamma, LogNormal, and MultipleIndependent.
        Unbounded priors (Gaussian, MultivariateNormal) return 0.0 - their
        density is NOT added since the SNPE estimator already encodes it.
        RestrictedPrior must be rejected at construction time (not here).
        """
        import jax.numpy as jnp
        from ._prior import BoxUniform, Beta, Gamma, LogNormal, MultipleIndependent

        _IN  = jnp.array(0.0,   dtype="f")
        _OUT = jnp.array(-1e38, dtype="f")

        if isinstance(prior, BoxUniform):
            low  = jnp.array(prior.low,  dtype="f")
            high = jnp.array(prior.high, dtype="f")
            return jnp.where(jnp.all((theta >= low) & (theta <= high)), _IN, _OUT)

        if isinstance(prior, Beta):
            return jnp.where(jnp.all((theta > 0.0) & (theta < 1.0)), _IN, _OUT)

        if isinstance(prior, (Gamma, LogNormal)):
            return jnp.where(jnp.all(theta > 0.0), _IN, _OUT)

        if isinstance(prior, MultipleIndependent):
            mask = _IN
            offset = 0
            for p in prior._priors:
                d     = p.dim
                mask  = mask + HMC._jax_prior_support_mask(p, theta[offset: offset + d])
                offset += d
            return mask   # -1e38 if any component is out of support

        # Gaussian, MultivariateNormal, CustomPrior: assume full support
        return _IN

    def _make_log_prob_fn(self, x_obs_j):
        """Return a JAX-differentiable log posterior function."""
        import jax.numpy as jnp

        est   = self._est
        prior = self._prior
        w     = {k: jnp.array(v) for k, v in est.weights.items()}

        def log_prob_fn(theta):
            theta_2d = theta[None]   # (1, D)
            x_emb = est._emb.forward(w, x_obs_j) if est._emb is not None else x_obs_j
            lp = est._get_log_prob(w, x_emb, theta_2d)[0]
            if prior is not None:
                lp = lp + HMC._jax_prior_support_mask(prior, theta)
            return lp

        return log_prob_fn

    def run(
        self,
        x_obs,
        n_samples: int,
        n_warmup: int = 500,
        seed: int | None = None,
        init_theta: np.ndarray | None = None,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Run HMC and return posterior samples.

        Uses dual-averaging step-size adaptation during warmup, then fixes
        the step size for the sampling phase.

        Parameters
        ----------
        x_obs : array  (1, feature_dim) or (feature_dim,)
        n_samples : int
        n_warmup : int  Warmup steps (discarded; step-size adapted here).
        seed : int | None
        init_theta : ndarray (param_dim,) | None
            Starting point.  Defaults to a draw from the prior when available,
            otherwise zeros.

        Returns
        -------
        ndarray  (n_samples, param_dim)
        """
        import jax
        import jax.numpy as jnp
        from tqdm.auto import trange

        x_obs   = np.atleast_2d(np.asarray(x_obs, dtype="f"))
        x_obs_j = jnp.array(x_obs)
        key     = jax.random.PRNGKey(seed if seed is not None else 0)
        D       = self._est.param_dim

        if init_theta is not None:
            theta_cur = jnp.array(init_theta, dtype="f")
        elif self._prior is not None:
            try:
                theta_cur = jnp.array(
                    np.array(self._prior.sample((1,), seed=seed), dtype="f").flatten()
                )
            except Exception:
                theta_cur = jnp.zeros(D)
        else:
            theta_cur = jnp.zeros(D)

        log_prob_fn   = self._make_log_prob_fn(x_obs_j)
        log_prob_grad = jax.value_and_grad(log_prob_fn)

        step  = self._step_size
        delta = 0.65   # target acceptance rate for dual averaging
        mu    = math.log(10 * step)
        log_step_bar, H_bar, m_adapt = 0.0, 0.0, 1.0
        gamma, t0, kappa = 0.05, 10, 0.75

        samples         = np.empty((n_samples, D), dtype="f")
        n_total         = n_warmup + n_samples
        warmup_accepted = 0
        sample_accepted = 0

        it = trange(n_total, desc="HMC sampling", disable=not show_progress)
        for i in it:
            key, subkey = jax.random.split(key)
            r0 = jax.random.normal(subkey, (D,))

            lp_cur, g_cur = log_prob_grad(theta_cur)
            theta_p, r_p, lp_p, g_p, n_tree, s = self._build_tree(
                theta_cur, r0, lp_cur, g_cur,
                log_prob_grad, step, self._num_leapfrog_steps, subkey,
            )

            log_alpha = float(lp_p - lp_cur - 0.5 * (jnp.dot(r_p, r_p) - jnp.dot(r0, r0)))
            log_u = math.log(float(jax.random.uniform(key)) + 1e-300)
            if log_u < log_alpha:
                theta_cur = theta_p
                if i < n_warmup:
                    warmup_accepted += 1
                else:
                    sample_accepted += 1
            alpha = 1.0 if log_alpha >= 0 else math.exp(log_alpha)

            # Dual-averaging step-size adaptation during warmup
            if i < n_warmup:
                H_bar = (1 - 1 / (m_adapt + t0)) * H_bar + (1 / (m_adapt + t0)) * (delta - alpha)
                log_step = mu - math.sqrt(m_adapt) / gamma * H_bar
                step = math.exp(log_step)
                log_step_bar = m_adapt ** (-kappa) * log_step + (1 - m_adapt ** (-kappa)) * log_step_bar
                m_adapt += 1
            else:
                if i == n_warmup:
                    step = math.exp(log_step_bar)   # fix step after warmup

            if i >= n_warmup:
                samples[i - n_warmup] = np.array(theta_cur)

        warmup_rate = warmup_accepted / max(n_warmup, 1)
        sample_rate = sample_accepted / max(n_samples, 1)
        log.info("HMC warmup acceptance: %.3f  sampling acceptance: %.3f  final step_size: %.4f",
                 warmup_rate, sample_rate, step)
        return samples

    @staticmethod
    def _build_tree(theta, r, lp, g, log_prob_grad, step, num_leapfrog_steps, key):
        """Fixed-length leapfrog trajectory."""
        import jax.numpy as jnp

        def leapfrog(th, rv, g_val, eps):
            rv_half = rv + 0.5 * eps * g_val
            th_new  = th + eps * rv_half
            lp_new, g_new = log_prob_grad(th_new)
            rv_new  = rv_half + 0.5 * eps * g_new
            return th_new, rv_new, lp_new, g_new

        theta_p, r_p, lp_p, g_p = theta, r, lp, g
        for _ in range(num_leapfrog_steps):
            theta_p, r_p, lp_p, g_p = leapfrog(theta_p, r_p, g_p, step)

        return theta_p, r_p, lp_p, g_p, num_leapfrog_steps, True


# Backward-compatible alias
NUTS = HMC
