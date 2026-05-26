"""
sbi-compatible high-level inference API.

SNPE mirrors ``sbi.inference.SNPE`` exactly — same method names, same kwarg
names, same call signatures — but operates on numpy arrays and has no torch
dependency.

Migration from sbi:
    # Before
    from sbi.inference import SNPE
    theta = torch.tensor(theta_np, dtype=torch.float32)
    x     = torch.tensor(x_np,     dtype=torch.float32)

    # After
    from vbi.inference import SNPE
    theta, x = theta_np, x_np   # plain numpy — nothing else changes
"""
from __future__ import annotations

import logging
from typing import Literal

import numpy as np

from ._estimators import MAFEstimator, MDNEstimator, ConditionalDensityEstimator
from ._posterior   import Posterior
from ._backends    import resolve_backend

log = logging.getLogger(__name__)

# Map sbi density_estimator strings to internal classes
_DE_MAP = {
    "maf": MAFEstimator,
    "mdn": MDNEstimator,
    # "nsf": NSFEstimator,   # TODO: MI1+
    # "made": MADEEstimator, # TODO: future
}


class SNPE:
    """
    Sequential Neural Posterior Estimation — torch-free, numpy-native.

    Mirrors ``sbi.inference.SNPE`` (NPE-C) API.  Multi-round sequential
    inference is supported via repeated ``append_simulations`` calls;
    the current training always fits a single amortised posterior over
    all accumulated simulations (full APT / round weighting added in MI3).

    Parameters
    ----------
    prior : BoxUniform | Gaussian | CustomPrior | None
        Prior distribution.  Used by ``build_posterior`` to add the prior
        log-probability to ``posterior.log_prob``.
    density_estimator : 'maf' | 'mdn'
        Which neural architecture to use.  'maf' (Masked Autoregressive
        Flow) is the default, matching sbi's default.
    backend : 'auto' | 'numpy' | 'numba' | 'jax'
        Computation backend.  'auto' picks JAX when available, else numpy.
        (numba and jax backends added in MI-numba / MI1.)
    show_progress_bars : bool
        Show tqdm progress during training.

    Examples
    --------
    >>> from vbi.inference import SNPE, BoxUniform
    >>> import numpy as np
    >>> prior = BoxUniform(low=np.array([0.]), high=np.array([1.]))
    >>> inference = SNPE(prior=prior, density_estimator='maf')
    >>> inference = inference.append_simulations(theta, x)
    >>> estimator = inference.train(training_batch_size=256, stop_after_epochs=20)
    >>> posterior = inference.build_posterior(estimator)
    >>> samples   = posterior.sample((1000,), x=x_obs)
    """

    def __init__(
        self,
        prior=None,
        density_estimator: Literal["maf", "mdn"] = "maf",
        backend: str = "auto",
        show_progress_bars: bool = True,
        **kwargs,
    ):
        if density_estimator not in _DE_MAP:
            raise ValueError(
                f"density_estimator={density_estimator!r} not supported. "
                f"Choose from: {list(_DE_MAP)}."
            )
        self._prior_obj        = prior
        self._de_type          = density_estimator
        self._backend          = resolve_backend(backend)
        self._show_progress    = show_progress_bars
        self._rounds: list[tuple[np.ndarray, np.ndarray, object]] = []
        self._estimator: ConditionalDensityEstimator | None = None

    # ------------------------------------------------------------------
    # sbi-compatible method chain
    # ------------------------------------------------------------------

    def append_simulations(
        self,
        theta,
        x,
        proposal=None,
        exclude_invalid_x: bool = True,
        data_device: str | None = None,
    ) -> "SNPE":
        """
        Store a round of simulations.

        Parameters
        ----------
        theta : array (n, d_theta)
        x     : array (n, d_x)
        proposal : posterior object | None
            Proposal used to generate theta (for importance-weighted
            multi-round training).  Currently stored but not used;
            full APT weighting is added in MI3.
        exclude_invalid_x : bool
            Drop rows where x contains NaN/inf.

        Returns
        -------
        self  (for method chaining)
        """
        theta = np.asarray(theta, dtype=np.float32)
        x     = np.asarray(x,     dtype=np.float32)

        if theta.ndim == 1: theta = theta[:, None]
        if x.ndim     == 1: x     = x[:, None]

        if exclude_invalid_x:
            ok    = np.all(np.isfinite(x), axis=1) & np.all(np.isfinite(theta), axis=1)
            theta, x = theta[ok], x[ok]
            dropped = (~ok).sum()
            if dropped:
                log.info("append_simulations: dropped %d rows with non-finite values.", dropped)

        self._rounds.append((theta, x, proposal))
        return self

    def train(
        self,
        # sbi-compatible kwargs (same names and defaults as sbi 0.26)
        training_batch_size: int        = 200,
        learning_rate: float            = 5e-4,
        validation_fraction: float      = 0.1,
        stop_after_epochs: int          = 20,
        max_num_epochs: int             = 2000,
        clip_max_norm: float | None     = 5.0,
        num_atoms: int                  = 10,   # SNPE-C; stored, ignored until MI3
        show_train_summary: bool        = False,
        verbose: bool | None            = None,   # overrides show_progress_bars + show_train_summary
        # Collapse-prevention (not in sbi; vbi.inference extension)
        early_stopping_delta: float | None = None,
        lr_schedule: str | None         = "cosine",
        lr_min: float                   = 1e-5,
        lr_period: int                  = 500,
        monitor_collapse: bool          = False,
        x_check                         = None,
        collapse_threshold: float       = 0.05,
        check_every: int                = 10,
        n_check: int                    = 200,
        # Extra kwargs forwarded to the estimator (e.g. seed, batch_size)
        **kwargs,
    ) -> ConditionalDensityEstimator:
        """
        Train the density estimator on all accumulated simulations.

        All parameter names and defaults match ``sbi.inference.SNPE.train()``.

        Parameters
        ----------
        training_batch_size : int
            Mini-batch size per gradient step.
        learning_rate : float
        validation_fraction : float
            Fraction of training data held out for early stopping.
        stop_after_epochs : int
            Stop if validation loss does not improve for this many epochs.
        max_num_epochs : int
            Hard cap on training epochs.  With ``monitor_collapse=True`` the
            training will stop automatically before this limit when the
            posterior collapses, so you rarely need to tune this manually.
        clip_max_norm : float | None
            Global gradient norm clip.  None disables clipping.
        num_atoms : int
            SNPE-C atoms; accepted for API compatibility, used in MI3.
        show_train_summary : bool
            Alias for verbose tqdm output.
        early_stopping_delta : float | None
            Minimum validation-loss improvement to reset patience counter.
            None → auto: 1e-4 when lr_schedule='cosine', else 0.0.
            With cosine LR, tiny late-epoch improvements (< lr_min × grad)
            would otherwise keep patience at 0 forever.
        lr_schedule : 'cosine' | None
            Cosine-anneal ``learning_rate`` → ``lr_min`` over the first
            ``lr_period`` epochs, then stay at lr_min.  None keeps lr fixed.
        lr_min : float
            Floor for cosine LR schedule.
        lr_period : int
            Epochs over which cosine annealing runs (default 500).
        monitor_collapse : bool
            **Opt-in vbi extension (default False, not in sbi-compatible path).**
            If True, periodically samples the posterior and restores the
            last-healthy checkpoint when std drops below threshold.
            Use with care — may stop training earlier than desired.
        collapse_threshold : float
            Collapse declared when std < threshold × max_seen_std.
        check_every : int
            Epochs between collapse checks.

        Returns
        -------
        estimator : ConditionalDensityEstimator
            Trained estimator (pass to ``build_posterior``).
        """
        if not self._rounds:
            raise RuntimeError(
                "No simulations appended.  Call append_simulations(theta, x) first."
            )

        # Concatenate all rounds
        theta_all = np.concatenate([r[0] for r in self._rounds], axis=0)
        x_all     = np.concatenate([r[1] for r in self._rounds], axis=0)

        # Build fresh estimator
        self._estimator = _DE_MAP[self._de_type]()

        # verbose kwarg overrides the constructor-level show_progress_bars
        if verbose is None:
            verbose = self._show_progress or show_train_summary

        # Dispatch to estimator.train() with mapped kwargs
        common = dict(
            params                = theta_all,
            features              = x_all,
            n_iter                = max_num_epochs,
            learning_rate         = learning_rate,
            batch_size            = training_batch_size,
            verbose               = verbose,
            validation_fraction   = validation_fraction,
            stop_after_epochs     = stop_after_epochs,
            early_stopping_delta  = early_stopping_delta,
            clip_max_norm         = clip_max_norm,
            **kwargs,              # forward seed, etc.
        )

        if isinstance(self._estimator, MAFEstimator):
            common.update(dict(
                lr_schedule         = lr_schedule,
                lr_min              = lr_min,
                lr_period           = lr_period,
                monitor_collapse    = monitor_collapse,
                x_check             = x_check,
                collapse_threshold  = collapse_threshold,
                check_every         = check_every,
                n_check             = n_check,
            ))
            self._estimator.train(**common)
        else:
            # MDNEstimator uses base train() — subset of kwargs
            base_kwargs = {k: v for k, v in common.items()
                           if k in ("params", "features", "n_iter",
                                    "learning_rate", "batch_size",
                                    "verbose", "patience")}
            base_kwargs["patience"] = stop_after_epochs
            self._estimator.train(**base_kwargs)

        return self._estimator

    def build_posterior(
        self,
        density_estimator=None,
        prior=None,
        sample_with: Literal["direct", "mcmc", "rejection"] = "direct",
        **kwargs,
    ) -> Posterior:
        """
        Wrap the trained estimator in a Posterior object.

        Parameters
        ----------
        density_estimator : ConditionalDensityEstimator | None
            Trained estimator returned by ``train()``.  Uses the internally
            stored one if None.
        prior : prior object | None
            Uses the prior passed to ``__init__`` if None.
        sample_with : 'direct' | 'mcmc' | 'rejection'
            'direct' (default) — ancestral sampling from the flow/mixture.
            'mcmc'             — Metropolis-Hastings (added in MI4).
            'rejection'        — rejection sampling against prior (added in MI4).

        Returns
        -------
        Posterior
        """
        est = density_estimator if density_estimator is not None else self._estimator
        if est is None:
            raise RuntimeError(
                "No trained estimator available.  Call train() first, or pass "
                "a trained estimator to build_posterior()."
            )

        p = prior if prior is not None else self._prior_obj

        if sample_with != "direct":
            raise NotImplementedError(
                f"sample_with={sample_with!r} is not yet implemented.  "
                "MCMC and rejection sampling are planned for MI4.  "
                "Use sample_with='direct' (the default)."
            )

        return Posterior(estimator=est, prior=p)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def n_rounds(self) -> int:
        """Number of simulation rounds appended so far."""
        return len(self._rounds)

    @property
    def n_simulations(self) -> int:
        """Total number of (theta, x) pairs across all rounds."""
        return sum(r[0].shape[0] for r in self._rounds)

    def __repr__(self):
        return (f"SNPE(density_estimator={self._de_type!r}, "
                f"backend={self._backend!r}, "
                f"rounds={self.n_rounds}, "
                f"n_sims={self.n_simulations})")


class SNLE:
    """
    Sequential Neural Likelihood Estimation — placeholder.

    Full implementation planned for a future milestone.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "SNLE is not yet implemented in vbi.inference.  "
            "Use SNPE, or install sbi for the full SNLE implementation."
        )
