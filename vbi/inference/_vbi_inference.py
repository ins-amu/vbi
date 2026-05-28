"""
VBIInference — end-to-end SBI workflow object.  MI6.

Owns the full loop: prior sampling → sweep simulation → feature extraction
→ SNPE training → posterior.  Raw recordings can optionally be cached to
disk so features can be recomputed with different pipeline settings later.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ._api import SNPE
from ._utils import (
    simulate_for_vbi_sweep,
    simulate_for_vbi_sweep_cached,
    extract_from_cache as _extract_from_cache,
    _extract_from_cache_impl,
)

if TYPE_CHECKING:
    from ._posterior import Posterior
    from ._estimators import ConditionalDensityEstimator

log = logging.getLogger(__name__)


class VBIInference:
    """
    End-to-end VBI inference workflow.

    Owns the full loop: prior sampling → sweep simulation → feature
    extraction → SNPE training → posterior.

    Parameters
    ----------
    sim_spec : SimulationSpec
    prior : prior object
        Must expose ``._resolved_param_names``, ``.sample()``, ``.log_prob()``.
    pipeline : FeaturePipeline
    density_estimator : 'maf' | 'mdn' | 'nsf'
    sim_backend : str
        Backend for the Sweeper ('numpy' | 'numba' | 'cuda' | 'jax').
    backend : str
        Backend for the density estimator ('auto' | 'numpy' | 'jax').
    show_progress_bars : bool
    embedding_net : EmbeddingNet | None

    Examples
    --------
    >>> inf = VBIInference(sim_spec, prior, pipeline)
    >>> theta, x = inf.simulate(num_simulations=2000, duration=5000.0)
    >>> est = inf.train(training_batch_size=256, stop_after_epochs=30)
    >>> post = inf.build_posterior(est)
    >>> samples = post.sample((1000,), x=x_obs)
    >>> inf.save("run.npz")
    """

    def __init__(
        self,
        sim_spec,
        prior,
        pipeline,
        density_estimator: str = "maf",
        sim_backend: str = "numba",
        backend: str = "auto",
        show_progress_bars: bool = True,
        embedding_net=None,
    ):
        self._sim_spec          = sim_spec
        self._prior             = prior
        self._pipeline          = pipeline
        self._sim_backend       = sim_backend
        self._de_type           = density_estimator
        self._snpe              = SNPE(
            prior               = prior,
            density_estimator   = density_estimator,
            backend             = backend,
            show_progress_bars  = show_progress_bars,
            embedding_net       = embedding_net,
        )
        self._feature_labels: list[str] | None = None
        self._param_names:    list[str] | None = None
        self._default_train_kwargs: dict       = {}
        self._last_estimator                   = None

    # ------------------------------------------------------------------
    # Core workflow
    # ------------------------------------------------------------------

    def simulate(
        self,
        num_simulations: int,
        duration: float,
        seed: int | None = None,
        proposal=None,
        x_obs=None,
        cache_dir=None,
        chunk_size: int = 500,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Sample parameters and run the sweep simulator.

        Appends the resulting ``(theta, x)`` pairs to the internal SNPE
        object, ready for training.

        Parameters
        ----------
        num_simulations : int
        duration : float  simulation length in ms
        seed : int | None
        proposal : Posterior | None
            When set, sample theta from this posterior instead of the prior.
            Requires ``x_obs``.
        x_obs : ndarray | None  required when proposal is not None
        cache_dir : str | Path | None
            When set, raw monitor recordings are written to this directory
            in chunked ``.npz`` files.  Use ``extract_from_cache()`` later
            to re-extract features with a different pipeline.
        chunk_size : int
            Simulations per chunk (only used when cache_dir is set).
            Tune to stay within GPU/RAM budget.

        Returns
        -------
        theta : (n, d_theta) ndarray
        x     : (n, d_x)     ndarray
        """
        common = dict(
            sim_spec        = self._sim_spec,
            prior           = self._prior,
            pipeline        = self._pipeline,
            num_simulations = num_simulations,
            duration        = duration,
            sim_backend     = self._sim_backend,
            seed            = seed,
            proposal        = proposal,
            x_obs           = x_obs,
        )

        if cache_dir is None:
            theta, x, param_names, feat_labels = simulate_for_vbi_sweep(**common)
        else:
            theta, x, param_names, feat_labels = simulate_for_vbi_sweep_cached(
                **common, cache_dir=cache_dir, chunk_size=chunk_size
            )

        if self._param_names is None:
            self._param_names = param_names
        if self._feature_labels is None:
            self._feature_labels = feat_labels

        self._snpe.append_simulations(theta, x, proposal=proposal)
        return theta, x

    def train(self, **train_kwargs) -> "ConditionalDensityEstimator":
        """
        Train the density estimator on all accumulated simulations.

        Keyword arguments are forwarded to ``SNPE.train()``.  Any kwargs
        stored via ``from_config`` serve as defaults and can be overridden
        here.

        Returns
        -------
        estimator : ConditionalDensityEstimator
        """
        if self._snpe.n_simulations == 0:
            raise RuntimeError(
                "No simulations available.  Call simulate() before train()."
            )
        merged = {**self._default_train_kwargs, **train_kwargs}
        self._last_estimator = self._snpe.train(**merged)
        return self._last_estimator

    def build_posterior(
        self,
        estimator=None,
        sample_with: str = "direct",
        **kwargs,
    ) -> "Posterior":
        """
        Wrap the trained estimator in a Posterior object.

        Parameters
        ----------
        estimator : ConditionalDensityEstimator | None
            Uses the last one returned by ``train()`` if None.
        sample_with : 'direct' | 'mcmc' | 'rejection'

        Returns
        -------
        Posterior
        """
        est = estimator if estimator is not None else self._last_estimator
        return self._snpe.build_posterior(
            density_estimator=est, sample_with=sample_with, **kwargs
        )

    def get_simulations(self, starting_round: int = 0):
        """
        Return all accumulated (theta, x, proposals) from ``starting_round``.

        Returns
        -------
        theta     : (N, d_theta)
        x         : (N, d_x)
        proposals : list
        """
        return self._snpe.get_simulations(starting_round)

    # ------------------------------------------------------------------
    # Cache helper (static — usable without an instance)
    # ------------------------------------------------------------------

    @staticmethod
    def extract_from_cache(cache_dir, pipeline) -> tuple[np.ndarray, np.ndarray]:
        """
        Load cached raw recordings and extract features with a new pipeline.

        Parameters
        ----------
        cache_dir : str | Path
        pipeline  : FeaturePipeline

        Returns
        -------
        theta : (n, d_theta) float64
        x     : (n, d_x)     float64
        """
        return _extract_from_cache(cache_dir, pipeline)

    # ------------------------------------------------------------------
    # Save / load  (Step 4)
    # ------------------------------------------------------------------

    def save(self, path) -> None:
        """Persist inference state to a ``.npz`` checkpoint.  Step 4."""
        raise NotImplementedError("save/load coming in Step 4.")

    @classmethod
    def load(cls, path, sim_spec, pipeline, prior=None, **kwargs) -> "VBIInference":
        """Restore from a checkpoint written by ``save()``.  Step 4."""
        raise NotImplementedError("save/load coming in Step 4.")

    # ------------------------------------------------------------------
    # Config loader  (Step 5)
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config) -> "VBIInference":
        """Build from a YAML / JSON config dict or file path.  Step 5."""
        raise NotImplementedError("from_config coming in Step 5.")

    # ------------------------------------------------------------------
    # Diagnostics  (Step 6)
    # ------------------------------------------------------------------

    def plot_loss(self):
        """Plot training / validation loss curves.  Step 6."""
        raise NotImplementedError("Diagnostics coming in Step 6.")

    def pairplot(self, x_obs, num_samples: int = 1000, **kwargs):
        """Pairplot of posterior samples at x_obs.  Step 6."""
        raise NotImplementedError("Diagnostics coming in Step 6.")

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"VBIInference("
            f"density_estimator={self._de_type!r}, "
            f"sim_backend={self._sim_backend!r}, "
            f"n_rounds={self._snpe.n_rounds}, "
            f"n_sims={self._snpe.n_simulations})"
        )
