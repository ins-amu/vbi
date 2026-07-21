"""
vbi.inference - torch-free SBI engine with sbi-compatible API.

Quick start
-----------
>>> from vbi.inference import SNPE, BoxUniform
>>> import numpy as np
>>>
>>> prior     = BoxUniform(low=np.array([0., -5.]), high=np.array([2., 0.]))
>>> inference = SNPE(prior=prior, density_estimator='maf')
>>> inference = inference.append_simulations(theta, x)
>>> estimator = inference.train(training_batch_size=256, stop_after_epochs=20)
>>> posterior = inference.build_posterior(estimator)
>>> samples   = posterior.sample((1000,), x=x_obs)
>>> log_probs = posterior.log_prob(theta, x=x_obs)

Migration from sbi
------------------
Replace::

    from sbi.inference import SNPE
    from sbi.utils     import BoxUniform
    theta = torch.tensor(theta_np, dtype=torch.float32)

With::

    from vbi.inference import SNPE, BoxUniform
    theta = theta_np   # plain numpy - nothing else changes
"""

from ._api               import SNPE, SNLE, set_jax_device
from ._inference_pipeline import InferencePipeline, use_sbi
from ._training        import TrainingOptions
from ._mcmc       import MetropolisHastings, HMC, NUTS, r_hat, effective_sample_size
from ._prior      import (
    BoxUniform, Gaussian, CustomPrior,
    MultivariateNormal, LogNormal, Gamma, Beta,
    MultipleIndependent, RestrictedPrior,
)
from ._posterior  import Posterior
from ._embedding  import EmbeddingNet
from ._utils      import (
    simulate_for_vbi, simulate_for_sbi, process_prior,
    simulate_for_vbi_sweep,
    simulate_for_vbi_sweep_cached,
    extract_from_cache,
)
from ._estimators import (
    ConditionalDensityEstimator,
    MDNEstimator,
    MAFEstimator,
    MAFEstimator0,
    NSFEstimator,
    MAF,
    MDN,
    NSF,
)
from ._diagnostics import (
    run_sbc,
    check_sbc,
    sbc_rank_plot,
    pp_plot,
    run_tarp,
    check_tarp,
    plot_tarp,
    c2st,
    lc2st,
    pairplot,
    conditional_pairplot,
    plot_loss,
)

__all__ = [
    # High-level sbi-compatible API
    "SNPE",
    "SNLE",
    "set_jax_device",
    # End-to-end workflow
    "InferencePipeline",
    "use_sbi",
    "TrainingOptions",
    # Priors
    "BoxUniform",
    "Gaussian",
    "CustomPrior",
    "MultivariateNormal",
    "LogNormal",
    "Gamma",
    "Beta",
    "MultipleIndependent",
    "RestrictedPrior",
    # Posterior
    "Posterior",
    # Embedding
    "EmbeddingNet",
    # Utilities
    "simulate_for_vbi",
    "simulate_for_sbi",
    "process_prior",
    "simulate_for_vbi_sweep",
    "simulate_for_vbi_sweep_cached",
    "extract_from_cache",
    # Low-level estimators (for direct use / subclassing)
    "ConditionalDensityEstimator",
    "MDNEstimator",
    "MAFEstimator",
    "MAFEstimator0",
    "NSFEstimator",
    # Backend-selecting estimator factories
    "MAF",
    "MDN",
    "NSF",
    # MCMC samplers (MI4)
    "MetropolisHastings",
    "HMC",
    "NUTS",   # backward-compatible alias for HMC
    "r_hat",
    "effective_sample_size",
    # Diagnostics
    "run_sbc",
    "check_sbc",
    "sbc_rank_plot",
    "pp_plot",
    "run_tarp",
    "check_tarp",
    "plot_tarp",
    "c2st",
    "lc2st",
    "pairplot",
    "conditional_pairplot",
    "plot_loss",
]
