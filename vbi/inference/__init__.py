"""
vbi.inference — torch-free SBI engine with sbi-compatible API.

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
    theta = theta_np   # plain numpy — nothing else changes
"""

from ._api        import SNPE, SNLE
from ._prior      import BoxUniform, Gaussian, CustomPrior
from ._posterior  import Posterior
from ._embedding  import EmbeddingNet
from ._utils      import simulate_for_vbi, simulate_for_sbi
from ._estimators import (
    ConditionalDensityEstimator,
    MDNEstimator,
    MAFEstimator,
    MAFEstimator0,
)

__all__ = [
    # High-level sbi-compatible API
    "SNPE",
    "SNLE",
    # Priors
    "BoxUniform",
    "Gaussian",
    "CustomPrior",
    # Posterior
    "Posterior",
    # Embedding
    "EmbeddingNet",
    # Utilities
    "simulate_for_vbi",
    "simulate_for_sbi",
    # Low-level estimators (for direct use / subclassing)
    "ConditionalDensityEstimator",
    "MDNEstimator",
    "MAFEstimator",
    "MAFEstimator0",
]
