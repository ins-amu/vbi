"""
Simulation and training utilities — sbi-compatible helpers.

MI0-utils: simulate_for_sbi, process_prior
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


def simulate_for_vbi(
    simulator_fn,
    prior,
    num_simulations: int,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run a simulator for ``num_simulations`` parameter draws and collect
    ``(theta, x)`` pairs, mirroring ``sbi.utils.simulate_for_sbi``.

    Failed simulations (exceptions or non-finite x) are replaced with NaN
    rows so that ``SNPE.append_simulations(..., exclude_invalid_x=True)``
    silently filters them out.

    Parameters
    ----------
    simulator_fn   : callable  ``theta_1d -> x_1d``  (one simulation at a time)
    prior          : prior object with ``.sample((n,))``
    num_simulations : int
    seed           : int | None   RNG seed for prior sampling

    Returns
    -------
    theta : ndarray  (num_simulations, d_theta)
    x     : ndarray  (num_simulations, d_x)

    Examples
    --------
    >>> theta, x = simulate_for_sbi(my_sim, prior, num_simulations=1000)
    >>> inference.append_simulations(theta, x)
    """
    process_prior(prior)

    rng       = np.random.default_rng(seed)
    prior_seed = int(rng.integers(0, 2 ** 31))
    theta     = prior.sample((num_simulations,), seed=prior_seed)

    x_list: list[np.ndarray | None] = []
    x_dim:  int | None = None
    failed = 0

    for th in theta:
        try:
            x_i = np.asarray(simulator_fn(th), dtype=np.float32)
            if x_i.ndim == 0:
                x_i = x_i.reshape(1)
            if x_dim is None:
                x_dim = int(x_i.shape[0])
            x_list.append(x_i)
        except Exception:
            x_list.append(None)
            failed += 1

    # Replace None placeholders (failures before x_dim was known, or just None)
    fill = np.full(x_dim or 1, np.nan, dtype=np.float32)
    x_list = [xi if xi is not None else fill for xi in x_list]

    if failed:
        log.warning(
            "simulate_for_sbi: %d / %d simulations failed (NaN rows added; "
            "will be filtered by append_simulations).",
            failed, num_simulations,
        )

    x = np.stack(x_list)
    return theta, x


def process_prior(prior) -> object:
    """
    Validate that *prior* exposes the required ``.sample`` and ``.log_prob``
    interface, mirroring ``sbi.utils.process_prior``.

    Parameters
    ----------
    prior : object

    Returns
    -------
    prior  (returned unchanged for chaining)

    Raises
    ------
    ValueError  if the prior is missing required methods.
    """
    if not (hasattr(prior, "sample") and callable(prior.sample)):
        raise ValueError(
            "prior must have a callable .sample(sample_shape) method. "
            f"Got {type(prior).__name__!r}."
        )
    if not (hasattr(prior, "log_prob") and callable(prior.log_prob)):
        raise ValueError(
            "prior must have a callable .log_prob(theta) method. "
            f"Got {type(prior).__name__!r}."
        )
    return prior


# For backward compatibility
simulate_for_sbi = simulate_for_vbi
