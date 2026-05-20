import numpy as np
import pytest
from vbi.simulator.spec import (
    SimulationSpec, IntegratorSpec, CouplingSpec, MonitorSpec,
)


# ---- Custom marks ----
# cuda_fast : deterministic, short duration — select with  pytest -m cuda_fast
# cuda_slow : stochastic, large N, pipeline, throughput benchmarks
def pytest_configure(config):
    config.addinivalue_line("markers",
        "cuda_fast: deterministic CUDA tests with short duration; fast to run")
    config.addinivalue_line("markers",
        "cuda_slow: stochastic/large-N CUDA tests; excluded in fast CI runs")
from vbi.simulator.models.mpr import mpr


def make_weights(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    W = np.abs(rng.standard_normal((n, n)))
    np.fill_diagonal(W, 0.0)
    # Row-sum normalise so that sum_j w_ij <= 1 for every node i
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    W /= row_sums
    D = rng.uniform(1.0, 100.0, (n, n))
    np.fill_diagonal(D, 0.0)
    return W, D


def make_mpr_spec(n_nodes: int = 2, dt: float = 0.01,
                  stochastic: bool = False,
                  method: str = "heun",
                  monitors: tuple = (MonitorSpec("tavg", period=1.0),),
                  seed: int = 0) -> SimulationSpec:
    W, D = make_weights(n_nodes, seed=seed)
    nsig = np.array([1e-3, 1e-3]) if stochastic else None
    return SimulationSpec(
        model=mpr,
        integrator=IntegratorSpec(method=method, dt=dt,
                                   stochastic=stochastic, noise_nsig=nsig),
        coupling=CouplingSpec("linear", a=1.0),
        monitors=monitors,
        weights=W,
        tract_lengths=D,
        speed=4.0,
    )
