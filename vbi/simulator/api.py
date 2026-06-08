from __future__ import annotations
import numpy as np

from vbi.simulator.spec.simulation import SimulationSpec
from vbi.simulator.spec.sweep import SweepSpec
from vbi.simulator.backend.base import load_backend, load_sweep_backend


class Simulator:
    """
    Single-run interface - exploration, debugging, TVB validation.

    Parameters
    ----------
    spec : SimulationSpec
    backend : str
        'numpy' (now) | 'numba' | 'cpp' | 'cuda' | 'jax'

    Examples
    --------
    sim = Simulator(spec, backend="numpy")
    result = sim.run(duration=5000.0)   # ms
    t, state = result["tavg"]           # (n_windows,), (n_windows, n_voi, n_nodes)
    """

    def __init__(self, spec: SimulationSpec, backend: str = "numpy"):
        self._impl = load_backend(backend)()
        self._impl.build(spec)

    def run(self, duration: float) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """
        Run the simulation.

        Parameters
        ----------
        duration : float   simulation length in ms

        Returns
        -------
        dict  {monitor_kind: (times, data)}
        """
        return self._impl.run(duration)


class Sweeper:
    """
    Parameter sweep interface - primary SBI use case.

    Parameters
    ----------
    spec : SimulationSpec   base simulation (model, connectivity, …)
    sweep_spec : SweepSpec  which params to vary and how many runs
    backend : str           'numpy' | 'numba' | 'cpp' | 'cuda' | 'jax'
    n_workers : int | None
        Backend worker count. For ``backend="numba"``, sets the global Numba
        thread count before the sweep backend is built. For ``backend="cpp"``,
        enables the C++ thread-pool sweep with this many workers. Ignored by
        backends without a worker-count concept.

    Examples
    --------
    # Without pipeline - returns list of monitor dicts
    sweeper = Sweeper(spec, sweep_spec, backend="numpy")
    results = sweeper.run(duration=5000.0)

    # Numba sweep pinned to 8 CPU threads
    sweeper = Sweeper(spec, sweep_spec, backend="numba", n_workers=8)
    results = sweeper.run(duration=5000.0)

    # With pipeline - returns (labels, values) array
    labels, values = sweeper.run(duration=5000.0)

    # As DataFrame (requires pandas)
    df = sweeper.run_df(duration=5000.0)
    """

    def __init__(
        self,
        spec: SimulationSpec,
        sweep_spec: SweepSpec,
        backend: str = "numpy",
        n_workers: int | None = None,
    ):
        if n_workers is not None and n_workers < 1:
            raise ValueError("n_workers must be a positive integer or None")

        backend_cls = load_sweep_backend(backend)
        if backend == "numba" and n_workers is not None:
            import numba

            numba.set_num_threads(n_workers)
            self._impl = backend_cls(spec, sweep_spec)
        elif backend == "cpp":
            self._impl = backend_cls(spec, sweep_spec, n_workers=n_workers)
        else:
            self._impl = backend_cls(spec, sweep_spec)

    def run(self, duration: float):
        return self._impl.run(duration)

    def run_df(self, duration: float):
        return self._impl.run_df(duration)
