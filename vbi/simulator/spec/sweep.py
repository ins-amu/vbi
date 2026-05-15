from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class SweepSpec:
    """
    Describes a parameter sweep for SBI training data generation.

    Parameters
    ----------
    params : dict[str, np.ndarray] | np.ndarray
        Either a dict {param_name: 1-D values} — outer product gives the grid —
        or a 2-D array of shape (n_samples, n_params) with param_names provided.
    param_names : tuple[str, ...] | None
        Required when params is an ndarray; ignored when params is a dict.
    t_cut : float
        Milliseconds of burn-in to discard before feature extraction.
    pipeline : object | None
        If set, sweeper.run() returns (labels, values); sweeper.run_df() returns
        a DataFrame. The object must expose ``extract(result) -> (labels, values)``.
        If None, sweeper returns the raw monitor output dict.

    Examples
    --------
    # 50x50 grid (2 500 runs):
    SweepSpec(params={"G": np.linspace(0.5, 5, 50), "eta": np.linspace(-5.5, -3, 50)})

    # Latin-hypercube / arbitrary samples (5 000 runs, 3 params):
    SweepSpec(params=theta_array, param_names=("G", "eta", "noise_amp"))
    """
    params: dict[str, np.ndarray] | np.ndarray
    param_names: tuple[str, ...] | None = None
    t_cut: float = 500.0
    pipeline: object | None = None

    @property
    def _param_names_list(self) -> list[str]:
        if isinstance(self.params, dict):
            return list(self.params.keys())
        if self.param_names is not None:
            return list(self.param_names)
        raise ValueError("param_names must be provided when params is an ndarray")

    @property
    def param_sets(self) -> np.ndarray:
        """Always returns (n_samples, n_params) float64."""
        if isinstance(self.params, np.ndarray):
            return self.params.astype(np.float64)
        names = list(self.params.keys())
        grids = np.meshgrid(*[self.params[n] for n in names], indexing="ij")
        return np.stack([g.ravel() for g in grids], axis=1).astype(np.float64)

    @property
    def n_samples(self) -> int:
        return self.param_sets.shape[0]
