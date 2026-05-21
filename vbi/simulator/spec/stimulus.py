from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np


@dataclass(frozen=True)
class StimSpec:
    """
    Additive external stimulus injected into one state variable at each step.

    At every integration step t (ms) the coupling array is augmented:

        coupling[cvar_idx] += waveform(t) * amplitude

    where ``cvar_idx`` is the position of ``sv_name`` in ``model.cvar``.
    The stimulus must target a state variable that is also a coupling variable
    so that it enters the model dfun via the ``c`` term.

    Parameters
    ----------
    sv_name : str
        Name of the target state variable.  Must appear in ``ModelSpec.cvar``.
    amplitude : float or (n_nodes,) array
        Per-node spatial pattern.  A scalar is broadcast to all nodes.
    waveform : (n_steps,) array or callable(t_ms) -> float
        Temporal waveform.  Provide either a pre-sampled NumPy array (indexed
        by step number) or a callable that takes the current time in ms and
        returns a scalar value.  If ``waveform`` is an array shorter than the
        simulation, out-of-range steps evaluate to 0.
    onset : float
        Start time [ms].  Stimulus is zero before this time.  Default 0.
    offset : float
        End time [ms].  Stimulus is zero at or after this time.  Default inf.

    Examples
    --------
    Constant stimulus on node 0:

    >>> StimSpec(sv_name="r", amplitude=np.array([1.0, 0, 0, 0]),
    ...          waveform=lambda t: 0.1)

    Pulse train on all nodes:

    >>> def pulse(t):
    ...     return 0.5 if (100.0 <= t < 200.0) else 0.0
    >>> StimSpec(sv_name="r", amplitude=np.ones(n), waveform=pulse)

    Pre-sampled sinusoid:

    >>> t = np.arange(n_steps) * dt
    >>> StimSpec(sv_name="theta", amplitude=np.ones(n),
    ...          waveform=0.2 * np.sin(2 * np.pi * 0.01 * t))
    """

    sv_name: str
    amplitude: float | np.ndarray
    waveform: np.ndarray | Callable[[float], float]
    onset: float = 0.0
    offset: float = float("inf")

    def evaluate(self, step: int, t_ms: float) -> float:
        """Return the scalar waveform value at this step, or 0 outside onset/offset."""
        if t_ms < self.onset or t_ms >= self.offset:
            return 0.0
        if callable(self.waveform):
            return float(self.waveform(t_ms))
        # pre-sampled array
        if step < len(self.waveform):
            return float(self.waveform[step])
        return 0.0
