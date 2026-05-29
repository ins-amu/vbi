from __future__ import annotations
import numpy as np


class History:
    """
    Circular delay buffer for network coupling with conduction delays.

    Shape: (horizon, n_cvar, n_nodes)
    Write index: step % horizon
    Read:  for each (src, tgt) pair, retrieve buf[(step - delay[src,tgt]) % horizon, cvar, src]

    Parameters
    ----------
    horizon : int    max_delay_steps + 1
    n_cvar  : int    number of coupling variables
    n_nodes : int    number of network nodes
    """

    def __init__(self, horizon: int, n_cvar: int, n_nodes: int,
                 dtype=np.float64):
        self.horizon = horizon
        self.n_cvar = n_cvar
        self.n_nodes = n_nodes
        self.buf = np.zeros((horizon, n_cvar, n_nodes), dtype=dtype)
        self._step = 0

    def initialize(self, cvar_state: np.ndarray) -> None:
        """Fill all buffer slots with a fixed initial state and reset the step counter.

        Parameters
        ----------
        cvar_state : (n_cvar, n_nodes)
        """
        self.buf[:] = cvar_state[np.newaxis]
        self._step = 0

    def write(self, cvar_state: np.ndarray) -> None:
        """Store current coupling-variable state at write head.

        Parameters
        ----------
        cvar_state : (n_cvar, n_nodes)
        """
        self.buf[self._step % self.horizon] = cvar_state
        self._step += 1

    def read_delayed(self, delay_steps: np.ndarray) -> np.ndarray:
        """Retrieve delayed states for all (src, tgt) pairs.

        Parameters
        ----------
        delay_steps : (n_nodes, n_nodes) int32
            delay_steps[src, tgt] = propagation delay from src to tgt in steps.

        Returns
        -------
        np.ndarray, shape (n_cvar, n_nodes, n_nodes)
            out[cvar, src, tgt] = buf[(current_step - delay[src,tgt]) % horizon,
                                      cvar, src]
        """
        # Matches TVB DenseHistory.query exactly:
        #   TVB: buf[(tvb_step - 1 - d + n) % n]  after update(tvb_step, state)
        #   VBI: buf[(_step  - 1 - d + h) % h]    before write(_step, state)
        # Both formulas are identical when _step == tvb_step.
        # Semantic: delay d reads the state written d+1 writes ago,
        # i.e. coupling for loop step s uses state(s-1-d).
        # d=0 → state(s-1) [most recent written]; d=k → state(s-1-k).
        step = self._step - 1
        idx = (step - delay_steps) % self.horizon   # (n_nodes, n_nodes)
        src_idx = np.arange(self.n_nodes)            # (n_nodes,)

        # Advanced indexing: buf[idx[src,tgt], cvar, src]
        # idx  shape (n_nodes, n_nodes) - varies over tgt for fixed src
        # src  shape (n_nodes, 1)       - selects the source node
        # result shape (n_cvar, n_nodes, n_nodes) = (cvar, src, tgt)
        out = np.empty((self.n_cvar, self.n_nodes, self.n_nodes))
        for cv in range(self.n_cvar):
            # buf[idx, cv, src_idx[:,None]] -> (n_nodes, n_nodes) = (src, tgt)
            out[cv] = self.buf[idx, cv, src_idx[:, np.newaxis]]
        return out
