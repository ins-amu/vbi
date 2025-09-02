"""
Numba reimplementation of the Virtual Epileptic Patient (VEP) model.

Reference dynamics taken from the C++ header (`vep.hpp`):
  dx_i   = 1 - x_i^3 - 2 x_i^2 - y_i + i_ext_i
  dy_i   = (4*(x_i - eta_i) - y_i - G * sum_j W_ij (x_j - x_i)) / tau
where (x_i, y_i) are the two state variables per node i, and the
network coupling uses a Laplacian form \sum_j W_ij (x_j - x_i).

Noise is additive, identical strength for all state dimensions, applied
as sqrt(dt) * sigma * N(0,1) per integration substep (Euler once; Heun twice),
mirroring the C++ implementation.

The Python-facing wrapper class is `VEP_sde`.
It returns a dict with keys:
  - 't': 1D array of times (after t_cut, optional decimation via `record_step`)
  - 'x': 2D array of shape (nn, nt_saved) with only the first variable (x)
        recorded, as in the C++ code.

Example
-------
>>> import numpy as np
>>> from vep_numba import VEP_sde
>>> nn = 4
>>> W = (np.ones((nn, nn)) - np.eye(nn)) * 0.5
>>> model = VEP_sde({
...     'weights': W,
...     'G': 1.0,
...     'tau': 10.0,
...     'noise_seed': 123,
...     'seed': 99,          # initial-state seed
...     'dt': 0.01,
...     't_end': 2.0,
...     't_cut': 0.2,
...     'method': 'heun',
... })
>>> out = model.run()
>>> out['t'][:5]
array([0.2 , 0.21, 0.22, 0.23, 0.24], dtype=float32)
>>> out['x'].shape
(4, 180)
"""

import warnings
import numpy as np
from numba import njit, jit
from numba.experimental import jitclass
from numba.extending import register_jitable
from numba import float64, int64, types
from numba.core.errors import NumbaPerformanceWarning

warnings.simplefilter("ignore", category=NumbaPerformanceWarning)

# ------------------------------
# Low-level utilities
# ------------------------------


@njit
def seed_rng(seed: int):
    if seed >= 0:
        np.random.seed(seed)


# ------------------------------
# JIT params (jitclass) — like in mpr.py
# ------------------------------

vep_spec = [
    ("G", float64),
    ("dt", float64),
    ("tau", float64),
    ("eta", float64[:]),
    ("iext", float64[:]),
    ("weights", float64[:, :]),
    ("t_cut", float64),
    ("t_end", float64),
    ("nn", int64),
    ("method", types.string),
    ("seed", int64),  # initial-state seed
    ("initial_state", float64[:]),
    ("sigma", float64),  # noise_sigma
    ("record_step", int64),  # decimation factor for recording
    ("output", types.string),
]


@jitclass(vep_spec)
class ParVEP:
    def __init__(
        self,
        G=1.0,
        dt=0.01,
        tau=10.0,
        eta=np.array([-1.5]),
        iext=np.array([0.0]),
        weights=np.array([[], []]),
        t_cut=0.0,
        t_end=100.0,
        method="euler",
        seed=-1,
        initial_state=np.array([0.0, 0.0]),
        sigma=0.1,
        record_step=1,
        output="output",
    ):
        self.G = G
        self.dt = dt
        self.tau = tau
        self.eta = eta
        self.iext = iext
        self.weights = weights
        self.t_cut = t_cut
        self.t_end = t_end
        self.method = method
        self.seed = seed
        self.initial_state = initial_state
        self.sigma = sigma
        self.record_step = record_step
        self.output = output
        self.nn = len(weights)


# ------------------------------
# Model dynamics & integrators
# ------------------------------


@njit
def f_vep(x, t, P):
    """Right-hand side of VEP dynamics (see header notes).

    x : shape (2*nn,)
      [0:nn] -> x-variable, [nn:2*nn] -> y-variable
    """
    nn = P.nn
    dxdt = np.zeros_like(x)
    x0 = x[:nn]
    x1 = x[nn:]

    # Laplacian coupling: sum_j W_ij*(x_j - x_i) = (W @ x) - (row_sum)*x
    Wx = P.weights.dot(x0)
    row_sum = P.weights.sum(1)
    gx = Wx - row_sum * x0

    dxdt[:nn] = 1.0 - x0 * x0 * x0 - 2.0 * x0 * x0 - x1 + P.iext
    dxdt[nn:] = (4.0 * (x0 - P.eta) - x1 - P.G * gx) / P.tau
    return dxdt


@njit
def euler_step(x, t, P):
    nn = P.nn
    dxdt = f_vep(x, t, P)
    noise = np.sqrt(P.dt) * P.sigma * np.random.randn(2 * nn)
    return x + P.dt * dxdt + noise


@njit
def heun_step(x, t, P):
    nn = P.nn
    k1 = f_vep(x, t, P)
    noise = np.sqrt(P.dt) * P.sigma * np.random.randn(2 * nn)
    xtemp = x + P.dt * k1 + noise
    k2 = f_vep(xtemp, t + P.dt, P)
    return x + 0.5 * P.dt * (k1 + k2) + noise


@njit
def set_initial_state_jit(nn: int, seed: int):
    """Match the Python/C++ initializer used in vep.py/vep.hpp.
    x0[:nn] ~ U(-3, -2), x0[nn:] ~ U(0, 3.5)
    """
    if seed >= 0:
        np.random.seed(seed)
    x0 = np.zeros(2 * nn)
    x0[:nn] = np.random.uniform(-3.0, -2.0, nn)  # in [-3, -2]
    x0[nn:] = np.random.uniform(0.0, 3.5, nn)  # in [ 0, 3.5]
    return x0


# ------------------------------
# Integration driver
# ------------------------------


@njit
def _integrate(P):
    # Seed Numba RNG for the SDE noise draws
    seed_rng(P.seed)

    nn = P.nn
    dt = P.dt
    nt = int(P.t_end / dt)
    idx_cut = int(P.t_cut / dt)
    bufsize = nt - idx_cut
    step_dec = P.record_step

    # Pre-allocate (with decimation)
    count_est = bufsize // step_dec + (1 if bufsize % step_dec != 0 else 0)
    x_current = P.initial_state.copy()
    t = 0.0
    states = np.zeros((count_est, nn), dtype=np.float32)
    times = np.zeros(count_est, dtype=np.float32)
    counter = 0

    for it in range(nt):
        if it >= idx_cut and ((it - idx_cut) % step_dec == 0):
            if counter < count_est:
                # Record only x (first nn dimensions), like the C++ code
                for i in range(nn):
                    states[counter, i] = x_current[i]
                times[counter] = t
                counter += 1

        t += dt
        if P.method == "euler":
            x_current = euler_step(x_current, t, P)
        else:
            x_current = heun_step(x_current, t, P)

    # Trim to the actual count
    states = states[:counter, :]
    times = times[:counter]
    return times, states


# ------------------------------
class VEP_sde:
    def __init__(self, par_vep: dict = {}):
        # Accepted parameter names
        self.valid_par = [item[0] for item in vep_spec]
        self.P = self._make_par(par_vep)
        self.seed = self.P.seed

    def __str__(self) -> str:
        lines = ["VEP model (Numba) Parameters:", "---------------------------------"]
        for key in self.valid_par:
            if hasattr(self.P, key):
                lines.append(f"{key} = {getattr(self.P, key)}")
        lines.append("---------------------------------")
        return "\n".join(lines)

    # --- helpers
    def _check_keys(self, par: dict):
        for key in par.keys():
            if key not in self.valid_par:
                raise ValueError(f"Invalid parameter: {key}")

    def _make_par(self, par: dict):
        p = dict(par) if par else {}
        if "weights" not in p:
            raise ValueError("'weights' (square connectivity matrix) is required.")
        p["weights"] = np.array(p["weights"], dtype=np.float64)
        assert (
            p["weights"].ndim == 2 and p["weights"].shape[0] == p["weights"].shape[1]
        ), "weights must be a square 2D array"
        nn = p["weights"].shape[0]

        # Broadcast eta and iext to length nn if given as scalars
        if "eta" in p:
            p["eta"] = np.array(p["eta"], dtype=np.float64)
            if p["eta"].ndim == 0 or (p["eta"].ndim == 1 and p["eta"].size == 1):
                p["eta"] = np.ones(nn) * p["eta"].ravel()[0]
            else:
                assert p["eta"].size == nn, "eta must have length nn"
        else:
            p["eta"] = np.ones(nn) * -1.5

        if "iext" in p:
            p["iext"] = np.array(p["iext"], dtype=np.float64)
            if p["iext"].ndim == 0 or (p["iext"].ndim == 1 and p["iext"].size == 1):
                p["iext"] = np.ones(nn) * p["iext"].ravel()[0]
            else:
                assert p["iext"].size == nn, "iext must have length nn"
        else:
            p["iext"] = np.zeros(nn)

        # Initial state placeholder — will be set in run() if None/not provided
        if "initial_state" in p and p["initial_state"] is not None:
            p["initial_state"] = np.array(p["initial_state"], dtype=np.float64)
            self._initial_state_provided = True
        else:
            p["initial_state"] = np.zeros(2 * nn, dtype=np.float64)
            self._initial_state_provided = False

        return ParVEP(**p)

    def set_initial_state(self):
        x0 = set_initial_state_jit(self.P.nn, self.P.seed)
        self.P.initial_state = x0

    def _check_input(self):
        assert self.P.weights is not None
        assert self.P.weights.shape[0] == self.P.weights.shape[1]
        assert self.P.initial_state is not None
        assert self.P.initial_state.size == 2 * self.P.nn
        assert self.P.t_cut < self.P.t_end, "t_cut must be less than t_end"
        # Ensure eta/iext sizes (broadcast if user changed nn)
        if self.P.eta.size != self.P.nn:
            self.P.eta = np.ones(self.P.nn) * self.P.eta.ravel()[0]
        if self.P.iext.size != self.P.nn:
            self.P.iext = np.ones(self.P.nn) * self.P.iext.ravel()[0]

    # --- main entry
    def run(self, par: dict = None, x0=None, verbose: bool = False):
        if x0 is None or x0 is False:
            # Only use random initial state if no initial_state was provided in input parameters
            if not self._initial_state_provided:
                self.set_initial_state()
        else:
            self.P.initial_state = np.array(x0, dtype=np.float64)

        if par:
            self._check_keys(par)
            for key, val in par.items():
                if key in ("weights", "eta", "iext", "initial_state"):
                    val = np.array(val, dtype=np.float64)
                setattr(self.P, key, val)
                # Update the flag if initial_state is provided in par
                if key == "initial_state":
                    self._initial_state_provided = True

        self._check_input()
        t, s = _integrate(self.P)
        # Match the C++/Python wrapper: times and only the first variable per node (x)
        return {"t": t, "x": s.T}
