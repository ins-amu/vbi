
import warnings
import numpy as np
from numba import njit, jit
from numba.experimental import jitclass
from numba.extending import register_jitable
from numba import float64, boolean, int64, types
from numba.core.errors import NumbaPerformanceWarning

warnings.simplefilter("ignore", category=NumbaPerformanceWarning)


# ---------- utilities ----------

def _to_1d_array(x):
    x = np.array(x, dtype=np.float64)
    if x.ndim == 0:
        x = x.reshape(1)
    return x

def _to_2d_array(x):
    x = np.array(x, dtype=np.float64)
    if x.ndim == 1:
        # try to guess a square matrix if possible
        n = int(np.sqrt(x.size))
        if n * n == x.size:
            x = x.reshape(n, n)
        else:
            raise ValueError("weights must be square (nxn).")
    return x

def check_vec_size(x, nn):
    """Return a length-nn vector from scalar/1-vector or already-length-nn input."""
    arr = np.array(x, dtype=np.float64)
    if arr.ndim == 0:
        return np.ones(nn, dtype=np.float64) * float(arr)
    if arr.size == 1:
        return np.ones(nn, dtype=np.float64) * float(arr[0])
    if arr.size != nn:
        raise ValueError(f"Vector parameter has size {arr.size} but nn={nn}.")
    return arr.astype(np.float64)


@register_jitable
def set_seed_compat(x):
    np.random.seed(x)


# ---------- core model (Numba) ----------

wc_spec = [
    ("c_ee", float64[:]),
    ("c_ei", float64[:]),
    ("c_ie", float64[:]),
    ("c_ii", float64[:]),
    ("tau_e", float64[:]),
    ("tau_i", float64[:]),
    ("a_e", float64),
    ("a_i", float64),
    ("b_e", float64),
    ("b_i", float64),
    ("c_e", float64),
    ("c_i", float64),
    ("theta_e", float64),
    ("theta_i", float64),
    ("r_e", float64),
    ("r_i", float64),
    ("k_e", float64),
    ("k_i", float64),
    ("alpha_e", float64),
    ("alpha_i", float64),
    ("P", float64[:]),
    ("Q", float64[:]),
    ("g_e", float64),
    ("g_i", float64),
    ("dt", float64),
    ("t_end", float64),
    ("t_cut", float64),
    ("nn", int64),
    ("weights", float64[:, :]),
    ("seed", int64),
    ("noise_amp", float64),
    ("decimate", int64),
    ("RECORD_EI", types.string),
    ("initial_state", float64[:]),
    ("shift_sigmoid", boolean),
]


@jitclass(wc_spec)
class ParWC:
    def __init__(
        self,
        c_ee=np.array([16.0]),
        c_ei=np.array([12.0]),
        c_ie=np.array([15.0]),
        c_ii=np.array([3.0]),
        tau_e=np.array([8.0]),
        tau_i=np.array([8.0]),
        a_e=1.3,
        a_i=2.0,
        b_e=4.0,
        b_i=3.7,
        c_e=1.0,
        c_i=1.0,
        theta_e=0.0,
        theta_i=0.0,
        r_e=1.0,
        r_i=1.0,
        k_e=0.994,
        k_i=0.999,
        alpha_e=1.0,
        alpha_i=1.0,
        P=np.array([0.0]),
        Q=np.array([0.0]),
        g_e=0.0,
        g_i=0.0,
        dt=0.01,
        t_end=300.0,
        t_cut=0.0,
        weights=np.empty((0, 0), dtype=np.float64),
        seed=-1,
        noise_amp=0.0,
        decimate=1,
        RECORD_EI="E",
        initial_state=np.empty(0, dtype=np.float64),
        shift_sigmoid=False,
    ):
        self.c_ee = c_ee
        self.c_ei = c_ei
        self.c_ie = c_ie
        self.c_ii = c_ii
        self.tau_e = tau_e
        self.tau_i = tau_i
        self.a_e = a_e
        self.a_i = a_i
        self.b_e = b_e
        self.b_i = b_i
        self.c_e = c_e
        self.c_i = c_i
        self.theta_e = theta_e
        self.theta_i = theta_i
        self.r_e = r_e
        self.r_i = r_i
        self.k_e = k_e
        self.k_i = k_i
        self.alpha_e = alpha_e
        self.alpha_i = alpha_i
        self.P = P
        self.Q = Q
        self.g_e = g_e
        self.g_i = g_i
        self.dt = dt
        self.t_end = t_end
        self.t_cut = t_cut
        self.nn = len(weights)
        self.weights = weights
        self.seed = seed
        self.noise_amp = noise_amp
        self.decimate = decimate
        self.RECORD_EI = RECORD_EI
        self.initial_state = initial_state
        self.shift_sigmoid = shift_sigmoid


@njit
def sigmoid_vec(x, a, b, c, shift_sigmoid):
    y = np.empty_like(x)
    if shift_sigmoid:
        # c * (sigmoid(a(x-b)) - sigmoid(-ab))
        base = 1.0 / (1.0 + np.exp(-a * (-b)))
        for i in range(x.size):
            y[i] = c * (1.0 / (1.0 + np.exp(-a * (x[i] - b))) - base)
    else:
        for i in range(x.size):
            y[i] = c / (1.0 + np.exp(-a * (x[i] - b)))
    return y


@njit
def f_wc(x, t, P):
    """
    Wilson-Cowan ODE right-hand side (per-node, single simulation).
    x: shape (2*nn,)
    """
    nn = P.nn
    dxdt = np.zeros_like(x)

    E = x[:nn]
    I = x[nn:]

    # Linear coupling (weights @ state)
    lc_e = P.g_e * np.dot(P.weights, E) if P.g_e != 0.0 else np.zeros(nn)
    lc_i = P.g_i * np.dot(P.weights, I) if P.g_i != 0.0 else np.zeros(nn)

    # Inputs to sigmoids
    x_e = P.alpha_e * (P.c_ee * E - P.c_ei * I + P.P - P.theta_e + lc_e)
    x_i = P.alpha_i * (P.c_ie * E - P.c_ii * I + P.Q - P.theta_i + lc_i)

    s_e = sigmoid_vec(x_e, P.a_e, P.b_e, P.c_e, P.shift_sigmoid)
    s_i = sigmoid_vec(x_i, P.a_i, P.b_i, P.c_i, P.shift_sigmoid)

    # Time constants (vectorized)
    inv_tau_e = 1.0 / P.tau_e
    inv_tau_i = 1.0 / P.tau_i

    # dE/dt
    for i in range(nn):
        dxdt[i] = inv_tau_e[i] * (-E[i] + (P.k_e - P.r_e * E[i]) * s_e[i])
    # dI/dt
    for i in range(nn):
        dxdt[nn + i] = inv_tau_i[i] * (-I[i] + (P.k_i - P.r_i * I[i]) * s_i[i])

    return dxdt


@njit
def heun_sde(x, t, P):
    dt = P.dt
    coeff = P.noise_amp * np.sqrt(dt)
    dW = coeff * np.random.randn(x.size)

    k1 = f_wc(x, t, P)
    x1 = x + dt * k1 + dW
    k2 = f_wc(x1, t + dt, P)
    x_out = x + 0.5 * dt * (k1 + k2) + dW
    return x_out


@njit
def set_initial_state(nn, seed=-1):
    if seed >= 0:
        set_seed_compat(seed)
    y0 = np.random.rand(2 * nn)
    return y0


# ---------- high-level API (Python) ----------

class WC_sde_numba:
    """
    Numba implementation of the Wilson-Cowan SDE, modeled after mpr.py and
    translated from the CuPy/Numpy reference.
    """

    def __init__(self, par: dict = {}):
        # Prepare raw dict and build jitclass
        self.P = self._get_par_wc(par)

        # Seed
        if self.P.seed >= 0:
            np.random.seed(self.P.seed)

    def __call__(self):
        return self.P

    def __str__(self) -> str:
        params = [
            "nn", "dt", "t_end", "t_cut", "decimate", "noise_amp",
            "g_e", "g_i", "a_e", "a_i", "b_e", "b_i", "k_e", "k_i",
        ]
        s = ["Wilson-Cowan (Numba) parameters:"]
        for k in params:
            s.append(f"{k} = {getattr(self.P, k)}")
        return "\n".join(s)

    # ----- builders & checks -----
    def _get_par_wc(self, par: dict):
        par = dict(par)  # shallow copy

        # weights first (to infer nn)
        if "weights" not in par:
            raise ValueError("weights (nxn) must be provided.")
        W = _to_2d_array(par["weights"])
        nn = W.shape[0]

        # convert possibly-scalar/vector params to length-nn arrays
        vec_keys = ["c_ee","c_ei","c_ie","c_ii","tau_e","tau_i","P","Q"]
        for k in vec_keys:
            if k in par:
                par[k] = check_vec_size(par[k], nn)

        # defaults for any missing vector keys
        defaults = {
            "c_ee": 16.0, "c_ei": 12.0, "c_ie": 15.0, "c_ii": 3.0,
            "tau_e": 8.0, "tau_i": 8.0, "P": 0.0, "Q": 0.0
        }
        for k, v in defaults.items():
            if k not in par:
                par[k] = np.ones(nn) * v

        # set weights and nn
        par["weights"] = W
        
        # initial_state (optional)
        if "initial_state" in par:
            arr = np.array(par["initial_state"], dtype=np.float64)
            if arr.size != 0 and arr.size != 2 * nn:
                raise ValueError(f"initial_state must have length {2*nn}.")
            par["initial_state"] = arr
        else:
            par["initial_state"] = np.empty(0, dtype=np.float64)

        # strings/flags
        if "RECORD_EI" not in par:
            par["RECORD_EI"] = "E"
        if "decimate" not in par:
            par["decimate"] = 1
        if "noise_amp" not in par:
            par["noise_amp"] = 0.0

        # build jitclass
        P = ParWC(**par)
        return P

    def set_initial_state(self):
        self.P.initial_state = set_initial_state(self.P.nn, self.P.seed)

    def check_input(self):
        P = self.P
        assert P.weights.shape[0] == P.weights.shape[1], "weights must be square"
        assert P.nn == P.weights.shape[0], "nn must match weights shape"
        if P.initial_state.size == 0:
            self.set_initial_state()
        assert P.initial_state.size == 2 * P.nn, "initial_state length mismatch"
        assert P.t_cut < P.t_end, "t_cut must be less than t_end"

        # ensure vector parameters are length-nn (already enforced in builder)
        # but re-check shapes at runtime for safety
        for k in ["c_ee","c_ei","c_ie","c_ii","tau_e","tau_i","P","Q"]:
            v = getattr(P, k)
            assert v.size == P.nn, f"{k} must be length nn"

    def run(self, par: dict = None, x0=None, verbose: bool = True):
        # update parameters if provided
        if par:
            # (rebuild jitclass when structure-changing params come in)
            merged = {**self._par_to_dict(), **par}
            self.P = self._get_par_wc(merged)

        # set external initial state if provided
        if x0 is not None:
            x0 = np.array(x0, dtype=np.float64)
            if x0.size != 2 * self.P.nn:
                raise ValueError(f"x0 must be length {2*self.P.nn}")
            self.P.initial_state = x0

        # checks
        self.check_input()

        return integrate(self.P, verbose=verbose)

    def _par_to_dict(self):
        P = self.P
        d = {
            "c_ee": np.array(P.c_ee),
            "c_ei": np.array(P.c_ei),
            "c_ie": np.array(P.c_ie),
            "c_ii": np.array(P.c_ii),
            "tau_e": np.array(P.tau_e),
            "tau_i": np.array(P.tau_i),
            "a_e": P.a_e,
            "a_i": P.a_i,
            "b_e": P.b_e,
            "b_i": P.b_i,
            "c_e": P.c_e,
            "c_i": P.c_i,
            "theta_e": P.theta_e,
            "theta_i": P.theta_i,
            "r_e": P.r_e,
            "r_i": P.r_i,
            "k_e": P.k_e,
            "k_i": P.k_i,
            "alpha_e": P.alpha_e,
            "alpha_i": P.alpha_i,
            "P": np.array(P.P),
            "Q": np.array(P.Q),
            "g_e": P.g_e,
            "g_i": P.g_i,
            "dt": P.dt,
            "t_end": P.t_end,
            "t_cut": P.t_cut,
            "weights": np.array(P.weights),
            "seed": P.seed,
            "noise_amp": P.noise_amp,
            "decimate": P.decimate,
            "RECORD_EI": P.RECORD_EI,
            "initial_state": np.array(P.initial_state),
            "shift_sigmoid": P.shift_sigmoid,
        }
        return d


def integrate(P: ParWC, verbose=True):
    """
    Pure-Python driver (Numba-accelerated inner steps).
    Returns dict with t, E, I (float32).
    """
    nn = P.nn
    dt = P.dt
    nt = int(P.t_end / dt)
    dec = max(1, int(P.decimate))

    # buffers sized after decimation & cut
    # we'll first allocate full decimated length, then trim by t_cut
    nbuf = nt // dec
    record_e = "e" in P.RECORD_EI.lower()
    record_i = "i" in P.RECORD_EI.lower()

    t_buf = np.zeros(nbuf, dtype=np.float32)
    E_buf = np.zeros((nbuf, nn), dtype=np.float32) if record_e else None
    I_buf = np.zeros((nbuf, nn), dtype=np.float32) if record_i else None

    x = P.initial_state.copy()
    buf_idx = 0

    for i in range(nt):
        t_curr = i * dt
        x = heun_sde(x, t_curr, P)

        if (i % dec) == 0 and buf_idx < nbuf:
            t_buf[buf_idx] = t_curr
            if record_e:
                E_buf[buf_idx] = x[:nn].astype(np.float32)
            if record_i:
                I_buf[buf_idx] = x[nn:].astype(np.float32)
            buf_idx += 1

    # trim to actual filled length
    t_buf = t_buf[:buf_idx]
    if record_e: E_buf = E_buf[:buf_idx]
    if record_i: I_buf = I_buf[:buf_idx]

    # apply t_cut
    keep = t_buf >= P.t_cut
    t_out = t_buf[keep]
    E_out = E_buf[keep] if record_e else None
    I_out = I_buf[keep] if record_i else None

    return {"t": t_out, "E": E_out, "I": I_out}


WC_sde = WC_sde_numba  # alias