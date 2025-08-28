# gpt5

import warnings
import numpy as np
from copy import copy
from numba import njit, jit
from numba.experimental import jitclass
from numba.extending import register_jitable
from numba import float64, boolean, int64, types
from numba.core.errors import NumbaPerformanceWarning

warnings.simplefilter("ignore", category=NumbaPerformanceWarning)


# -----------------------------
# Helper utilities
# -----------------------------

# @register_jitable
# def set_seed_compat(x):
#     np.random.seed(x)

@jit(nopython=True)
def initialize_random_state(seed):
    """Call this once to set the seed in Numba context"""
    np.random.seed(seed)



def check_vec_size_1d(x, nn):
    """Return a 1D vector of size nn, broadcasting scalar if needed (no numba)."""
    x = np.array(x, dtype=np.float64) if np.ndim(x) > 0 else np.array([x], dtype=np.float64)
    return np.ones(nn, dtype=np.float64) * x if x.size != nn else x.astype(np.float64)


# -----------------------------
# BOLD model parameters (same structure as in mpr.py)
# -----------------------------

bold_spec = [
    ("kappa", float64),
    ("gamma", float64),
    ("tau", float64),
    ("alpha", float64),
    ("epsilon", float64),
    ("Eo", float64),
    ("TE", float64),
    ("vo", float64),
    ("r0", float64),
    ("theta0", float64),
    ("t_min", float64),
    ("rtol", float64),
    ("atol", float64),
]


@jitclass(bold_spec)
class ParBold:
    def __init__(
        self,
        kappa=0.65,
        gamma=0.41,
        tau=0.98,
        alpha=0.32,
        epsilon=0.34,
        Eo=0.4,
        TE=0.04,
        vo=0.08,
        r0=25.0,
        theta0=40.3,
        t_min=0.0,
        rtol=1e-5,
        atol=1e-8,
    ):
        self.kappa = kappa
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.epsilon = epsilon
        self.Eo = Eo
        self.TE = TE
        self.vo = vo
        self.r0 = r0
        self.theta0 = theta0
        self.t_min = t_min
        self.rtol = rtol
        self.atol = atol


@jit(nopython=True)
def do_bold_step(r_in, s, f, ftilde, vtilde, qtilde, v, q, dtt, P):
    """
    One BOLD step for all nodes (vectorized over nn). Same as mpr.py.
    r_in should be non-negative neural drive per node (here we use S_exc).
    """
    kappa = P.kappa
    gamma = P.gamma
    ialpha = 1.0 / P.alpha
    tau = P.tau
    Eo = P.Eo

    s[1] = s[0] + dtt * (r_in - kappa * s[0] - gamma * (f[0] - 1.0))
    # keep f[0] >= 1 to avoid log issues
    f[0] = np.maximum(f[0], 1.0)
    ftilde[1] = ftilde[0] + dtt * (s[0] / f[0])
    fv = v[0] ** ialpha  # outflow
    vtilde[1] = vtilde[0] + dtt * ((f[0] - fv) / (tau * v[0]))
    q[0] = np.maximum(q[0], 0.01)
    ff = (1.0 - (1.0 - Eo) ** (1.0 / f[0])) / Eo  # oxygen extraction
    qtilde[1] = qtilde[0] + dtt * ((f[0] * ff - fv * q[0] / v[0]) / (tau * q[0]))

    # exponentiate back
    f[1] = np.exp(ftilde[1])
    v[1] = np.exp(vtilde[1])
    q[1] = np.exp(qtilde[1])

    # roll state
    f[0] = f[1]
    s[0] = s[1]
    ftilde[0] = ftilde[1]
    vtilde[0] = vtilde[1]
    qtilde[0] = qtilde[1]
    v[0] = v[1]
    q[0] = q[1]


# -----------------------------
# Wong–Wang model params (Numba jitclass)
# -----------------------------

ww_spec = [
    # local population parameters
    ("a_exc", float64),
    ("a_inh", float64),
    ("b_exc", float64),
    ("b_inh", float64),
    ("d_exc", float64),
    ("d_inh", float64),
    ("tau_exc", float64),
    ("tau_inh", float64),
    ("gamma_exc", float64),
    ("gamma_inh", float64),
    ("W_exc", float64),
    ("W_inh", float64),
    ("ext_current", float64[:]),
    ("J_NMDA", float64),
    ("J_I", float64),
    ("w_plus", float64),
    ("lambda_inh_exc", float64),
    # global / simulation parameters
    ("t_end", float64),
    ("t_cut", float64),
    ("dt", float64),
    ("G_exc", float64),
    ("G_inh", float64),
    ("weights", float64[:, :]),
    ("tr", float64),
    ("s_decimate", int64),
    ("sigma", float64),
    ("nn", int64),
    ("seed", int64),
    ("output", types.string),
    ("dtype", types.string),
    ("initial_state", float64[:]),
    ("RECORD_S", boolean),
    ("RECORD_BOLD", boolean),
]


@jitclass(ww_spec)
class ParWW:
    def __init__(
        self,
        # exc/inh params (Wong & Wang 2006 / Deco et al.)
        a_exc=310.0,
        a_inh=0.615,
        b_exc=125.0,
        b_inh=177.0,
        d_exc=0.16,
        d_inh=0.087,
        tau_exc=100.0,  # ms
        tau_inh=10.0,   # ms
        gamma_exc=0.641 / 1000.0,
        gamma_inh=1.0 / 1000.0,
        W_exc=1.0,
        W_inh=0.7,
        ext_current=np.array([0.382]),  # nA
        J_NMDA=0.15,
        J_I=1.0,
        w_plus=1.4,
        lambda_inh_exc=0.0,
        # simulation
        t_end=1000.0,
        t_cut=0.0,
        dt=0.1,
        G_exc=0.0,
        G_inh=0.0,
        weights=np.array([[], []]),
        tr=300.0,         # ms
        s_decimate=1,
        sigma=0.0,
        nn=1,
        seed=-1,
        output="output",
        dtype="f",
        initial_state=np.array([0.0]),
        RECORD_S=False,
        RECORD_BOLD=True,
    ):
        # assign
        self.a_exc = a_exc
        self.a_inh = a_inh
        self.b_exc = b_exc
        self.b_inh = b_inh
        self.d_exc = d_exc
        self.d_inh = d_inh
        self.tau_exc = tau_exc
        self.tau_inh = tau_inh
        self.gamma_exc = gamma_exc
        self.gamma_inh = gamma_inh
        self.W_exc = W_exc
        self.W_inh = W_inh
        self.ext_current = ext_current
        self.J_NMDA = J_NMDA
        self.J_I = J_I
        self.w_plus = w_plus
        self.lambda_inh_exc = lambda_inh_exc

        self.t_end = t_end
        self.t_cut = t_cut
        self.dt = dt
        self.G_exc = G_exc
        self.G_inh = G_inh
        self.weights = weights
        self.tr = tr
        self.s_decimate = s_decimate
        self.sigma = sigma
        self.nn = nn
        self.seed = seed
        self.output = output
        self.dtype = dtype
        self.initial_state = initial_state
        self.RECORD_S = RECORD_S
        self.RECORD_BOLD = RECORD_BOLD


# -----------------------------
# Wong–Wang dynamics (Numba)
# -----------------------------

@njit
def firing_rate(current, a, b, d):
    """
    r(I) = (a I - b) / (1 - exp(-d (a I - b)))
    Safe for vector inputs.
    """
    u = a * current - b
    den = 1.0 - np.exp(-d * u)
    # avoid division by ~0; if u ~ 0 => limit is a/d
    out = np.zeros_like(current)
    for i in range(current.shape[0]):
        if np.abs(den[i]) < 1e-12:
            out[i] = a * u[i] * 0.5  # very small; fallback (won't really occur)
        else:
            out[i] = u[i] / den[i]
    return out


@njit
def f_ww(S, t, P):
    """
    Right-hand side for Wong–Wang model.
    S: length 2*nn vector [S_exc, S_inh]
    returns dS/dt shape (2*nn,)
    """
    nn = P.nn
    S_exc = S[:nn]
    S_inh = S[nn:]

    # network couplings
    network_exc_exc = P.weights.dot(S_exc)
    if P.lambda_inh_exc > 0.0:
        network_inh_exc = P.weights.dot(S_inh)
    else:
        network_inh_exc = np.zeros_like(S_exc)

    # currents
    current_exc = (
        P.W_exc * P.ext_current
        + P.w_plus * P.J_NMDA * S_exc
        + P.G_exc * P.J_NMDA * network_exc_exc
        - P.J_I * S_inh
    )

    current_inh = (
        P.W_inh * P.ext_current
        + P.J_NMDA * S_inh
        - S_inh
        + P.G_inh * P.J_NMDA * network_inh_exc
    )

    # firing rates
    r_exc = firing_rate(current_exc, P.a_exc, P.b_exc, P.d_exc)
    r_inh = firing_rate(current_inh, P.a_inh, P.b_inh, P.d_inh)

    dSdt = np.zeros(2 * nn)

    # exc
    dSdt[:nn] = (-S_exc / P.tau_exc) + (1.0 - S_exc) * P.gamma_exc * r_exc
    # inh
    dSdt[nn:] = (-S_inh / P.tau_inh) + P.gamma_inh * r_inh

    return dSdt


@jit(nopython=True)
def heun_sde(S, t, P):
    """
    One Heun stochastic step for S (2*nn vector).
    """
    dt = P.dt
    nn = P.nn

    dW = P.sigma * np.sqrt(dt) * np.random.randn(2 * nn)

    k1 = f_ww(S, t, P)
    y_ = S + dt * k1 + dW
    k2 = f_ww(y_, t + dt, P)
    S = S + 0.5 * dt * (k1 + k2) + dW

    return S


# -----------------------------
# Public-facing class (mirror mpr.py style)
# -----------------------------

class WW_sde:
    def __init__(self, par: dict = None, Bpar: dict = None) -> None:
        if par is None:
            par = {}
        if Bpar is None:
            Bpar = {}

        # sanity & defaults
        nn = par.get("nn", None)
        weights = par.get("weights", None)
        if weights is None:
            # default single node
            weights = np.zeros((1, 1), dtype=np.float64)
        weights = np.array(weights, dtype=np.float64)
        if nn is None:
            nn = weights.shape[0]
        par.setdefault("nn", nn)

        # broadcast scalars to vectors where necessary
        par.setdefault("ext_current", 0.382)
        par["ext_current"] = check_vec_size_1d(par["ext_current"], nn)

        # dt-based noise scalars are computed inside heun_sde (uses sigma directly)

        # initial state
        if "initial_state" in par:
            par["initial_state"] = np.array(par["initial_state"], dtype=np.float64)
        else:
            par["initial_state"] = set_initial_state(nn, par.get("seed", -1))

        par.setdefault("dtype", "f")  # kept for compatibility
        par.setdefault("output", "output")

        # create numba jitclass param holders
        self.P = ParWW(
            a_exc=par.get("a_exc", 310.0),
            a_inh=par.get("a_inh", 0.615),
            b_exc=par.get("b_exc", 125.0),
            b_inh=par.get("b_inh", 177.0),
            d_exc=par.get("d_exc", 0.16),
            d_inh=par.get("d_inh", 0.087),
            tau_exc=par.get("tau_exc", 100.0),
            tau_inh=par.get("tau_inh", 10.0),
            gamma_exc=par.get("gamma_exc", 0.641 / 1000.0),
            gamma_inh=par.get("gamma_inh", 1.0 / 1000.0),
            W_exc=par.get("W_exc", 1.0),
            W_inh=par.get("W_inh", 0.7),
            ext_current=par["ext_current"].astype(np.float64),
            J_NMDA=par.get("J_NMDA", 0.15),
            J_I=par.get("J_I", 1.0),
            w_plus=par.get("w_plus", 1.4),
            lambda_inh_exc=par.get("lambda_inh_exc", 0.0),
            t_end=par.get("t_end", 1000.0),
            t_cut=par.get("t_cut", 0.0),
            dt=par.get("dt", 0.1),
            G_exc=par.get("G_exc", 0.0),
            G_inh=par.get("G_inh", 0.0),
            weights=weights,
            tr=par.get("tr", 300.0),
            s_decimate=int(par.get("s_decimate", 1)),
            sigma=par.get("sigma", 0.0),
            nn=nn,
            seed=int(par.get("seed", -1)),
            output=par.get("output", "output"),
            dtype=par.get("dtype", "f"),
            initial_state=par["initial_state"],
            RECORD_S=bool(par.get("RECORD_S", False)),
            RECORD_BOLD=bool(par.get("RECORD_BOLD", True)),
        )

        # Bold parameters
        self.B = ParBold(
            kappa=Bpar.get("kappa", 0.65),
            gamma=Bpar.get("gamma", 0.41),
            tau=Bpar.get("tau", 0.98),
            alpha=Bpar.get("alpha", 0.32),
            epsilon=Bpar.get("epsilon", 0.34),
            Eo=Bpar.get("Eo", 0.4),
            TE=Bpar.get("TE", 0.04),
            vo=Bpar.get("vo", 0.08),
            r0=Bpar.get("r0", 25.0),
            theta0=Bpar.get("theta0", 40.3),
            t_min=Bpar.get("t_min", 0.0),
            rtol=Bpar.get("rtol", 1e-5),
            atol=Bpar.get("atol", 1e-8),
        )

        # seeding
        if self.P.seed >= 0:
            # set_seed_compat(self.P.seed)
            initialize_random_state(self.P.seed)
            # print(f"WW_sde: setting random seed to {self.P.seed}")

    def __str__(self) -> str:
        lines = [
            "Wong-Wang (Numba) model",
            "Parameters: --------------------------------",
        ]
        for name in [
            "nn", "dt", "t_end", "t_cut", "G_exc", "G_inh", "sigma", "tr",
            "a_exc", "b_exc", "d_exc", "tau_exc", "gamma_exc",
            "a_inh", "b_inh", "d_inh", "tau_inh", "gamma_inh",
            "W_exc", "W_inh", "w_plus", "J_NMDA", "J_I",
        ]:
            lines.append(f"{name} = {getattr(self.P, name)}")
        lines.append("--------------------------------------------")
        return "\n".join(lines)

    # -----------------------------
    # Simulation
    # -----------------------------
    def run(self, par: dict = None, x0=None, verbose=True):
        """
        Run simulation and return dict with:
        - 'S': recorded S_exc if RECORD_S (shape [T, nn])
        - 't': times for S (ms)
        - 'bold_t': times for BOLD (ms)
        - 'bold_d': BOLD signal [T_bold, nn]
        """
        # update runtime parameters if provided
        if par:
            for key, val in par.items():
                if key == "ext_current":
                    val = check_vec_size_1d(val, self.P.nn).astype(np.float64)
                setattr(self.P, key, val)

        # initial state
        if x0 is None:
            S = copy(self.P.initial_state)
        else:
            S = np.array(x0, dtype=np.float64)

        # sanity
        assert self.P.weights is not None
        assert self.P.weights.shape[0] == self.P.weights.shape[1]
        assert len(S) == 2 * self.P.nn, "x0 must be length 2*nn"
        assert self.P.t_cut < self.P.t_end

        # time grid
        nt = int(np.floor(self.P.t_end / self.P.dt))
        t = np.arange(nt) * self.P.dt
        valid_mask = t > self.P.t_cut
        s_buffer_len = int(np.sum(valid_mask) // max(1, self.P.s_decimate))

        # buffers
        t_buf = np.zeros((s_buffer_len,), dtype=np.float32)
        S_rec = np.zeros((s_buffer_len, self.P.nn), dtype=np.float32) if self.P.RECORD_S else np.array([])

        # BOLD buffers
        tr = self.P.tr
        bold_decimate = int(np.round(tr / self.P.dt))
        dtt = self.P.dt / 1000.0  # seconds
        s = np.zeros((2, self.P.nn))
        f = np.zeros((2, self.P.nn))
        ftilde = np.zeros((2, self.P.nn))
        vtilde = np.zeros((2, self.P.nn))
        qtilde = np.zeros((2, self.P.nn))
        v = np.zeros((2, self.P.nn))
        q = np.zeros((2, self.P.nn))
        vv = np.zeros((nt // max(1, bold_decimate), self.P.nn), dtype=np.float64)
        qq = np.zeros_like(vv)

        # init BOLD states
        s[0] = 1.0
        f[0] = 1.0
        v[0] = 1.0
        q[0] = 1.0
        ftilde[0] = 0.0
        vtilde[0] = 0.0
        qtilde[0] = 0.0

        # main loop
        s_idx = 0
        for i in range(nt):
            t_curr = i * self.P.dt
            S = heun_sde(S, t_curr, self.P)

            if (t_curr > self.P.t_cut) and (i % max(1, self.P.s_decimate) == 0):
                if s_idx < s_buffer_len:
                    t_buf[s_idx] = t_curr
                    if self.P.RECORD_S:
                        S_rec[s_idx] = S[: self.P.nn].astype(np.float32)
                    s_idx += 1

            if self.P.RECORD_BOLD:
                do_bold_step(S[: self.P.nn], s, f, ftilde, vtilde, qtilde, v, q, dtt, self.B)
                if (i % max(1, bold_decimate) == 0) and ((i // max(1, bold_decimate)) < vv.shape[0]):
                    vv[i // max(1, bold_decimate)] = v[1]
                    qq[i // max(1, bold_decimate)] = q[1]

        # finalize BOLD
        bold_t = np.linspace(0.0, self.P.t_end - self.P.dt * max(1, bold_decimate), vv.shape[0])
        if self.P.RECORD_BOLD:
            # cut off t <= t_cut
            valid = bold_t > self.P.t_cut
            bold_t = bold_t[valid]
            if bold_t.size > 0:
                vv = vv[valid]
                qq = qq[valid]
                k1 = 4.3 * self.B.theta0 * self.B.Eo * self.B.TE
                k2 = self.B.epsilon * self.B.r0 * self.B.Eo * self.B.TE
                k3 = 1.0 - self.B.epsilon
                bold_d = self.B.vo * (k1 * (1.0 - qq) + k2 * (1.0 - qq / vv) + k3 * (1.0 - vv))
            else:
                bold_d = np.array([])
        else:
            bold_d = np.array([])
            bold_t = np.array([])

        return {
            "S": S_rec,
            "t": t_buf,
            "bold_t": bold_t.astype(np.float32),
            "bold_d": bold_d.astype(np.float32),
        }


# -----------------------------
# API helpers
# -----------------------------

def set_initial_state(nn, seed=-1):
    if seed is not None and seed >= 0:
        np.random.seed(seed)
        # initialize_random_state(seed)
    y0 = np.random.rand(2 * nn) * 0.1  # small positive
    return y0.astype(np.float64)
