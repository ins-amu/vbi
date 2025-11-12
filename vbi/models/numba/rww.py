import numpy as np
from copy import copy
from numba import njit, jit
from numba.experimental import jitclass
from numba import float64, int64, boolean, types
from vbi.models.numba.bold import ParBold, bold_spec, do_bold_step
from vbi.models.numba.utils import check_vec_size_1d, check_parameters

# ----------------------------------------
# Reduced Wong–Wang spec (excitatory only)
# ----------------------------------------

rww_spec = [
    ("a", float64),
    ("b", float64),
    ("d", float64),
    ("tau_s", float64),
    ("gamma", float64),
    ("w", float64),
    ("J", float64),
    ("G", float64),
    ("I_ext", float64[:]),
    ("weights", float64[:, :]),
    ("sigma", float64),
    ("dt", float64),
    ("t_end", float64),
    ("t_cut", float64),
    ("nn", int64),
    ("seed", int64),
    ("initial_state", float64[:]),
    ("RECORD_S", boolean),
    ("RECORD_BOLD", boolean),
    ("tr", float64),
    ("s_decimate", int64),
]


@jitclass(rww_spec)
class ParRWW:
    def __init__(
        self,
        a=270.0,
        b=108.0,
        d=0.154,
        tau_s=100.0,  # ms
        gamma=0.641 / 1000.0,
        w=1.0,
        J=0.15,
        G=0.0,
        I_ext=np.array([0.382]),
        weights=np.array([[0.0]]),
        sigma=0.0,
        dt=0.1,
        t_end=1000.0,
        t_cut=0.0,
        tr=300.0,
        nn=1,
        seed=-1,
        initial_state=np.array([0.0]),
        RECORD_S=True,
        RECORD_BOLD=True,
        s_decimate=1,
    ):
        self.a = a
        self.b = b
        self.d = d
        self.tau_s = tau_s
        self.gamma = gamma
        self.w = w
        self.J = J
        self.G = G
        self.I_ext = I_ext
        self.weights = weights
        self.sigma = sigma
        self.dt = dt
        self.t_end = t_end
        self.t_cut = t_cut
        self.nn = nn
        self.seed = seed
        self.tr = tr
        self.initial_state = initial_state
        self.RECORD_S = RECORD_S
        self.RECORD_BOLD = RECORD_BOLD
        self.s_decimate = s_decimate


# -----------------------------
# Transfer function H(x)
# -----------------------------
@njit
def H(x, a, b, d):
    u = a * x - b
    out = np.zeros_like(x)
    for i in range(x.shape[0]):
        den = 1.0 - np.exp(-d * u[i])
        if np.abs(den) < 1e-12:
            out[i] = a * u[i] * 0.5
        else:
            out[i] = u[i] / den
    return out


# -----------------------------
# RHS of reduced model
# -----------------------------
@njit
def f_rww(S, t, P):
    nn = P.nn
    # recurrent & global input
    x = P.w * P.J * S + P.G * P.J * P.weights.dot(S) + P.I_ext
    r = H(x, P.a, P.b, P.d)
    dSdt = -S / P.tau_s + (1.0 - S) * P.gamma * r
    return dSdt


# -----------------------------
# Heun SDE integrator
# -----------------------------
@njit
def heun_sde(S, t, P):
    dt = P.dt
    nn = P.nn
    dW = P.sigma * np.sqrt(dt) * np.random.randn(nn)
    k1 = f_rww(S, t, P)
    y_ = S + dt * k1 + dW
    k2 = f_rww(y_, t + dt, P)
    return S + 0.5 * dt * (k1 + k2) + dW


# -----------------------------
# Public API class
# -----------------------------
class RWW_sde:
    def __init__(self, par: dict = {}, Bpar: dict = {}) -> None:
        
        self.valid_par = [rww_spec[i][0] for i in range(len(rww_spec))]
        check_parameters(par, self.valid_par, rww_spec)
        
        nn = par.get("nn", None)
        weights = par.get("weights", None)
        if weights is None:
            weights = np.zeros((1, 1), dtype=np.float64)
        weights = np.array(weights, dtype=np.float64)
        if nn is None:
            nn = weights.shape[0]
        par.setdefault("nn", nn)

        # initial state
        if "initial_state" in par:
            par["initial_state"] = np.array(par["initial_state"], dtype=np.float64)
        else:
            par["initial_state"] = set_initial_state(nn, seed=par.get("seed", -1))

        self.P = ParRWW(
            a=par.get("a", 270.0),
            b=par.get("b", 108.0),
            d=par.get("d", 0.154),
            tau_s=par.get("tau_s", 100.0),
            gamma=par.get("gamma", 0.641 / 1000.0),
            w=par.get("w", 1.0),
            J=par.get("J", 0.2609),
            G=par.get("G", 0.0),
            I_ext=check_vec_size_1d(par.get("I_ext", 0.382), nn),
            weights=weights,
            sigma=par.get("sigma", 0.0),
            dt=par.get("dt", 1.0),
            t_end=par.get("t_end", 1000.0),
            t_cut=par.get("t_cut", 0.0),
            nn=nn,
            seed=int(par.get("seed", -1)),
            tr=par.get("tr", 300.0),
            initial_state=par["initial_state"],
            RECORD_S=bool(par.get("RECORD_S", True)),
            RECORD_BOLD=bool(par.get("RECORD_BOLD", True)),
            s_decimate=int(par.get("s_decimate", 1)),
        )
        
        # --- Bold parameters ---
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
    def run(self, par: dict = {}, x0=None):
        if x0 is None:
            S = copy(self.P.initial_state)
        else:
            S = np.array(x0, dtype=np.float64)
            
        check_parameters(par, self.valid_par, rww_spec)

        if par:
            for k, v in par.items():
                if k == "I_ext":
                    v = check_vec_size_1d(v, self.P.nn)
                setattr(self.P, k, v)

        # --- sanity checks ---
        assert self.P.weights is not None
        assert self.P.weights.shape[0] == self.P.weights.shape[1]
        assert len(S) == self.P.nn, "x0 must be length nn"
        assert self.P.t_cut < self.P.t_end
        
        # --- time grid ---
        nt = int(np.floor(self.P.t_end / self.P.dt))
        t = np.arange(nt) * self.P.dt
        valid_mask = t > self.P.t_cut
        s_buffer_len = int(np.sum(valid_mask) // max(1, self.P.s_decimate))
        
        
        # --- BOLD buffers ---
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
        
        s[0] = 1.0
        f[0] = 1.0
        v[0] = 1.0
        q[0] = 1.0
        ftilde[0] = 0.0
        vtilde[0] = 0.0
        qtilde[0] = 0.0


        t_buf = np.zeros((s_buffer_len,), dtype=np.float32)
        S_rec = (
            np.zeros((s_buffer_len, self.P.nn), dtype=np.float32)
            if self.P.RECORD_S
            else np.array([])
        )

        # --- main loop ---
        s_idx = 0
        for i in range(nt):
            t_curr = i * self.P.dt
            S = heun_sde(S, t_curr, self.P)

            if (t_curr > self.P.t_cut) and (i % max(1, self.P.s_decimate) == 0):
                if s_idx < s_buffer_len:
                    t_buf[s_idx] = t_curr
                    if self.P.RECORD_S:
                        S_rec[s_idx] = S
                    s_idx += 1
            # BOLD step
            if self.P.RECORD_BOLD:
                do_bold_step(S[: self.P.nn], s, f, ftilde, vtilde, qtilde, v, q, dtt, self.B)
                if (i % max(1, bold_decimate) == 0) and ((i // max(1, bold_decimate)) < vv.shape[0]):
                    vv[i // max(1, bold_decimate)] = v[1]
                    qq[i // max(1, bold_decimate)] = q[1]
                
        # finalize BOLD
        if self.P.RECORD_BOLD:
            bold_t = np.linspace(0.0, self.P.t_end - self.P.dt * max(1, bold_decimate), vv.shape[0])
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
            "bold_d": bold_d.astype(np.float32),
            "bold_t": bold_t.astype(np.float32),
        }


# -----------------------------
# API helpers
# -----------------------------


def set_initial_state(nn, seed=-1):
    """
    Generate random initial conditions for the Reduced Wong-Wang model.

    Creates small positive random values for both excitatory
    synaptic gating variables, ensuring the system starts in a biologically
    plausible state.

    Parameters
    ----------
    nn : int
        Number of brain regions/nodes.
    seed : int, optional
        Random seed for reproducibility. If -1 or None, no seeding is applied.

    Returns
    -------
    np.ndarray
        Initial state vector of shape (nn,) containing small positive
        random values for [S_exc_0, ..., S_exc_n].
    """
    if seed is not None and seed >= 0:
        np.random.seed(seed)
        # initialize_random_state(seed)
    y0 = np.random.rand(nn) * 0.1  # small positive
    return y0.astype(np.float64)
