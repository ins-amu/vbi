import warnings
import numpy as np
from copy import copy
from numba import njit, jit
from numba.experimental import jitclass
from numba.extending import register_jitable
from numba import float64, boolean, int64, types
from numba.core.errors import NumbaPerformanceWarning

warnings.simplefilter("ignore", category=NumbaPerformanceWarning)
np.random.seed(42)


@njit
def f_mpr(x, t, P):
    """
    MPR model
    """

    dxdt = np.zeros_like(x)
    nn = P.nn
    x0 = x[:nn]
    x1 = x[nn:]
    delta_over_tau_pi = P.delta / (P.tau * np.pi)
    J_tau = P.J * P.tau
    pi2 = np.pi * np.pi
    tau2 = P.tau * P.tau
    rtau = 1.0 / P.tau

    coupling = np.dot(P.weights, x0)
    dxdt[:nn] = rtau * (delta_over_tau_pi + 2 * x0 * x1)
    dxdt[nn:] = rtau * (
        x1 * x1 + P.eta + P.iapp + J_tau * x0 - (pi2 * tau2 * x0 * x0) + P.G * coupling
    )
    return dxdt


@njit
def heun_sde(x, t, P):
    nn = P.nn
    dt = P.dt
    dW_r = P.sigma_r * np.random.randn(nn)
    dW_v = P.sigma_v * np.random.randn(nn)
    k1 = f_mpr(x, t, P)
    x1 = x + dt * k1
    x1[:nn] += dW_r
    x1[nn:] += dW_v

    k2 = f_mpr(x1, t + dt, P)
    x += 0.5 * dt * (k1 + k2)
    x[:nn] += dW_r
    x[:nn] = (x[:nn] > 0.0) * x[:nn]
    x[nn:] += dW_v
    return x


@njit
def do_bold_step(r_in, s, f, ftilde, vtilde, qtilde, v, q, dtt, P):
    kappa = P.kappa
    gamma = P.gamma
    ialpha = 1 / P.alpha
    tau = P.tau
    Eo = P.Eo

    s[1] = s[0] + dtt * (r_in - kappa * s[0] - gamma * (f[0] - 1))
    f[0] = np.clip(f[0], 1, None)
    ftilde[1] = ftilde[0] + dtt * (s[0] / f[0])
    fv = v[0] ** ialpha  # outflow
    vtilde[1] = vtilde[0] + dtt * ((f[0] - fv) / (tau * v[0]))
    q[0] = np.clip(q[0], 0.01, None)
    ff = (1 - (1 - Eo) ** (1 / f[0])) / Eo  # oxygen extraction
    qtilde[1] = qtilde[0] + dtt * ((f[0] * ff - fv * q[0] / v[0]) / (tau * q[0]))

    f[1] = np.exp(ftilde[1])
    v[1] = np.exp(vtilde[1])
    q[1] = np.exp(qtilde[1])

    f[0] = f[1]
    s[0] = s[1]
    ftilde[0] = ftilde[1]
    vtilde[0] = vtilde[1]
    qtilde[0] = qtilde[1]
    v[0] = v[1]
    q[0] = q[1]


def integrate(P, B):

    nn = P.nn
    tr = P.tr
    dt = P.dt
    dt = P.dt
    rv_decimate = P.rv_decimate
    r_period = P.dt * 10 # extenting time 
    bold_decimate = int(np.round(tr / r_period))

    dtt = r_period / 1000.0  # in seconds
    k1 = 4.3 * B.theta0 * B.Eo * B.TE
    k2 = B.epsilon * B.r0 * B.Eo * B.TE
    k3 = 1 - B.epsilon
    vo = B.vo

    nt = int(P.t_end / P.dt)
    rv_current = P.initial_state
    RECORD_RV = P.RECORD_RV
    RECORD_BOLD = P.RECORD_BOLD

    rv_d = np.array([])
    rv_t = np.zeros([])

    bold_d = np.array([])
    bold_t = np.array([])

    if P.RECORD_RV:
        rv_d = np.zeros((nt // rv_decimate, 2 * nn), dtype=np.float32)
        rv_t = np.zeros((nt // rv_decimate), dtype=np.float32)

    def compute():
        nonlocal rv_d, rv_t, bold_d, bold_t

        bold_d = np.array([])
        bold_t = np.array([])
        s = np.zeros((2, nn))
        f = np.zeros((2, nn))
        ftilde = np.zeros((2, nn))
        vtilde = np.zeros((2, nn))
        qtilde = np.zeros((2, nn))
        v = np.zeros((2, nn))
        q = np.zeros((2, nn))
        vv = np.zeros((nt // bold_decimate, nn))
        qq = np.zeros((nt // bold_decimate, nn))
        s[0] = 1
        f[0] = 1
        v[0] = 1
        q[0] = 1
        ftilde[0] = 0
        vtilde[0] = 0
        qtilde[0] = 0

        for i in range(nt - 1):
            t_current = i * dt
            heun_sde(rv_current, t_current, P)

            if RECORD_RV:
                if ((i % rv_decimate) == 0) and ((i // rv_decimate) < rv_d.shape[0]):
                    rv_d[i // rv_decimate, :] = rv_current
                    rv_t[i // rv_decimate] = t_current

            if RECORD_BOLD:
                do_bold_step(
                    rv_current[:nn], s, f, ftilde, vtilde, qtilde, v, q, dtt, B
                )
                if (i % bold_decimate == 0) and ((i // bold_decimate) < vv.shape[0]):
                    vv[i // bold_decimate] = v[1]
                    qq[i // bold_decimate] = q[1]
                    
        if RECORD_RV:
            rv_d = rv_d[rv_t >= P.t_cut, :]
            rv_t = rv_t[rv_t >= P.t_cut]

        if RECORD_BOLD:
            bold_d = vo * (k1 * (1 - qq) + k2 * (1 - qq / vv) + k3 * (1 - vv))
            bold_t = np.linspace(0, P.t_end - dt * bold_decimate, len(bold_d))
            bold_d = bold_d[bold_t >= P.t_cut, :]
            bold_t = bold_t[bold_t >= P.t_cut]

        return rv_t, rv_d, bold_t, bold_d

    rv_t, rv_d, bold_t, bold_d = compute()

    return {
        "rv_t": rv_t * 10,
        "rv_d": rv_d,
        "bold_t": bold_t.astype("f") * 10,
        "bold_d": bold_d.astype("f"),
    }


class MPR_sde:
    def __init__(self, par_mpr: dict = {}) -> None:
        self.valid_par = [mpr_spec[i][0] for i in range(len(mpr_spec))]
        self.check_parameters(par_mpr)
        self.P = self.get_par_mpr(par_mpr)
        self.B = ParBold()

        self.seed = self.P.seed
        if self.seed > 0:
            np.random.seed(self.seed)

    def __str__(self) -> str:
        print("MPR model")
        print("Parameters: --------------------------------")
        for key in self.valid_par:
            print(f"{key} = {getattr(self.P, key)}")
        print("--------------------------------------------")
        return ""

    def check_parameters(self, par: dict) -> None:
        for key in par.keys():
            if key not in self.valid_par:
                raise ValueError(f"Invalid parameter: {key}")

    def get_par_mpr(self, par: dict):
        """
        return default parameters of MPR model and update with user defined parameters.
        """
        if "initial_state" in par.keys():
            par["initial_state"] = np.array(par["initial_state"])
        if "weights" in par.keys():
            assert par["weights"] is not None
            par["weights"] = np.array(par["weights"])
            assert par["weights"].shape[0] == par["weights"].shape[1]
        parP = ParMPR(**par)
        return parP

    def set_initial_state(self):
        self.initial_state = set_initial_state(self.P.nn, self.seed)
        self.INITIAL_STATE_SET = True

    def check_input(self):
        assert self.P.weights is not None
        assert self.P.weights.shape[0] == self.P.weights.shape[1]
        assert self.P.initial_state is not None
        assert len(self.P.initial_state) == 2 * self.P.weights.shape[0]
        assert self.P.t_cut < self.P.t_end, "t_cut must be less than t_end"
        self.P.eta = check_vec_size(self.P.eta, self.P.nn)
        self.P.t_end /= 10
        self.P.t_cut /= 10

    def run(self, par={}, x0=None, verbose=True):

        if x0 is None:
            self.seed = self.P.seed if self.P.seed > 0 else None
            self.set_initial_state()
            self.P.initial_state = self.initial_state
        else:
            self.P.initial_state = x0
            # self.P.nn = len(x0) // 2 # is it necessary?
        if par:
            self.check_parameters(par)
            for key in par.keys():
                setattr(self.P, key, par[key])

        self.check_input()

        return integrate(self.P, self.B)


@njit
def set_initial_state(nn, seed=None):

    if seed is not None:
        set_seed_compat(seed)

    y0 = np.random.rand(2 * nn)
    y0[:nn] = y0[:nn] * 1.5
    y0[nn:] = y0[nn:] * 4 - 2
    return y0


mpr_spec = [
    ("G", float64),
    ("dt", float64),
    ("J", float64),
    ("eta", float64[:]),
    ("tau", float64),
    ("weights", float64[:, :]),
    ("delta", float64),
    ("t_init", float64),
    ("t_cut", float64),
    ("t_end", float64),
    ("nn", int64),
    ("method", types.string),
    ("seed", int64),
    ("initial_state", float64[:]),
    ("noise_amp", float64),
    ("sigma_r", float64),
    ("sigma_v", float64),
    ("iapp", float64),
    ("output", types.string),
    ("RECORD_RV", boolean),
    ("RECORD_BOLD", boolean),
    ("rv_decimate", int64),
    ("tr", float64),
]


@jitclass(mpr_spec)
class ParMPR:
    def __init__(
        self,
        G=0.5,
        dt=0.01,
        J=14.5,
        eta=np.array([-4.6]),
        tau=1.0,
        delta=0.7,
        rv_decimate=1.0,
        noise_amp=0.037,
        weights=np.array([[], []]),
        t_init=0.0,
        t_cut=0.0,
        t_end=1000.0,
        iapp=0.0,
        seed=-1,
        output="output",
        RECORD_RV=True,
        RECORD_BOLD=True,
        tr=500.0,  # TR in milliseconds
    ):

        self.G = G
        self.dt = dt
        self.J = J
        self.eta = eta
        self.tau = tau
        self.delta = delta
        self.rv_decimate = rv_decimate
        self.noise_amp = noise_amp
        self.t_init = t_init
        self.t_cut = t_cut
        self.t_end = t_end
        self.iapp = iapp
        self.nn = len(weights)
        self.seed = seed
        self.output = output
        self.weights = weights
        self.RECORD_RV = RECORD_RV
        self.RECORD_BOLD = RECORD_BOLD
        self.sigma_r = np.sqrt(dt) * np.sqrt(2 * noise_amp)
        self.sigma_v = np.sqrt(dt) * np.sqrt(4 * noise_amp)
        self.tr = tr


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


def check_vec_size(x, nn):
    return np.ones(nn) * x if len(x) != nn else np.array(x)


@register_jitable
def set_seed_compat(x):
    np.random.seed(x)
