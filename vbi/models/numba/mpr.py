import warnings
import numpy as np
from copy import copy
from numba import njit
from numba.experimental import jitclass
from numba.core.errors import NumbaPerformanceWarning
from numba import float64, boolean, int64, types

warnings.simplefilter("ignore", category=NumbaPerformanceWarning)


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
def f_fmri(xin, x, t, B):
    """
    system function for Balloon model.
    """
    E0 = B.E0
    nn = B.nn
    inv_tauf = B.inv_tauf
    inv_tauo = B.inv_tauo
    inv_taus = B.inv_taus
    inv_alpha = B.inv_alpha

    dxdt = np.zeros(4 * nn)
    s = x[:nn]
    f = x[nn : 2 * nn]
    v = x[2 * nn : 3 * nn]
    q = x[3 * nn :]

    dxdt[:nn] = xin - inv_taus * s - inv_tauf * (f - 1.0)
    dxdt[nn : (2 * nn)] = s
    dxdt[(2 * nn) : (3 * nn)] = inv_tauo * (f - v ** (inv_alpha))
    dxdt[3 * nn :] = (inv_tauo) * (
        (f * (1.0 - (1.0 - E0) ** (1.0 / f)) / E0) - (v ** (inv_alpha)) * (q / v)
    )
    return dxdt


@njit
def integrate_fmri(yin, y, t, B):
    """
    Integrate Balloon model

    Parameters
    ----------
    yin : array [nn]
        r and v time series, r is used as input
    y : array [4*nn]
        state, update in place
    t : float
        time

    Returns
    -------
    yb : array [nn]
        BOLD signal

    """

    V0 = B.V0
    K1 = B.K1
    K2 = B.K2
    K3 = B.K3

    nn = yin.shape[0]
    y = heun_ode(yin, y, t, B)
    yb = V0 * (
        K1 * (1.0 - y[(3 * nn) :])
        + K2 * (1.0 - y[(3 * nn) :] / y[(2 * nn) : (3 * nn)])
        + K3 * (1.0 - y[(2 * nn) : (3 * nn)])
    )
    return y, yb


@njit
def integrate(P, B, intg=None):

    if intg is None:
        intg = heun_sde
    
    t = np.arange(0, P.t_end, P.dt)
    nt = len(t)
    nn = P.nn
    rs = P.ts_decimate
    dec = P.fmri_decimate
        
    n_steps = np.ceil(P.t_end / P.dt).astype(int)
    i_cut = np.ceil(P.t_cut / P.dt).astype(int)
    y0 = np.zeros((4 * nn))
    y0[nn:] = 1.0
    x0 = copy(P.initial_state)

    if P.RECORD_TS:
        _nt = np.ceil((n_steps - i_cut) / rs).astype(int)
        rv_d = np.zeros((_nt, 2 * nn), dtype="f")
        rv_t = np.zeros((_nt))
    else:
        rv_d = np.array([], dtype="f")
        rv_t = np.zeros([], dtype="f")

    if P.RECORD_FMRI:
        _nt = np.ceil((n_steps) / (rs * dec)).astype(int)
        fmri_d = np.zeros((_nt, nn), dtype="f")
        fmri_t = np.zeros((_nt))

    ii = 0  # index for rv_d
    jj = 0  # index for fmri_D

    for it in range(n_steps):

        t = it * P.dt
        intg(x0, t, P)

        if (it % rs) == 0:

            if P.RECORD_TS and (it >= i_cut):
                rv_d[ii, :] = x0
                rv_t[ii] = t
                ii += 1

            if P.RECORD_FMRI:
                y0, fmri_i = integrate_fmri(x0[:nn], y0, t[it], B)

        if P.RECORD_FMRI:
            if it % (dec * rs) == 0:
                fmri_d[jj, :] = fmri_i
                fmri_t[jj] = t[it]
                jj += 1

    return rv_t, rv_d, fmri_t, fmri_d


class MPR_sde:
    def __init__(self, par: dict = {}, parB: dict = {}) -> None:
        self.valid_parP = [mpr_spec[i][0] for i in range(len(mpr_spec))]
        self.valid_parB = [b_spec[i][0] for i in range(len(b_spec))]
        self.valid_par = self.valid_parP + self.valid_parB

        self.check_parameters(par)
        self.P = self.get_par_mpr_obj(par)
        self.B = self.get_par_baloon_obj(parB)

    def __str__(self) -> str:
        print("MPR model")
        print("Parameters: --------------------------------")
        for key in self.valid_parP:
            print(f"{key} = {getattr(self.P, key)}")
        print("Baloon model")
        print("Parameters: --------------------------------")
        for key in self.valid_parB:
            print(f"{key} = {getattr(self.B, key)}")
        print("--------------------------------------------")
        return ""

    def check_parameters(self, par: dict) -> None:
        for key in par.keys():
            if key not in self.valid_par:
                raise ValueError(f"Invalid parameter: {key}")

    def get_par_mpr_obj(self, par: dict):
        """
        return default parameters of MPR model.
        """
        if "initial_state" in par.keys():
            par["initial_state"] = np.array(par["initial_state"])
        if "weights" in par.keys():
            assert par["weights"] is not None
            par["weights"] = np.array(par["weights"])
            assert par["weights"].shape[0] == par["weights"].shape[1]
        parP = ParMPR(**par)
        return parP

    def get_par_baloon_obj(self, par: dict):
        """
        return default parameters of Baloon model.
        """
        parB = ParBaloon(**par)
        return parB

    def set_initial_state(self):
        self.initial_state = set_initial_state(self.num_nodes, self.seed)
        self.INITIAL_STATE_SET = True

    def check_input(self):
        assert self.P.weights is not None
        assert self.P.weights.shape[0] == self.P.weights.shape[1]
        assert self.P.initial_state is not None
        assert len(self.P.initial_state) == 2 * self.P.weights.shape[0]
        self.B.nn = self.P.nn

    def run(self, par={}, parB={}, x0=None, verbose=True):

        if x0 is None:
            self.seed = self.P.seed if self.P.seed > 0 else None
            self.set_initial_state(self.seed)
            self.P.initial_state = self.initial_state
        else:
            self.P.initial_state = x0
            self.P.nn = len(x0) // 2

        if par:
            self.check_parameters(par)
            for key in par.keys():
                setattr(self.P, key, par[key])

        if parB:
            self.check_parameters(parB)
            for key in parB.keys():
                setattr(self.B, key, parB[key])
        self.check_input()

        t, rv, t_fmri, d_fmri = integrate(self.P, self.B)

        return {"t": t, "x": rv, "t_fmri": t_fmri, "d_fmri": d_fmri}


def set_initial_state(nn, seed=None):

    if seed is not None:
        np.random.seed(seed)

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
    ("fmri_decimate", int64),
    ("t_init", float64),
    ("t_cut", float64),
    ("t_end", float64),
    ("nn", int64),
    ("method", types.string),
    ("seed", int64),
    ("initial_state", float64[:]),
    ("noise_sigma", float64),
    ("sigma_r", float64),
    ("sigma_v", float64),
    ("iapp", float64),
    ("output", types.string),
    ("RECORD_TS", boolean),
    ("RECORD_FMRI", boolean),
    ("ts_decimate", int64),
    ("fmri_decimate", int64),
]

b_spec = [
    ("eps", float64),
    ("E0", float64),
    ("V0", float64),
    ("alpha", float64),
    ("inv_alpha", float64),
    ("K1", float64),
    ("K2", float64),
    ("K3", float64),
    ("taus", float64),
    ("tauo", float64),
    ("tauf", float64),
    ("inv_tauo", float64),
    ("inv_taus", float64),
    ("inv_tauf", float64),
    ("nn", int64),
    ("dt", float64),
]


@jitclass(mpr_spec)
class ParMPR:
    def __init__(
        self,
        G=0.5,
        dt=0.1,
        J=14.5,
        eta=np.array([]),
        tau=1.0,
        delta=0.7,
        fmri_decimate=1,
        ts_decimate=1,
        noise_sigma=0.037,
        weights=np.array([[], []]),
        t_init=0.0,
        t_cut=100.0,
        t_end=1000.0,
        iapp=0.0,
        output="output",
        RECORD_TS=True,
        RECORD_FMRI=True,
    ):

        self.G = G
        self.dt = dt
        self.J = J
        self.eta = eta
        self.tau = tau
        self.delta = delta
        self.fmri_decimate = fmri_decimate
        self.ts_decimate = ts_decimate
        self.noise_sigma = noise_sigma
        self.t_init = t_init
        self.t_cut = t_cut
        self.t_end = t_end
        self.iapp = iapp
        self.nn = 0
        self.output = output
        self.weights = weights
        self.RECORD_TS = RECORD_TS
        self.RECORD_FMRI = RECORD_FMRI
        self.sigma_r = np.sqrt(dt) * np.sqrt(2 * noise_sigma)
        self.sigma_v = np.sqrt(dt) * np.sqrt(4 * noise_sigma)


@jitclass(b_spec)
class ParBaloon:
    def __init__(
        self, eps=0.5, E0=0.4, V0=4.0, alpha=0.32, taus=1.54, tauo=0.98, tauf=1.44
    ):
        self.eps = eps
        self.E0 = E0
        self.V0 = V0
        self.alpha = alpha
        self.inv_alpha = 1.0 / alpha
        self.K1 = 7.0 * E0
        self.K2 = 2 * E0
        self.K3 = 1 - eps
        self.taus = taus
        self.tauo = tauo
        self.tauf = tauf
        self.inv_tauo = 1.0 / tauo
        self.inv_taus = 1.0 / taus
        self.inv_tauf = 1.0 / tauf
        self.dt = 0.001


@njit
def euler_sde(x, t, P):
    dW = np.sqrt(P.dt) * P.sigma_noise * np.random.randn(P.nn)
    return x + P.dt * f_mpr(x, t, P) + dW


@njit
def heun_sde(x, t, P):
    dW = np.sqrt(P.dt) * P.sigma_noise * np.random.randn(P.nn)
    k0 = f_mpr(x, t, P)
    x1 = x + P.dt * k0 + dW
    k1 = f_mpr(x1, t, P)
    return x + 0.5 * P.dt * (k0 + k1) + dW


@njit
def heun_ode(yin, y, t, B):
    """Heun scheme."""

    dt = B.dt
    k1 = f_fmri(yin, y, t, B)
    tmp = y + dt * k1
    k2 = f_fmri(yin, tmp, t + dt, B)
    y += 0.5 * dt * (k1 + k2)
    return y
