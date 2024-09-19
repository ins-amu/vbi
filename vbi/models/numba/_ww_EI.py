import warnings
import numpy as np
from numba import njit
from numba.experimental import jitclass
from numba.core.errors import NumbaPerformanceWarning
from numba import float64, boolean, int64, types

warnings.simplefilter("ignore", category=NumbaPerformanceWarning)


w_spec = [
    ("G", float64),
    
    ("a_I", float64),
    ("b_I", float64),
    ("d_I", float64),
    ("tau_I", float64),
    
    ("a_E", float64),
    ("b_E", float64),
    ("d_E", float64),
    ("tau_E", float64),
    
    ("w_II", float64),
    ("w_EE", float64),
    ("w_IE", float64),
    ("w_EI", float64),
    
    ("W_E", float64),
    ("W_I", float64),
    
    ("gamma", float64),
    ("dt", float64),
    ("J_NMDA", float64),
    ("J_I", float64),
    
    ("I_I", float64[:]),
    ("I_E", float64[:]),
    
    # ("sigma_I", float64),
    # ("sigma_E", float64),
    
    ("initial_state", float64[:]),
    ("weights", float64[:, :]),
    ("seed", int64),
    ("method", types.string),
    ("t_end", float64),
    ("t_cut", float64),
    ("nn", int64),
    ("ts_decimate", int64),
    ("fmri_decimate", int64),
    ("RECORD_TS", boolean),
    ("RECORD_FMRI", boolean),
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
    ("dt_bold", float64),
]


@jitclass(w_spec)
class ParWW:
    def __init__(
        self,
        G=0.0,
        a_I=615.0,
        b_I=177.0,
        d_I=0.087,
        tau_I=0.01,
        
        a_E=310.0,
        b_E=125.0,
        d_E=0.16,
        tau_E=0.1,
        
        gamma=0.641,
        
        w_II=1.0,
        w_IE=1.4,
        w_EI=1.0,
        w_EE=1.0,
        dt=0.01,
        
        W_E=1.0,
        W_I =0.7,
        
        I0 = 0.382,
        J_NMDA=0.15,
        
        
        I_I=np.array([0.296]),  # 0.296
        I_E=np.array([0.377]),  # 0.377
        sigma_I=0.001,
        sigma_E=0.001,
        initial_state=np.array([]),
        weights=np.array([[], []]),
        seed=-1,
        method="heun",
        t_end=300.0,
        t_cut=0.0,
        ts_decimate=10,
        fmri_decimate=10,
        RECORD_TS=True,
        RECORD_FMRI=True,
    ):
        self.G = G
        self.a_I = a_I
        self.b_I = b_I
        self.d_I = d_I
        self.tau_I = tau_I
        
        self.a_E = a_E
        self.b_E = b_E
        self.d_E = d_E
        self.tau_E = tau_E

        self.w_II = w_II
        self.w_IE = w_IE
        self.w_EI = w_EI
        self.w_EE = w_EE
        self.gamma = gamma

        self.dt = dt
        
        self.W_E = W_E
        self.W_I = W_I
        
        self.I0 = I0
        self.I_E = I_E
        self.I_I = I_I
        self.J_NMDA = J_NMDA

        self.sigma_I = sigma_I
        self.sigma_E = sigma_E

        self.initial_state = initial_state
        self.weights = weights
        self.seed = seed
        self.method = method
        self.t_end = t_end
        self.t_cut = t_cut
        self.ts_decimate = ts_decimate
        self.fmri_decimate = fmri_decimate
        self.RECORD_TS = RECORD_TS
        self.RECORD_FMRI = RECORD_FMRI
        if len(initial_state) > 0:
            self.nn = len(initial_state)
        else:
            self.nn = -1


@jitclass(b_spec)
class ParBaloon:
    def __init__(
        self, eps=0.5, E0=0.4, V0=4.0, 
        alpha=0.32, taus=1.54, tauo=0.98, tauf=1.44
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
        self.dt_bold = 0.01


@njit
def f_ww(S, P):
    """
    system function for Wong-Wang model.
    """
    coupling = np.dot(P.weights, S)
    x = P.w * P.J_N * S + P.I_o + P.G * P.J_N * coupling
    H = (P.a * x - P.b) / (1 - np.exp(-P.d * (P.a * x - P.b)))
    dS = -(S / P.tau_s) + (1 - S) * H * P.gamma
    return dS


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
def euler_sde_step(S, P):
    dW = np.sqrt(P.dt) * P.sigma_noise * np.random.randn(P.nn)
    return S + P.dt * f_ww(S, P) + dW


@njit
def heun_sde_step(S, P):
    dW = np.sqrt(P.dt) * P.sigma_noise * np.random.randn(P.nn)
    k0 = f_ww(S, P)
    S1 = S + P.dt * k0 + dW
    k1 = f_ww(S1, P)
    return S + 0.5 * P.dt * (k0 + k1) + dW


@njit
def heun_ode_step(yin, y, t, B):
    """Heun scheme."""

    dt = B.dt_bold
    k1 = f_fmri(yin, y, t, B)
    tmp = y + dt * k1
    k2 = f_fmri(yin, tmp, t + dt, B)
    y += 0.5 * dt * (k1 + k2)
    return y


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
    y = heun_ode_step(yin, y, t, B)
    yb = V0 * (
        K1 * (1.0 - y[(3 * nn) :])
        + K2 * (1.0 - y[(3 * nn) :] / y[(2 * nn) : (3 * nn)])
        + K3 * (1.0 - y[(2 * nn) : (3 * nn)])
    )
    return y, yb


@njit
def integrate(P, B, intg=heun_sde_step):
    """
    integrate Wong-Wang model and Balloon model.
    """
    t = np.arange(0, P.t_end, P.dt)
    nt = len(t)
    nn = P.nn

    if P.RECORD_TS:
        T = np.empty(int(np.ceil(nt / P.ts_decimate)))
        S = np.empty((int(np.ceil(nt / P.ts_decimate)), nn))
    else:
        T = np.empty(0)
        S = np.empty((0, 1))

    if P.RECORD_FMRI:
        t_fmri = np.empty(int(np.ceil(nt / P.fmri_decimate)))
        d_fmri = np.empty((int(np.ceil(nt / P.fmri_decimate)), nn))
    else:
        t_fmri = np.empty(0)
        d_fmri = np.empty((0, 1))

    x0 = P.initial_state
    y0 = np.zeros((4 * nn))
    y0[nn:] = 1.0

    jj = 0
    ii = 0
    for i in range(1, nt):

        t = i * P.dt
        t_bold = i * B.dt_bold
        x0 = intg(x0, P)

        if P.RECORD_TS:
            if i % P.ts_decimate == 0:
                S[ii, :] = x0
                T[ii] = t
                ii += 1
        if P.RECORD_FMRI:
            y0, fmri_i = integrate_fmri(x0, y0, t, B)
            if i % P.fmri_decimate == 0:
                d_fmri[jj, :] = fmri_i
                # t_fmri[jj] = t[i]
                t_fmri[jj] = t_bold
                jj += 1
    S = S[T >= P.t_cut, :]
    T = T[T >= P.t_cut]
    d_fmri = d_fmri[t_fmri >= P.t_cut, :]
    t_fmri = t_fmri[t_fmri >= P.t_cut]

    return T, S, t_fmri, d_fmri


class WW_sde(object):
    r"""
    Wong-Wang model.
    
    .. math::
                    x_k       &=   w\,J_N \, S_k + I_o + G\,J_N \sum_j \, C_{kj} \,Sj \\
                    H(x_k)    &=  \dfrac{ax_k - b}{1 - \exp(-d(ax_k -b))}\\
                    \dot{S}_k &= -\dfrac{S_k}{\tau_s} + (1 - S_k) \, H(x_k) \, \gamma + \sigma \, \Xi_k
                    
    - Kong-Fatt Wong and Xiao-Jing Wang, A Recurrent Network Mechanism of Time Integration in Perceptual Decisions. Journal of Neuroscience 26(4), 1314-1328, 2006.       
    - Deco Gustavo, Ponce Alvarez Adrian, Dante Mantini, Gian Luca Romani, Patric Hagmann and Maurizio Corbetta. Resting-State Functional Connectivity Emerges from Structurally and Dynamically Shaped Slow Linear Fluctuations. The Journal of Neuroscience 32(27), 11239-11252, 2013. Equations taken from DPA 2013 , page 11242.
    """

    def __init__(self, par: dict = {}, parB: dict = {}) -> None:

        self.valid_parW = [w_spec[i][0] for i in range(len(w_spec))]
        self.valid_parB = [b_spec[i][0] for i in range(len(b_spec))]
        self.valid_par = self.valid_parW + self.valid_parB

        self.check_parameters(par)
        self.P = self.get_par_WW_obj(par)
        self.B = self.get_par_Baloon_obj(parB)

    def __str__(self):
        print("Wong-Wang model of neural population dynamics")
        print("Parameters:----------------------------")
        for key in self.valid_parW:
            print(key, ": ", getattr(self.P, key))
        print("---------------------------------------")
        for key in self.valid_parB:
            print(key, ": ", getattr(self.B, key))
        return ""

    def get_par_WW_obj(self, par: dict = {}) -> ParWW:
        """
        return default parameters for Wong-Wang model.
        """
        if "initial_state" in par.keys():
            par["initial_state"] = np.array(par["initial_state"])
        if "weights" in par.keys():
            assert par["weights"] is not None
            par["weights"] = np.array(par["weights"])
            assert par["weights"].shape[0] == par["weights"].shape[1]
        parobj = ParWW(**par)

        return parobj

    def get_par_Baloon_obj(self, par: dict = {}) -> ParBaloon:
        """
        return default parameters for Balloon model.
        """
        parobj = ParBaloon(**par)
        return parobj

    def check_parameters(self, par: dict) -> None:
        for key in par.keys():
            if key not in self.valid_par:
                raise ValueError(f"Invalid parameter {key}")

    def set_initial_state(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.P.nn = self.P.weights.shape[0]
        self.initial_state = np.random.rand(self.P.nn)
        self.B.nn = self.P.nn

    def check_input(self):

        assert self.P.weights is not None
        assert self.P.weights.shape[0] == self.P.weights.shape[1]
        assert self.P.initial_state is not None
        assert len(self.P.initial_state) == self.P.weights.shape[0]
        self.B.nn = self.P.nn
        # self.B.dt = self.P.dt

    def run(self, par={}, parB={}, x0=None, verbose=True):

        if x0 is None:
            self.seed = self.P.seed if self.P.seed > 0 else None
            self.set_initial_state(self.seed)
            self.P.initial_state = self.initial_state
        else:
            self.P.initial_state = x0
            self.P.nn = len(x0)

        if par:
            self.check_parameters(par)
            for key in par.keys():
                setattr(self.P, key, par[key])
        if parB:
            self.check_parameters(parB)
            for key in parB.keys():
                setattr(self.B, key, parB[key])
        self.check_input()

        T, S, t_fmri, d_fmri = integrate(self.P, self.B)

        return {"t": T, "s": S, "t_fmri": t_fmri, "d_fmri": d_fmri}
