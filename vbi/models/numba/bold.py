import warnings
import numpy as np
from numba import jit
from numba import float64
from numba.experimental import jitclass

# from copy import copy
# from numba.core.errors import NumbaPerformanceWarning
# from numba.extending import register_jitable
# from vbi.utils import print_valid_parameters

# warnings.simplefilter("ignore", category=NumbaPerformanceWarning)

# ---------------------
# BOLD model parameters
# ---------------------

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
    """
    Parameter class for BOLD signal generation in the Wong-Wang model.
    
    This Numba jitclass holds parameters for the hemodynamic response model
    that converts neural activity to BOLD signal. Based on the Balloon-Windkessel
    model for simulating fMRI BOLD responses.
    
    Parameters
    ----------
    kappa : float, default 0.65
        Signal decay parameter
    gamma : float, default 0.41
        Feedback regulation parameter
    tau : float, default 0.98
        Hemodynamic transit time (s)
    alpha : float, default 0.32
        Grubb's vessel stiffness exponent
    epsilon : float, default 0.34
        Efficacy of oxygen extraction
    Eo : float, default 0.4
        Oxygen extraction fraction at rest
    TE : float, default 0.04
        Echo time (s)
    vo : float, default 0.08
        Resting venous volume fraction
    r0 : float, default 25.0
        Slope parameter for intravascular signal
    theta0 : float, default 40.3
        Frequency offset at the outer surface of magnetized vessels (Hz)
    t_min : float, default 0.0
        Minimum integration time
    rtol : float, default 1e-5
        Relative tolerance for integration
    atol : float, default 1e-8
        Absolute tolerance for integration
    """
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
    Perform one integration step of the BOLD hemodynamic response model.
    
    Implements the Balloon-Windkessel model to convert neural activity into
    BOLD signal by simulating the hemodynamic cascade: neural activity →
    vasodilatory signal → blood flow → blood volume → deoxyhemoglobin →
    BOLD signal.
    
    Parameters
    ----------
    r_in : np.ndarray
        Neural activity input (excitatory synaptic gating variables)
    s : np.ndarray, shape (2, nn)
        Vasodilatory signal state variables
    f : np.ndarray, shape (2, nn)
        Blood flow state variables
    ftilde : np.ndarray, shape (2, nn)
        Log-transformed blood flow variables
    vtilde : np.ndarray, shape (2, nn)
        Log-transformed blood volume variables
    qtilde : np.ndarray, shape (2, nn)
        Log-transformed deoxyhemoglobin content variables
    v : np.ndarray, shape (2, nn)
        Blood volume state variables
    q : np.ndarray, shape (2, nn)
        Deoxyhemoglobin content state variables
    dtt : float
        Integration time step (seconds)
    P : ParBold
        BOLD parameter object
        
    Notes
    -----
    This function modifies the state arrays in-place and uses a log-transform
    approach to ensure numerical stability of the hemodynamic state variables.
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
