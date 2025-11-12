import warnings
import numpy as np
from copy import copy
from numba import njit, jit
from numba.experimental import jitclass
from numba.extending import register_jitable
from vbi.utils import print_valid_parameters
from numba import float64, boolean, int64, types
from numba.core.errors import NumbaPerformanceWarning
from vbi.models.numba.utils import check_vec_size_1d

warnings.simplefilter("ignore", category=NumbaPerformanceWarning)


# -----------------------------
# Helper utilities
# -----------------------------

@jit(nopython=True)
def initialize_random_state(seed):
    """
    Initialize the random number generator with a specific seed.
    
    This function sets the random seed in the Numba context to ensure
    reproducible stochastic simulations.
    
    Parameters
    ----------
    seed : int
        Random seed value for reproducibility
    """
    np.random.seed(seed)



# def check_vec_size_1d(x, nn):
#     """
#     Return a 1D vector of size nn, broadcasting scalar if needed.
    
#     This utility function ensures that parameter inputs are properly
#     formatted as arrays of the correct size for multi-region simulations.
    
#     Parameters
#     ----------
#     x : scalar or array-like
#         Input value(s) to be broadcast or validated
#     nn : int
#         Required array size (number of brain regions)
        
#     Returns
#     -------
#     np.ndarray
#         Array of shape (nn,) with input values properly broadcast
#     """
#     x = np.array(x, dtype=np.float64) if np.ndim(x) > 0 else np.array([x], dtype=np.float64)
#     return np.ones(nn, dtype=np.float64) * x if x.size != nn else x.astype(np.float64)


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
    ("sigma", float64[:]),
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
    """
    Parameter class for the Wong-Wang full neural mass model.
    
    This Numba jitclass holds all parameters required for Wong-Wang simulation
    including biophysical parameters for excitatory and inhibitory populations,
    coupling strengths, noise levels, and simulation settings. Uses Numba for 
    high-performance compilation.
    
    Parameters
    ----------
    a_exc : float, default 310.0
        Excitatory population gain parameter (n/C)
    a_inh : float, default 0.615
        Inhibitory population gain parameter (nC⁻¹)
    b_exc : float, default 125.0
        Excitatory population threshold parameter (Hz)
    b_inh : float, default 177.0
        Inhibitory population threshold parameter (Hz)
    d_exc : float, default 0.16
        Excitatory population saturation parameter (s)
    d_inh : float, default 0.087
        Inhibitory population saturation parameter (s)
    tau_exc : float, default 100.0
        Excitatory synaptic time constant (ms)
    tau_inh : float, default 10.0
        Inhibitory synaptic time constant (ms)
    gamma_exc : float, default 0.641/1000
        Excitatory kinetic parameter (ms⁻¹)
    gamma_inh : float, default 1.0/1000
        Inhibitory kinetic parameter (ms⁻¹)
    W_exc : float, default 1.0
        Excitatory population local weight
    W_inh : float, default 0.7
        Inhibitory population local weight
    ext_current : array-like, default [0.382]
        External current input per region (nA)
    J_NMDA : float, default 0.15
        NMDA synaptic coupling strength (nA)
    J_I : float, default 1.0
        Inhibitory synaptic coupling strength (nA)
    w_plus : float, default 1.4
        Local excitatory recurrence strength
    lambda_inh_exc : float, default 0.0
        Long-range feedforward inhibition switch
    """
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
        sigma=np.array([0.0]),
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
    Compute the firing rate using the Wong-Wang transfer function.
    
    Implements the input-output relationship for neural populations:
    r(I) = (a*I - b) / (1 - exp(-d*(a*I - b)))
    
    This function captures the nonlinear relationship between synaptic
    current and population firing rate, including saturation effects.
    
    Parameters
    ----------
    current : np.ndarray
        Synaptic current input to the population.
    a : float
        Gain parameter controlling the slope of the transfer function.
    b : float
        Threshold parameter determining the firing threshold.
    d : float
        Saturation parameter controlling the saturation behavior.
        
    Returns
    -------
    np.ndarray
        Population firing rate (Hz).
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
    Compute the right-hand side of the Wong-Wang full model equations.
    
    This function implements the deterministic part of the Wong-Wang model,
    computing the time derivatives of synaptic gating variables for both
    excitatory and inhibitory populations across all brain regions.
    
    The implemented equations are:
    dS_exc/dt = -S_exc/τ_exc + (1-S_exc)*γ_exc*r_exc(I_exc)
    dS_inh/dt = -S_inh/τ_inh + γ_inh*r_inh(I_inh)
    
    Parameters
    ----------
    S : np.ndarray
        Current state vector of shape (2*nn,) containing stacked arrays:
        [S_exc_0, ..., S_exc_n, S_inh_0, ..., S_inh_n] where nn is the 
        number of brain regions.
    t : float
        Current time (not used in autonomous system).
    P : ParWW
        Parameter object containing all model parameters.
        
    Returns
    -------
    np.ndarray
        Derivative vector dS/dt of shape (2*nn,).
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
    Perform one heun integration step for the stochastic Wong-Wang model.
    
    This function implements the Heun method (predictor-corrector) for 
    numerical integration of stochastic differential equations, providing 
    second-order accuracy for the deterministic part while properly handling 
    additive Gaussian noise.
    
    Parameters
    ----------
    S : np.ndarray
        Current state vector of shape (2*nn,) containing synaptic gating variables.
    t : float
        Current time (ms).
    P : ParWW
        Parameter object containing model parameters including dt and sigma.
        
    Returns
    -------
    np.ndarray
        Updated state vector after one integration step.
    """
    dt = P.dt
    nn = P.nn

    # Handle sigma as either scalar or array
    if P.sigma.size == 1:
        # Scalar case: apply same noise to all variables
        dW = P.sigma[0] * np.sqrt(dt) * np.random.randn(2 * nn)
    else:
        # Array case: apply different noise to exc and inh populations
        dW = np.zeros(2 * nn)
        dW[:nn] = P.sigma * np.sqrt(dt) * np.random.randn(nn)
        dW[nn:] = P.sigma * np.sqrt(dt) * np.random.randn(nn)

    k1 = f_ww(S, t, P)
    y_ = S + dt * k1 + dW
    k2 = f_ww(y_, t + dt, P)
    S = S + 0.5 * dt * (k1 + k2) + dW

    return S


# -----------------------------
# Public-facing class (mirror mpr.py style)
# -----------------------------

class WW_sde:
    """
    Numba implementation of the Wong-Wang full neural mass model with stochastic dynamics.
    
    The Wong-Wang full model is a biophysically realistic neural mass model that explicitly 
    captures the dynamics of both excitatory and inhibitory neural populations. This model 
    is based on the original work of Wong and Wang [Wong2006]_ and has been extended for 
    whole-brain network simulations [Deco2013]_, [Deco2014]_. The model provides a detailed 
    representation of recurrent network mechanisms underlying decision-making processes and 
    has been widely used to study brain dynamics in health and disease.

    The model describes the temporal evolution of synaptic gating variables for excitatory 
    (S_exc) and inhibitory (S_inh) populations at each brain region. The dynamics are 
    governed by the balance between synaptic decay, activity-dependent facilitation, and 
    network coupling. The firing rates of each population are determined by input-output 
    transfer functions that capture the relationship between synaptic currents and 
    population firing rates.

    The Wong-Wang full model equations are:

    .. math::

        \\frac{dS_{exc,i}}{dt} &= -\\frac{S_{exc,i}}{\\tau_{exc}} + (1 - S_{exc,i}) \\gamma_{exc} r_{exc,i}(t) + \\sigma \\xi_i(t) \\\\
        \\frac{dS_{inh,i}}{dt} &= -\\frac{S_{inh,i}}{\\tau_{inh}} + \\gamma_{inh} r_{inh,i}(t) + \\sigma \\xi_i(t)

    where the firing rates are computed using:

    .. math::

        r_{exc,i}(t) &= \\frac{a_{exc} I_{exc,i} - b_{exc}}{1 - \\exp(-d_{exc}(a_{exc} I_{exc,i} - b_{exc}))} \\\\
        r_{inh,i}(t) &= \\frac{a_{inh} I_{inh,i} - b_{inh}}{1 - \\exp(-d_{inh}(a_{inh} I_{inh,i} - b_{inh}))}

    The total synaptic currents for each population are:

    .. math::

        I_{exc,i} &= W_{exc} I_{ext} + w_{plus} J_{NMDA} S_{exc,i} + G_{exc} J_{NMDA} \\sum_{j=1}^{N} SC_{ij} S_{exc,j} - J_I S_{inh,i} \\\\
        I_{inh,i} &= W_{inh} I_{ext} + J_{NMDA} S_{exc,i} - S_{inh,i} + G_{inh} J_{NMDA} \\lambda_{inh,exc} \\sum_{j=1}^{N} SC_{ij} S_{inh,j}

    
    .. list-table:: Parameters
        :widths: 25 50 25
        :header-rows: 1

        * - Name
          - Explanation
          - Default Value
        * - `a_exc`
          - Excitatory population gain parameter (n/C)
          - 310.0
        * - `a_inh`
          - Inhibitory population gain parameter (nC⁻¹)
          - 0.615
        * - `b_exc`
          - Excitatory population threshold parameter (Hz)
          - 125.0
        * - `b_inh`
          - Inhibitory population threshold parameter (Hz)
          - 177.0
        * - `d_exc`
          - Excitatory population saturation parameter (s)
          - 0.16
        * - `d_inh`
          - Inhibitory population saturation parameter (s)
          - 0.087
        * - `tau_exc`
          - Excitatory synaptic time constant (ms)
          - 100.0
        * - `tau_inh`
          - Inhibitory synaptic time constant (ms)
          - 10.0
        * - `gamma_exc`
          - Excitatory kinetic parameter (ms⁻¹)
          - 0.641/1000
        * - `gamma_inh`
          - Inhibitory kinetic parameter (ms⁻¹)
          - 1.0/1000
        * - `W_exc`
          - Excitatory population local weight
          - 1.0
        * - `W_inh`
          - Inhibitory population local weight
          - 0.7
        * - `ext_current`
          - External current input (nA)
          - 0.382
        * - `J_NMDA`
          - NMDA synaptic coupling strength (nA)
          - 0.15
        * - `J_I`
          - Inhibitory synaptic coupling strength (nA)
          - 1.0
        * - `w_plus`
          - Local excitatory recurrence strength
          - 1.4
        * - `lambda_inh_exc`
          - Long-range feedforward inhibition switch
          - 0.0
        * - `G_exc`
          - Global excitatory coupling strength
          - 0.0
        * - `G_inh`
          - Global inhibitory coupling strength
          - 0.0
        * - `sigma`
          - Noise amplitude
          - 0.0
        * - `weights`
          - Structural connectivity matrix (nn x nn)
          - zeros
        * - `dt`
          - Integration time step (ms)
          - 0.1
        * - `t_end`
          - Simulation end time (ms)
          - 1000.0
        * - `t_cut`
          - Initial time to discard (ms)
          - 0.0
        

    Usage example:
        >>> import numpy as np
        >>> from vbi.models.numba.ww import WW_sde
        >>> W = np.eye(2) * 0.1  # 2-node connectivity
        >>> I_ext = np.array([0.4, 0.5])  # External currents
        >>> ww = WW_sde({
        ...     "weights": W, 
        ...     "ext_current": I_ext,
        ...     "G_exc": 0.5,
        ...     "G_inh": 0.2,
        ...     "dt": 0.1, 
        ...     "t_end": 1000.0, 
        ...     "t_cut": 100.0,
        ...     "sigma": 0.01
        ... })
        >>> data = ww.run()
        >>> print(data['bold_d'].shape)  # BOLD data shape
        (2, 2)
        >>> print(ww.P.tr)  # Repetition time
        300.0
        

    Notes
    -----
    The Wong-Wang model is particularly useful for studying:
    - Decision-making processes and attractor dynamics
    - Excitatory-inhibitory balance in cortical circuits
    - Biophysically realistic neural mass dynamics
    - Working memory mechanisms
    - Brain state transitions and criticality
    
    The model's strength lies in its biophysical realism while maintaining
    computational efficiency through mean-field approximations.
    
    References
    ----------
    .. [Wong2006] Wong, K. F., & Wang, X. J. (2006). A recurrent network 
       mechanism of time integration in perceptual decisions. Journal of 
       Neuroscience, 26(4), 1314-1328.
    .. [Deco2013] Deco, G., Ponce-Alvarez, A., Mantini, D., Romani, G. L., 
       Hagmann, P., & Corbetta, M. (2013). Resting-state functional 
       connectivity emerges from structurally and dynamically shaped slow 
       linear fluctuations. Journal of Neuroscience, 33(27), 11239-11252.
    .. [Deco2014] Deco, G., Ponce-Alvarez, A., Hagmann, P., Romani, G. L., 
       Mantini, D., & Corbetta, M. (2014). How local excitation–inhibition 
       ratio impacts the whole brain dynamics. Journal of Neuroscience, 
       34(23), 7886-7898.
    """
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

        # handle sigma parameter - can be scalar or array
        par.setdefault("sigma", 0.0)
        sigma_val = par["sigma"]
        if np.isscalar(sigma_val):
            par["sigma"] = np.array([sigma_val], dtype=np.float64)
        else:
            sigma_val = np.array(sigma_val, dtype=np.float64)
            if sigma_val.size != 1 and sigma_val.size != nn:
                raise ValueError(f"sigma must be scalar or array of size {nn}, got size {sigma_val.size}")
            par["sigma"] = sigma_val

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
            sigma=par["sigma"].astype(np.float64),
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
    
    def check_parameters(self, par: dict) -> None:
        """
        Validate that all provided parameters are recognized.
        
        Parameters
        ----------
        par : dict
            Dictionary of parameters to validate.
            
        Raises
        ------
        ValueError
            If any parameter name is not recognized.
        """
        for key in par.keys():
            if key not in self.valid_par:
                print(f"Invalid parameter: {key}")
                print_valid_parameters(ww_spec)
                raise ValueError(f"Invalid parameter: {key}")
    # -----------------------------
    # Simulation
    # -----------------------------
    def run(self, par: dict = None, x0=None, verbose=True):
        """
        Run the Wong-Wang full model simulation.
        
        Executes the Euler-Maruyama numerical integration scheme to simulate
        the stochastic Wong-Wang dynamics. Includes optional BOLD signal
        generation and data downsampling.
        
        Parameters
        ----------
        par : dict, optional
            Dictionary of parameters to update before simulation.
            Can include any parameter from the ParWW class.
        x0 : array-like, optional
            Initial state vector of length 2*nn (S_exc and S_inh for each region).
            If None, uses the stored initial_state from the parameter object.
        verbose : bool, default True
            Whether to print simulation progress and parameter information.
            
        Returns
        -------
        dict
            Dictionary containing simulation results with keys:
            
            - 'S' : ndarray, shape (T, nn)
                Recorded excitatory synaptic gating variables if RECORD_S=True
            - 't' : ndarray, shape (T,)
                Time points for neural data (ms)
            - 'bold_t' : ndarray, shape (T_bold,)
                Time points for BOLD signal (ms)
            - 'bold_d' : ndarray, shape (T_bold, nn)
                BOLD signal time series if RECORD_BOLD=True
        
        Notes
        -----
        The simulation uses the Heun-Euler method for stochastic integration,
        which provides second-order accuracy for the deterministic part while
        handling the stochastic noise appropriately.
        """
        
        self.valid_par = [ww_spec[i][0] for i in range(len(ww_spec))]
        self.check_parameters(par)
        
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
    """
    Generate random initial conditions for the Wong-Wang model.
    
    Creates small positive random values for both excitatory and inhibitory
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
        Initial state vector of shape (2*nn,) containing small positive
        random values for [S_exc_0, ..., S_exc_n, S_inh_0, ..., S_inh_n].
    """
    if seed is not None and seed >= 0:
        np.random.seed(seed)
        # initialize_random_state(seed)
    y0 = np.random.rand(2 * nn) * 0.1  # small positive
    return y0.astype(np.float64)
