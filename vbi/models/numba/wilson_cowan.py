
import warnings
import numpy as np
from numba import njit, jit
from numba.experimental import jitclass
from numba.extending import register_jitable
from numba import float64, boolean, int64, types
from numba.core.errors import NumbaPerformanceWarning
from vbi.utils import print_valid_parameters

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
    """
    Parameter class for the Wilson-Cowan neural mass model.
    
    This Numba jitclass holds all parameters required for Wilson-Cowan simulation
    including synaptic strengths, time constants, sigmoid parameters, noise levels,
    and simulation settings. Uses Numba for high-performance compilation.
    
    Parameters
    ----------
    c_ee : array-like, default [16.0]
        Excitatory to excitatory synaptic strength
    c_ei : array-like, default [12.0] 
        Excitatory to inhibitory synaptic strength
    c_ie : array-like, default [15.0]
        Inhibitory to excitatory synaptic strength
    c_ii : array-like, default [3.0]
        Inhibitory to inhibitory synaptic strength
    tau_e : array-like, default [8.0]
        Excitatory population time constant
    tau_i : array-like, default [8.0]
        Inhibitory population time constant
    a_e : float, default 1.3
        Excitatory sigmoid maximum firing rate parameter
    a_i : float, default 2.0
        Inhibitory sigmoid maximum firing rate parameter
    b_e : float, default 4.0
        Excitatory sigmoid steepness parameter
    b_i : float, default 3.7
        Inhibitory sigmoid steepness parameter
    c_e : float, default 1.0
        Excitatory sigmoid center parameter
    c_i : float, default 1.0
        Inhibitory sigmoid center parameter  
    theta_e : float, default 0.0
        Excitatory threshold parameter
    theta_i : float, default 0.0
        Inhibitory threshold parameter
    r_e : float, default 1.0
        Excitatory refractory parameter
    r_i : float, default 1.0
        Inhibitory refractory parameter
    k_e : float, default 0.994
        Excitatory maximum response parameter
    k_i : float, default 0.999
        Inhibitory maximum response parameter
    alpha_e : float, default 1.0
        Excitatory scaling parameter
    alpha_i : float, default 1.0
        Inhibitory scaling parameter
    P : array-like, default [0.0]
        External input to excitatory population
    Q : array-like, default [0.0]
        External input to inhibitory population
    g_e : float, default 0.0
        Excitatory global coupling strength
    g_i : float, default 0.0
        Inhibitory global coupling strength
    dt : float, default 0.01
        Integration time step (ms)
    t_end : float, default 300.0
        Simulation end time (ms)
    t_cut : float, default 0.0
        Initial time to discard (ms)
    """
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
    """
    Vectorized sigmoidal transfer function for Wilson-Cowan model.
    
    Computes either standard sigmoid or shifted sigmoid that maps synaptic
    input to firing rate, capturing saturation effects and thresholds.
    
    Parameters
    ----------
    x : np.ndarray
        Input values (synaptic drive).
    a : float
        Sigmoid slope parameter.
    b : float
        Sigmoid threshold parameter.
    c : float
        Maximum output of sigmoid.
    shift_sigmoid : bool
        If True, uses shifted sigmoid: c * (sigmoid(a(x-b)) - sigmoid(-ab))
        If False, uses standard sigmoid: c / (1 + exp(-a(x-b)))
        
    Returns
    -------
    np.ndarray
        Sigmoid-transformed firing rates.
    """
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
    Compute the right-hand side of the Wilson-Cowan neural mass model.
    
    This function implements the deterministic part of the Wilson-Cowan equations
    for a network of coupled excitatory-inhibitory neural populations. The model
    describes the temporal evolution of mean firing rates using nonlinear 
    differential equations with sigmoidal transfer functions.
    
    The equations are:
    
    .. math::
    
        \\tau_e \\frac{dE}{dt} = -E + (k_e - r_e E) \\cdot S_e(\\alpha_e(c_{ee}E - c_{ei}I + P - \\theta_e + g_e\\sum w_{ij}E_j))
        
        \\tau_i \\frac{dI}{dt} = -I + (k_i - r_i I) \\cdot S_i(\\alpha_i(c_{ie}E - c_{ii}I + Q - \\theta_i + g_i\\sum w_{ij}I_j))
    
    Parameters
    ----------
    x : np.ndarray
        Current state vector of shape (2*nn,) containing stacked arrays:
        [E₀, E₁, ..., Eₙ, I₀, I₁, ..., Iₙ] where nn is the number of nodes.
        E represents excitatory population activity, I represents inhibitory.
    t : float
        Current time (not used in autonomous system).
    P : ParWC
        Parameter object containing all model parameters.
        
    Returns
    -------
    np.ndarray
        Derivative vector of shape (2*nn,) containing dx/dt.
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
    Numba implementation of the Wilson-Cowan neural mass model with stochastic dynamics.
    
    The Wilson-Cowan model is a seminal neural mass model that describes the dynamics 
    of connected excitatory and inhibitory neural populations at the cortical microcolumn 
    level. It models the temporal evolution of mean firing rates using nonlinear 
    differential equations with sigmoidal transfer functions that capture saturation 
    effects and thresholds in neural response.
    
    This implementation includes:
    - Coupled excitatory (E) and inhibitory (I) populations per brain region
    - Network connectivity through structural connectivity matrix
    - Additive Gaussian noise for stochastic dynamics
    - Flexible sigmoid functions (standard or shifted)
    - Efficient Numba compilation for high-performance simulation
    
    The model equations are:
    
    .. math::
    
        \\tau_e \\frac{dE}{dt} = -E + (k_e - r_e E) \\cdot S_e(\\alpha_e(c_{ee}E - c_{ei}I + P - \\theta_e + g_e\\sum w_{ij}E_j)) + \\text{noise}
        
        \\tau_i \\frac{dI}{dt} = -I + (k_i - r_i I) \\cdot S_i(\\alpha_i(c_{ie}E - c_{ii}I + Q - \\theta_i + g_i\\sum w_{ij}I_j)) + \\text{noise}
    
    .. list-table:: Parameters
        :widths: 25 50 25
        :header-rows: 1

        * - Name
          - Explanation
          - Default Value
        * - `c_ee`
          - Excitatory to excitatory synaptic strength. If array-like, should be of length `nn`.
          - 16.0
        * - `c_ei`
          - Inhibitory to excitatory synaptic strength. If array-like, should be of length `nn`.
          - 12.0
        * - `c_ie`
          - Excitatory to inhibitory synaptic strength. If array-like, should be of length `nn`.
          - 15.0
        * - `c_ii`
          - Inhibitory to inhibitory synaptic strength. If array-like, should be of length `nn`.
          - 3.0
        * - `tau_e`
          - Time constant of excitatory population. If array-like, should be of length `nn`.
          - 8.0
        * - `tau_i`
          - Time constant of inhibitory population. If array-like, should be of length `nn`.
          - 8.0
        * - `a_e`
          - Sigmoid slope for excitatory population.
          - 1.3
        * - `a_i`
          - Sigmoid slope for inhibitory population.
          - 2.0
        * - `b_e`
          - Sigmoid threshold for excitatory population.
          - 4.0
        * - `b_i`
          - Sigmoid threshold for inhibitory population.
          - 3.7
        * - `c_e`
          - Maximum output of sigmoid for excitatory population.
          - 1.0
        * - `c_i`
          - Maximum output of sigmoid for inhibitory population.
          - 1.0
        * - `theta_e`
          - Firing threshold for excitatory population.
          - 0.0
        * - `theta_i`
          - Firing threshold for inhibitory population.
          - 0.0
        * - `r_e`
          - Refractoriness of excitatory population.
          - 1.0
        * - `r_i`
          - Refractoriness of inhibitory population.
          - 1.0
        * - `k_e`
          - Scaling constant for excitatory output.
          - 0.994
        * - `k_i`
          - Scaling constant for inhibitory output.
          - 0.999
        * - `alpha_e`
          - Gain of excitatory population.
          - 1.0
        * - `alpha_i`
          - Gain of inhibitory population.
          - 1.0
        * - `P`
          - External input to excitatory population. If array-like, should be of length `nn`.
          - 0.0
        * - `Q`
          - External input to inhibitory population. If array-like, should be of length `nn`.
          - 0.0
        * - `g_e`
          - Global coupling strength for excitatory populations.
          - 0.0
        * - `g_i`
          - Global coupling strength for inhibitory populations.
          - 0.0
        * - `weights`
          - Structural connectivity matrix of shape (`nn`, `nn`). Must be provided.
          - None
        * - `dt`
          - Integration time step.
          - 0.01
        * - `t_end`
          - End time of simulation.
          - 300.0
        * - `t_cut`
          - Time from which to start collecting output (burn-in period).
          - 0.0
        * - `noise_amp`
          - Amplitude of additive Gaussian noise.
          - 0.0
        * - `decimate`
          - Decimation factor for output time series (every `decimate`-th point is saved).
          - 1
        * - `RECORD_EI`
          - Which populations to record: "E" (excitatory only), "I" (inhibitory only), "EI" (both).
          - "E"
        * - `shift_sigmoid`
          - Whether to use shifted sigmoid function.
          - False
        * - `seed`
          - Random seed for reproducible simulations. If -1, no seeding is applied.
          - -1
        * - `initial_state`
          - Initial state vector of shape (2*nn,). If None, random initial conditions are generated.
          - None

    Usage example:
        >>> import numpy as np
        >>> from vbi.models.numba.wilson_cowan import WC_sde_numba
        >>> W = np.eye(2) * 0.1  # 2-node demo connectivity
        >>> P_ext = np.array([0.5, 0.8])  # External inputs to excitatory populations
        >>> wc = WC_sde_numba({
        ...     "weights": W, 
        ...     "P": P_ext,
        ...     "g_e": 0.2,
        ...     "dt": 0.01, 
        ...     "t_end": 100.0, 
        ...     "t_cut": 20.0,
        ...     "noise_amp": 0.01,
        ...     "RECORD_EI": "EI"
        ... })
        >>> result = wc.run()
        >>> t, E, I = result["t"], result["E"], result["I"]  # Time series data

    Notes
    -----
    The Wilson-Cowan model is widely used for understanding:
    - Oscillations and wave propagation in neural tissue
    - Pattern formation and spatial dynamics
    - Responses to external stimuli and perturbations
    - Brain dysfunction in neurological conditions (e.g., Parkinson's disease)
    - Local field potentials (LFPs) and EEG signal generation
    
    The model's nonlinear dynamics arise from the sigmoidal transfer functions
    that map synaptic input to firing rate, allowing for rich dynamical behaviors
    including fixed points, limit cycles, and complex spatiotemporal patterns.
    
    References
    ----------
    Wilson, H. R., & Cowan, J. D. (1972). Excitatory and inhibitory interactions 
    in localized populations of model neurons. Biophysical journal, 12(1), 1-24.
    """
    def __init__(self, par: dict = {}):
        """
        Initialize the Wilson-Cowan model.

        Parameters
        ----------
        par : dict, optional
            Dictionary containing model parameters. See class documentation for 
            available parameters. The 'weights' parameter is required.
        """
        # Prepare raw dict and build jitclass
        self.P = self._get_par_wc(par)

        # Seed
        if self.P.seed >= 0:
            np.random.seed(self.P.seed)

    def __call__(self):
        return self.P

    def __str__(self) -> str:
        """
        Return a string representation of the model parameters.
        
        Returns
        -------
        str
            Formatted string showing key model parameters and their values.
        """
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
        """
        Generate random initial conditions for the model.
        
        Sets random initial state for both excitatory and inhibitory populations
        with values appropriate for Wilson-Cowan dynamics.
        """
        self.P.initial_state = set_initial_state(self.P.nn, self.P.seed)
        
    
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
                print_valid_parameters(wc_spec)
                raise ValueError(f"Invalid parameter: {key}")

    def check_input(self):
        """
        Check shape and consistency of input parameters.
        
        Returns
        -------
        bool
            True if input is provided, False otherwise.
        """
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
        """
        Run the Wilson-Cowan model simulation.

        Executes the numerical integration to simulate
        the stochastic Wilson-Cowan dynamics.
        
        Parameters
        ----------
        par : dict, optional
            Dictionary of parameters to update before simulation
        x0 : array-like, optional
            Initial state vector of length 2*nn (E and I for each region)
        verbose : bool, default True
            Whether to print simulation progress information
            
        Returns
        -------
        tuple
            Dictionary with keys 't', 'E', 'I' containing time vector and 
            excitatory/inhibitory time series arrays.
        """
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
    Run Wilson-Cowan model integration with Heun's method.
    
    Performs stochastic integration of the Wilson-Cowan equations using 
    Heun's method for improved accuracy. Returns time series data for
    excitatory and inhibitory populations.
    
    Parameters
    ----------
    P : ParWC
        Parameter object containing all model parameters and settings
    verbose : bool, default True
        Whether to print integration progress
        
    Returns
    -------
    dict
        Dictionary containing:
        - 't' : ndarray, time points
        - 'E' : ndarray or None, excitatory population activity
        - 'I' : ndarray or None, inhibitory population activity
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