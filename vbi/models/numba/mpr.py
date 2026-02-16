import warnings
import numpy as np
from copy import copy
from typing import Dict, Any, List
from numba import njit, jit
from numba.experimental import jitclass
from numba.extending import register_jitable
from numba import float64, boolean, int64, types
from numba.core.errors import NumbaPerformanceWarning
from vbi.utils import print_valid_parameters
from vbi.models.numba.base import BaseNumbaModel

warnings.simplefilter("ignore", category=NumbaPerformanceWarning)
np.random.seed(42)


# ---------- Default Parameters - Single Source of Truth ----------

# Default parameter values for Montbrió-Pazó-Roxin model
MPR_DEFAULTS = {
    'G': 0.5,
    'dt': 0.01,
    'J': 14.5,
    'eta': -4.6,
    'tau': 1.0,
    'delta': 0.7,
    'rv_decimate': 1,
    'noise_amp': 0.037,
    'weights': None,  # Required - must be provided by user
    'initial_state': None,  # Optional - if None, will be set to small random values
    't_init': 0.0,
    't_cut': 0.0,
    't_end': 1000.0,
    'iapp': 0.0,
    'seed': -1,
    'output': 'output',
    'RECORD_RV': True,
    'RECORD_BOLD': True,
    'tr': 500.0,
}


@njit
def f_mpr(x, t, P):
    """
    Compute the right-hand side of the Montbrió neural mass model.
    
    This function implements the deterministic part of the Montbrió model equations
    for a network of coupled neural populations. The model describes the macroscopic
    dynamics of quadratic integrate-and-fire (QIF) neuron populations in terms of
    population firing rate (r) and mean membrane potential (v).
    
    The equations are:
    τ dr/dt = Δ/(τπ) + 2rv
    τ dv/dt = v² + η + I_app + Jr - (πτr)² + G·coupling
    
    Parameters
    ----------
    x : np.ndarray
        Current state vector of shape (2*nn,) containing stacked arrays:
        [r₀, r₁, ..., rₙ, v₀, v₁, ..., vₙ] where nn is the number of nodes.
        r represents the population firing rate and v represents the mean membrane potential.
    t : float
        Current time (not used in autonomous system).
    P : ParMPR
        Parameter object containing all model parameters.
        
    Returns
    -------
    np.ndarray
        Derivative vector of shape (2*nn,) containing dx/dt.
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

    coupling = np.dot(np.ascontiguousarray(P.weights), np.ascontiguousarray(x0))
    dxdt[:nn] = rtau * (delta_over_tau_pi + 2 * x0 * x1)
    dxdt[nn:] = rtau * (
        x1 * x1 + P.eta + P.iapp + J_tau * x0 - (pi2 * tau2 * x0 * x0) + P.G * coupling
    )
    return dxdt


@njit
def heun_sde(x, t, P):
    """
    Perform one step of Heun's method for stochastic differential equations.
    
    This implements the Heun scheme for integrating the Montbrió model with 
    additive noise applied to both firing rate and membrane potential variables.
    
    Parameters
    ----------
    x : np.ndarray
        Current state vector of shape (2*nn,).
    t : float
        Current time.
    P : ParMPR
        Parameter object containing dt, sigma_r, sigma_v and other parameters.
        
    Returns
    -------
    np.ndarray
        Updated state vector after one integration step.
    """
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
    """
    Perform one step of BOLD signal computation using the Balloon-Windkessel model.
    
    This function implements the hemodynamic response model that converts neural
    activity (firing rate) into BOLD signal through a cascade of physiological
    processes including blood flow, volume, and oxygenation changes.
    
    Parameters
    ----------
    r_in : float
        Input neural activity (firing rate).
    s, f, ftilde, vtilde, qtilde, v, q : np.ndarray
        BOLD state variables (flow, volume, deoxygenation).
    dtt : float
        Time step for BOLD integration.
    P : ParBold
        BOLD parameter object.
    """
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
    """
    Integrate the Montbrió model over time with BOLD signal computation.
    
    This function performs the main time integration loop for the Montbrió model,
    using the Heun stochastic integration scheme. It optionally computes both
    neural activity (r, v) and hemodynamic BOLD signals.
    
    Parameters
    ----------
    P : ParMPR
        Parameter object containing all simulation parameters.
    B : ParBold
        BOLD parameter object for hemodynamic response computation.
        
    Returns
    -------
    dict
        Dictionary with keys:
        - 'rv_t': np.ndarray of time points for neural activity
        - 'rv_d': np.ndarray of shape (n_steps, 2*nn) containing [r, v] time series
        - 'bold_t': np.ndarray of time points for BOLD signal
        - 'bold_d': np.ndarray of shape (n_steps, nn) containing BOLD time series
    """

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


class MPR_sde(BaseNumbaModel):
    """
    Numba implementation of the Montbrió neural mass model with BOLD signal generation.
    
    This model implements the exact macroscopic dynamics derived from infinitely 
    all-to-all coupled quadratic integrate-and-fire (QIF) neurons in the thermodynamic 
    limit. The model describes the collective firing activity (r) and mean membrane 
    potential (v) of neural populations, providing a rigorous mathematical foundation 
    for whole-brain network modeling.
    
    The neural dynamics follow the Montbrió equations:
    τ dr/dt = 2rv + Δ/(πτ)
    τ dv/dt = v² - (πτr)² + Jτr + η + G·coupling + I_stim + noise
    
    Where:
    - r is the population firing rate
    - v is the mean membrane potential  
    - τ is the characteristic time constant
    - J is the synaptic weight
    - Δ is the spread of heterogeneous excitabilities
    - η is the mean excitability
    - G is the global coupling strength
    
    The model can operate in bistable regime, exhibiting down-state (low firing) 
    and up-state (high firing) attractors, which is fundamental for realistic 
    brain dynamics and functional connectivity patterns.
    
    .. list-table:: Parameters
        :widths: 25 50 25
        :header-rows: 1

        * - Name
          - Explanation
          - Default Value
        * - `G`
          - Global coupling strength scaling network connections.
          - 0.5
        * - `dt`
          - Integration time step in milliseconds.
          - 0.01
        * - `J`
          - Synaptic weight strength in ms⁻¹.
          - 14.5
        * - `eta`
          - Mean excitability parameter. If array-like, it should be of length `nn`.
          - np.array([-4.6])
        * - `tau`
          - Characteristic time constant in ms.
          - 1.0
        * - `delta`
          - Spread of heterogeneous excitability distribution in ms⁻¹.
          - 0.7
        * - `rv_decimate`
          - Decimation factor for neural activity recording.
          - 1.0
        * - `noise_amp`
          - Amplitude of additive Gaussian noise.
          - 0.037
        * - `weights`
          - Structural connectivity matrix of shape (`nn`, `nn`). Must be provided.
          - np.array([[], []])
        * - `t_init`
          - Initial time of simulation.
          - 0.0
        * - `t_cut`
          - Time from which to start collecting output (burn-in period).
          - 0.0
        * - `t_end`
          - End time of simulation in milliseconds.
          - 1000.0
        * - `iapp`
          - External applied current (stimulation).
          - 0.0
        * - `seed`
          - Random seed for reproducible simulations. If -1, no seeding is applied.
          - -1
        * - `output`
          - Output directory name.
          - "output"
        * - `RECORD_RV`
          - Whether to record neural activity (r, v) time series.
          - True
        * - `RECORD_BOLD`
          - Whether to record BOLD signal time series.
          - True
        * - `tr`
          - BOLD repetition time in milliseconds.
          - 500.0

    Usage example:
        >>> import numpy as np
        >>> from vbi.models.numba.mpr import MPR_sde
        >>> W = np.eye(2) * 0.1  # 2-node demo connectivity
        >>> eta = np.array([-4.6, -4.5])  # Excitability parameters
        >>> mpr = MPR_sde({
        ...     "weights": W, 
        ...     "eta": eta,
        ...     "G": 0.5,
        ...     "dt": 0.01, 
        ...     "t_end": 1000.0, 
        ...     "t_cut": 200.0,
        ...     "J": 14.5,
        ...     "tau": 1.0,
        ...     "delta": 0.7
        ... })
        >>> result = mpr.run()
        >>> rv_t, rv_d = result["rv_t"], result["rv_d"]  # Neural activity
        >>> bold_t, bold_d = result["bold_t"], result["bold_d"]  # BOLD signals

    Notes
    -----
    The Montbrió model provides an exact mathematical description derived from 
    microscopic spiking neuron networks, making it particularly suitable for:
    - Whole-brain network modeling with rigorous theoretical foundation
    - Studying bistable dynamics and metastable states
    - Investigating the relationship between neural activity and BOLD signals
    - Parameter estimation and model validation against empirical data
    
    The model includes sophisticated BOLD signal generation using the 
    Balloon-Windkessel hemodynamic response model, enabling direct comparison 
    with fMRI measurements.
    
    References
    ----------
    Montbrió, E., et al. (2015). Macroscopic description for networks of spiking 
    neurons. Physical Review X, 5(2), 021028.
    """
    def __init__(self, par_mpr: dict = {}) -> None:
        """
        Initialize the Montbrió model.

        Parameters
        ----------
        par_mpr : dict, optional
            Dictionary containing model parameters. See class documentation for 
            available parameters. The 'weights' parameter is required for proper 
            functionality.
        """
        super().__init__()
        
        # Define valid parameters from MPR_DEFAULTS, excluding derived parameters
        # 'nn' is derived from weights.shape, 'sigma_r'/'sigma_v' derived from noise_amp
        self.valid_params = [k for k in MPR_DEFAULTS.keys() if k not in ['nn', 'sigma_r', 'sigma_v']]
        
        self.check_parameters(par_mpr)
        self.P = self.get_par_mpr(par_mpr)
        self.B = ParBold()

        self.seed = self.P.seed
        if self.seed > 0:
            np.random.seed(self.seed)
    
    def get_default_parameters(self) -> Dict[str, Any]:
        """
        Get the default parameters for the Montbrió model.
        
        Returns a copy of the MPR_DEFAULTS dictionary with additional
        required/derived parameters.
        
        Returns
        -------
        dict
            Dictionary containing default parameter values.
            
        Examples
        --------
        >>> from vbi.models.numba.mpr import MPR_sde
        >>> model = MPR_sde({"weights": np.eye(2)})
        >>> defaults = model.get_default_parameters()
        >>> print(defaults['G'])
        0.5
        """
        defaults = MPR_DEFAULTS.copy()
        # Add required/derived parameters with None values
        defaults['weights'] = None  # Required parameter (must be provided by user)
        defaults['initial_state'] = None  # Optional (auto-generated if not provided)
        defaults['nn'] = None  # Derived from weights.shape
        defaults['sigma_r'] = None  # Derived from sqrt(dt) * sqrt(2 * noise_amp)
        defaults['sigma_v'] = None  # Derived from sqrt(dt) * sqrt(4 * noise_amp)
        return defaults
    
    def get_parameter_descriptions(self) -> Dict[str, tuple]:
        """
        Get descriptions of all Montbrió model parameters.
        
        Returns
        -------
        dict
            Dictionary mapping parameter names to tuples of (description, type).
            
        Examples
        --------
        >>> model = MPR_sde({"weights": np.eye(2)})
        >>> descriptions = model.get_parameter_descriptions()
        >>> print(descriptions['G'])
        ('Global coupling strength', 'scalar')
        """
        return {
            'G': ('Global coupling strength scaling network connections', 'scalar'),
            'dt': ('Integration time step (ms)', 'scalar'),
            'J': ('Synaptic weight strength (ms⁻¹)', 'scalar'),
            'eta': ('Mean excitability parameter', 'vector'),
            'tau': ('Characteristic time constant (ms)', 'scalar'),
            'delta': ('Spread of heterogeneous excitability distribution (ms⁻¹)', 'scalar'),
            'rv_decimate': ('Decimation factor for neural activity recording', 'int'),
            'noise_amp': ('Amplitude of additive Gaussian noise', 'scalar'),
            'weights': ('Structural connectivity matrix (nn x nn)', 'matrix'),
            't_init': ('Initial time of simulation (ms)', 'scalar'),
            't_cut': ('Time from which to start collecting output (ms)', 'scalar'),
            't_end': ('End time of simulation (ms)', 'scalar'),
            'iapp': ('External applied current (stimulation)', 'scalar'),
            'seed': ('Random seed for reproducibility (-1 for no seeding)', 'int'),
            'output': ('Output directory name', 'string'),
            'RECORD_RV': ('Whether to record neural activity (r, v) time series', 'bool'),
            'RECORD_BOLD': ('Whether to record BOLD signal time series', 'bool'),
            'tr': ('BOLD repetition time (ms)', 'scalar'),
            'nn': ('[Derived] Number of nodes (from weights.shape)', 'int'),
            'initial_state': ('[Derived] Initial state vector (2*nn)', 'vector'),
            'sigma_r': ('[Derived] Noise std for firing rate (from noise_amp)', 'scalar'),
            'sigma_v': ('[Derived] Noise std for membrane potential (from noise_amp)', 'scalar'),
        }
    
    def list_parameters(self) -> List[str]:
        """
        List all valid parameter names.
        
        Returns
        -------
        list of str
            List of valid parameter names.
        """
        return self.valid_params.copy()

    def __str__(self) -> str:
        """
        Return a string representation of the model parameters.
        
        Returns
        -------
        str
            Formatted string showing all model parameters and their values.
        """
        return self._format_parameters_table("MPR")

    def _par_to_dict(self):
        """
        Convert parameter jitclass to dictionary.
        
        Returns
        -------
        dict
            Dictionary containing all parameter values.
        """
        P = self.P
        d = {
            "G": P.G,
            "dt": P.dt,
            "J": P.J,
            "eta": np.array(P.eta),
            "tau": P.tau,
            "delta": P.delta,
            "rv_decimate": P.rv_decimate,
            "noise_amp": P.noise_amp,
            "t_init": P.t_init,
            "t_cut": P.t_cut,
            "t_end": P.t_end,
            "iapp": P.iapp,
            "seed": P.seed,
            "output": P.output,
            "weights": np.array(P.weights),
            "RECORD_RV": P.RECORD_RV,
            "RECORD_BOLD": P.RECORD_BOLD,
            "tr": P.tr,
            "initial_state": np.array(P.initial_state),
        }
        return d

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
            if key not in self.valid_params:
                print(f"Invalid parameter: {key}")
                print(f"Valid parameters: {self.valid_params}")
                raise ValueError(f"Invalid parameter: {key}")

    def get_par_mpr(self, par: dict):
        """
        Create parameter object from dictionary with validation.
        
        Parameters
        ----------
        par : dict
            Parameter dictionary.
            
        Returns
        -------
        ParMPR
            Numba jitclass parameter object with validated parameters.
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
        """
        Generate random initial conditions for the model.
        
        Sets the initial state for both firing rate (r) and membrane potential (v)
        variables using random values appropriate for the Montbrió model dynamics.
        """
        self.initial_state = set_initial_state(self.P.nn, self.seed)
        self.INITIAL_STATE_SET = True

    def check_input(self):
        """
        Validate model parameters before simulation.
        
        Raises
        ------
        AssertionError
            If weights matrix is None, not square, if initial_state length doesn't 
            match 2*nn, or if t_cut >= t_end.
        """
        assert self.P.weights is not None
        assert self.P.weights.shape[0] == self.P.weights.shape[1]
        assert self.P.initial_state is not None
        assert len(self.P.initial_state) == 2 * self.P.weights.shape[0]
        assert self.P.t_cut < self.P.t_end, "t_cut must be less than t_end"
        self.P.eta = check_vec_size(self.P.eta, self.P.nn)
        self.P.t_end /= 10
        self.P.t_cut /= 10

    def run(self, par={}, x0=None, verbose=True):
        """
        Run the Montbrió simulation.

        Parameters
        ----------
        par : dict, optional
            Dictionary of parameters to update for this simulation run.
            Any parameter from the class documentation can be updated.
        x0 : np.ndarray, optional
            Initial state vector of shape (2*nn,) containing [r₀, v₀]. 
            If None, random initial conditions are generated.
        verbose : bool, optional
            Whether to print verbose output (currently unused).

        Returns
        -------
        dict
            Dictionary containing simulation results:
            
            - 'rv_t': np.ndarray of time points for neural activity (in ms)
            - 'rv_d': np.ndarray of shape (n_steps, 2*nn) containing 
              [r, v] time series (firing rate and membrane potential)
            - 'bold_t': np.ndarray of time points for BOLD signal (in ms)
            - 'bold_d': np.ndarray of shape (n_steps, nn) containing 
              simulated BOLD signals for each brain region
        """

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
    """
    Generate random initial state for the Montbrió model.
    
    Creates initial conditions with firing rates in [0, 1.5] and 
    membrane potentials in [-2, 2], appropriate for the model dynamics.
    
    Parameters
    ----------
    nn : int
        Number of nodes/brain regions.
    seed : int, optional
        Random seed for reproducible initial conditions.
        
    Returns
    -------
    np.ndarray
        Initial state vector of shape (2*nn,) with [r₀, v₀] values.
    """

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
    """
    Numba jitclass container for Montbrió model parameters.
    
    This class holds all parameters needed for the Montbrió neural mass model
    in a format optimized for Numba compilation. It stores both scalar parameters
    and array parameters like connectivity weights and excitability values.
    
    Note: This is an internal class used by the MPR_sde class. Users should
    not instantiate this class directly.
    
    Default values are defined in MPR_DEFAULTS dictionary.
    """
    def __init__(
        self,
        G=MPR_DEFAULTS['G'],
        dt=MPR_DEFAULTS['dt'],
        J=MPR_DEFAULTS['J'],
        eta=np.array([MPR_DEFAULTS['eta']]),
        tau=MPR_DEFAULTS['tau'],
        delta=MPR_DEFAULTS['delta'],
        rv_decimate=MPR_DEFAULTS['rv_decimate'],
        noise_amp=MPR_DEFAULTS['noise_amp'],
        weights=np.array([[], []]),
        t_init=MPR_DEFAULTS['t_init'],
        t_cut=MPR_DEFAULTS['t_cut'],
        t_end=MPR_DEFAULTS['t_end'],
        iapp=MPR_DEFAULTS['iapp'],
        seed=MPR_DEFAULTS['seed'],
        output=MPR_DEFAULTS['output'],
        RECORD_RV=MPR_DEFAULTS['RECORD_RV'],
        RECORD_BOLD=MPR_DEFAULTS['RECORD_BOLD'],
        tr=MPR_DEFAULTS['tr'],
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
    """
    Numba jitclass container for BOLD hemodynamic model parameters.
    
    This class holds parameters for the Balloon-Windkessel model used to 
    convert neural activity into simulated BOLD signals. Based on the 
    Friston 2003 hemodynamic response model.
    
    Note: This is an internal class used by the MPR_sde class. Users should
    not instantiate this class directly.
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


def check_vec_size(x, nn):
    """
    Ensure parameter array has correct size for the number of nodes.
    
    Parameters
    ----------
    x : array-like
        Parameter array to check/broadcast.
    nn : int
        Required number of nodes.
        
    Returns
    -------
    np.ndarray
        Array of length nn, either the original array or broadcasted scalar.
    """
    return np.ones(nn) * x if len(x) != nn else np.array(x)


@register_jitable
def set_seed_compat(x):
    """Numba-compatible random seed setter."""
    np.random.seed(x)


# Alias for consistency with naming convention
MPR_sde_numba = MPR_sde
