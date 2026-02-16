import numpy as np
from copy import copy
from numba import njit
from numba.experimental import jitclass
from numba import float64, int64, boolean, types
from numba.extending import register_jitable
from vbi.models.numba.utils import check_parameters
from vbi.models.numba.base import BaseNumbaModel
# ----------------------------------------
# Stuart-Landau model spec
# ----------------------------------------

sl_spec = [
    ("a", float64),
    ("omega", float64),
    ("G", float64),
    ("sigma", float64),
    ("dt", float64),
    ("t_end", float64),
    ("t_cut", float64),
    ("nn", int64),
    ("seed", int64),
    ("speed", float64),
    ("weights", float64[:, :]),
    ("tr_len", float64[:, :]),
    ("initial_state", types.complex128[:]),
    ("RECORD_X", boolean),
    ("x_decimate", int64),
]

# Default parameter values - single source of truth
SL_DEFAULTS = {
    'a': 0.1,
    'omega': 2.0 * np.pi * 0.040,  # rad/ms (40 Hz oscillation)
    'G': 0.0,
    'sigma': 0.01,
    'dt': 0.1,
    't_end': 1000.0,
    't_cut': 0.0,
    'seed': -1,
    'speed': 5.0,
    'weights': None,  # Required - must be provided by user
    'tr_len': None,  # Optional - defaults to zeros if not provided
    'initial_state': None,  # Optional - auto-generated if not provided
    'RECORD_X': True,
    'x_decimate': 1,
}


@jitclass(sl_spec)
class ParSL:
    """
    Numba jitclass container for Stuart-Landau model parameters.
    
    This class holds all parameters needed for the Stuart-Landau oscillator model
    in a format optimized for Numba compilation. It stores both scalar parameters
    and array parameters like connectivity weights and transmission delays.
    
    Note: This is an internal class used by the SL_sde class. Users should
    not instantiate this class directly.
    
    Default values are defined in SL_DEFAULTS dictionary.
    """
    def __init__(
        self,
        a=SL_DEFAULTS['a'],
        omega=SL_DEFAULTS['omega'],
        G=SL_DEFAULTS['G'],
        sigma=SL_DEFAULTS['sigma'],
        dt=SL_DEFAULTS['dt'],
        t_end=SL_DEFAULTS['t_end'],
        t_cut=SL_DEFAULTS['t_cut'],
        nn=1,  # Will be set from weights.shape
        seed=SL_DEFAULTS['seed'],
        speed=SL_DEFAULTS['speed'],
        weights=np.array([[0.0]]),
        tr_len=np.array([[0.0]]),
        initial_state=np.array([0.01 + 0.01j]),
        RECORD_X=SL_DEFAULTS['RECORD_X'],
        x_decimate=SL_DEFAULTS['x_decimate'],
    ):
        self.a = a
        self.omega = omega
        self.G = G
        self.sigma = sigma
        self.dt = dt
        self.t_end = t_end
        self.t_cut = t_cut
        self.nn = nn
        self.seed = seed
        self.speed = speed
        self.weights = weights
        self.tr_len = tr_len
        self.initial_state = initial_state
        self.RECORD_X = RECORD_X
        self.x_decimate = x_decimate


# --------------------------------
# Stuart-Landau dynamics
# --------------------------------
@njit
def f_sl(Z, delays_state, W, a, omega, G):
    """
    Compute the Stuart-Landau equations with delayed coupling.
    
    dz_i/dt = z_i * (a + i*omega - |z_i|^2) + G * sum_j W_ij * (z_j_delayed - z_i)
    
    Parameters
    ----------
    Z : complex array
        Current state of all nodes (nn,)
    delays_state : complex array
        Delayed states for each connection (nn, nn)
    W : float array
        Connectivity weights (nn, nn)
    a : float
        Bifurcation parameter (dimensionless)
    omega : float
        Angular frequency (rad/ms). For f Hz: omega = 2*pi*f/1000
    G : float
        Global coupling strength (dimensionless)
        
    Returns
    -------
    dzdt : complex array
        Time derivatives for each node
    """
    nn = Z.shape[0]
    dzdt = np.zeros_like(Z)
    
    # Precompute complex constant for local dynamics
    a_omega = a + 1j * omega
    
    for i in range(nn):
        z_i = Z[i]
        # Local dynamics: z_i * (a + i*omega - |z_i|^2)
        z_norm_sq = (z_i * np.conj(z_i)).real
        local = z_i * (a_omega - z_norm_sq)
        
        # Coupling term: sum_j W_ij * (z_j_delayed - z_i)
        coupling = 0.0j
        for j in range(nn):
            coupling += W[i, j] * (delays_state[i, j] - z_i)
        
        dzdt[i] = local + G * coupling
    
    return dzdt


# --------------------------------
# Heun SDE integrator for complex variables
# --------------------------------
@njit
def heun_sde_sl(Z, delays_state, W, P):
    """
    Heun method for the stochastic Stuart-Landau model with delays.
    
    Parameters
    ----------
    Z : complex array
        Current state (nn,)
    delays_state : complex array
        Delayed states (nn, nn)
    W : float array
        Weights (nn, nn)
    P : ParSL
        Model parameters
        
    Returns
    -------
    Z_new : complex array
        Updated state after one timestep
    """
    dt = P.dt
    nn = P.nn
    
    # Stochastic increments (separate real and imaginary parts)
    dW_real = P.sigma * np.sqrt(dt) * np.random.randn(nn)
    dW_imag = P.sigma * np.sqrt(dt) * np.random.randn(nn)
    dW = dW_real + 1j * dW_imag
    
    # First stage
    k1 = f_sl(Z, delays_state, W, P.a, P.omega, P.G)
    
    # Prediction
    Z_pred = Z + dt * k1 + dW
    
    # Delayed states stay the same for the prediction step
    k2 = f_sl(Z_pred, delays_state, W, P.a, P.omega, P.G)
    
    # Corrector
    Z_new = Z + 0.5 * dt * (k1 + k2) + dW
    
    return Z_new


# --------------------------------
# Public API class
# --------------------------------
class SL_sde(BaseNumbaModel):
    """
    Numba implementation of the Stuart-Landau oscillator model with delayed coupling.

    This model implements a system of coupled Stuart-Landau oscillators, which are
    complex-valued oscillators that can exhibit limit cycle behavior. The model is
    particularly suitable for simulating whole-brain neural activity with distance-
    dependent conduction delays, providing a simplified yet mathematically tractable
    framework for studying collective neural dynamics.

    The neural dynamics follow the Stuart-Landau equations:
    dz_i/dt = z_i * (a + i*ω - |z_i|²) + G * Σ_j W_ij * (z_j_delayed - z_i)

    Where:
    - z_i is the complex oscillator state for node i
    - a is the bifurcation parameter (controls oscillatory behavior)
    - ω is the angular frequency (rad/ms)
    - G is the global coupling strength
    - W_ij are the structural connectivity weights
    - z_j_delayed is the delayed state from node j

    The model supports both deterministic and stochastic dynamics, with optional
    distance-dependent delays that can be specified as either uniform (scalar) or
    pairwise (matrix) values.

    .. list-table:: Parameters
        :widths: 25 50 25
        :header-rows: 1

        * - Name
          - Explanation
          - Default Value
        * - `a`
          - Bifurcation parameter controlling oscillatory behavior. Values > 0 produce limit cycles.
          - 0.1
        * - `omega`
          - Angular frequency in rad/ms. For f Hz: omega = 2*pi*f/1000. Controls oscillation frequency.
          - 2.0 * π * 0.040 (40 Hz)
        * - `G`
          - Global coupling strength scaling all network connections.
          - 0.0
        * - `sigma`
          - Noise amplitude for stochastic dynamics. 0.0 = deterministic, >0.0 = stochastic.
          - 0.01
        * - `dt`
          - Integration time step in milliseconds. Must be small for numerical stability.
          - 0.1
        * - `t_end`
          - End time of simulation in milliseconds.
          - 1000.0
        * - `t_cut`
          - Time from which to start collecting output (burn-in period) in milliseconds.
          - 0.0
        * - `nn`
          - Number of oscillators/nodes. Inferred from weights matrix if not provided.
          - 1
        * - `seed`
          - Random seed for reproducible simulations. If -1, no seeding is applied.
          - -1
        * - `speed`
          - Conduction velocity in mm/ms. Used to convert tr_len to delay steps.
          - 5.0
        * - `weights`
          - Structural connectivity matrix of shape (nn, nn). Must be provided.
          - np.array([[0.0]])
        * - `tr_len`
          - Transmission length/distance in mm. Can be scalar (uniform) or (nn, nn) matrix.
          - np.array([[0.0]])
        * - `initial_state`
          - Initial complex state vector of shape (nn,). If None, random initialization is used.
          - np.array([0.01+0.01j])
        * - `RECORD_X`
          - Whether to record complex neural activity time series.
          - True
        * - `x_decimate`
          - Decimation factor for neural activity recording (record every x_decimate steps).
          - 1

    Usage example:
        >>> import numpy as np
        >>> from vbi.models.numba.sl import SL_sde
        >>> W = np.eye(2) * 0.1  # 2-node demo connectivity
        >>> sl = SL_sde({
        ...     "weights": W,
        ...     "a": 0.20,
        ...     "omega": 2*np.pi*0.040,  # 40 Hz
        ...     "G": 0.5,
        ...     "dt": 0.01,
        ...     "t_end": 1000.0,
        ...     "t_cut": 200.0
        ... })
        >>> result = sl.run()
        >>> X, t = result["X"], result["t"]  # Complex neural activity

    Notes
    -----
    The Stuart-Landau model provides a mathematically elegant framework for studying:
    - Collective synchronization in neural networks
    - The effects of network topology on oscillatory dynamics
    - Phase transitions and bifurcation phenomena
    - The impact of conduction delays on brain dynamics

    The model exhibits rich dynamical behavior:
    - For a < 0: Stable fixed point (no oscillations)
    - For a > 0: Limit cycle oscillations with amplitude sqrt(a)
    - Coupling can induce synchronization, clustering, or more complex patterns

    The complex representation z = x + i*y allows natural representation of:
    - Amplitude: |z| = sqrt(x² + y²)
    - Phase: arg(z) = atan2(y, x)
    - Frequency: d(arg(z))/dt

    References
    ----------
    Selivanov AA, Lehnert J, Dahms T, Hövel P, Fradkov AL, Schöll E. 2012. Adaptive synchronization in delay-coupled networks of Stuart-Landau oscillators. Physical Review E 85:016201. `DOI: 10.1103/PhysRevE.85.016201 <https://doi.org/10.1103/PhysRevE.85.016201>`_

    .. list-table:: State Variables
        :widths: 25 50 25
        :header-rows: 1

        * - Variable
          - Description
          - Units
        * - `z_i`
          - Complex oscillator state for node i: z_i = x_i + i*y_i
          - dimensionless
        * - `|z_i|`
          - Oscillator amplitude (neural activity strength)
          - dimensionless
        * - `arg(z_i)`
          - Oscillator phase (neural synchronization)
          - radians
        * - `d(arg(z_i))/dt`
          - Instantaneous frequency
          - rad/ms

    """
    
    def __init__(self, par: dict = {}) -> None:
        
        self.valid_par = [sl_spec[i][0] for i in range(len(sl_spec))]
        check_parameters(par, self.valid_par, sl_spec)
        
        # Handle connectivity matrix
        nn = par.get("nn", None)
        weights = par.get("weights", None)
        if weights is None:
            weights = np.zeros((1, 1), dtype=np.float64)
        weights = np.array(weights, dtype=np.float64)
        if nn is None:
            nn = weights.shape[0]
        par.setdefault("nn", nn)
        
        # Handle tr_len: can be scalar or (nn, nn) matrix
        tr_len = par.get("tr_len", None)
        if tr_len is None:
            tr_len = np.zeros((nn, nn), dtype=np.float64)
        else:
            tr_len = np.array(tr_len, dtype=np.float64)
            if tr_len.ndim == 0:  # Scalar
                tr_len = np.ones((nn, nn), dtype=np.float64) * float(tr_len)
            elif tr_len.shape != (nn, nn):
                raise ValueError(f"tr_len must be scalar or shape ({nn}, {nn}), got {tr_len.shape}")
        par["tr_len"] = tr_len
        
        # Handle random seed
        seed = par.get("seed", -1)
        if seed > 0:
            set_seed_compat(seed)
        
        # Set initial state
        if "initial_state" in par:
            par["initial_state"] = np.array(
                par["initial_state"], dtype=np.complex128
            )
        else:
            par["initial_state"] = set_initial_state_sl(nn, seed=seed)
        
        # Initialize SL parameters
        self.P = ParSL()
        for k, v in par.items():
            if k == "weights":
                v = np.array(v, dtype=np.float64)
            setattr(self.P, k, v)
        
        # Set valid_params for BaseNumbaModel (excluding 'nn' which is derived)
        self.valid_params = [k for k in SL_DEFAULTS.keys() if k != 'nn']
    
    def get_default_parameters(self):
        """
        Get the default parameters for the Stuart-Landau model.
        
        Returns a copy of the SL_DEFAULTS dictionary with additional
        required/derived parameters.
        
        Returns
        -------
        dict
            Dictionary containing default parameter values.
            
        Examples
        --------
        >>> from vbi.models.numba.sl import SL_sde
        >>> import numpy as np
        >>> model = SL_sde({"weights": np.eye(2)})
        >>> defaults = model.get_default_parameters()
        >>> print(defaults['a'])
        0.1
        """
        defaults = SL_DEFAULTS.copy()
        defaults['nn'] = None  # Derived from weights.shape
        return defaults
    
    def get_parameter_descriptions(self):
        """
        Get descriptions for all Stuart-Landau model parameters.
        
        Returns
        -------
        dict
            Dictionary mapping parameter names to tuples of (description, type).
            
        Examples
        --------
        >>> model = SL_sde({"weights": np.eye(2)})
        >>> descriptions = model.get_parameter_descriptions()
        >>> print(descriptions['a'])
        ('Bifurcation parameter', 'scalar')
        """
        return {
            'a': ('Bifurcation parameter controlling oscillatory behavior', 'scalar'),
            'omega': ('Angular frequency (rad/ms)', 'scalar'),
            'G': ('Global coupling strength', 'scalar'),
            'sigma': ('Noise amplitude for stochastic dynamics', 'scalar'),
            'dt': ('Integration time step (ms)', 'scalar'),
            't_end': ('Simulation end time (ms)', 'scalar'),
            't_cut': ('Burn-in period (ms)', 'scalar'),
            'nn': ('[Derived] Number of oscillators/nodes', 'int'),
            'seed': ('Random seed (-1 = no seed)', 'int'),
            'speed': ('Conduction velocity (mm/ms)', 'scalar'),
            'weights': ('Structural connectivity matrix', 'matrix'),
            'tr_len': ('Transmission distances (mm)', 'matrix'),
            'initial_state': ('[Derived] Initial complex states', 'vector'),
            'RECORD_X': ('Record activity time series', 'bool'),
            'x_decimate': ('Recording decimation factor', 'int'),
        }
    
    def run(self, par: dict = {}, x0=None):
        """
        Run the Stuart-Landau simulation.
        
        Parameters
        ----------
        par : dict, optional
            Runtime parameter overrides
        x0 : array-like, optional
            Initial state. If None, uses self.P.initial_state
            
        Returns
        -------
        dict
            Dictionary with keys:
            - 'X': complex neural activity (n_timepoints, nn)
            - 't': time points (n_timepoints,)
        """
        
        # Set initial condition
        if x0 is None:
            Z = copy(self.P.initial_state)
        else:
            Z = np.array(x0, dtype=np.complex128)
        
        # Reset RNG if seed provided
        if self.P.seed > 0:
            set_seed_compat(self.P.seed)
            reset_numba_rng(self.P.seed)
        
        # Check and apply parameter overrides
        check_parameters(par, self.valid_par, sl_spec)
        if par:
            for k, v in par.items():
                if k == "weights":
                    v = np.array(v, dtype=np.float64)
                setattr(self.P, k, v)
        
        # Sanity checks
        assert self.P.weights is not None
        assert self.P.weights.shape[0] == self.P.weights.shape[1]
        assert len(Z) == self.P.nn, "x0 must be length nn"
        assert self.P.t_cut < self.P.t_end
        
        # Time grid
        nt = int(np.floor(self.P.t_end / self.P.dt))
        t = np.arange(nt) * self.P.dt
        valid_mask = t > self.P.t_cut
        x_buffer_len = int(np.sum(valid_mask) // max(1, self.P.x_decimate))
        
        # Compute delays (in steps)
        delays = np.zeros((self.P.nn, self.P.nn), dtype=np.int64)
        for i in range(self.P.nn):
            for j in range(self.P.nn):
                if self.P.weights[i, j] > 0:
                    delay_steps = int(np.floor(self.P.tr_len[i, j] / self.P.speed / self.P.dt))
                    # Allow zero delay! Don't force minimum to 1
                    delays[i, j] = np.clip(delay_steps, 0, nt - 1)
                else:
                    delays[i, j] = 0
        
        maxdelay = int(np.max(delays))
        
        # Initialize state history for delays - fill with initial state
        state_history = np.zeros((self.P.nn, maxdelay + 1), dtype=np.complex128)
        for i in range(maxdelay + 1):
            state_history[:, i] = Z
        
        # Output buffers
        t_buf = np.zeros((x_buffer_len,), dtype=np.float32)
        X_rec = (
            np.zeros((x_buffer_len, self.P.nn), dtype=np.complex128)
            if self.P.RECORD_X
            else np.array([], dtype=np.complex128)
        )
        
        # Main integration loop
        x_idx = 0
        for it in range(nt):
            t_curr = it * self.P.dt
            
            # Get delayed states for all connections
            delays_state = np.zeros((self.P.nn, self.P.nn), dtype=np.complex128)
            for i in range(self.P.nn):
                for j in range(self.P.nn):
                    delay_idx = min(delays[i, j], maxdelay)
                    delays_state[i, j] = state_history[j, -(delay_idx + 1)]
            
            # Integration step
            Z = heun_sde_sl(Z, delays_state, self.P.weights, self.P)
            
            # Update state history
            state_history[:, :-1] = state_history[:, 1:]
            state_history[:, -1] = Z
            
            # Record activity if past t_cut
            if (t_curr > self.P.t_cut) and (it % max(1, self.P.x_decimate) == 0):
                if x_idx < x_buffer_len:
                    t_buf[x_idx] = t_curr
                    if self.P.RECORD_X:
                        X_rec[x_idx] = Z
                    x_idx += 1
        
        return {
            "X": X_rec,
            "t": t_buf,
        }

    def __str__(self) -> str:
        """
        Return a formatted string representation of the model parameters.
        
        Returns
        -------
        str
            Formatted string showing parameter names, explanations, and current values/shapes.
        """
        # Parameter explanations from docstring
        param_explanations = {
            "a": "Bifurcation parameter",
            "omega": "Angular frequency (rad/ms)",
            "G": "Global coupling strength",
            "sigma": "Noise amplitude",
            "dt": "Integration time step (ms)",
            "t_end": "Simulation end time (ms)",
            "t_cut": "Burn-in period (ms)",
            "nn": "Number of oscillators",
            "seed": "Random seed (-1 = no seed)",
            "speed": "Conduction velocity (mm/ms)",
            "weights": "Connectivity matrix (nn, nn)",
            "tr_len": "Transmission distances (mm)",
            "initial_state": "Initial complex states (nn,)",
            "RECORD_X": "Record activity time series",
            "x_decimate": "Recording decimation factor"
        }
        
        print("Stuart-Landau Oscillator Model (Numba)")
        print("=" * 50)
        print(f"{'Parameter':<16} {'Explanation':<42} {'Value/Shape'}")
        print("-" * 80)
        
        for param_name in self.valid_par:
            explanation = param_explanations.get(param_name, "No description available")
            value = getattr(self.P, param_name)
            
            # Format value/shape display
            if hasattr(value, 'shape'):
                # Array parameter - show shape
                value_str = f"shape {value.shape}"
            else:
                # Scalar parameter - show value
                value_str = str(value)
            
            print(f"{param_name:<16} {explanation:<30} {value_str}")
        
        print("-" * 80)
        return ""


# --------------------------------
# Helper functions
# --------------------------------

@njit
def set_initial_state_sl(nn, seed=-1):
    """
    Generate random initial conditions for the Stuart-Landau model.
    
    Creates small complex values for each oscillator.
    
    Parameters
    ----------
    nn : int
        Number of oscillators/brain regions
    seed : int, optional
        Random seed. If -1 or None, no seeding is applied.
        
    Returns
    -------
    np.ndarray
        Complex initial state of shape (nn,)
    """
    if seed is not None and seed >= 0:
        set_seed_compat(seed)
    
    # Small random complex perturbations
    z0 = (0.1 + 0.1j) * (np.random.randn(nn) + 1j * np.random.randn(nn))
    return z0.astype(np.complex128)


@register_jitable
def set_seed_compat(x):
    """Numba-compatible random seed setter."""
    np.random.seed(x)


@njit
def reset_numba_rng(seed):
    """Reset numba's random number generator."""
    np.random.seed(seed)
