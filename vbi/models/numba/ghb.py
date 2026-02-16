import warnings
import numpy as np
from numba import njit
from numba.experimental import jitclass
from numba.core.errors import NumbaPerformanceWarning
from numba import float64, int64
from vbi.models.numba.base import BaseNumbaModel

warnings.simplefilter("ignore", category=NumbaPerformanceWarning)


@njit(nogil=True)
def run(P, times):
    """
    Run the Generic Hopf Bifurcation model simulation.
    
    This function integrates the coupled Generic Hopf oscillators with BOLD signal
    generation using Euler-Maruyama stochastic integration.
    
    Parameters
    ----------
    P : ParGHB
        Parameter object containing all model parameters.
    times : np.ndarray
        Time array for the simulation.
        
    Returns
    -------
    tuple
        (t_bold, bold) where t_bold is time array and bold is BOLD signal
        of shape (nn, n_timepoints).
    """

    G = P.G
    dt = P.dt
    eta = P.eta
    SC = P.weights
    sigma = P.sigma
    omega = P.omega
    tcut = P.tcut
    decimate = P.decimate
    initial_state = P.initial_state

    epsilon = 0.5
    itaus = 1.25
    itauf = 2.5
    itauo = 1.02040816327
    ialpha = 5.0
    Eo = 0.4
    V0 = 4.0
    k1 = 2.77264
    k2 = 0.572
    k3 = -0.43

    nt = times.shape[0]
    nn = SC.shape[0]

    # state variables
    n_buffer = np.int64(np.floor(nt / decimate) + 1)
    x = np.zeros(nn)
    bold = np.zeros((nn, n_buffer))
    y = np.zeros(nn)
    z = np.array([0.0] * nn + [1.0] * 3 * nn)
    # act = np.zeros((nn, nt))

    # initial conditions (similar value for all regions)
    x_init, y_init = initial_state[:nn], initial_state[nn:]
    x[:] = x_init
    y[:] = y_init

    for i in range(nn):
        bold[i, 0] = V0 * (
            k1
            - k1 * z[3 * nn + i]
            + k2
            - k2 * (z[3 * nn + i] / z[2 * nn + i])
            + k3
            - k3 * z[2 * nn + i]
        )

    ii = 0 # counter for decimation
    for it in range(nt - 1):
        for i in range(nn):
            gx, gy = 0.0, 0.0
            for j in range(nn):
                gx = gx + SC[i, j] * (x[j] - x[i])
                gy = gy + SC[i, j] * (y[j] - y[i])
            dx = (
                (x[i] * (eta[i] - (x[i] * x[i]) - (y[i] * y[i])))
                - (omega[i] * y[i])
                + (G * gx)
            )
            dy = (
                (y[i] * (eta[i] - (x[i] * x[i]) - (y[i] * y[i])))
                + (omega[i] * x[i])
                + (G * gy)
            )
            dz0 = epsilon * x[i] - itaus * z[i] - itauf * (z[nn + i] - 1)
            dz1 = z[i]
            dz2 = itauo * (z[nn + i] - z[2 * nn + i] ** ialpha)
            dz3 = itauo * (
                z[nn + i] * (1 - (1 - Eo) ** (1 / z[nn + i])) / Eo
                - (z[2 * nn + i] ** ialpha) * z[3 * nn + i] / z[2 * nn + i]
            )

            x[i] = x[i] + dt * dx + np.sqrt(dt) * sigma * np.random.randn()
            y[i] = y[i] + dt * dy + np.sqrt(dt) * sigma * np.random.randn()

            z[i] = z[i] + dt * dz0
            z[nn + i] = z[nn + i] + dt * dz1
            z[2 * nn + i] = z[2 * nn + i] + dt * dz2
            z[3 * nn + i] = z[3 * nn + i] + dt * dz3
            if (it%decimate == 0):
                bold[i, ii + 1] = V0 * (
                    k1
                    - k1 * z[3 * nn + i]
                    + k2
                    - k2 * (z[3 * nn + i] / z[2 * nn + i])
                    + k3
                    - k3 * z[2 * nn + i]
                )
        if (it%decimate == 0):
            ii += 1
    bold = bold[:, times[::decimate]>tcut]
    t_bold = times[times[::decimate]>tcut]
    return t_bold, bold


class GHB_sde(BaseNumbaModel):
    """
    Numba implementation of the Generic Hopf Bifurcation model with BOLD signal generation.
    
    This model implements a network of coupled Stuart-Landau oscillators (Generic Hopf 
    Bifurcation) with additive noise and hemodynamic response modeling to generate 
    BOLD signals. Each brain region is modeled as a 2D oscillator with intrinsic 
    frequency and bifurcation parameter.
    
    The neural dynamics follow:
    dx/dt = x(η - x² - y²) - ωy + G·coupling_x + noise
    dy/dt = y(η - x² - y²) + ωx + G·coupling_y + noise
    
    Where:
    - (x, y) are the neural state variables in Cartesian coordinates
    - η (eta) is the bifurcation parameter controlling oscillation amplitude
    - ω (omega) is the intrinsic frequency of oscillation
    - G is the global coupling strength
    - The (x² + y²) term provides amplitude-dependent damping
    
    .. list-table:: Parameters
        :widths: 25 50 25
        :header-rows: 1

        * - Name
          - Explanation
          - Default Value
        * - `G`
          - Global coupling strength scaling network connections.
          - 1.0
        * - `dt`
          - Integration time step in seconds.
          - 0.001
        * - `sigma`
          - Noise amplitude (standard deviation of additive Gaussian noise).
          - 0.1
        * - `tend`
          - End time of simulation in seconds.
          - 10.0
        * - `tcut`
          - Time from which to start collecting output (burn-in period).
          - 0.0
        * - `eta`
          - Bifurcation parameter array of length `nn`. Controls oscillation amplitude. If array-like, it should be of length `nn` (number of nodes).
          - np.array([])
        * - `omega`
          - Intrinsic frequency array of length `nn`. Sets the natural oscillation frequency for each region. If array-like, it should be of length `nn`.
          - np.array([])
        * - `weights`
          - Structural connectivity matrix of shape (`nn`, `nn`). Must be provided.
          - np.array([[], []])
        * - `decimate`
          - Decimation factor for output time series (every `decimate`-th point is saved).
          - 1
        * - `initial_state`
          - Initial state vector of shape (2*nn,) containing [x₀, y₀]. If None, random initial conditions are generated.
          - np.array([])
        * - `seed`
          - Random seed for reproducible simulations. If -1, no seeding is applied.
          - -1

    Usage example:
        >>> import numpy as np
        >>> from vbi.models.numba.ghb import GHB_sde
        >>> W = np.eye(2)  # 2-node demo connectivity
        >>> eta = np.array([0.1, 0.1])  # Bifurcation parameters
        >>> omega = np.array([40.0, 40.0])  # Frequencies in Hz
        >>> ghb = GHB_sde({
        ...     "weights": W, 
        ...     "eta": eta, 
        ...     "omega": omega,
        ...     "dt": 0.01, 
        ...     "tend": 10.0, 
        ...     "tcut": 2.0
        ... })
        >>> result = ghb.run()
        >>> t, bold = result["t"], result["bold"]  # bold has shape (nn, n_timepoints)

    Notes
    -----
    The model combines neural oscillator dynamics with a simplified BOLD hemodynamic
    response based on the Balloon-Windkessel model. The BOLD signal is computed using
    the Friston 2003 formulation with fixed parameters optimized for realistic
    hemodynamic responses.
    
    The Generic Hopf Bifurcation model is particularly useful for studying:
    - Oscillatory brain dynamics and synchronization
    - Transitions between different dynamical regimes
    - Frequency-specific network interactions
    - BOLD signal generation from neural oscillations
    """
    def __init__(self, par: dict = {}) -> None:
        """
        Initialize the Generic Hopf Bifurcation model.

        Parameters
        ----------
        par : dict, optional
            Dictionary containing model parameters. See class documentation for 
            available parameters. The 'weights', 'eta', and 'omega' parameters 
            are required for proper functionality.
        """
        super().__init__()
        self.valid_par = [par_spec[i][0] for i in range(len(par_spec))]
        # Set valid_params for BaseNumbaModel (excluding derived parameters if any)
        self.valid_params = [k for k in GHB_DEFAULTS.keys()]
        self.check_parameters(par)
        self.P = self.get_par_obj(par)
    
    def get_default_parameters(self):
        """
        Get the default parameters for the Generic Hopf Bifurcation model.
        
        Returns a copy of the GHB_DEFAULTS dictionary.
        
        Returns
        -------
        dict
            Dictionary containing default parameter values.
            
        Examples
        --------
        >>> from vbi.models.numba.ghb import GHB_sde
        >>> import numpy as np
        >>> model = GHB_sde({"weights": np.eye(2), "eta": np.array([0.1, 0.1]), "omega": np.array([40.0, 40.0])})
        >>> defaults = model.get_default_parameters()
        >>> print(defaults['G'])
        1.0
        """
        return GHB_DEFAULTS.copy()
    
    def get_parameter_descriptions(self):
        """
        Get descriptions for all Generic Hopf Bifurcation model parameters.
        
        Returns
        -------
        dict
            Dictionary mapping parameter names to tuples of (description, type).
            
        Examples
        --------
        >>> model = GHB_sde({"weights": np.eye(2), "eta": np.array([0.1, 0.1]), "omega": np.array([40.0, 40.0])})
        >>> descriptions = model.get_parameter_descriptions()
        >>> print(descriptions['G'])
        ('Global coupling strength', 'scalar')
        """
        return {
            'G': ('Global coupling strength scaling network connections', 'scalar'),
            'dt': ('Integration time step (seconds)', 'scalar'),
            'sigma': ('Noise amplitude (standard deviation)', 'scalar'),
            'tend': ('End time of simulation (seconds)', 'scalar'),
            'tcut': ('Burn-in time (seconds)', 'scalar'),
            'eta': ('Bifurcation parameter per node', 'vector'),
            'omega': ('Intrinsic frequency per node (Hz)', 'vector'),
            'weights': ('Structural connectivity matrix', 'matrix'),
            'decimate': ('Decimation factor for output', 'int'),
            'initial_state': ('[Derived] Initial state vector (2*nn)', 'vector'),
            'seed': ('Random seed (-1 = no seed)', 'int'),
        }

    def get_par_obj(self, par: dict):
        """
        Create parameter object from dictionary.
        
        Parameters
        ----------
        par : dict
            Parameter dictionary.
            
        Returns
        -------
        ParGHB
            Numba jitclass parameter object.
        """
        if "initial_state" in par.keys():
            par["initial_state"] = np.array(par["initial_state"])
        if "weights" in par.keys():
            par["weights"] = np.array(par["weights"])
        return ParGHB(**par)

    def __str__(self) -> str:
        """
        Return a formatted string representation of the model parameters.
        
        Returns
        -------
        str
            Formatted string showing all model parameters and their values.
        """
        return self._format_parameters_table("Generic Hopf Bifurcation")

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
                raise ValueError(f"Invalid parameter: {key}")

    def set_initial_state(self, seed=None):
        """
        Generate random initial conditions for the model.
        
        Parameters
        ----------
        seed : int, optional
            Random seed for reproducible initial conditions.
            
        Returns
        -------
        np.ndarray
            Initial state vector of shape (2*nn,) with random values in [0, 1].
        """
        if seed is not None:
            np.random.seed(seed)
        assert self.P.weights is not None
        return np.random.uniform(0, 1, 2 * self.P.weights.shape[0])

    def check_input(self):
        """
        Validate model parameters before simulation.
        
        Raises
        ------
        AssertionError
            If weights matrix is None, not square, or if eta/omega arrays 
            don't match the number of nodes.
        """
        assert self.P.weights is not None
        assert self.P.weights.shape[0] == self.P.weights.shape[1]
        assert self.P.eta is not None
        assert self.P.omega is not None
        assert self.P.weights.shape[0] == self.P.eta.shape[0]

    def run(self, par={}, tspan=None, x0=None, verbose=True):
        """
        Run the Generic Hopf Bifurcation simulation.

        Parameters
        ----------
        par : dict, optional
            Dictionary of parameters to update for this simulation run.
            Any parameter from the class documentation can be updated.
        tspan : tuple or None, optional
            Time span as (start_time, end_time). If None, uses (0, tend).
        x0 : np.ndarray, optional
            Initial state vector of shape (2*nn,). If None, random initial 
            conditions are generated using set_initial_state().
        verbose : bool, optional
            Whether to print verbose output (currently unused).

        Returns
        -------
        dict
            Dictionary containing simulation results:
            
            - 't': np.ndarray of time points (after burn-in and decimation)
            - 'bold': np.ndarray of shape (nn, n_timepoints) containing 
              simulated BOLD signals for each brain region
        """
        if x0 is None:
            self.seed = self.P.seed if self.P.seed > 0 else None
            self.P.initial_state = self.set_initial_state(seed=self.seed)
        else:
            self.P.initial_state = x0

        if tspan is None:
            times = np.arange(0, self.P.tend, self.P.dt)
        else:
            times = np.arange(tspan[0], tspan[1], self.P.dt)

        if par:
            self.check_parameters(par)
            for key in par.keys():
                setattr(self.P, key, par[key])

        self.check_input()
        t, b = run(self.P, times)
        return {'t': t, 'bold': b}


par_spec = [
    ("G", float64),
    ("dt", float64),
    ("seed", int64),
    ("tend", float64),
    ("tcut", float64),
    ("sigma", float64),
    ("eta", float64[:]),
    ("decimate", int64),
    ("omega", float64[:]),
    ("weights", float64[:, :]),
    ("initial_state", float64[:]),
]

# Default parameter values - single source of truth
GHB_DEFAULTS = {
    'G': 1.0,
    'dt': 0.001,
    'sigma': 0.1,
    'tend': 10.0,
    'tcut': 0.0,
    'eta': None,  # Required - must be provided by user
    'omega': None,  # Required - must be provided by user
    'weights': None,  # Required - must be provided by user
    'decimate': 1,
    'initial_state': None,  # Optional - auto-generated if not provided
    'seed': -1,
}


@jitclass(par_spec)
class ParGHB:
    """
    Numba jitclass container for Generic Hopf Bifurcation model parameters.
    
    This class holds all parameters needed for the Generic Hopf Bifurcation model
    in a format optimized for Numba compilation. It stores both scalar parameters
    and array parameters like connectivity weights, bifurcation parameters, and frequencies.
    
    Note: This is an internal class used by the GHB_sde class. Users should
    not instantiate this class directly.
    
    Default values are defined in GHB_DEFAULTS dictionary.
    """
    def __init__(
        self,
        G=GHB_DEFAULTS['G'],
        dt=GHB_DEFAULTS['dt'],
        sigma=GHB_DEFAULTS['sigma'],
        tend=GHB_DEFAULTS['tend'],
        tcut=GHB_DEFAULTS['tcut'],
        eta=np.array([]),
        initial_state=np.array([]),
        omega=np.array([]),
        weights=np.array([[], []]),
        decimate=GHB_DEFAULTS['decimate'],
        seed=-1,
    ):
        """
        Initialize the parameter container.
        
        Parameters
        ----------
        G : float
            Global coupling strength.
        dt : float
            Integration time step.
        sigma : float
            Noise amplitude.
        tend : float
            End time of simulation.
        tcut : float
            Burn-in time.
        eta : np.ndarray
            Bifurcation parameters for each node.
        initial_state : np.ndarray
            Initial state vector.
        omega : np.ndarray
            Intrinsic frequencies for each node.
        weights : np.ndarray
            Connectivity matrix.
        decimate : int
            Decimation factor.
        """
        self.G = G
        self.dt = dt
        self.seed = -1
        self.eta = eta
        self.tend = tend
        self.tcut = tcut
        self.sigma = sigma
        self.omega = omega
        self.weights = weights
        self.decimate = decimate
        self.initial_state = initial_state


# Alias for consistency with naming convention
GHB_sde_numba = GHB_sde
