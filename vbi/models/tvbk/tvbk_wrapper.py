import numpy as np
import collections
from vbi.models.tvbk.utils import prepare_vec, setup_connectivity


try:
    import tvbk as m

    TVBK_AVAILABLE = True
except ImportError:
    TVBK_AVAILABLE = False


class MPR:
    """
    Montbrio Population Rate (MPR) model wrapper for TVBK simulation.
    
    This class provides a Python interface to the TVBK (The Virtual Brain Kernel) 
    implementation of the Montbrio population rate model. The MPR model describes 
    the dynamics of neural populations using mean-field approximations.
    
    The model simulates the evolution of neural state variables across multiple nodes
    with configurable connectivity, delays, and noise. It supports batch processing
    and can optionally record regional variables (RV) and BOLD signals.
    
    Attributes
    ----------
    MPRTheta : namedtuple
        Named tuple containing the model parameters (tau, I, Delta, J, eta, G).
    num_svar : int
        Number of state variables in the model (default: 2).
    num_parm : int
        Number of parameters in the model (default: 6).
    valid_parameters : list
        List of valid parameter names accepted by the model.
    
    Parameters
    ----------
    par : dict, optional
        Dictionary of model parameters to override defaults. Valid keys include:
        - tau : float, time constant
        - I : float, external input
        - Delta : float, heterogeneity parameter
        - J : float, synaptic coupling strength
        - eta : float, excitability
        - G : float, global coupling scaling
        - num_batch : int, number of parallel simulations
        - horizon : int, circular buffer horizon length
        - width : int, SIMD vector width
        - dt : float, integration time step
        - dtype : numpy.dtype, numerical precision
        - weights : array_like, connectivity weight matrix
        - delays : array_like, connectivity delay matrix
        - num_node : int, number of network nodes
        - noise_amp : float, noise amplitude
        - num_time : int, total simulation time steps
        - decimate_rv : int, decimation factor for recording
        - RECORD_RV : bool, whether to record regional variables
        - RECORD_BOLD : bool, whether to record BOLD signals
    
    Examples
    --------
    >>> weights = np.random.rand(10, 10)
    >>> delays = np.ones((10, 10)) * 5
    >>> model = MPR(par={'weights': weights, 'delays': delays, 'G': 1.5})
    >>> results = model.run()
    
    Notes
    -----
    Requires the TVBK library to be installed. The model uses circular buffers
    to efficiently handle delayed interactions between nodes.
    """

    MPRTheta = collections.namedtuple(
        typename="MPRTheta", field_names="tau I Delta J eta G".split(" ")
    )
    num_svar = 2  # number of state variables
    num_parm = 6  # number of parameters

    def __init__(self, par: dict = {}) -> None:
        """
        Initialize the MPR model with specified parameters.
        
        Sets up the model by merging user-provided parameters with defaults,
        validating parameter names, and creating a named tuple of model parameters.
        
        Parameters
        ----------
        par : dict, optional
            Dictionary of parameters to override defaults. Keys must be valid
            parameter names as defined in `get_default_parameters()`.
        
        Raises
        ------
        ValueError
            If any key in `par` is not a valid parameter name.
        
        See Also
        --------
        get_default_parameters : Returns the default parameter dictionary.
        check_parameters : Validates parameter names.
        """

        self._par = self.get_default_parameters()
        self.valid_parameters = list(self._par.keys())
        self.check_parameters(par)
        self._par.update(par)

        for item in self._par.items():
            setattr(self, item[0], item[1])

        self.mpr_default_theta = self.MPRTheta(
            tau=self._par["tau"],
            I=self._par["I"],
            Delta=self._par["Delta"],
            J=self._par["J"],
            eta=self._par["eta"],
            G=self._par["G"],
        )

    def __str__(self) -> str:
        """
        Return a string representation of the MPR model.
        
        Returns
        -------
        str
            A formatted string showing the model name and its parameters.
        """
        return f"MPR model with parameters: {self._par}"

    def get_default_parameters(self) -> dict:
        """
        Get the default parameters for the MPR model.
        
        Returns a dictionary containing all default parameter values including
        model parameters (tau, I, Delta, J, eta, G), simulation settings 
        (dt, num_time, etc.), and recording options.
        
        Returns
        -------
        dict
            Dictionary with parameter names as keys and default values.
            Includes model parameters, simulation configuration, and recording flags.
        
        Notes
        -----
        Default model parameters are based on the original Montbrio et al. model.
        Simulation parameters can be adjusted based on computational resources
        and desired temporal resolution.
        """
        return {
            "tau": 1.0,
            "Delta": 1.0,
            "I": 0.0,
            "J": 15.0,
            "eta": -5.0,
            "G": 1.0,
            "num_batch": 1,
            "horizon": 256,
            "width": 8,
            "dt": 0.01,
            "dtype": np.float32,
            "weights": None,
            "delays": None,
            "num_node": None,
            "noise_amp": None, 
            "num_time": 1000,
            "decimate_rv": 10,
            "RECORD_RV": True,
            "RECORD_BOLD": False, # TODO: Add BOLD recording
        }

    def check_parameters(self, par):
        """
        Validate that all provided parameters are recognized.
        
        Checks each key in the provided parameter dictionary against the list
        of valid parameters to ensure no invalid parameters are passed.
        
        Parameters
        ----------
        par : dict
            Dictionary of parameters to validate.
        
        Raises
        ------
        ValueError
            If any parameter name in `par` is not in the valid parameters list.
        """
        for key in par.keys():
            if key not in self.valid_parameters:
                raise ValueError(f"Invalid parameter {key:s} provided.")

    
    def initialize_buffers(self):
        """
        Initialize circular buffers for the TVBK simulation.
        
        Creates and initializes the circular buffer structure (Cx8s) used by TVBK
        to store state variable history for handling delayed interactions.
        The buffer is initialized with linearly spaced values for testing purposes.
        
        Notes
        -----
        The buffer size is determined by num_batch * num_node * horizon * width.
        Initial values are set to a linearly increasing sequence scaled by 4.0.
        The coupling variables cx1 and cx2 are initialized to zero.
        
        Raises
        ------
        AttributeError
            If num_batch, num_node, horizon, width, or dtype are not set.
        """

        total_volume = self.num_batch * self.num_node * self.horizon * self.width
        self.cx = m.Cx8s(self.num_node, self.horizon, self.num_batch)
        buf_val = (
            np.r_[: 1.0 : 1j * total_volume]
            .reshape(self.num_batch, self.num_node, self.horizon, self.width)
            .astype(self.dtype)
            * 4.0
        )
        self.cx.buf[:] = buf_val
        self.cx.cx1[:] = self.cx.cx2[:] = 0.0
        

    def prepare_input(self):
        """
        Prepare and validate all input arrays for the TVBK simulation.
        
        This method performs several critical steps:
        1. Validates and processes connectivity weights and delays
        2. Broadcasts scalar parameters to batch and SIMD vector dimensions
        3. Organizes parameters into the required tensor format
        4. Initializes circular buffers
        5. Sets up connectivity structure
        
        The parameter tensor has shape (num_batch, num_node, num_parm, width)
        where parameters are ordered as: [tau, I, Delta, J, eta, G].
        
        Raises
        ------
        AssertionError
            If weights matrix is not provided.
        
        Notes
        -----
        This method must be called before running the simulation. It modifies
        several instance attributes including G, J, I, eta, tau, Delta, p, cx, and conn.
        All parameters are broadcast to match the batch size and SIMD width.
        """

        width = self.width
        num_batch = self.num_batch
        num_parm = self.num_parm
        
        assert self.weights is not None, "weights must be provided"
        self.weights = np.array(self.weights)
        num_node = self.num_node = self.weights.shape[0]
        
        self.G = prepare_vec(self.G, num_batch, self.dtype)
        self.J = prepare_vec(self.J, num_batch, self.dtype)
        self.I = prepare_vec(self.I, num_batch, self.dtype)
        self.eta = prepare_vec(self.eta, num_batch, self.dtype)
        self.tau = prepare_vec(self.tau, num_batch, self.dtype)
        self.Delta = prepare_vec(self.Delta, num_batch, self.dtype)        
        self.noise_amp = np.array(self.noise_amp, self.dtype)
        self.p = np.zeros((num_batch, num_node, num_parm, width), self.dtype)
        
        self.p[:, :, 0, :] = self.tau
        self.p[:, :, 1, :] = self.I
        self.p[:, :, 2, :] = self.Delta
        self.p[:, :, 3, :] = self.J
        self.p[:, :, 4, :] = self.eta
        self.p[:, :, 5, :] = self.G
        
        self.initialize_buffers()
        self.conn = setup_connectivity(self.weights, self.delays)
        

    def run(self):
        """
        Execute the MPR model simulation using TVBK.
        
        Runs the main simulation loop, advancing the model state through time
        using the TVBK step function. Optionally records regional variables (RV)
        and BOLD signals based on configuration flags.
        
        The simulation proceeds in discrete steps determined by `decimate_rv`,
        with state variables stored at each recorded time point if RECORD_RV is True.
        
        Returns
        -------
        dict
            Dictionary containing simulation results with keys:
            - 'rv_t' : array_like or Ellipsis
                Time points for regional variable recordings (not yet implemented).
            - 'rv_d' : ndarray, shape (num_samples, num_batch, num_svar, num_node)
                Regional variable data recorded at each time step.
                Contains the state variables across all nodes and batches.
                Note: Only one SIMD width index is stored (the 8 width values are 
                identical, representing the same physical quantity computed in parallel).
            - 'fmri_t' : array_like or Ellipsis
                Time points for BOLD signal (not yet implemented).
            - 'fmri_d' : None or ndarray
                BOLD signal data (not yet implemented, currently returns None).
        
        Notes
        -----
        The simulation uses a circular buffer to efficiently handle delays.
        Recording frequency is controlled by `decimate_rv` parameter.
        BOLD signal calculation is planned but not yet implemented.
        
        The TVBK step_mpr function performs numerical integration of the
        differential equations with noise injection and delayed coupling.
        
        TVBK internally uses SIMD width=8 for performance (AVX vectorization),
        but only one width index is stored in the output since all 8 values
        represent the same physical state (computed in parallel for speed).
        This reduces memory usage by 8x compared to storing all SIMD lanes.
        
        See Also
        --------
        prepare_input : Must be called before run() to set up simulation inputs.
        initialize_buffers : Initializes the circular buffer structure.
        """
        
        self.prepare_input()

        x = np.zeros((self.num_batch, self.num_svar, self.num_node, self.width), self.dtype)
        y = np.zeros_like(x)
        z = np.zeros((self.num_batch, self.num_svar, 8), self.dtype)+ self.noise_amp
        seed = np.zeros((self.num_batch, 8, 4), np.uint64)
        num_samples = self.num_time // self.decimate_rv + 1
        
        if self.RECORD_RV:
            # Store only one SIMD width index to save memory (8x reduction)
            # All 8 width indices should be identical/nearly identical
            trace_c = np.zeros(
                (
                    num_samples,
                    self.num_batch,
                    self.num_svar,
                    self.num_node,
                ),
                dtype=self.dtype
            )

        for i in range(num_samples):
            if self.RECORD_RV:
                m.step_mpr(
                    self.cx,
                    self.conn,
                    x,
                    y,
                    z, 
                    self.p,
                    i * self.decimate_rv,
                    self.decimate_rv,
                    self.dt,
                    seed
                )
            if self.RECORD_RV:
                # Store only the first SIMD width index [0]
                # Since all 8 width values represent the same physical quantity
                trace_c[i] = x[:, :, :, 0]
            
            # TODO: Calculate BOLD signal
            if self.RECORD_BOLD:
                pass
                # add BOLD signal calculation pytorch code here
            
        
        return {
            "rv_t": ...,
            "rv_d": trace_c, 
            "fmri_t": ...,
            "fmri_d": None,
            }
        
