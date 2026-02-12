import os
import numpy as np
from os.path import join

from vbi.models.cpp.base import BaseModel

_JR_sde = None
_JR_sdde = None
_JR_SDE_AVAILABLE = False
_JR_SDDE_AVAILABLE = False

try:
    from vbi.models.cpp._src.jr_sde import JR_sde as _JR_sde
    _JR_SDE_AVAILABLE = True
except ImportError as e:
    print(f"Could not import JR_sde C++ module: {e}")
    print("JR_sde class will be available but run() will raise an error.")

try:
    from vbi.models.cpp._src.jr_sdde import JR_sdde as _JR_sdde
    _JR_SDDE_AVAILABLE = True
except ImportError as e:
    print(f"Could not import JR_sdde C++ module: {e}")
    print("JR_sdde class will be available but run() will raise an error.")

if not _JR_SDE_AVAILABLE and not _JR_SDDE_AVAILABLE:
    print("Please compile C++ extensions or use Python/Numba implementations instead.")


class JR_sde(BaseModel):
    r"""
    Jansen-Rit model C++ implementation.

    This model supports heterogeneous parameters across brain regions.
    Parameters marked as "scalar|vector" in the parameter descriptions can be 
    specified as either single values (applied to all regions) or arrays 
    (one value per region).

    Parameters
    ----------

    par: dict
        Including the following:
        - **A** : [mV] determine the maximum amplitude of the excitatory PSP (EPSP) - can be heterogeneous
        - **B** : [mV] determine the maximum amplitude of the inhibitory PSP (IPSP)
        - **a** : [Hz]  1/tau_e,  :math:`\\sum` of the reciprocal of the time constant of passive membrane and all other spatially distributed  delays in the dendritic network
        - **b** : [Hz] 1/tau_i
        - **r**  [mV] the steepness of the sigmoidal transformation.
        - **v0** parameter of nonlinear sigmoid function
        - **vmax** parameter of nonlinear sigmoid function
        - **C_i** [list or np.array] average number of synaptic contacts in th inhibitory and excitatory feedback loops - can be heterogeneous
        - **noise_amp**
        - **noise_mu**

        - **dt** [second] integration time step
        - **t_initial** [s] initial time
        - **t_end** [s] final time
        - **method** [str] method of integration
        - **t_transition** [s] time to reach steady state
        - **dim** [int] dimention of the system

    """

    def __init__(self, par: dict = {}):
        import warnings

        super().__init__()
        
        # Backward compatibility mapping
        param_mapping = {
            'noise_std': 'noise_amp'
        }
        
        # Apply backward compatibility
        for old_name, new_name in param_mapping.items():
            if old_name in par:
                warnings.warn(f"Parameter '{old_name}' is deprecated. Use '{new_name}' instead.", 
                            DeprecationWarning, stacklevel=2)
                par[new_name] = par.pop(old_name)
        
        self._par = self.get_default_parameters()
        self.valid_params = list(self._par.keys())
        self.check_parameters(par)
        self._par.update(par)

        for item in self._par.items():
            name = item[0]
            value = item[1]
            setattr(self, name, value)

        if self.seed is not None:
            np.random.seed(self.seed)

        self.N = self.num_nodes = np.asarray(self.weights).shape[0]

        if self.initial_state is None:
            self.INITIAL_STATE_SET = False

        self.noise_seed = 1 if self.noise_seed else 0
        os.makedirs(join(self.output), exist_ok=True)

    def get_parameter_descriptions(self):
        """
        Get descriptions and types for Jansen-Rit SDE model parameters.
        
        Returns
        -------
        dict
            Dictionary mapping parameter names to (description, type) tuples.
        """
        return {
            "G": ("Global coupling strength", "scalar"),
            "A": ("Maximum amplitude of EPSP (mV)", "scalar|vector"),
            "B": ("Maximum amplitude of IPSP (mV)", "scalar"),
            "a": ("Inverse of excitatory time constant (Hz)", "scalar"),
            "b": ("Inverse of inhibitory time constant (Hz)", "scalar"),
            "noise_mu": ("Mean of noise input", "scalar"),
            "noise_amp": ("Noise amplitude", "scalar"),
            "vmax": ("Maximum firing rate parameter", "scalar"),
            "v0": ("Firing threshold", "scalar"),
            "r": ("Steepness of sigmoid", "scalar"),
            "initial_state": ("Initial state of the system", "vector"),
            "weights": ("Structural connectivity matrix", "matrix"),
            "C0": ("Synaptic contacts (pyr→exc)", "scalar|vector"),
            "C1": ("Synaptic contacts (exc→pyr)", "scalar|vector"),
            "C2": ("Synaptic contacts (pyr→inh)", "scalar|vector"),
            "C3": ("Synaptic contacts (inh→pyr)", "scalar|vector"),
            "noise_seed": ("Seed for noise generation", "scalar"),
            "seed": ("Random seed for reproducibility", "-"),
            "dt": ("Integration time step", "scalar"),
            "dim": ("Dimension of the system", "scalar"),
            "method": ("Integration method", "string"),
            "t_transition": ("Transition time", "scalar"),
            "t_end": ("End time of simulation", "scalar"),
            "output": ("Output directory", "string"),
            "RECORD_AVG": ("Record average time series", "bool"),
        }

    def __str__(self) -> str:
        return self._format_parameters_table()

    def __call__(self):
        return self._par

    def get_default_parameters(self):
        """
        return default parameters for the Jansen-Rit sde model.
        """

        par = {
            "G": 0.5,  # global coupling strength
            "A": 3.25,  # mV
            "B": 22.0,  # mV
            "a": 0.1,  # 1/ms
            "b": 0.05,  # 1/ms
            "noise_mu": 0.24,
            "noise_amp": 0.3,
            "vmax": 0.005,
            "v0": 6,  # mV
            "r": 0.56,  # mV
            "initial_state": None,
            "weights": None,
            "C0": 135.0 * 1.0,
            "C1": 135.0 * 0.8,
            "C2": 135.0 * 0.25,
            "C3": 135.0 * 0.25,
            "noise_seed": 0,
            "seed": None,
            "dt": 0.05,  # ms
            "dim": 6,
            "method": "heun",
            "t_transition": 500.0,  # ms
            "t_end": 2501.0,  # ms
            "output": "output",  # output directory
            "RECORD_AVG": False,  # true to store large time series in file
        }
        return par

    # ---------------------------------------------------------------
    def set_initial_state(self):
        """
        Set initial state for the system of JR equations with N nodes.
        """

        self.initial_state = set_initial_state(self.num_nodes, self.seed)
        self.INITIAL_STATE_SET = True

    # -------------------------------------------------------------------------

    def prepare_input(self):
        """
        prepare input parameters for passing to C++ engine.
        """

        self.N = int(self.N)
        self.weights = np.asarray(self.weights)
        self.dt = float(self.dt)
        self.t_transition = float(self.t_transition)
        self.t_end = float(self.t_end)
        self.G = float(self.G)
        self.B = float(self.B)
        self.a = float(self.a)
        self.b = float(self.b)
        self.r = float(self.r)
        self.v0 = float(self.v0)
        self.vmax = float(self.vmax)
        self.A = check_sequence(self.A, self.N)
        self.C0 = check_sequence(self.C0, self.N)
        self.C1 = check_sequence(self.C1, self.N)
        self.C2 = check_sequence(self.C2, self.N)
        self.C3 = check_sequence(self.C3, self.N)
        self.noise_mu = float(self.noise_mu)
        self.noise_amp = float(self.noise_amp)
        self.noise_seed = int(self.noise_seed)
        self.initial_state = np.asarray(self.initial_state)

    # -------------------------------------------------------------------------
    def run(self, par: dict = {}, x0: np.ndarray = None, verbose: bool = False):
        """
        Integrate the system of equations for Jansen-Rit sde model.

        Parameters
        ----------

        par: dict
            parameters to control the Jansen-Rit sde model.
        x0: np.array
            initial state
        verbose: bool
            print the message if True

        Returns
        -------
        dict
            - **t** : time series
            - **x** : state variables

        """
        if not _JR_SDE_AVAILABLE:
            raise RuntimeError(
                "JR_sde C++ module is not available. "
                "Please compile C++ extensions using 'python setup.py build_ext --inplace' "
                "or use Python/Numba JR implementations instead."
            )

        if x0 is None:
            if not self.INITIAL_STATE_SET:
                self.set_initial_state()
                if verbose:
                    print("initial state set by default")
        else:
            self.INITIAL_STATE_SET = True
            self.initial_state = x0

        for key in par.keys():
            if key not in self.valid_params:
                raise ValueError("Invalid parameter: " + key)
            setattr(self, key, par[key])

        self.prepare_input()

        obj = _JR_sde(
            self.N,
            self.dt,
            self.t_transition,
            self.t_end,
            self.G,
            self.weights,
            self.initial_state,
            self.A,
            self.B,
            self.a,
            self.b,
            self.r,
            self.v0,
            self.vmax,
            self.C0,
            self.C1,
            self.C2,
            self.C3,
            self.noise_mu,
            self.noise_amp,
            self.noise_seed,
        )

        if self.method == "euler":
            obj.eulerIntegrate()
        elif self.method == "heun":
            obj.heunIntegrate()
        else:
            print("unkown integratiom method")
            exit(0)

        sol = np.asarray(obj.get_coordinates()).T
        times = np.asarray(obj.get_times())

        del obj

        return {"t": times, "x": sol}


############################ Jansen-Rit sdde ##################################


class JR_sdde(BaseModel):
    pass

    # -------------------------------------------------------------------------

    def __init__(self, par={}) -> None:
        super().__init__()
        
        self._par = self.get_default_parameters()
        self.valid_params = list(self._par.keys())
        self.check_parameters(par)
        self._par.update(par)

        for item in self._par.items():
            setattr(self, item[0], item[1])

        if self.seed is not None:
            np.random.seed(self.seed)

        self.noise_seed = 1 if self.noise_seed else 0
        assert self.weights is not None, "weights must be provided"
        assert self.delays is not None, "delays must be provided"
        self.N = self.num_nodes = len(self.weights)

        self.C0 = check_sequence(self.C0, self.N)
        self.C1 = check_sequence(self.C1, self.N)
        self.C2 = check_sequence(self.C2, self.N)
        self.C3 = check_sequence(self.C3, self.N)
        self.sti_amplitude = check_sequence(self.sti_amplitude, self.N)

        if self.initial_state is None:
            self.INITIAL_STATE_SET = False
        os.makedirs(join(self.output), exist_ok=True)

    # -------------------------------------------------------------------------

    def get_default_parameters(self):
        """
        get default parameters for the system of JR equations.
        """

        param = {
            "weights": None,
            "delays": None,
            "dt": 0.01,
            "G": 0.01,
            "mu": 0.22,
            "sigma": 0.005,
            "dim": 6,
            "A": 3.25,
            "a": 0.1,
            "B": 22.0,
            "b": 0.05,
            "v0": 6.0,
            "vmax": 0.005,
            "r": 0.56,
            "C": None,  # kept for backward compatibility
            "C0": 135.0 * 1.0,
            "C1": 135.0 * 0.8,
            "C2": 135.0 * 0.25,
            "C3": 135.0 * 0.25,
            "nstart": None,
            "record_step": 1,
            "sti_ti": 0.0,
            "sti_duration": 0.0,
            "sti_amplitude": 0.0,  # scalar or sequence of length N
            "sti_gain": 0.0,
            "noise_seed": False,
            "seed": None,
            "initial_state": None,
            "method": "heun",
            "output": "output",
            "t_end": 2000.0,
            "t_transition": 1000.0,
        }

        return param

    # -------------------------------------------------------------------------

    def get_parameter_descriptions(self):
        """
        Get descriptions and types for Jansen-Rit SDDE model parameters.
        
        Returns
        -------
        dict
            Dictionary mapping parameter names to (description, type) tuples.
        """
        return {
            "weights": ("Structural connectivity matrix", "matrix"),
            "delays": ("Delay matrix", "matrix"),
            "dt": ("Integration time step", "scalar"),
            "G": ("Global coupling strength", "scalar"),
            "mu": ("Mean of noise input", "scalar"),
            "sigma": ("Standard deviation of noise", "scalar"),
            "dim": ("Dimension of the system", "scalar"),
            "A": ("Maximum amplitude of EPSP", "scalar"),
            "a": ("Inverse of excitatory time constant", "scalar"),
            "B": ("Maximum amplitude of IPSP", "scalar"),
            "b": ("Inverse of inhibitory time constant", "scalar"),
            "v0": ("Firing threshold", "scalar"),
            "vmax": ("Maximum firing rate parameter", "scalar"),
            "r": ("Steepness of sigmoid", "scalar"),
            "C": ("Legacy synaptic contacts parameter", "-"),
            "C0": ("Synaptic contacts (pyr→exc)", "scalar|vector"),
            "C1": ("Synaptic contacts (exc→pyr)", "scalar|vector"),
            "C2": ("Synaptic contacts (pyr→inh)", "scalar|vector"),
            "C3": ("Synaptic contacts (inh→pyr)", "scalar|vector"),
            "nstart": ("Start index for recording", "-"),
            "record_step": ("Recording step interval", "scalar"),
            "sti_ti": ("Stimulation start time", "scalar"),
            "sti_duration": ("Stimulation duration", "scalar"),
            "sti_amplitude": ("Stimulation amplitude per node", "scalar|vector"),
            "sti_gain": ("Stimulation gain factor", "scalar"),
            "noise_seed": ("Seed for noise generation", "scalar"),
            "seed": ("Random seed for reproducibility", "-"),
            "initial_state": ("Initial state of the system", "vector"),
            "method": ("Integration method", "string"),
            "output": ("Output directory", "string"),
            "t_end": ("End time of simulation", "scalar"),
            "t_transition": ("Transition time", "scalar"),
        }

    # -------------------------------------------------------------------------

    def prepare_stimulus(self, sti_gain, sti_ti):
        """
        prepare stimulation parameteres
        """
        if np.abs(sti_gain) > 0.0:
            assert (
                sti_ti >= self.t_transition
            ), "stimulation must start after transition"

    # -------------------------------------------------------------------------

    def set_initial_state(self):
        """
        set initial state for the system of JR equations with N nodes.
        """
        self.initial_state = set_initial_state(self.num_nodes, self.seed)
        self.INITIAL_STATE_SET = True

    # -------------------------------------------------------------------------

    # def set_C(self, label, val_dict):
    #     indices = val_dict['indices']

    #     if indices is None:
    #         indices = [list(range(self.N))]

    #     values = val_dict['value']
    #     if isinstance(values, np.ndarray):
    #         values = values.tolist()
    #     if not isinstance(values, list):
    #         values = [values]

    #     assert (len(indices) == len(values))
    #     C = getattr(self, label)

    #     for i in range(len(values)):
    #         C[indices[i]] = values[i]
    # -------------------------------------------------------------------------

    def prepare_input(self):
        """
        prepare input parameters for C++ code.
        """

        self.dt = float(self.dt)
        self.t_transition = float(self.t_transition)
        self.t_end = float(self.t_end)
        self.G = float(self.G)
        self.A = float(self.A)
        self.B = float(self.B)
        self.a = float(self.a)
        self.b = float(self.b)
        self.r = float(self.r)
        self.v0 = float(self.v0)
        self.vmax = float(self.vmax)
        self.C0 = np.asarray(self.C0)
        self.C1 = np.asarray(self.C1)
        self.C2 = np.asarray(self.C2)
        self.C3 = np.asarray(self.C3)
        self.sti_amplitude = np.asarray(self.sti_amplitude)
        self.sti_gain = float(self.sti_gain)
        self.sti_ti = float(self.sti_ti)
        self.sti_duration = float(self.sti_duration)
        self.mu = float(self.mu)
        self.sigma = float(self.sigma)
        self.noise_seed = int(self.noise_seed)
        self.initial_state = np.asarray(self.initial_state)
        self.weights = np.asarray(self.weights)
        self.delays = np.asarray(self.delays)

    # -------------------------------------------------------------------------

    def run(self, par: dict = {}, x0: np.ndarray = None, verbose: bool = False):
        """
        Integrate the system of equations for Jansen-Rit model.
        """
        if not _JR_SDDE_AVAILABLE:
            raise RuntimeError(
                "JR_sdde C++ module is not available. "
                "Please compile C++ extensions using 'python setup.py build_ext --inplace' "
                "or use Python/Numba JR implementations instead."
            )

        if x0 is None:
            if not self.INITIAL_STATE_SET:
                self.set_initial_state()
                if verbose:
                    print("initial state set by default")
        else:
            assert len(x0) == self.num_nodes * self.dim
            self.initial_state = x0
            self.INITIAL_STATE_SET = True

        for key in par.keys():
            if key not in self.valid_params:
                raise ValueError("Invalid parameter: " + key)
            setattr(self, key, par[key])

        self.prepare_input()
        obj = _JR_sdde(
            self.dt,
            self.initial_state,
            self.weights,
            self.delays,
            self.G,
            self.dim,
            self.A,
            self.B,
            self.a,
            self.b,
            self.r,
            self.v0,
            self.vmax,
            self.C0,
            self.C1,
            self.C2,
            self.C3,
            self.sti_amplitude,
            self.sti_gain,
            self.sti_ti,
            self.sti_duration,
            self.mu,
            self.sigma,
            self.t_transition,
            self.t_end,
            self.noise_seed,
        )
        obj.integrate(self.method)
        nstart = int((np.max(self.delays)) / self.dt) + 1
        t = np.asarray(obj.get_t())[:-nstart]
        y = np.asarray(obj.get_y())[:, :-nstart]
        sti_vector = np.asarray(obj.get_sti_vector())[:-nstart]

        return {"t": t, "x": y, "sti": sti_vector}


############################# helper functions ################################


def check_sequence(x, n):
    """
    check if x is a scalar or a sequence of length n

    parameters
    ----------
    x: scalar or sequence of length n
    n: number of nodes

    returns
    -------
    x: sequence of length n
    """
    if isinstance(x, (np.ndarray, list, tuple)):
        assert len(x) == n, f" variable must be a sequence of length {n}"
        return x
    else:
        return x * np.ones(n)


def set_initial_state(nn, seed=None):
    """
    set initial state for the system of JR equations with N nodes.

    parameters
    ----------
    nn: number of nodes
    seed: random seed

    returns
    -------
    y: initial state of length 6N

    """
    if seed is not None:
        np.random.seed(seed)

    y0 = np.random.uniform(-1, 1, nn)
    y1 = np.random.uniform(-500, 500, nn)
    y2 = np.random.uniform(-50, 50, nn)
    y3 = np.random.uniform(-6, 6, nn)
    y4 = np.random.uniform(-20, 20, nn)
    y5 = np.random.uniform(-500, 500, nn)

    return np.hstack((y0, y1, y2, y3, y4, y5))
