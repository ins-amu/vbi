import os
import numpy as np
from os.path import join

try:
    from vbi.models.cpp._src.jr_sde import JR_sde as _JR_sde
    from vbi.models.cpp._src.jr_sdde import JR_sdde as _JR_sdde
except ImportError as e:
    print(f"Could not import modules: {e}, probably C++ code is not compiled or properly linked.")


class JR_sde:
    '''
    Jansen-Rit model C++ implementation.

    Parameters
    ----------

    par: dict
        Including the following:
        - **A** : [mV] determine the maximum amplitude of the excitatory PSP (EPSP)
        - **B** : [mV] determine the maximum amplitude of the inhibitory PSP (IPSP)
        - **a** : [Hz]  1/tau_e,  :math:`\sum` of the reciprocal of the time constant of passive membrane and all other spatially distributed  delays in the dendritic network
        - **b** : [Hz] 1/tau_i
        - **r**  [mV] the steepness of the sigmoidal transformation.
        - **v0** parameter of nonlinear sigmoid function
        - **vmax** parameter of nonlinear sigmoid function
        - **C_i** [list or np.array] average number of synaptic contacts in th inhibitory and excitatory feedback loops
        - **noise_amp**
        - **noise_std**

        - **dt** [second] integration time step
        - **t_initial** [s] initial time
        - **t_end** [s] final time
        - **method** [str] method of integration
        - **t_transition** [s] time to reach steady state
        - **dim** [int] dimention of the system

    '''
    valid_params = [
        "noise_seed", "seed", "G", "weights", "A", "B", "a", "b",
        "noise_mu", "noise_std", "vmax", "v0", "r",
        "C0", "C1", "C2", "C3", "dt", "method", "t_transition",
        "t_end", "control", "output", "RECORD_AVG",
        "initial_state"
    ]

    def __init__(self, par={}):

        self.check_parameters(par)
        self._par = self.get_default_parameters()
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

        # self.C0 = self.C0 * np.ones(self.N)
        # self.C1 = self.C1 * np.ones(self.N)
        # self.C2 = self.C2 * np.ones(self.N)
        # self.C3 = self.C3 * np.ones(self.N)
        self.noise_seed = 1 if self.noise_seed else 0
        os.makedirs(join(self.output), exist_ok=True)

    def __str__(self) -> str:
        print("Jansen-Rit sde model")
        print("----------------")
        for item in self._par.items():
            name = item[0]
            value = item[1]
            print(f"{name} = {value}")
        return ""

    def __call__(self):
        print("Jansen-Rit sde model")
        return self._par

    def check_parameters(self, par):
        '''
        Check if the parameters are valid.
        '''
        for key in par.keys():
            if key not in self.valid_params:
                raise ValueError("Invalid parameter: " + key)

    def get_default_parameters(self):
        '''
        return default parameters for the Jansen-Rit sde model.
        '''

        par = {
            'G': 0.5,                   # global coupling strength
            "A": 3.25,                  # mV
            "B": 22.0,                  # mV
            "a": 0.1,                   # 1/ms
            "b": 0.05,                  # 1/ms
            "noise_mu": 0.24,
            "noise_std": 0.3,
            "vmax": 0.005,
            "v0": 6,                    # mV
            "r": 0.56,                  # mV
            "initial_state": None,

            'weights': None,
            "C0": 135.0 * 1.0,
            "C1": 135.0 * 0.8,
            "C2": 135.0 * 0.25,
            "C3": 135.0 * 0.25,

            "noise_seed": 0,
            "seed": None,

            "dt": 0.05,                 # ms
            "dim": 6,
            "method": "heun",
            "t_transition": 500.0,      # ms
            "t_end": 2501.0,            # ms
            "output": "output",         # output directory
            "RECORD_AVG": False         # true to store large time series in file
        }
        return par

    # ---------------------------------------------------------------
    def set_initial_state(self):
        '''
        Set initial state for the system of JR equations with N nodes.
        '''

        self.initial_state = set_initial_state(self.num_nodes, self.seed)
        self.INITIAL_STATE_SET = True

    # -------------------------------------------------------------------------

    # def set_C(self, label, val_dict):
    #     '''
    #     set the value of C0, C1, C2, C3.

    #     Parameters
    #     ----------
    #     label: str
    #         C0, C1, C2, C3
    #     val_dict: dict
    #         {'indices': [list or multiple list seperated with comma],
    #          'value': [list or multiple list seperated with comma]}

    #     '''
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

    def prepare_input(self):
        '''
        prepare input parameters for passing to C++ engine.
        '''

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
        self.noise_std = float(self.noise_std)
        self.noise_seed = int(self.noise_seed)
        self.initial_state = np.asarray(self.initial_state)

    # -------------------------------------------------------------------------
    def run(self, par={}, x0=None, verbose=False):
        '''
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

        '''

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

        obj = _JR_sde(self.N,
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
                      self.noise_std,
                      self.noise_seed)

        if self.method == 'euler':
            obj.eulerIntegrate()
        elif self.method == 'heun':
            obj.heunIntegrate()
        else:
            print("unkown integratiom method")
            exit(0)

        sol = np.asarray(obj.get_coordinates()).T
        times = np.asarray(obj.get_times())

        del obj

        return {"t": times, "x": sol}


############################ Jansen-Rit sdde ##################################

class JR_sdde:
    pass

    valid_params = ["weights", "delays", "dt", "t_end", "G", "A", "a", "B", "b", "mu",
                    "nstart", "t_end", "t_transition", "sigma", "C", "record_step",
                    "C0", "C1", "C2", "C3", "vmax", "r", "v0", "output",
                    'sti_ti', 'sti_duration', 'sti_amplitude', 'sti_gain',
                    "noise_seed", "seed", "method"]
    # -------------------------------------------------------------------------

    def __init__(self, par={}) -> None:

        self.check_parameters(par)
        _par = self.get_default_parameters()
        _par.update(par)

        for item in _par.items():
            setattr(self, item[0], item[1])

        if self.seed is not None:
            np.random.seed(self.seed)

        self.noise_seed = 1 if self.noise_seed else 0
        assert (self.weights is not None), "weights must be provided"
        assert (self.delays is not None), "delays must be provided"
        self.N = self.num_nodes = len(self.weights)

        self.C0 = check_sequence(self.C0, self.N)
        self.C1 = check_sequence(self.C1, self.N)
        self.C2 = check_sequence(self.C2, self.N)
        self.C3 = check_sequence(self.C3, self.N)
        self.sti_amplitude = check_sequence(self.sti_amplitude, self.N)

        if self.initial_state is None:
            self.INITIAL_STATE_SET = False
        os.makedirs(join(self.output), exist_ok=True)

    def check_parameters(self, par):
        '''
        check if the parameters are valid
        '''
        for key in par.keys():
            if key not in self.valid_params:
                raise ValueError("Invalid parameter: " + key)
    # -------------------------------------------------------------------------

    def get_default_parameters(self):
        '''
        get default parameters for the system of JR equations.
        '''

        param = {
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
            "C0": 135.0 * 1.0,
            "C1": 135.0 * 0.8,
            "C2": 135.0 * 0.25,
            "C3": 135.0 * 0.25,
            'sti_ti': 0.0,
            'sti_duration': 0.0,
            'sti_amplitude': 0.0,  # scalar or sequence of length N
            'sti_gain': 0.0,
            "noise_seed": False,
            "seed": None,
            "initial_state": None,
            "method": "heun",
            "output": "output",
            "t_end": 2000.0,
            "t_transition": 1000.0
        }

        return param
    # -------------------------------------------------------------------------

    def prepare_stimulus(self, sti_gain, sti_ti):
        '''
        prepare stimulation parameteres
        '''
        if np.abs(sti_gain) > 0.0:
            assert (
                sti_ti >= self.t_transition), "stimulation must start after transition"
    # -------------------------------------------------------------------------

    def set_initial_state(self):
        '''
        set initial state for the system of JR equations with N nodes.
        '''
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
        '''
        prepare input parameters for C++ code.
        '''

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

    def run(self, par={}, x0=None, verbose=False):
        '''
        Integrate the system of equations for Jansen-Rit model.
        '''

        if x0 is None:
            if not self.INITIAL_STATE_SET:
                self.set_initial_state()
                if verbose:
                    print("initial state set by default")
        else:
            assert (len(x0) == self.num_nodes * self.dim)
            self.initial_state = x0
            self.INITIAL_STATE_SET = True

        for key in par.keys():
            if key not in self.valid_params:
                raise ValueError("Invalid parameter: " + key)
            # if key in ["C0", "C1", "C2", "C3"]:
            #     self.set_C(key, par[key])
            # else:
            setattr(self, key, par[key])

        self.prepare_input()
        obj = _JR_sdde(self.dt,
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
                       self.noise_seed)
        obj.integrate(self.method)
        nstart = int((np.max(self.delays)) / self.dt) + 1
        t = np.asarray(obj.get_t())[:-nstart]
        y = np.asarray(obj.get_y())[:, :-nstart]
        sti_vector = np.asarray(obj.get_sti_vector())[:-nstart]

        return {"t": t, "x": y, "sti": sti_vector}

############################# helper functions ################################


def check_sequence(x, n):
    '''
    check if x is a scalar or a sequence of length n

    parameters
    ----------
    x: scalar or sequence of length n
    n: number of nodes

    returns
    -------
    x: sequence of length n
    '''
    if isinstance(x, (np.ndarray, list, tuple)):
        assert (len(x) == n), f" variable must be a sequence of length {n}"
        return x
    else:
        return x * np.ones(n)


def set_initial_state(nn, seed=None):
    '''
    set initial state for the system of JR equations with N nodes.

    parameters
    ----------
    nn: number of nodes
    seed: random seed

    returns
    -------
    y: initial state of length 6N

    '''
    if seed is not None:
        np.random.seed(seed)

    y0 = np.random.uniform(-1, 1, nn)
    y1 = np.random.uniform(-500, 500, nn)
    y2 = np.random.uniform(-50, 50, nn)
    y3 = np.random.uniform(-6, 6, nn)
    y4 = np.random.uniform(-20, 20, nn)
    y5 = np.random.uniform(-500, 500, nn)

    return np.hstack((y0, y1, y2, y3, y4, y5))
