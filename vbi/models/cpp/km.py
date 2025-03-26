import numpy as np

try:
    from vbi.models.cpp._src.km_sde import KM_sde as _KM_sde
except ImportError as e:
    print(f"Could not import modules: {e}, probably C++ code is not compiled or properly linked.")


class KM_sde:
    '''
    Kuramoto model with noise (sde), C++ implementation.

    Parameters
    ----------
    par : dict
        Dictionary of parameters.

    '''

    valid_parameters = [
        "G",              # global coupling strength
        "dt",             # time step
        "noise_amp",      # noise amplitude
        "omega",          # natural angular frequency
        "weights",        # weighted connection matrix
        "noise_seed",     # fix random seed for noise in Cpp code
        "seed",
        "alpha",          # frustration matrix
        "t_initial",      # initial time
        "t_transition",   # transition time
        "t_end",          # end time
        "output",         # output directory
        "num_threads",    # number of threads using openmp
        "initial_state",
        "type"            # output times series data type
    ]

    def __init__(self, par) -> None:

        self.check_parameters(par)
        self._par = self.get_default_parameters()
        self._par.update(par)

        for item in self._par.items():
            name = item[0]
            value = item[1]
            setattr(self, name, value)

        assert (self.omega is not None)

        if self.seed is not None:
            np.random.seed(self.seed)

        self.num_nodes = len(self.omega)

        if self.initial_state is None:
            self.INITIAL_STATE_SET = False

    def set_initial_state(self):
        self.INITIAL_STATE_SET = True
        self.initial_state = set_initial_state(self.num_nodes, self.seed)

    def __str__(self) -> str:
        print("Kuramoto model with noise (sde), C++ implementation.")
        print("----------------")
        for item in self._par.items():
            name = item[0]
            value = item[1]
            print(f"{name} = {value}")
        return ""

    def __call__(self):
        return self._par

    def get_default_parameters(self):
        return {
            "G": 1.0,                        # global coupling strength
            "dt": 0.01,                      # time step
            "noise_amp": 0.1,                # noise amplitude
            "weights": None,                 # weighted connection matrix
            "alpha": None,                   # frustration matrix
            "omega": None,                   # natural angular frequency
            "noise_seed": 0,                 # fix random seed for noise in Cpp code
            "seed": None,                    # fix random seed for initial state
            "t_initial": 0.0,                # initial time
            "t_transition": 0.0,             # transition time
            "t_end": 100.0,                  # end time
            "num_threads": 1,                # number of threads using openmp
            "output": "output",              # output directory
            "initial_state": None,           # initial state
            "type": np.float32
        }

    def check_parameters(self, par):
        for key in par.keys():
            if key not in self.valid_parameters:
                raise ValueError(f"Invalid parameter: {key}")

    def prepare_input(self):

        nn = self.num_nodes
        if self.weights is None:
            raise ValueError("Missing weights.")
        if self.omega is None:
            raise ValueError("Missing omega.")
        if not self.INITIAL_STATE_SET:
            self.set_initial_state()

        self.weights = np.array(self.weights, dtype=np.float64)
        self.omega = np.array(self.omega, dtype=np.float64)
        self.initial_state = np.array(self.initial_state, dtype=np.float64)
        self.G = float(self.G)
        self.dt = float(self.dt)
        self.noise_amp = float(self.noise_amp)
        self.t_initial = float(self.t_initial)
        self.t_transition = float(self.t_transition)
        self.t_end = float(self.t_end)
        self.noise_seed = int(self.noise_seed)
        if self.alpha is None:
            self.alpha = np.zeros_like(self.weights, dtype=np.float64)
        else:
            self.alpha = np.array(self.alpha, dtype=np.float64)
            assert (self.alpha.shape == (nn, nn))

    def run(self, par={}, x0=None, verbose=False):
        '''
        Simulate the model.

        Parameters
        ----------
        par : dict
            Dictionary of parameters.
        x0 : array
            Initial state.
        verbose : bool
            Print simulation progress.

        Returns
        -------
        dict
            t : array
                Time points.
            x : array
                State time series.
        '''

        if x0 is None:
            if not self.INITIAL_STATE_SET:
                self.set_initial_state()
                if verbose:
                    print("initial state set by default")
        else:
            assert (len(x0) == self.num_nodes)
            self.initial_state = x0
            self.INITIAL_STATE_SET = True

        for key in par.keys():
            if key not in self.valid_parameters:
                raise ValueError(f"Invalid parameter {key:s} provided.")
            else:
                setattr(self, key, par[key]['value'])
        self.prepare_input()

        obj = _KM_sde(self.dt,
                      self.t_initial,
                      self.t_transition,
                      self.t_end,
                      self.G,
                      self.noise_amp,
                      self.initial_state,
                      self.omega,
                      self.alpha,
                      self.weights,
                      self.noise_seed,
                      self.num_threads
                      )
        obj.IntegrateHeun()
        t = np.asarray(obj.get_times())
        x = np.asarray(obj.get_theta()).T.astype(self.type)

        return {"t": t, "x": x}


def set_initial_state(num_nodes, seed=None):
    if seed is not None:
        np.random.seed(seed)
    return np.random.uniform(0, 2*np.pi, num_nodes)
