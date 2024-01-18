import os
import numpy as np
from os.path import join
from copy import copy, deepcopy
from vbi.models.cpp._src.sl_sdde import Stuart_Landau as _sl_sdde


class SL_sdde:
    '''
    Stuart-Landau model, C++ implementation.

    Parameters
    ----------
    par : dict
        Dictionary of parameters.

    '''

    valid_parameters = [
        "G",              # global coupling strength
        'a',              # biforcation parameter
        "dt",             # time step [s]
        "sigma_r",        # noise strength
        "sigma_v",        # noise strength
        "omega",          # natural angular frequency [Hz]
        "noise_seed",       # fix random seed for noise in Cpp code
        "seed",
        "velocity",       # velocity        [m/s]
        "t_initial",      # initial time    [s]
        "t_transition",   # transition time [s]
        "t_end",        # end time        [s]
        "method",         # integration method
        "weights",            # weighted connection matrix
        "distances",      # distance matrix [m]
        "initial_state",  # initial state
        "record_step",    # sampling every n step from time series
        "data_path",      # output directory
        "RECORD_TS",      # true to store large time series in file
        "verbose"         # true to print more information
    ]

    def __init__(self, par) -> None:

        self.check_parameters(par)
        self._par = self.get_default_parameters()
        self._par.update(par)

        for item in self._par.items():
            name = item[0]
            value = item[1]
            setattr(self, name, value)

        assert (self.weights is not None)
        assert (self.distances is not None)
        assert (self.omega is not None)

        self.delays = self.distances / self.velocity
        self.num_nodes = self.weights.shape[0]

        if self.seed is not None:
            np.random.seed(self.seed)

        if self.initial_state is None:
            self.INITIAL_STATE_SET = False

    def set_initial_state(self):
        self.initial_state = set_initial_state(self.num_nodes, 0.01, self.seed)
        self.INITIAL_STATE_SET = True

    def __str__(self) -> str:
        print(f"Stuart-Landau model.")
        print("----------------")
        for item in self._par.items():
            name = item[0]
            value = item[1]
            print(f"{name} = {value}")
        return ""

    def __call__(self):
        print(f"Stuart-Landau model.")
        return self._par

    def check_parameters(self, par):
        for key in par.keys():
            if key not in self.valid_parameters:
                raise ValueError(
                    f"Invalid parameter {key} for Stuart-Landau model.")

    def get_default_parameters(self):

        params = {
            "G": 1000.0,
            "a": -5.0,
            "dt": 1e-4,
            'sigma_r': 1e-4,
            'sigma_v': 1e-4,
            'omega': None,
            "noise_seed": 0,
            "seed": None,
            "velocity": 6.0,
            "t_initial": 0.0,
            "t_transition": 2.0,
            "t_end": 10.0,
            "method": "euler",
            "weights": None,
            "distances": None,
            "initial_state": None,
            "record_step": 1,
            "data_path": "output",
            "RECORD_TS": 0,
        }

        return params

    def prepare_input(self):
        self.dt = float(self.dt)
        self.t_initial = float(self.t_initial)
        self.t_transition = float(self.t_transition)
        self.t_end = float(self.t_end)
        self.record_step = int(self.record_step)
        self.noise_seed = int(self.noise_seed)
        self.num_nodes = int(self.num_nodes)
        self.omega = np.asarray(self.omega)
        self.weights = np.asarray(self.weights)
        self.delays = np.asarray(self.delays)
        self.G = float(self.G)
        self.a = float(self.a)
        self.sigma_r = float(self.sigma_r)
        self.sigma_v = float(self.sigma_v)
        assert (self.omega.shape[0] == self.num_nodes)


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
        x : array
            State time series.
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
            if key not in self.valid_parameters:
                raise ValueError(f"Invalid parameter {key:s} provided.")
            else:
                setattr(self, key, par[key]['value'])
        self.prepare_input()

        obj = _sl_sdde(self.dt,
                       self.initial_state,
                       self.weights,
                       self.delays,
                       self.G,
                       self.a,
                       self.omega,
                       self.sigma_r,
                       self.sigma_v,
                       self.t_initial,
                       self.t_transition,
                       self.t_end,
                       self.noise_seed)

        if self.method == 'euler':
            y = obj.integrate_euler()
        elif self.method == 'heun':
            y = obj.integrate_heun()
        else:
            raise ValueError(
                f"Invalid integration method {self.method} provided.")

        t = np.asarray(obj.get_time())
        y = np.asarray(y).astype(np.float32)
        index_trans = np.where(t >= self.t_transition)[0][0]
        nstart = int(np.max(self.delays) / self.dt)
        nn = self.num_nodes
        t = t[nstart+1+index_trans::self.record_step]
        y = y[:nn, nstart+1+index_trans::self.record_step]

        return {"t": t, "x": y}


def set_initial_state(nn, amp=0.01, seed=None):
    if seed is not None:
        np.random.seed(seed)
    return np.random.randn(2 * nn) * amp
