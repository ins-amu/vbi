
import os
from typing import Any
import tqdm
import numpy as np
from os.path import join
from copy import copy, deepcopy
from vbi.models.cpp._src.mpr_sde import MPR_sde as _MPR_sde


class MPR_sde:
    '''
    Montbrio-Pazo-Roxin model C++ implementation.

    Parameters
    ----------
    par : dict
        Dictionary of parameters.


    '''

    valid_parameters = [
        'weights', 'eta',
        'noise_seed', 'seed', 'J', 'tau', 'delta',
        'iapp', 'sti_indices', 'square_wave_stimulation',
        'dt', 'dt_bold', 'G', 'noise_amp', 'decimate',
        'RECORD_AVG', 'alpha_sc', 'beta_sc', 'weights_A', 'weights_B',
        't_initial', 't_transition', 't_end', 'record_step',
        'output']

    def __init__(self, par) -> None:

        self.check_parameters(par)
        self._par = self.get_default_parameters()
        self._par.update(par)

        for item in self._par.items():
            name = item[0]
            value = item[1]
            setattr(self, name, value)

        self.num_nodes = self.weights.shape[0]

        if self.seed is not None:
            np.random.seed(self.seed)

        if self.initial_state is None:
            self.INITIAL_STATE_SET = False
    # -------------------------------------------------------------------------
            
    def set_initial_state(self):
        return set_initial_state(self.num_nodes, self.seed)

    def __str__(self) -> str:
        return f"MPR sde model."
    # -------------------------------------------------------------------------

    def __call__(self):
        return self._par
    # -------------------------------------------------------------------------

    def check_parameters(self, par):
        for key in par.keys():
            if key not in self.valid_parameters:
                raise ValueError(f"Invalid parameter {key:s} provided.")

    def get_default_parameters(self):

        params = {
            "G": 0.733,                         # global coupling strength
            "dt": 0.01,                         # for mpr model [ms]
            "dt_bold": 0.001,                   # for Balloon model [s]
            "J": 14.5,                          # model parameter
            "eta": -4.6,                        # model parameter
            "tau": 1.0,                         # model parameter
            "delta": 0.7,                       # model parameter
            "decimate": 500,                    # sampling from mpr time series

            "noise_amp": 0.037,                 # amplitude of noise
            "noise_seed": 0,                    # fix seed for noise
            "iapp": 0.0,                        # constant applyed current
            "seed": None,
            "square_wave_stimulation": 0,
            "sti_indices": [],                  # indices of stimulated nodes


            "alpha_sc": 0.0,                    # interhemispheric mask gain
            "beta_sc": 0.0,                     # region mask gain

            "initial_state": None,              # initial condition of the system

            "t_initial": 0.0,                   # initial time * 10[ms]
            "t_transition": 10_000.0,           # transition time [ms]
            "t_end": 250_000.0,                 # end time  [ms]
            "weights": None,                    # weighted connection matrix
            "weights_A": None,                  # weighted connection matrix
            "weights_B": None,                  # weighted connection matrix
            "record_step": 10,                  # sampling every n step from mpr time series
            "data_path": "output",              # output directory
            "RECORD_AVG": 0                     # true to store large time series in file
        }

        return params
    
    def set_eta(self, key, val_dict):
        indices = val_dict['indices']

        if indices is None:
            indices = [list(range(self.num_nodes))]
        values = val_dict['value']
        if isinstance(values, np.ndarray):
            values = values.tolist()
        if not isinstance(values, list):
            values = [values]
        
        assert(len(indices) == len(values))
        eta = getattr(self, key)
        for i in range(len(values)):
            eta[indices[i]] = values[i]

    # -------------------------------------------------------------------------

    def prepare_input(self):
        ''' 
        Prepare input parameters for passing to C++ engine.
        '''

        self.dt = float(self.dt)
        self.dt_bold = float(self.dt_bold)
        self.decimate = int(self.decimate)
        self.initial_state = np.asarray(self.initial_state).astype(np.float64)
        self.weights = np.asarray(self.weights).astype(np.float64)
        if self.weights_A is None:
            self.weights_A = np.zeros_like(self.weights)
        self.weights_A = np.asarray(self.weights_A).astype(np.float64)
        if self.weights_B is None:
            self.weights_B = np.zeros_like(self.weights)
        self.weights_B = np.asarray(self.weights_B).astype(np.float64)
        self.G = float(self.G)
        self.eta = check_sequence(self.eta, self.num_nodes) # check eta be array-like
        self.eta = np.asarray(self.eta).astype(np.float64)
        self.alpha_sc = float(self.alpha_sc)
        self.beta_sc = float(self.beta_sc)
        #!TODO check sti_indices be array-like
        #!TODO add sti_ti for initial time of stimulation
        #!TODO add sti_duration for duration of stimulation
        #!TODO add sti_amp for amplitude of stimulation as an array of shape (nn,)
        #!TODO add sti_gain for grain of stimulation as scalar
        self.J = float(self.J)
        self.tau = float(self.tau)
        self.delta = float(self.delta)
        self.iapp = float(self.iapp)
        self.square_wave_stimulation = int(self.square_wave_stimulation)
        self.noise_amp = float(self.noise_amp)
        self.record_step = int(self.record_step)
        self.t_initial = float(self.t_initial) / 10.0 
        self.t_transition = float(self.t_transition) / 10.0
        self.t_end = float(self.t_end) / 10.0
        self.RECORD_AVG = int(self.RECORD_AVG)
        self.noise_seed = int(self.noise_seed)
        
    # -------------------------------------------------------------------------

    def simulate(self, par={}, x0=None, verbose=False):
        '''
        Integrate the MPR model with the given parameters.

        Parameters
        ----------
        par : dict
            Dictionary of parameters.
        x0 : array_like
            Initial condition of the system.
        verbose : bool
            If True, print the progress of the simulation.

        Returns
        -------
        bold : array_like
            Simulated BOLD signal.
        '''

        if x0 is None:
            if not self.INITIAL_STATE_SET:
                self.initial_state = self.set_initial_state()
                self.INITIAL_STATE_SET = True
                if verbose:
                    print("initial state set by default")
        else:
            assert (len(x0) == self.num_nodes * self.dim)
            self.initial_state = x0
            self.INITIAL_STATE_SET = True

        for key in par.keys():
            if key not in self.valid_parameters:
                raise ValueError(f"Invalid parameter {key:s} provided.")
            if key in ['eta']:
                self.set_eta(key, par[key])
            else:
                setattr(self, key, par[key]['value'])

        self.prepare_input()

        obj = _MPR_sde(self.dt,
                       self.dt_bold,
                       self.decimate,
                       self.initial_state,
                       self.weights,
                       self.weights_A,
                       self.weights_B,
                       self.G,
                       self.eta,
                       self.alpha_sc,
                       self.beta_sc,
                       [int(i) for i in self.sti_indices],
                       self.J,
                       self.tau,
                       self.delta,
                       self.iapp,
                       int(self.square_wave_stimulation),
                       self.noise_amp,
                       self.record_step,
                       self.t_initial,
                       0.0, #self.t_transition,
                       self.t_end,
                       self.RECORD_AVG,
                       self.noise_seed)

        obj.heunStochasticIntegrate()
        bold = np.asarray(obj.get_bold()).astype(np.float32).T
        t = np.asarray(obj.get_time())
        
        if bold.ndim == 2:
            bold = bold[:, t >= self.t_transition]
            t = t[t >= self.t_transition] * 10.0 # [ms]


        return {"t": t, "x": bold}


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

    if seed is not None:
        np.random.seed(seed)
        
    y0 = np.random.rand(2*nn)
    y0[:nn] = y0[:nn]*1.5
    y0[nn:] = y0[nn:]*4-2
    return y0
