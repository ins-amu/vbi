
import os
from typing import Any
import tqdm
import torch
import numpy as np
from os.path import join
from copy import copy, deepcopy
from vbi.models.cpp._src.mpr_sde import MPR_sde as _MPR_sde


class MPR_sde:
    valid_parameters = [
        'weights', 'eta', 
        'noise_seed', 'seed', 'J', 'tau', 'delta',
        'iapp', 'sti_indices', 'square_wave_stimulation', 
        'dt', 'dt_bold', 'G', 'noise_amp', 'decimate', 'RECORD_AVG', 'alpha_sc', 'beta_sc',
        't_initial', 't_transition', 't_end', 'record_step', 'output', 'weights_A', 'weights_B']

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
            
            "alpha_sc": 0.0,                    # interhemispheric mask gain
            "beta_sc": 0.0,                     # region mask gain

            "t_initial": 0.0,                   # initial time * 10[ms]
            "t_transition": 5000.0,             # transition time * 10 [ms]
            "t_end": 25_000.0,                # end time * 10 [ms]
            "weights": None,                    # weighted connection matrix
            "weights_A": None,                  # weighted connection matrix
            "weights_B": None,                  # weighted connection matrix
            "record_step": 10,                  # sampling every n step from mpr time series
            "data_path": "output",              # output directory
            "RECORD_AVG": 0                     # true to store large time series in file
        }

        return params

    # -------------------------------------------------------------------------

    def prepare_input(self):
        ''' 
        Prepare input parameters for passing to C++ engine.
        '''
        pass
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
        
        self.prepare_input()

        obj = _MPR_sde(self.dt,
                       self.dt_bold,
                       self.decimate,
                       self.initial_condition,
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
                       self.t_transition,
                       self.t_end,
                       self.RECORD_AVG,
                       self.noise_seed)

        obj.heunStochasticIntegrate(self.data_path)
        bold = np.asarray(obj.get_bold()).astype(np.float32).T
        t = np.asarray(obj.get_t())
        
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
