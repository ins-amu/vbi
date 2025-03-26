import os
import numpy as np
from copy import deepcopy
from os.path import join
from typing import Union

try:
    from vbi.models.cpp._src.vep import VEP as _VEP
except ImportError as e:
    print(f"Could not import modules: {e}, probably C++ code is not compiled or properly linked.")


class VEP:
    """
    Virtual Epileptic Patient (VEP) model
    """

    def __init__(self, par: dict = {}):

        par = deepcopy(par)
        self._par = self.get_default_parameters()
        self.valid_params = list(self._par.keys())
        self.check_parameters(par)
        self._par.update(par)

        for item in self._par.items():
            setattr(self, item[0], item[1])

        if self.seed is not None:
            np.random.seed(self.seed)
        
        self.INITIAL_STATE_SET = False
        if self.initial_state is not None:
            self.INITIAL_STATE_SET = True
        

    def set_initial_state(self):
        self.nn = self.weights.shape[0]
        self.initial_state = set_initial_state(self.nn, self.seed)
        self.INITIAL_STATE_SET = True

    def __str__(self) -> str:
        print("VEP model")
        print("---------")
        for item in self._par.items():
            print(f"{item[0]} = {item[1]}")
        return ""

    def __call__(self):
        return self._par

    def check_parameters(self, par: dict):
        for key in par.keys():
            if key not in self.valid_params:
                raise ValueError(f"Invalid parameter: {key}")

    def prepare_input(self):
        self.nn = self.weights.shape[0]
        self.iext = check_sequence(self.iext, self.nn)
        self.tau = float(self.tau)
        self.eta = check_sequence(self.eta, self.nn)
        self.sigma = float(self.noise_sigma)
        self.dt = float(self.dt)
        self.tend = float(self.tend)
        self.tcut = float(self.tcut)
        self.noise_seed = int(self.noise_seed)
        self.record_step = int(self.record_step)
        self.method = str(self.method)

    def get_default_parameters(self):
        params = {
            "G": 1.0,
            "seed": None,
            "initial_state": None,
            "weights": None,
            "tau": 10.0,
            "eta": -1.5,
            "noise_sigma": 0.1,
            "iext": 0.0,
            "dt": 0.01,
            "tend": 100.0,
            "tcut": 0.0,
            "noise_seed": 0,
            "record_step": 1,
            "method": "euler",
            "output": "output",
        }
        return params

    def run(self, par: dict = {}, x0: np.ndarray = None, verbose: bool = False):

        if x0 is None:
            if not self.INITIAL_STATE_SET:
                self.set_initial_state()
        else:
            self.initial_state = x0
            self.INITIAL_STATE_SET = True
        for key in par.keys():
            if key not in self.valid_params:
                raise ValueError(f"Invalid parameter: {key}")
            setattr(self, key, par[key])
        self.prepare_input()

        obj = _VEP(
            self.G,
            self.iext,
            self.eta,
            self.dt,
            self.tcut,
            self.tend,
            self.tau,
            self.noise_sigma,
            self.initial_state,
            self.weights,
            self.noise_seed,
            self.method,
        )
        obj.integrate()
        states = np.asarray(obj.get_states(), dtype=np.float32).T
        t = np.asarray(obj.get_times())
        return {"t": t, "x": states}


def set_initial_state(nn: int, seed: int = None):
    if seed is not None:
        np.random.seed(seed)
        x0 = np.zeros(2 * nn)
        x0[:nn] = np.random.uniform(-3.0, -2.0, nn)
        x0[nn:] = np.random.uniform(0.0, 3.5, nn)
    return x0


def check_sequence(x: Union[int, float, np.ndarray], n: int):
    """
    check if x is a scalar or a sequence of length n

    parameters
    ----------
    x: scalar or sequence
    n: number of elements

    returns
    -------
    x: sequence of length n
    """
    if isinstance(x, (np.ndarray, list, tuple)):
        assert len(x) == n, f" variable must be a sequence of length {n}"
        return x
    else:
        return x * np.ones(n)
