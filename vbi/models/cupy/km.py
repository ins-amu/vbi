import tqdm
import cupy as cp
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from numpy.matlib import repmat
from vbi.models.cupy.utils import *


class KM_sde:

    valid_parameters = [
        "num_sim",        # number of simulations
        "G",              # global coupling strength
        "dt",             # time step
        "noise_amp",      # noise amplitude
        "omega",          # natural angular frequency
        "weights",        # weighted connection matrix
        "seed",
        "alpha",          # frustration matrix
        "t_initial",      # initial time
        "t_transition",   # transition time
        "t_end",          # end time
        "output",         # output directory
        "num_threads",    # number of threads using openmp
        "initial_state",
        "type",           # output times series data type
        "engine",         # cpu or gpu
    ]

    def __init__(self, par={}) -> None:

        self.check_parameters(par)
        self._par = self.get_default_parameters()
        self._par.update(par)

        for item in self._par.items():
            name = item[0]
            value = item[1]
            setattr(self, name, value)

        self.xp = get_module(self.engine)
        self.ns = self.num_sim

        assert (self.weights is not None), "weights must be provided"
        assert (self.omega is not None), "omega must be provided"

        self.nn = self.num_nodes = self.weights.shape[0]

        if self.seed is not None:
            self.xp.random.seed(self.seed)

        if self.initial_state is None:
            self.INITIAL_STATE_SET = False

    def set_initial_state(self):
        self.INITIAL_STATE_SET = True
        self.initial_state = set_initial_state(
            self.nn, self.ns, self.xp, self.seed)

    def __str__(self) -> str:
        return f"Kuramoto model with noise (sde), {self.engine} implementation."

    def __call__(self):
        print(
            f"Kuramoto model with noise (sde), {self.engine} implementation.")
        return self._par

    def get_default_parameters(self):

        return {
            "G": 1.0,                        # global coupling strength
            "dt": 0.01,                      # time step
            "noise_amp": 0.1,                # noise amplitude
            "weights": None,                 # weighted connection matrix
            "omega": None,                   # natural angular frequency
            "seed": None,                    # fix random seed for initial state
            "t_initial": 0.0,                # initial time
            "t_transition": 0.0,             # transition time
            "t_end": 100.0,                  # end time
            "num_threads": 1,                # number of threads using openmp
            "output": "output",              # output directory
            "initial_state": None,           # initial state
            "engine": "cpu",                 # cpu or gpu
            "type": np.float32,              # output times series data type
            "alpha": None,                   # frustration matrix
            "num_sim": 1,                    # number of simulations
            "method": "heun",                # integration method

        }

    def check_parameters(self, par):
        for key in par.keys():
            if key not in self.valid_parameters:
                raise ValueError(f"Invalid parameter: {key}")

    def prepare_input(self):

        self.weights = self.xp.array(
            self.weights.reshape(self.weights.shape+(1,)))
        self.omega = prepare_vec(self.omega, self.ns, self.engine)
        self.weights = move_data(self.weights, self.engine)

    def f_sys(self, x, t):
        return self.omega + self.G * self.xp.sum(self.weights * self.xp.sin(x - x[:, None]), axis=1)

    def euler(self, x, t):
        ''' Euler's method integration'''
        coef = self.xp.sqrt(self.dt)
        dW = self.xp.random.normal(0, self.noise_amp, size=x.shape)
        return x + self.dt * self.f_sys(x, t) + coef * dW

    def heun(self, x, t):
        ''' Heun's method integration'''
        coef = self.xp.sqrt(self.dt)
        dW = self.xp.random.normal(0, self.noise_amp, size=x.shape)
        k1 = self.f_sys(x, t) * self.dt
        tmp = x + k1 + coef * dW
        k2 = self.f_sys(tmp, t + self.dt) * self.dt
        return x + 0.5 * (k1 + k2) + coef * dW

    def integrate(self, t):
        ''' Integrate the model'''
        x = self.initial_state
        xs = []
        integrator = self.euler if self.method == "euler" else self.heun
        n_transition = int(self.t_transition /
                           self.dt) if self.t_transition > 0 else 1

        for it in tqdm.tqdm(range(1, len(t))):
            x = integrator(x, t[it])
            if it >= n_transition:
                if self.engine == "gpu":
                    xs.append(x.get())
                else:
                    xs.append(x)
        xs = np.asarray(xs).astype(self.type)
        t = t[n_transition:]

        return {"t": t, "x": xs}

    def run(self, par={}, x0=None, verbose=False):
        '''
        run the model

        Parameters
        ----------
        par: dict
            parameters
        x0: array
            initial state
        verbose: bool
            print progress bar

        Returns
        -------
        dict
            x: array
                time series data
            t: array
                time points

        '''

        if x0 is not None:
            self.initial_state = x0
            self.INITIAL_STATE_SET = True
        else:
            self.set_initial_state()
            if verbose:
                print("Initial state set randomly.")
        # self.check_parameters(par)
        # for key in par.keys():
        #     setattr(self, key, par[key]['value'])
        self.prepare_input()
        t = self.xp.arange(self.t_initial, self.t_end, self.dt)
        data = self.integrate(t)
        return data


def set_initial_state(nn, ns=1, xp=np, seed=None):
    '''
    set initial state

    Parameters
    ----------

    nn: int
        number of nodes
    ns: int
        number of states
    xp: module
        numpy or cupy
    seed: int
        set random seed if not None

    Returns
    -------
    x: array
        initial state

    '''
    if seed is not None:
        xp.random.seed(seed)

    return xp.random.uniform(0, 2*np.pi, size=(nn, ns))


def prepare_vec(vec, ns, engine):
    '''
    repeat vector ns times

    '''
    vec = repmat(vec, ns, 1).T
    vec = move_data(vec, engine)
    return vec
