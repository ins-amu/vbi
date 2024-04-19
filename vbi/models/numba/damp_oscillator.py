import os
import numpy as np
from numba.experimental import jitclass
from numba import float64, types
from numba import njit

jit_spec = [('a', float64),
            ('b', float64),
            ('dt', float64),
            ('t_start', float64),
            ('t_end', float64),
            ('t_cut', float64),
            ('output', types.string),
            ('initial_state', float64[:]),
            ('method', types.string)
            ]

@jitclass(jit_spec)
class Param:
    def __init__(self,
                 a=0.1,
                 b=0.05,
                 dt=0.01,
                 t_start=0,
                 t_end=100.0,
                 t_cut=20,
                 output="output",
                 method="euler",
                 initial_state=np.array([0.5, 1.0])
                 ):
        self.a = a
        self.b = b
        self.dt = dt
        self.t_start = t_start
        self.t_end = t_end
        self.t_cut = t_cut
        self.output = output
        self.method = method
        self.initial_state = initial_state


@njit
def _f_sys(x, P):
    '''
    system function for damp oscillator model.
    '''
    a = P.a
    b = P.b
    return np.array([x[0] - x[0]*x[1] - a * x[0] * x[0],
                        x[0]*x[1] - x[1] - b * x[1] * x[1]])


@njit
def euler(x, P):
    '''
    euler integration for damp oscillator model.
    '''
    return x + P.dt * _f_sys(x, P)

@njit
def heun(x, P):
    '''
    heun integration for damp oscillator model.
    '''
    k0 = _f_sys(x, P)
    x1 = x + P.dt * k0
    k1 = _f_sys(x1, P)
    return x + 0.5 * P.dt * (k0 + k1)

@njit 
def rk4(x, P):
    '''
    runge-kutta integration for damp oscillator model.
    '''
    k1 = _f_sys(x, P)
    k2 = _f_sys(x + 0.5 * P.dt * k1, P)
    k3 = _f_sys(x + 0.5 * P.dt * k2, P)
    k4 = _f_sys(x + P.dt * k3, P)
    return x + P.dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0


@njit
def _integrate(x, P, intg=euler):

    t0 = np.arange(P.t_start, P.t_cut, P.dt)
    
    for i in range(len(t0)):
        x = intg(x, P)

    t = np.arange(P.t_cut, P.t_end, P.dt)
    x_out = np.zeros((len(t), len(x)))

    for i in range(len(t)):
        x = intg(x, P)
        x_out[i, :] = x
    return t, x_out


class DO_nb:
    '''
    Damper Oscillator model class.
    '''

    def __init__(self, par={}):

        self.valid_params = [jit_spec[i][0] for i in range(len(jit_spec))]
        self.check_parameters(par)
        self.P = self.get_parobj(par)

        self.P.output = "output" if self.P.output is None else self.P.output
        os.makedirs(self.P.output, exist_ok=True)

    def __str__(self) -> str:
        print("Damp Oscillator model")
        print("----------------")
        for key in self.valid_params:
            print(key, ": ", getattr(self.P, key))
        return ""

    def check_parameters(self, par):
        '''
        check if the parameters are valid.
        '''
        for key in par.keys():
            if key not in self.valid_params:
                raise ValueError("Invalid parameter: " + key)

    def get_parobj(self, par={}):
        '''
        return default parameters for damp oscillator model.
        '''
        if "initial_state" in par.keys():
            par["initial_state"] = np.array(par["initial_state"])

        parobj = Param(**par)

        return parobj
    
    def update_par(self, par={}):
        
        if par:
            self.check_parameters(par)
            for key in par.keys():
                setattr(self.P, key, par[key])

    def run(self, par={}, x0=None):

        self.update_par(par)
        if x0 is not None:
            assert len(x0) == 2, "Invalid initial state"
            self.P.initial_state = x0

        method = self.P.method 
        if method == "euler":
            intg = euler
        elif method == "heun":
            intg = heun
        elif method == "rk4":
            intg = rk4
        
        return _integrate(self.P.initial_state, self.P, intg=intg)
        