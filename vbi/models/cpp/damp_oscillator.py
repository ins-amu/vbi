import os
from typing import Any
import numpy as np

try:
    from vbi.models.cpp._src.do import DO as _DO
except ImportError as e:
    print(f"Could not import modules: {e}, probably C++ code is not compiled or properly linked.")

class DO:

    '''
    Damp Oscillator model class.
    '''

    valid_params = ["a", "b", "dt", "t_start", "t_end", "t_transition",
                    "initial_state", "method", "output"]

    # ---------------------------------------------------------------
    def __init__(self, par={}):
        '''
        Parameters
        ----------
        par : dictionary
            parameters which includes the following:
            - **dt** [double] time step.
            - **t_start** [double] initial time for simulation.
            - **t_end** [double] final time for simulation.
            - **initial_state** [list] initial state of the system.

        '''
        self.check_parameters(par)
        self._par = self.get_default_parameters()
        self._par.update(par)

        for item in self._par.items():
            name = item[0]
            value = item[1]
            setattr(self, name, value)

    def __str__(self) -> str:
        print("Damp Oscillator model")
        print("----------------")
        for item in self._par.items():
            name = item[0]
            value = item[1]
            print(f"{name} = {value}")
        return ""

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        print("Damp Oscillator model")
        return self._par

    def check_parameters(self, par):
        '''
        check if the parameters are valid.
        '''
        for key in par.keys():
            if key not in self.valid_params:
                raise ValueError("Invalid parameter: " + key)

    def get_default_parameters(self):
        '''
        return default parameters for damp oscillator model.
        '''

        params = {
            "a": 0.1,
            "b": 0.05,
            "dt": 0.01,
            "t_start": 0,
            "method": "rk4",
            "t_end": 100.0,
            "t_transition": 20,
            "output": "output",
            "initial_state": [0.5, 1.0],
        }

        return params

    def prepare_input(self):
        '''
        prepare input for cpp model.
        '''
        self.t_start = float(self.t_start)
        self.t_end = float(self.t_end)
        self.dt = float(self.dt)
        self.a = float(self.a)
        self.b = float(self.b)

        if self.output is None:
            self.output = "output"
        if not os.path.exists(self.output):
            os.makedirs(self.output)

        if self.initial_state is None:
            self.initial_state = [0.5, 1.0]
        self.initial_state = np.asarray(self.initial_state, dtype=np.float64)

    # ---------------------------------------------------------------
    def run(self, par={}, x0=None, verbose=False):
        '''
        Integrate the damp oscillator system of equations

        Parameters
        ----------
        par : dictionary
            parameters to control the model parameters.

        '''

        if x0 is not None:
            assert(len(x0) == 2)
            self.initial_state = x0

        self.check_parameters(par)
        for key in par.keys():
            setattr(self, key, par[key])

        self.prepare_input()

        obj = _DO(self.dt,
                  self.a,
                  self.b,
                  self.t_start,
                  self.t_end,
                  self.initial_state)

        if self.method.lower() == 'euler':
            obj.eulerIntegrate()
        elif self.method.lower() == 'heun':
            obj.heunIntegrate()
        elif self.method.lower() == 'rk4':
            obj.rk4Integrate()
        else:
            print("unkown integratiom method")
            exit(0)

        sol = np.asarray(obj.get_coordinates())
        times = np.asarray(obj.get_times())
        del obj

        return {"t": times, "x": sol}
