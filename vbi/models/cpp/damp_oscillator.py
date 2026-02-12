import os
from typing import Any
import numpy as np

from vbi.models.cpp.base import BaseModel

try:
    from vbi.models.cpp._src.do import DO as _DO
except ImportError as e:
    print(
        f"Could not import modules: {e}, probably C++ code is not compiled or properly linked."
    )


class DO(BaseModel):
    """
    Damp Oscillator model class.
    """

    # ---------------------------------------------------------------
    def __init__(self, par: dict = {}):
        """
        Parameters
        ----------
        par : dictionary
            parameters which includes the following:
            - **dt** [double] time step.
            - **t_start** [double] initial time for simulation.
            - **t_end** [double] final time for simulation.
            - **initial_state** [list] initial state of the system.

        """
        super().__init__()
        self.valid_params = list(self.get_default_parameters().keys())
        self.check_parameters(par)
        self._par = self.get_default_parameters()
        self._par.update(par)

        for item in self._par.items():
            name = item[0]
            value = item[1]
            setattr(self, name, value)

    def __str__(self) -> str:
        """
        Return a string representation of the model parameters.

        Returns
        -------
        str
            Formatted string showing all model parameters and their values.
        """
        print("Damped Oscillator Model (C++)")
        print("-----------------------------")

        # Model parameters
        print(f"a = {self.a}")
        print(f"b = {self.b}")

        # Simulation parameters
        print(f"dt = {self.dt}")
        print(f"t_start = {self.t_start}")
        print(f"t_end = {self.t_end}")
        print(f"t_cut = {self.t_cut}")
        print(f"method = {self.method}")
        print(f"output = {self.output}")
        print(f"initial_state = {self.initial_state}")

        return ""

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        print("Damp Oscillator model")
        return self._par

    def get_default_parameters(self):
        """
        return default parameters for damp oscillator model.
        """

        params = {
            "a": 0.1,
            "b": 0.05,
            "dt": 0.01,
            "t_start": 0,
            "method": "euler",
            "t_end": 100.0,
            "t_cut": 20,
            "output": "output",
            "initial_state": [0.5, 1.0],
        }

        return params

    def update_par(self, par={}):
        """
        Update model parameters.

        Parameters
        ----------
        par : dict, optional
            Dictionary of parameters to update. Keys must be valid parameter names.
        """
        if par:
            self.check_parameters(par)
            for key in par.keys():
                setattr(self, key, par[key])

    def prepare_input(self):
        """
        prepare input for cpp model.
        """
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
    def run(self, par: dict = {}, x0: np.ndarray = None, verbose: bool = False):
        """
        Run the damped oscillator simulation.

        Parameters
        ----------
        par : dict, optional
            Dictionary of parameters to update for this simulation run.
            Any parameter from the class documentation can be updated.
        x0 : array-like, optional
            Initial state vector [x0, y0] of length 2. If None, uses the
            initial state set during initialization.
        verbose : bool, optional
            If True, print verbose output during simulation. Default is False.

        Returns
        -------
        dict
            Dictionary containing simulation results:

            - 't' : np.ndarray of shape (n_steps,) - time points
            - 'x' : np.ndarray of shape (n_steps, 2) - simulated trajectories
              where x[:, 0] is x(t) and x[:, 1] is y(t)
        """

        if x0 is not None:
            assert len(x0) == 2
            self.initial_state = x0

        self.check_parameters(par)
        for key in par.keys():
            setattr(self, key, par[key])

        self.prepare_input()

        obj = _DO(self.dt, self.a, self.b, self.t_start, self.t_end, self.initial_state)

        if self.method.lower() == "euler":
            obj.eulerIntegrate()
        elif self.method.lower() == "heun":
            obj.heunIntegrate()
        elif self.method.lower() == "rk4":
            obj.rk4Integrate()
        else:
            print("unkown integratiom method")
            exit(0)

        sol = np.asarray(obj.get_coordinates())
        times = np.asarray(obj.get_times())
        del obj

        return {"t": times, "x": sol}
