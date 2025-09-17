"""
Damped Oscillator Model - Numba Implementation

This module implements a nonlinear damped oscillator model using Numba for high-performance
numerical integration. The model exhibits rich dynamical behavior including fixed points,
limit cycles, and chaotic dynamics depending on parameter values.

The system is described by:
    dx/dt = x - xy - ax² \n
    dy/dt = xy - y - by²

Classes
-------
DO_nb : Main model class
Param : Parameter container (Numba jitclass)

Functions
---------
euler, heun, rk4 : Integration methods
_f_sys : System dynamics function
_integrate : Main integration routine
"""

import os
import numpy as np
from numba.experimental import jitclass
from numba import float64, types
from numba import njit
from typing import Any
from vbi.utils import print_valid_parameters

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
    """
    Parameter container for the damped oscillator model.
    
    A Numba-compiled parameter class that stores all model parameters
    for efficient access during numerical integration. This class is
    optimized for high-performance computing with Numba JIT compilation.
    
    Parameters
    ----------
    a : float, default 0.1
        Damping parameter for x equation (ax² term).
    b : float, default 0.05  
        Damping parameter for y equation (by² term).
    dt : float, default 0.01
        Integration time step.
    t_start : float, default 0
        Start time for integration.
    t_end : float, default 100.0
        End time for integration.
    t_cut : float, default 20
        Initial time to discard (burn-in period).
    output : str, default "output"
        Output directory name.
    method : str, default "euler"
        Integration method ("euler", "heun", or "rk4").
    initial_state : np.ndarray, default [0.5, 1.0]
        Initial conditions [x0, y0].
        
    Notes
    -----
    This class is compiled by Numba for efficient parameter access.
    The damped oscillator equations are:
    
    .. math::
        \\frac{dx}{dt} = x - xy - ax^2
        
        \\frac{dy}{dt} = xy - y - by^2
    """
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
    """
    System function for the damped oscillator model.
    
    Computes the derivatives for the nonlinear system:
    dx/dt = x - xy - ax²
    dy/dt = xy - y - by²
    
    Parameters
    ----------
    x : np.ndarray
        State vector [x, y] of length 2.
    P : Param
        Parameter object containing model parameters.
        
    Returns
    -------
    np.ndarray
        Derivative vector [dx/dt, dy/dt].
    """
    a = P.a
    b = P.b
    return np.array([x[0] - x[0]*x[1] - a * x[0] * x[0],
                        x[0]*x[1] - x[1] - b * x[1] * x[1]])


@njit
def euler(x, P):
    """
    Euler integration method for the damped oscillator.
    
    First-order explicit integration scheme: x(t+dt) = x(t) + dt * f(x, t)
    
    Parameters
    ----------
    x : np.ndarray
        Current state vector [x, y].
    P : Param
        Parameter object containing dt and other parameters.
        
    Returns
    -------
    np.ndarray
        Next state vector after one integration step.
    """
    return x + P.dt * _f_sys(x, P)

@njit
def heun(x, P):
    """
    Heun's integration method for the damped oscillator.
    
    Second-order predictor-corrector method that provides better accuracy
    than Euler method with moderate computational cost.
    
    Parameters
    ----------
    x : np.ndarray
        Current state vector [x, y].
    P : Param
        Parameter object containing dt and other parameters.
        
    Returns
    -------
    np.ndarray
        Next state vector after one integration step.
    """
    k0 = _f_sys(x, P)
    x1 = x + P.dt * k0
    k1 = _f_sys(x1, P)
    return x + 0.5 * P.dt * (k0 + k1)

@njit 
def rk4(x, P):
    """
    Fourth-order Runge-Kutta integration method for the damped oscillator.
    
    High-accuracy integration scheme that evaluates the system function
    four times per step for excellent precision.
    
    Parameters
    ----------
    x : np.ndarray
        Current state vector [x, y].
    P : Param
        Parameter object containing dt and other parameters.
        
    Returns
    -------
    np.ndarray
        Next state vector after one integration step.
    """
    k1 = _f_sys(x, P)
    k2 = _f_sys(x + 0.5 * P.dt * k1, P)
    k3 = _f_sys(x + 0.5 * P.dt * k2, P)
    k4 = _f_sys(x + P.dt * k3, P)
    return x + P.dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0


@njit
def _integrate(x, P, intg=euler):
    """
    Main integration function with burn-in period.
    
    Performs numerical integration of the damped oscillator system,
    including a burn-in period to allow transients to decay.
    
    Parameters
    ----------
    x : np.ndarray
        Initial state vector [x0, y0].
    P : Param
        Parameter object containing simulation parameters.
    intg : function, optional
        Integration function to use (euler, heun, or rk4).
        
    Returns
    -------
    tuple
        (t, x_out) where:
        - t : np.ndarray of time points
        - x_out : np.ndarray of shape (n_steps, 2) containing trajectories
    """
    t0 = np.arange(P.t_start, P.t_cut, P.dt)
    
    for i in range(len(t0)):
        x = intg(x, P)

    t = np.arange(P.t_cut, P.t_end, P.dt)
    x_out = np.zeros((len(t), len(x)))

    for i in range(len(t)):
        x = intg(x, P)
        x_out[i, :] = x
    return t, x_out


class DO:
    """
    Numba implementation of the damped oscillator model.
    
    The damped oscillator is a nonlinear dynamical system that exhibits complex behavior
    including limit cycles and chaotic dynamics. The system is described by the equations:
    
    .. math::
        \\frac{dx}{dt} = x - xy - ax^2 \n
        \\frac{dy}{dt} = xy - y - by^2
    
    where x and y are the state variables, and a and b are damping parameters.
    
    .. list-table:: Parameters
        :widths: 25 50 25
        :header-rows: 1

        * - Name
          - Explanation
          - Default Value
        * - `a`
          - Damping parameter for the first state variable x.
          - 0.1
        * - `b`
          - Damping parameter for the second state variable y.
          - 0.05
        * - `dt`
          - Integration time step.
          - 0.01
        * - `t_start`
          - Start time of simulation.
          - 0
        * - `t_end`
          - End time of simulation.
          - 100.0
        * - `t_cut`
          - Time from which to start collecting output (burn-in period).
          - 20
        * - `output`
          - Output directory path for saving results.
          - "output"
        * - `method`
          - Integration method. Options: "euler", "heun", "rk4".
          - "euler"
        * - `initial_state`
          - Initial state vector [x0, y0] of length 2.
          - [0.5, 1.0]

    Usage example:
        >>> import numpy as np
        >>> from vbi.models.numba.damp_oscillator import DO_nb
        >>> do = DO_nb({"a": 0.1, "b": 0.05, "dt": 0.01, "t_end": 100.0, "method": "rk4"})
        >>> t, x = do.run()
        >>> # x has shape (n_steps, 2) where x[:, 0] is x(t) and x[:, 1] is y(t)

    Notes
    -----
    The damped oscillator model is a classic example of a nonlinear dynamical system.
    Depending on the parameter values, the system can exhibit:
    
    - Fixed points (stable equilibria)
    - Limit cycles (periodic oscillations)
    - Chaotic behavior (sensitive dependence on initial conditions)
    
    The model supports three integration methods:
    
    - **Euler**: First-order explicit method, fastest but least accurate
    - **Heun**: Second-order predictor-corrector method, good balance of speed and accuracy  
    - **RK4**: Fourth-order Runge-Kutta method, most accurate but slowest
    
    The simulation includes a burn-in period (`t_cut`) to allow transients to decay
    before collecting output data.
    """

    def __init__(self, par={}):
        """
        Initialize the damped oscillator model.

        Parameters
        ----------
        par : dict, optional
            Dictionary containing model parameters. See class documentation for 
            available parameters. Default is an empty dictionary which uses 
            all default parameter values.
        """
        self.valid_params = [jit_spec[i][0] for i in range(len(jit_spec))]
        self.check_parameters(par)
        self.P = self.get_parobj(par)

        self.P.output = "output" if self.P.output is None else self.P.output
        os.makedirs(self.P.output, exist_ok=True)

    def __str__(self) -> str:
        """
        Return a string representation of the model parameters.
        
        Returns
        -------
        str
            Formatted string showing all model parameters and their values.
        """
        print("Damped Oscillator Model (Numba)")
        print("-------------------------------")
        
        # Model parameters
        print(f"a = {self.P.a}")
        print(f"b = {self.P.b}")
        
        # Simulation parameters
        print(f"dt = {self.P.dt}")
        print(f"t_start = {self.P.t_start}")
        print(f"t_end = {self.P.t_end}")
        print(f"t_cut = {self.P.t_cut}")
        print(f"method = {self.P.method}")
        print(f"output = {self.P.output}")
        print(f"initial_state = {self.P.initial_state}")
        
        return ""

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        print("Damped Oscillator Model (Numba)")
        print("-------------------------------")
        
        # Model parameters
        print(f"a = {self.P.a}")
        print(f"b = {self.P.b}")
        
        # Simulation parameters
        print(f"dt = {self.P.dt}")
        print(f"t_start = {self.P.t_start}")
        print(f"t_end = {self.P.t_end}")
        print(f"t_cut = {self.P.t_cut}")
        print(f"method = {self.P.method}")
        print(f"output = {self.P.output}")
        print(f"initial_state = {self.P.initial_state}")
        return self.P
    
    def check_parameters(self, par):
        """
        Validate model parameters.
        
        Parameters
        ----------
        par : dict
            Dictionary of parameters to validate.
            
        Raises
        ------
        ValueError
            If any parameter name is not recognized.
        """
        for key in par.keys():
            if key not in self.valid_params:
                print(f"Invalid parameter: {key}")
                print_valid_parameters(jit_spec)
                raise ValueError("Invalid parameter: " + key)

    def get_parobj(self, par={}):
        """
        Create parameter object with default values and user overrides.
        
        Parameters
        ----------
        par : dict, optional
            Dictionary of parameters to override defaults.
            
        Returns
        -------
        Param
            Numba jitclass parameter object.
        """
        if "initial_state" in par.keys():
            par["initial_state"] = np.array(par["initial_state"])

        parobj = Param(**par)
        return parobj
    
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
                setattr(self.P, key, par[key])

    def run(self, par={}, x0=None, verbose=False):
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
        
        t, x = _integrate(self.P.initial_state, self.P, intg=intg)
        return {"t": t, "x": x}


# Alias for consistency with naming convention
DO_nb_numba = DO
DO = DO