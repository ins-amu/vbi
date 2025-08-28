import tqdm
import numpy as np
from vbi.models.cupy.utils import *


class JR_sde:
    """
    Jansen-Rit model cupy implementation.
    
    .. list-table:: Parameters
        :widths: 25 50 25
        :header-rows: 1

        * - Name
          - Explanation
          - Default Value
        * - `A`
          - Excitatory post synaptic potential amplitude.
          - 3.25
        * - `B`
          - Inhibitory post synaptic potential amplitude.
          - 22.0
        * - `1/a`
          - Time constant of the excitatory postsynaptic potential.
          - a: 0.1 (1/a: 10.0)
        * - `1/b`
          - Time constant of the inhibitory postsynaptic potential.
          - b: 0.05 (1/b: 20.0)
        * - `C0`, `C1`
          - Average numbers of synapses between excitatory populations. If array-like, it should be the shape of (`num_nodes`, `num_sim`).
          - C0: 1.0 * 135.0, C1: 0.8 * 135.0
        * - `C2`, `C3`
          - Average numbers of synapses between inhibitory populations. If array-like, it should be the shape of (`num_nodes`, `num_sim`).
          - C2: 0.25 * 135.0, C3: 0.25 * 135.0
        * - `vmax`
          - Maximum firing rate
          - 0.005
        * - `v`
          - Potential at half of maximum firing rate
          - 6.0
        * - `r`
          - Slope of sigmoid function at `v_0`
          - 0.56
        * - `G`
          - Scaling the strength of network connections. If array-like, it should of length `num_sim`.
          - 1.0
        * - `mu`
          - Mean of the noise
          - 0.24
        * - `noise_amp`
          - Amplitude of the noise
          - 0.01
        * - `weights`
          - Weight matrix of shape (`num_nodes`, `num_nodes`)
          - None
        * - `num_sim`
          - Number of simulations
          - 1
        * - `dt`
          - Time step
          - 0.01
        * - `t_end`
          - End time of simulation
          - 1000.0
        * - `t_cut`
          - Cut time
          - 500.0
        * - `engine`
          - "cpu" or "gpu"
          - "cpu"
        * - `method`
          - "heun" or "euler" method for integration
          - "heun"
        * - `seed`
          - Random seed
          - None
        * - `initial_state`
          - Initial state of the system of shape (`num_nodes`, `num_sim`)
          - None
        * - `same_initial_state`
          - If True, all simulations have the same initial state
          - False
        * - `same_noise_per_sim`
          - If True, all simulations have the same noise
          - False
        * - `decimate`
          - Decimation factor for the output time series
          - 1
        * - `dtype`
          - Data type to use for the simulation, `float` for `float64` or `f` for `float32`.
          - "float"
    """

    def __init__(self, par: dict = {}):

        self.valid_parameters = list(self.get_default_parameters().keys())
        self.check_parameters(par)
        self._par = self.get_default_parameters()
        self._par.update(par)

        for item in self._par.items():
            name = item[0]
            value = item[1]
            setattr(self, name, value)

        self.xp = get_module(self.engine)
        if self.seed is not None:
            self.xp.random.seed(self.seed)

    def __str__(self) -> str:
        print("Jansen-Rit Model")
        print("----------------")
        for item in self._par.items():
            name = item[0]
            value = item[1]
            print(f"{name} = {value}")
        return ""

    def __call__(self):
        print("Jansen-Rit Model")
        return self._par

    def check_parameters(self, par):
        """
        Check if the provided parameters are valid.

        Parameters
        ----------
        par : dict
            Dictionary of parameters to check.

        Raises
        ------
        ValueError
            If any parameter in `par` is not valid.
        """
        for key in par.keys():
            if key not in self.valid_parameters:
                raise ValueError("Invalid parameter: " + key)

    def set_initial_state(self):
        self.initial_state = set_initial_state(
            self.nn,
            self.num_sim,
            self.engine,
            self.seed,
            self.same_initial_state,
            self.dtype,
        )

    def get_default_parameters(self) -> dict:
        """
        Default parameters for the Jansen-Rit model

        Parameters
        ----------
        nn : int
            number of nodes

        Returns
        -------
        params : dict
            default parameters
        """
        params = {
            "G": 1.0,
            "A": 3.25,
            "B": 22.0,
            "v": 6.0,
            "r": 0.56,
            "v0": 6.0,
            "vmax": 0.005,
            "C0": 1.0 * 135.0,
            "C1": 0.8 * 135.0,
            "C2": 0.25 * 135.0,
            "C3": 0.25 * 135.0,
            "a": 0.1,
            "b": 0.05,
            "mu": 0.24,
            "noise_amp": 0.01,
            "decimate": 1,
            "dt": 0.01,
            "t_end": 1000.0,
            "t_cut": 500.0,
            "engine": "cpu",
            "method": "heun",
            "num_sim": 1,
            "weights": None,
            "dtype": "float",
            "seed": None,
            "initial_state": None,
            "same_initial_state": False,
            "same_noise_per_sim": False,
        }
        return params

    def prepare_input(self):
        """Prepare input parameters for the Jansen-Rit model."""

        self.G = self.xp.array(self.G)
        assert self.weights is not None, "weights must be provided"
        self.weights = self.xp.array(self.weights) 
        self.weights = move_data(self.weights, self.engine)
        self.nn = self.num_nodes = self.weights.shape[0]

        if self.initial_state is None:
            self.set_initial_state()

        self.C0 = prepare_vec(self.C0, self.num_sim, self.engine, self.dtype)
        self.C1 = prepare_vec(self.C1, self.num_sim, self.engine, self.dtype)
        self.C2 = prepare_vec(self.C2, self.num_sim, self.engine, self.dtype)
        self.C3 = prepare_vec(self.C3, self.num_sim, self.engine, self.dtype)
        assert self.t_cut < self.t_end, "t_cut must be smaller than t_end"

    def S_(self, x):
        """
        Compute the sigmoid function.

        This function calculates the sigmoid of the input `x` using the parameters
        `vmax`, `r`, and `v0`.

        Parameters
        ----------
        x : float or array-like
            The input value(s) for which to compute the sigmoid function.

        Returns
        -------
        float or array-like
            The computed sigmoid value(s).

        """
        return self.vmax / (1.0 + self.xp.exp(self.r * (self.v0 - x)))

    def f_sys(self, x0, t):
        """
        Compute the derivatives of the Jansen-Rit neural mass model.

        Parameters
        ----------
        x0 : array_like
            Initial state vector of the system. It should have a shape of (6*nn, ns), where nn is the number of neurons and ns is the number of simulations.
        t : float
            Current time point (not used in the computation but required for compatibility with ODE solvers).

        Returns
        -------
        dx : array_like
            Derivatives of the state vector. It has the same shape as `x0`.

        The function computes the derivatives of the state vector based on the Jansen-Rit model equations.
        """

        nn = self.nn
        ns = self.num_sim
        mu = self.mu
        G = self.G
        C0 = self.C0
        C1 = self.C1
        C2 = self.C2
        C3 = self.C3
        A = self.A
        B = self.B
        a = self.a
        b = self.b
        Aa = A * a
        Bb = B * b
        bb = b * b
        aa = a * a
        SC = self.weights
        _xp = self.xp
        S = self.S_

        x = x0[:nn, :]
        y = x0[nn : 2 * nn, :]
        z = x0[2 * nn : 3 * nn, :]
        xp = x0[3 * nn : 4 * nn, :]
        yp = x0[4 * nn : 5 * nn, :]
        zp = x0[5 * nn : 6 * nn, :]

        dx = _xp.zeros((6 * nn, ns))
        couplings = S(SC.dot(y - z))

        dx[0:nn, :] = xp
        dx[nn : 2 * nn, :] = yp
        dx[2 * nn : 3 * nn, :] = zp
        dx[3 * nn : 4 * nn, :] = Aa * S(y - z) - 2 * a * xp - aa * x
        dx[4 * nn : 5 * nn, :] = (
            Aa * (mu + C1 * S(C0 * x) + G * couplings) - 2 * a * yp - aa * y
        )
        dx[5 * nn : 6 * nn, :] = Bb * C3 * S(C2 * x) - 2 * b * zp - bb * z

        return dx

    def euler(self, x0, t):
        """
        Perform one step of the Euler-Maruyama method for stochastic differential equations.

        Parameters
        ----------
        x0 : array_like
            The initial state of the system.
        t : float
            The current time.

        Returns
        -------
        array_like
            The updated state of the system after one Euler step.
        """

        _xp = self.xp
        nn = self.nn
        ns = self.num_sim
        dt = self.dt
        sqrt_dt = np.sqrt(dt)
        noise_amp = self.noise_amp
        randn = _xp.random.randn
        snps = self.same_noise_per_sim

        dW = randn(nn, 1) if snps else randn(nn, ns)
        dW = sqrt_dt * noise_amp * dW

        x0 = x0 + dt * self.f_sys(x0, t)
        x0[4 * nn : 5 * nn, :] += dW

        return x0

    def heun(self, x0, t):
        """
        Perform a single step of the Heun's method for stochastic differential equations.

        Parameters
        ----------
        x0 : ndarray
            The initial state of the system.
        t : float
            The current time.

        Returns
        -------
        ndarray
            The updated state of the system after one Heun step.
        """

        nn = self.nn
        ns = self.num_sim
        dt = self.dt
        _xp = self.xp
        sqrt_dt = np.sqrt(dt)
        noise_amp = self.noise_amp
        randn = _xp.random.randn
        snps = self.same_noise_per_sim

        dW = randn(nn, 1) if snps else randn(nn, ns)
        if snps:
            dW = np.tile(dW, (1, ns))
        dW = sqrt_dt * noise_amp * dW

        k1 = self.f_sys(x0, t) * dt
        x1 = x0 + k1
        x1[4 * nn : 5 * nn, :] += dW
        k2 = self.f_sys(x1, t + dt) * dt
        x0 = x0 + (k1 + k2) / 2.0
        x0[4 * nn : 5 * nn, :] += dW

        return x0

    def run(self, x0=None):
        """
        Simulate the Jansen-Rit model.

        Parameters
        ----------

        x0: array [num_nodes, num_sim]
            initial state

        Returns
        -------

        dict: simulation results
            t: array [n_step]
                time
            x: array [n_step, num_nodes, num_sim]
                y1-y2 time series

        """

        self.prepare_input()
        x = self.initial_state if x0 is None else x0
        self.integrator = self.euler if self.method == "euler" else self.heun
        dt = self.dt
        _xp = self.xp
        nn = self.nn
        ns = self.num_sim
        decimate = self.decimate
        t_end = self.t_end
        t_cut = self.t_cut

        tspan = _xp.arange(0, t_end, dt)
        i_cut = int(_xp.where(tspan >= t_cut)[0][0])

        n_step = int((len(tspan) - i_cut) / decimate)
        y = np.zeros((n_step, nn, ns), dtype="f")  # store in host
        ii = 0

        for i in tqdm.trange(len(tspan)):

            x = self.integrator(x, tspan[i])
            x_ = get_(x, self.engine, "f")

            if (i >= i_cut) and (i % decimate == 0):
                y[ii, :, :] = x_[nn : 2 * nn, :] - x_[2 * nn : 3 * nn, :]
                ii += 1

        t = get_(tspan[tspan >= t_cut][::decimate], self.engine, "f")

        return {"t": t, "x": y}


def set_initial_state(
    nn, ns, engine, seed=None, same_initial_state=False, dtype="float"
):
    """
    set initial state for the Jansen-Rit model

    Parameters
    ----------
    nn: int
        number of nodes
    ns: int
        number of simulations
    engine: str
        cpu or gpu
    seed: int
        random seed
    same_initial_state: bool
        if True, all simulations have the same initial state
    dtype: str
        data type

    Returns
    -------
    y0: array [nn, ns]
        initial state

    """

    if seed is not None:
        np.random.seed(seed)

    if same_initial_state:
        y0 = np.random.uniform(-1, 1, (nn, 1))
        y1 = np.random.uniform(-500, 500, (nn, 1))
        y2 = np.random.uniform(-50, 50, (nn, 1))
        y3 = np.random.uniform(-6, 6, (nn, 1))
        y4 = np.random.uniform(-20, 20, (nn, 1))
        y5 = np.random.uniform(-500, 500, (nn, 1))
        y = np.vstack((y0, y1, y2, y3, y4, y5))
        y = np.tile(y, (1, ns))
        y = move_data(y, engine)
    else:
        y0 = np.random.uniform(-1, 1, (nn, ns))
        y1 = np.random.uniform(-500, 500, (nn, ns))
        y2 = np.random.uniform(-50, 50, (nn, ns))
        y3 = np.random.uniform(-6, 6, (nn, ns))
        y4 = np.random.uniform(-20, 20, (nn, ns))
        y5 = np.random.uniform(-500, 500, (nn, ns))
        y = np.vstack((y0, y1, y2, y3, y4, y5))
        y = move_data(y, engine)

    return y.astype(dtype)
