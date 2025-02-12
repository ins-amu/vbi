import numpy as np
from typing import Union
from copy import deepcopy
from vbi.models.cpp._src.mpr_sde import MPR_sde as _MPR_sde


class MPR_sde:
    """
    Montbrio-Pazo-Roxin model C++ implementation.

    Parameters
    ----------
    par : dict
        Dictionary of parameters.


    """

    def __init__(self, par: dict = {}) -> None:

        par = deepcopy(par)
        self._par = self.get_default_parameters()
        self.valid_parameters = list(self._par.keys())
        self.check_parameters(par)
        self._par.update(par)

        for item in self._par.items():
            name = item[0]
            value = item[1]
            setattr(self, name, value)

        if self.seed is not None:
            np.random.seed(self.seed)

        if self.initial_state is None:
            self.INITIAL_STATE_SET = False

    # -------------------------------------------------------------------------

    def set_initial_state(self):
        self.num_nodes = self.weights.shape[0]
        self.initial_state = set_initial_state(self.num_nodes, self.seed)
        self.INITIAL_STATE_SET = True

    # -------------------------------------------------------------------------

    def __str__(self) -> str:
        print("MPR sde model.")
        print("----------------")
        for item in self._par.items():
            name = item[0]
            value = item[1]
            print(f"{name} = {value}")
        return ""

    # -------------------------------------------------------------------------

    def __call__(self):
        return self._par

    # -------------------------------------------------------------------------

    def check_parameters(self, par: dict):
        for key in par.keys():
            if key not in self.valid_parameters:
                raise ValueError(f"Invalid parameter {key:s} provided.")

    def get_default_parameters(self):

        params = {
            "G": 0.733,  # global coupling strength
            "dt": 0.01,  # for mpr model [ms]
            "dt_bold": 0.001,  # for Balloon model [s]
            "J": 14.5,  # model parameter
            "eta": -4.6,  # model parameter
            "tau": 1.0,  # model parameter
            "delta": 0.7,  # model parameter
            "tr": 500,  # sampling from mpr time series
            "noise_amp": 0.037,  # amplitude of noise
            "noise_seed": 0,  # fix seed for noise
            "iapp": 0.0,  # constant applyed current
            "seed": None,
            "initial_state": None,  # initial condition of the system
            "t_cut": 0.5 * 60 * 1000,  # transition time [ms]
            "t_end": 5 * 60 * 1000.0,  # end time  [ms]
            "weights": None,  # weighted connection matrix
            "rv_decimate": 10,  # sampling every n step from mpr time series
            "output": "output",  # output directory
            "RECORD_RV": 0,  # true to store large time series in file
            "RECORD_BOLD": 1,
        }

        return params

    # -------------------------------------------------------------------------

    def prepare_input(self):
        """
        Prepare input parameters for passing to C++ engine.
        """

        self.dt = float(self.dt)
        self.dt_bold = float(self.dt_bold)
        self.tr = int(self.tr)
        self.initial_state = np.asarray(self.initial_state).astype(np.float64)
        self.weights = np.asarray(self.weights).astype(np.float64)
        self.num_nodes = self.weights.shape[0]
        self.G = float(self.G)
        self.eta = check_sequence(self.eta, self.num_nodes)  # check eta be array-like
        self.eta = np.asarray(self.eta).astype(np.float64)
        
        self.J = float(self.J)
        self.tau = float(self.tau)
        self.delta = float(self.delta)
        self.iapp = float(self.iapp)
        self.noise_amp = float(self.noise_amp)
        self.rv_decimate = int(self.rv_decimate)
        self.t_cut = float(self.t_cut) / 10.0
        self.t_end = float(self.t_end) / 10.0
        self.RECORD_RV = int(self.RECORD_RV)
        self.noise_seed = int(self.noise_seed)

    # -------------------------------------------------------------------------

    def run(self, par: dict = {}, x0: np.ndarray = None, verbose: bool = False):
        """
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
        """

        if x0 is None:
            if not self.INITIAL_STATE_SET:
                self.set_initial_state()
                if verbose:
                    print("initial state set by default")
        else:
            assert len(x0) == self.num_nodes * 2
            self.initial_state = x0
            self.INITIAL_STATE_SET = True

        for key in par.keys():
            if key not in self.valid_parameters:
                raise ValueError(f"Invalid parameter {key:s} provided.")
            # if key in ['eta']:
            #     self.set_eta(key, par[key])
            # else:
            setattr(self, key, par[key])

        self.prepare_input()

        obj = _MPR_sde(
            self.dt,
            self.dt_bold,
            self.tr,
            self.initial_state,
            self.weights,
            self.G,
            self.eta,
            self.J,
            self.tau,
            self.delta,
            self.iapp,
            self.noise_amp,
            self.rv_decimate,
            0.0,  # self.t_cut,
            self.t_end,
            self.RECORD_RV,
            self.noise_seed,
        )

        obj.heunStochasticIntegrate()
        bold = np.asarray(obj.get_bold()).astype(np.float32).T
        t = np.asarray(obj.get_time())

        if bold.ndim == 2:
            bold = bold[:, t >= self.t_cut]
            t = t[t >= self.t_cut] * 10.0  # [ms]

        return {"t": t, "x": bold}


def check_sequence(x: Union[int, float, np.ndarray], n: int):
    """
    check if x is a scalar or a sequence of length n

    parameters
    ----------
    x: scalar or sequence of length n
    n: number of nodes

    returns
    -------
    x: sequence of length n
    """
    if isinstance(x, (np.ndarray, list, tuple)):
        assert len(x) == n, f" variable must be a sequence of length {n}"
        return x
    else:
        return x * np.ones(n)


def set_initial_state(nn, seed=None):

    if seed is not None:
        np.random.seed(seed)

    y0 = np.random.rand(2 * nn)
    y0[:nn] = y0[:nn] * 1.5
    y0[nn:] = y0[nn:] * 4 - 2
    return y0
