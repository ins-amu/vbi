import numpy as np
from typing import Union
from vbi.models.cpp._src.ww_sde import WW_sde as _WW_sde

########################### Wong Wang sde model ###############################
###############################################################################


class WW_sde(object):
    """
    Wong Wang model with SDE integration

    Parameters
    ----------

    par: dict
        Dictionary with parameters of the model.
        - **G**: float, coupling strength
        - **J_N**: float, synaptic coupling strength
        - **I_o**: float, external input
        - **a**: float, slope of the sigmoidal activation function
        - **b**: float, threshold of the sigmoidal activation function
        - **d**: float, inverse of the sigmoidal activation function
        - #TODO: add more parameters
    """

    def __init__(self, par: dict) -> None:
        self.valid_parameters = self.get_default_parameters().keys()
        self.check_parameters(par)
        self.par_ = self.get_default_parameters()
        self.par_.update(par)
        self.set_parameters(self.par_)

        if self.seed is not None:
            np.random.seed(self.seed)
            
        if "dt" in par.keys():
            self.dt_bold = self.par_["dt"] / 1000.0

    def __str__(self) -> str:
        print("Wong Wang model with SDE integration")
        print("-------------------------------------")
        for item in self.par_.items():
            print(f"{item[0]}: {item[1]}")
        return ""

    def __call__(self):
        print("Wong Wang model with SDE integration")
        return self.par_

    def set_parameters(self, par: dict) -> None:
        for item in par.items():
            setattr(self, item[0], item[1])

    def set_initial_state(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        assert self.weights is not None, "Weights are not set"
        nn = self.weights.shape[0]
        self.initial_state = np.random.rand(nn)

    def check_parameters(self, par: dict) -> None:
        for key in par.keys():
            if key not in self.valid_parameters:
                raise ValueError(f"Invalid parameter {key}")

    def get_default_parameters(self) -> dict:
        """get default model parameters."""
        par = {
            "G": 0.0,
            "a": 270.0,
            "b": 108.0,
            "d": 0.154,
            "gamma": 0.641,
            "tau_s": 0.1,
            "w": 0.9,
            "dt": 10.0,
            "dt_bold": 0.01,
            "J_N": 0.2609,
            "I_o": 0.33,
            "sigma_noise": 0.0,
            "initial_state": None,
            "weights": None,
            "seed": None,
            "method": "heun",
            "t_end": 5 * 60 * 1000.0,
            "t_cut": 0.0,
            "noise_seed": False,
            "fmri_decimate": 10,
            "ts_decimate": 10,
            "RECORD_TS": True,
            "RECORD_FMRI": True,
        }

        return par

    def prepare_input(self):
        assert self.weights is not None
        self.weights = np.array(self.weights)
        self.N = self.weights.shape[0]
        assert self.weights.shape == (self.N, self.N)
        self.dt = float(self.dt)
        self.t_end = float(self.t_end)
        self.t_cut = float(self.t_cut)
        assert self.t_cut < self.t_end
        self.G = float(self.G)
        self.a = float(self.a)
        self.b = float(self.b)
        self.d = float(self.d)
        self.gamma = float(self.gamma)
        self.tau_s = float(self.tau_s)
        self.J_N = float(self.J_N)
        self.I_o = check_sequence(self.I_o, self.N)
        self.w = check_sequence(self.w, self.N)
        self.sigma_noise = check_sequence(self.sigma_noise, self.N) / np.sqrt(0.001)
        self.initial_state = np.array(self.initial_state)
        assert self.initial_state.shape == (self.N,)
        self.noise_seed = int(self.noise_seed)

    def run(self, par={}, x0=None, verbose=False):
        """
        Integrates the model equations
        """

        if x0 is None:
            self.set_initial_state(self.seed)
        else:
            self.initial_state = x0

        for key in par.keys():
            if key not in self.valid_parameters:
                raise ValueError(f"Invalid parameter {key}")
            else:
                setattr(self, key, par[key])

        self.prepare_input()
        
        obj = _WW_sde(
            self.N,
            self.dt,
            self.dt_bold,
            self.G,
            self.t_cut,
            self.t_end,
            self.initial_state,
            self.weights,
            self.a,
            self.b,
            self.d,
            self.gamma,
            self.tau_s,
            self.w,
            self.J_N,
            self.I_o,
            self.sigma_noise,
            self.fmri_decimate,
            self.ts_decimate,
            self.RECORD_TS,
            self.RECORD_FMRI,
            self.noise_seed,
        )
        
        obj.integrate()
        d_fmri = np.asarray(obj.get_d_fmri(), dtype=np.float32)
        t_fmri = np.asarray(obj.get_t_fmri())
        x = np.asarray(obj.get_states(), dtype=np.float32)
        t = np.asarray(obj.get_times())
        del obj
        tcut = self.t_cut
        x = x[t > tcut, :]
        t = t[t > tcut]
        if d_fmri.ndim == 2:
            d_fmri = d_fmri[t_fmri > tcut/1000, :]
            t_fmri = t_fmri[t_fmri > tcut/1000]

        return {"t": t, "s": x.T, "t_fmri": t_fmri, "d_fmri": d_fmri.T}


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
