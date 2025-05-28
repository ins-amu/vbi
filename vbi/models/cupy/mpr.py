import os
import tqdm
import logging
import numpy as np
from copy import copy
from vbi.models.cupy.utils import *

try:
    import cupy as cp
except ImportError:
    logging.warning("Cupy is not installed. Using Numpy instead.")


class Bold:

    def __init__(self, par: dict = {}) -> None:

        self._par = self.get_default_parameters()
        self.valid_parameters = list(self._par.keys())
        self.check_parameters(par)
        self._par.update(par)

        for item in self._par.items():
            setattr(self, item[0], item[1])
        self.update_dependent_parameters()
            

    def get_default_parameters(self):
        """get balloon model parameters."""

        vo = 0.08
        theta = 40.3
        TE = 0.04
        Eo = 0.4
        r0 = 25.0
        epsilon = 0.34
        k1 = 4.3 * theta * Eo * TE
        k2 = epsilon * r0 * Eo * TE
        k3 = 1 - epsilon

        par = {
            "kappa": 0.65,
            "gamma": 0.41,
            "tau": 0.98,
            "alpha": 0.32,
            "epsilon": epsilon,
            "Eo": Eo,
            "TE": TE,
            "vo": vo,
            "r0": r0,
            "theta": theta,
            "t_min": 0.0,
            "rtol": 1e-5,
            "atol": 1e-8,
            "k1": k1,
            "k2": k2,
            "k3": k3
        }
        return par
    
    def update_dependent_parameters(self):
        self.k1 = 4.3 * self.theta * self.Eo * self.TE
        self.k2 = self.epsilon * self.r0 * self.Eo * self.TE
        self.k3 = 1 - self.epsilon

    def check_parameters(self, par):
        for key in par.keys():
            if key not in self.valid_parameters:
                raise ValueError(f"Invalid parameter {key:s} provided.")
            
    def allocate_memory(self, xp, nn, ns, n_steps, bold_decimate, dtype):
    
        self.s = xp.zeros((2, nn, ns), dtype=dtype)
        self.f = xp.zeros((2, nn, ns), dtype=dtype)
        self.ftilde = xp.zeros((2, nn, ns), dtype=dtype)
        self.vtilde = xp.zeros((2, nn, ns), dtype=dtype)
        self.qtilde = xp.zeros((2, nn, ns), dtype=dtype)
        self.v = xp.zeros((2, nn, ns), dtype=dtype)
        self.q = xp.zeros((2, nn, ns), dtype=dtype)
        self.vv = np.zeros((n_steps // bold_decimate, nn, ns), dtype="f")
        self.qq = np.zeros((n_steps // bold_decimate, nn, ns), dtype="f")
        self.s[0] = 1
        self.f[0] = 1
        self.v[0] = 1
        self.q[0] = 1
        self.ftilde[0] = 0
        self.vtilde[0] = 0
        self.qtilde[0] = 0

    def do_bold_step(self, r_in, dtt):

        Eo = self.Eo
        tau = self.tau
        kappa = self.kappa
        gamma = self.gamma
        alpha = self.alpha
        ialpha = 1 / alpha
        
        v = self.v
        q = self.q
        s = self.s
        f = self.f 
        ftilde = self.ftilde
        vtilde = self.vtilde
        qtilde = self.qtilde

        s[1] = s[0] + dtt * (r_in - kappa * s[0] - gamma * (f[0] - 1))
        f[0] = np.clip(f[0], 1, None)
        ftilde[1] = ftilde[0] + dtt * (s[0] / f[0])
        fv = v[0] ** ialpha  # outflow
        vtilde[1] = vtilde[0] + dtt * ((f[0] - fv) / (tau * v[0]))
        q[0] = np.clip(q[0], 0.01, None)
        ff = (1 - (1 - Eo) ** (1 / f[0])) / Eo  # oxygen extraction
        qtilde[1] = qtilde[0] + dtt * ((f[0] * ff - fv * q[0] / v[0]) / (tau * q[0]))

        f[1] = np.exp(ftilde[1])
        v[1] = np.exp(vtilde[1])
        q[1] = np.exp(qtilde[1])

        f[0] = f[1]
        s[0] = s[1]
        ftilde[0] = ftilde[1]
        vtilde[0] = vtilde[1]
        qtilde[0] = qtilde[1]
        v[0] = v[1]
        q[0] = q[1]



class MPR_sde:
    """
    Montbrio-Pazo-Roxin model Cupy and Numpy implementation.

    Parameters
    ----------

    G: float. np.ndarray
        global coupling strength
    dt: float
        time step
    dt_bold: float
        time step for Balloon model
    J: float, np.ndarray
        model parameter
    eta: float, np.ndarray
        model parameter
    tau:
        model parameter
    delta:
        model parameter
    tr: float
        repetition time of fMRI
    noise_amp: float, np.array
        amplitude of noise
    same_noise_per_sim:
        same noise for all simulations
    iapp: float, np.ndarray
        external input
    t_start: float
        initial time
    t_cut: float
        transition time
    t_end: float
        end time
    num_nodes: int
        number of nodes
    weights: np.ndarray
        weighted connection matrix
    rv_decimate: int
        sampling step from r and v
    output: str
        output directory
    RECORD_TS:  bool
        store r and v time series
    num_sim: int
        number of simulations
    method: str
        integration method
    engine: str
        cpu or gpu
    seed: int
        seed for random number generator
    dtype: str
        float or f
    initial_state: np.ndarray
        initial state
    same_initial_state: bool
        same initial state for all simulations

    """

    def __init__(self, par: dict = {}, Bpar:dict = {}) -> None:

        self._par = self.get_default_parameters()
        self.valid_parameters = list(self._par.keys())
        self.check_parameters(par)
        self._par.update(par)

        for item in self._par.items():
            name = item[0]
            value = item[1]
            setattr(self, name, value)
        
        self.update_dependent_parameters()
            
        self.B = Bold(Bpar)

        self.xp = get_module(self.engine)
        if self.seed is not None:
            self.xp.random.seed(self.seed)

        os.makedirs(self.output, exist_ok=True)

    def __call__(self):
        print("Montbrió, Pazó, Roxin model.")
        return self._par

    def __str__(self) -> str:
        print("Montbrió, Pazó, Roxin model.")
        print("----------------")
        for item in self._par.items():
            name = item[0]
            value = item[1]
            print(f"{name} = {value}")
        return ""

    def set_initial_state(self):
        self.initial_state = set_initial_state(
            self.nn,
            self.num_sim,
            self.engine,
            self.seed,
            self.same_initial_state,
            self.dtype,
        )

    def get_default_parameters(self):

        par = {
            "G": 0.72,  # global coupling strength
            "dt": 0.01,  # dt for mpr model [ms]
            "dt_bold": 0.001,  # dt for Balloon model [s]
            "J": 14.5,  # model parameter
            "eta": -4.6,
            "tau": 1.0,  # model parameter
            "delta": 0.7,  # model parameter
            "tr": 500.0,  # repetition time [ms]
            "noise_amp": 0.037,  # amplitude of noise
            "same_noise_per_sim": False,  # same noise for all simulations
            "sti_apply": False,  # apply stimulation
            "iapp": 0.0,  # external input
            "t_start": 0.0,  # initial time    [ms]
            "t_cut": 0,  # transition time [ms]
            "t_end": 300_000,  # end time        [ms]
            "num_nodes": None,  # number of nodes
            "weights": None,  # weighted connection matrix
            "rv_decimate": 10,  # sampling step from r and v
            "output": "output",  # output directory
            "RECORD_RV": False,  # store r and v time series
            "RECORD_BOLD": True,  # store BOLD signal
            "RECORD_AVG_r": False,  # store average_r
            "num_sim": 1,
            "method": "heun",
            "engine": "cpu",
            "seed": None,
            "dtype": "float",
            "initial_state": None,
            "same_initial_state": False,
        }
        dt = par["dt"]
        noise_amp = np.array(par["noise_amp"])
        sigma_r = np.sqrt(dt) * np.sqrt(2 * noise_amp)
        sigma_v = np.sqrt(dt) * np.sqrt(4 * noise_amp)
        par["sigma_r"] = sigma_r
        par["sigma_v"] = sigma_v
        # par.update(self.get_balloon_parameters())

        return par
    
    def update_dependent_parameters(self):
        dt = self.dt
        noise_amp = self.noise_amp
        if hasattr(noise_amp, "__iter__"):
            noise_amp = np.array(noise_amp)
        else:
            noise_amp = np.array([noise_amp])
            
        sigma_r = np.sqrt(dt) * np.sqrt(2 * noise_amp)
        sigma_v = np.sqrt(dt) * np.sqrt(4 * noise_amp)
        self.sigma_r = sigma_r
        self.sigma_v = sigma_v
        self._par["sigma_r"] = sigma_r
        self._par["sigma_v"] = sigma_v


    def check_parameters(self, par):
        for key in par.keys():
            if key not in self.valid_parameters:
                raise ValueError(f"Invalid parameter {key:s} provided.")
        
    def prepare_input(self):

        self.G = self.xp.array(self.G)
        self.sigma_r = self.xp.array(self.sigma_r)
        self.sigma_v = self.xp.array(self.sigma_v)
        assert self.weights is not None, "weights must be provided"
        self.weights = self.xp.array(self.weights)
        self.weights = move_data(self.weights, self.engine)
        self.nn = self.num_nodes = self.weights.shape[0]

        if self.initial_state is None:
            self.set_initial_state()

        self.t_end = self.t_end / 10.0
        self.t_start = self.t_start / 10.0
        self.t_cut = self.t_cut / 10.0
        self.eta = prepare_vec(self.eta, self.num_sim, self.engine, self.dtype)
        self.J = prepare_vec(self.J, self.num_sim, self.engine, self.dtype)
        self.delta = prepare_vec(self.delta, self.num_sim, self.engine, self.dtype)

    def f_mpr(self, x, t):
        """
        MPR model
        """

        G = self.G
        J = self.J
        xp = self.xp
        weights = self.weights
        tau = self.tau
        eta = self.eta
        iapp = self.iapp
        ns = self.num_sim
        delta = self.delta
        nn = self.num_nodes
        rtau = 1.0 / tau
        x0 = x[:nn, :]
        x1 = x[nn:, :]
        dxdt = xp.zeros((2 * nn, ns)).astype(self.dtype)
        tau_pi_inv = 1.0 / (tau * np.pi)
        pi2 = np.pi * np.pi
        tau2 = tau * tau

        coupling = weights @ x0
        dxdt[:nn, :] = rtau * (delta * tau_pi_inv + 2 * x0 * x1)
        dxdt[nn:, :] = rtau * (
            x1 * x1 + eta + iapp + J * tau * x0 - (pi2 * tau2 * x0 * x0) + G * coupling
        )

        return dxdt

    def heunStochastic(self, y, t, dt):
        """Heun scheme to integrate MPR model with noise."""

        xp = self.xp
        nn = self.num_nodes
        ns = self.num_sim

        if not self.same_noise_per_sim:
            dW_r = self.sigma_r * xp.random.randn(nn, ns)
            dW_v = self.sigma_v * xp.random.randn(nn, ns)
        else:
            dW_r = self.sigma_r * xp.random.randn(nn, 1)
            dW_v = self.sigma_v * xp.random.randn(nn, 1)

        k1 = self.f_mpr(y, t)
        tmp = y + dt * k1
        tmp[:nn, :] += dW_r
        tmp[nn:, :] += dW_v

        k2 = self.f_mpr(tmp, t + dt)
        y += 0.5 * dt * (k1 + k2)
        y[:nn, :] += dW_r
        y[:nn, :] = (y[:nn, :] > 0) * y[:nn, :]  # set zero if negative
        y[nn:, :] += dW_v

    def sync_(self, engine="gpu"):
        if engine == "gpu":
            cp.cuda.Stream.null.synchronize()
        else:
            pass

    def run(self, verbose=True):

        self.prepare_input()
        dt = self.dt
        rv_decimate = self.rv_decimate
        r_period = dt * 10  # extenting time
        dtt = r_period / 1000.0  # in seconds
        tr = self.tr
        xp = self.xp
        ns = self.num_sim
        nn = self.num_nodes
        engine = self.engine

        n_steps = int(self.t_end / dt)
        bold_decimate = int(np.round(tr / r_period))
        
        B = self.B
        B.allocate_memory(xp, nn, ns, n_steps, bold_decimate, self.dtype)
        
        rv_curr = copy(self.initial_state)
        rv_d = np.array([])
        rv_t = np.array([])
        avg_r = np.array([])
        bold_d = np.array([])
        bold_t = np.array([])

        if self.RECORD_RV:
            rv_d = np.zeros((n_steps // rv_decimate, 2 * nn, ns), dtype="f")
            rv_t = np.zeros((n_steps // rv_decimate), dtype="f")

        if self.RECORD_AVG_r:
            avg_r = np.zeros((nn, ns), dtype="f")

        cc = 0
        for i in tqdm.trange(n_steps - 1, disable=not verbose, desc="Integrating"):

            t_curr = i * dt
            self.heunStochastic(rv_curr, t_curr, dt)

            if ((i % rv_decimate) == 0) and ((i // rv_decimate) < rv_d.shape[0]):

                if self.RECORD_RV:
                    rv_d[i // rv_decimate] = get_(rv_curr, engine, "f")
                    rv_t[i // rv_decimate] = t_curr

                if self.RECORD_AVG_r and i > n_steps // 2:
                    avg_r += get_(rv_curr[:nn, :], engine, "f")
                    cc += 1

            if self.RECORD_BOLD:
                B.do_bold_step(rv_curr[:nn, :], dtt)
                
                if (i % bold_decimate == 0) and ((i // bold_decimate) < B.vv.shape[0]):
                    B.vv[i // bold_decimate] = get_(B.v[1], engine, "f")
                    B.qq[i // bold_decimate] = get_(B.q[1], engine, "f")

        if self.RECORD_BOLD:
            bold_d = B.vo * (B.k1 * (1 - B.qq) + B.k2 * (1 - B.qq / B.vv) + B.k3 * (1 - B.vv))
            bold_t = np.linspace(0, self.t_end - dt * bold_decimate, len(bold_d))
            bold_d = bold_d[bold_t > self.t_cut, ...]
            bold_t = bold_t[bold_t > self.t_cut]
            bold_t = bold_t * 10.0
        avg_r = avg_r / cc

        if self.RECORD_RV:
            rv_t = np.asarray(rv_t).astype("f")
            rv_d = rv_d[rv_t > self.t_cut, ...]
            rv_t = rv_t[rv_t > self.t_cut] * 10.0

        return {
            "rv_t": rv_t,
            "rv_d": rv_d,
            "fmri_t": bold_t,
            "fmri_d": bold_d,
            "avg_r": avg_r,
        }


def set_initial_state(nn, ns, engine, seed=None, same_initial_state=False, dtype=float):
    """
    Set initial state

    Parameters
    ----------
    nn : int
        number of nodes
    ns : int
        number of simulations
    engine : str
        cpu or gpu
    same_initial_condition : bool
        same initial condition for all simulations
    seed : int
        random seed
    dtype : str
        float: float64
        f    : float32
    """

    if seed is not None:
        np.random.seed(seed)

    if same_initial_state:
        y0 = np.random.rand(2 * nn)
        y0 = repmat_vec(y0, ns, engine)
    else:
        y0 = np.random.rand(2 * nn, ns)
        y0 = move_data(y0, engine)

    y0[:nn, :] = y0[:nn, :] * 1.5
    y0[nn:, :] = y0[nn:, :] * 4 - 2

    return y0.astype(dtype)
