import tqdm
import cupy as cp
import numpy as np
from copy import copy
from vbi.models.cupy.utils import *


class MPR_sde:
    '''
    Montbrio-Pazo-Roxin model Cupy and Numpy implementation.

    Parameters
    ----------
    par : dict
        Dictionary of parameters.
        - G: global coupling strength
        - dt: time step
        - dt_bold: time step for Balloon model
        - J: model parameter
        - eta: model parameter
        - tau: model parameter
        - delta: model parameter
        - decimate: sampling step from fmri
        - noise_amp: amplitude of noise
        - same_noise_per_sim: same noise for all simulations
        - iapp: external input
        - t_initial: initial time
        - t_transition: transition time
        - t_end: end time
        - num_nodes: number of nodes
        - weights: weighted connection matrix
        - record_step: sampling step from r and v
        - output: output directory
        - RECORD_TS: store r and v time series
        - num_sim: number of simulations
        - method: integration method
        - engine: cpu or gpu
        - seed: seed for random number generator
        - dtype: float or f
        - initial_state: initial state
        - same_initial_state: same initial state for all simulations
        #!TODO: add more details about the parameters
        #TODO: add references
    '''

    valid_parameters = [
        "G", "dt", "dt_bold", "J", "eta", "tau", "delta",
        "decimate", "noise_amp", "iapp", "t_initial", "t_transition",
        "t_end", "weights", "record_step", "same_initial_condition",
        "RECORD_TS", "num_sim", "method", "num_nodes", "initial_state",
        "seed", "engine", "dtype", "output", "alpha", "inv_alpha",
        "tauo", "taus", "tauf",  "K1", "K2", "K3", "E0", "V0", "eps",
        "same_initial_state"
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
        if self.seed is not None:
            self.xp.random.seed(self.seed)

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
            self.nn, self.num_sim, self.engine, self.seed,
            self.same_initial_state, self.dtype)

    def get_balloon_parameters(self):
        '''get balloon model parameters.'''
        alpha = 0.32
        E0 = 0.4
        eps = 0.5

        par = {
            "eps": eps,
            "E0": 0.4,
            "V0": 4.0,
            "alpha": alpha,
            "inv_alpha": 1.0 / alpha,
            "K1": 7.0 * E0,
            "K2": 2 * E0,
            "K3": 1 - eps,
            "taus": 1.54,
            "tauo": 0.98,
            "tauf": 1.44,
        }
        par['inv_tauo'] = 1.0 / par['tauo']
        par['inv_taus'] = 1.0 / par['taus']
        par['inv_tauf'] = 1.0 / par['tauf']
        return par

    def get_default_parameters(self):

        par = {
            "G": 0.72,                   # global coupling strength
            "dt": 0.01,                  # dt for mpr model [ms]
            "dt_bold": 0.001,            # dt for Balloon model [s]
            "J": 14.5,                   # model parameter
            "eta": -4.6,
            "tau": 1.0,                  # model parameter
            "delta": 0.7,                # model parameter

            "decimate": 500,             # sampling step from fmri
            "noise_amp": 0.037,          # amplitude of noise

            "same_noise_per_sim": False,  # same noise for all simulations

            "sti_apply": False,          # apply stimulation
            "iapp": 0.0,                 # external input
            "t_initial": 0.0,            # initial time    [ms]
            "t_transition": 30_000,      # transition time [ms]
            "t_end": 300_000,            # end time        [ms]

            "num_nodes": None,           # number of nodes
            "weights": None,             # weighted connection matrix
            "record_step": 10,           # sampling step from r and v
            "output": "output",          # output directory
            "RECORD_TS": 0,              # store r and v time series
            "num_sim": 1,
            "method": "heun",
            "engine": "cpu",
            "seed": None,
            "dtype": "float",
            "initial_state": None,
            "same_initial_state": False,
        }
        dt = par["dt"]
        noise_amp = par["noise_amp"]
        sigma_r = np.sqrt(dt) * np.sqrt(2 * noise_amp)
        sigma_v = np.sqrt(dt) * np.sqrt(4 * noise_amp)
        par['sigma_r'] = sigma_r
        par['sigma_v'] = sigma_v
        par.update(self.get_balloon_parameters())

        return par

    def check_parameters(self, par):
        for key in par.keys():
            assert key in self.valid_parameters, "Invalid parameter: " + key

    def prepare_input(self):

        self.G = self.xp.array(self.G)
        assert (self.weights is not None), "weights must be provided"
        self.weights = self.xp.array(self.weights).T  # ! Directed network #!TODO: check
        self.weights = move_data(self.weights, self.engine)
        self.nn = self.num_nodes = self.weights.shape[0]

        if self.initial_state is None:
            self.set_initial_state()

        self.t_end = self.t_end / 10.0
        self.t_initial = self.t_initial / 10.0
        self.t_transition = self.t_transition / 10.0
        self.eta = prepare_vec(self.eta, self.num_sim, self.engine, self.dtype)

    def f_mpr(self, x, t):
        '''
        MPR model
        '''

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
        dxdt = xp.zeros((2*nn, ns)).astype(self.dtype)
        delta_over_tau_pi = delta / (tau * np.pi)
        J_tau = J * tau
        pi2 = np.pi * np.pi
        tau2 = tau * tau

        coupling = weights @ x0
        dxdt[:nn, :] = rtau * (delta_over_tau_pi + 2 * x0 * x1)
        dxdt[nn:, :] = rtau * (x1 * x1 + eta + iapp + J_tau *
                               x0 - (pi2 * tau2 * x0 * x0) +
                               G * coupling)

        return dxdt

    def f_fmri(self, xin, x, t):

        E0 = self.E0
        xp = self.xp
        nn = self.num_nodes
        ns = self.num_sim
        inv_tauf = self.inv_tauf
        inv_tauo = self.inv_tauo
        inv_taus = self.inv_taus
        inv_alpha = self.inv_alpha

        dxdt = xp.zeros((4*nn, ns)).astype(self.dtype)
        s = x[:nn, :]
        f = x[nn:2*nn, :]
        v = x[2*nn:3*nn, :]
        q = x[3*nn:, :]
        x0in = xin[:nn, :]  # use r

        dxdt[:nn, :] = x0in - inv_taus * s - inv_tauf * (f - 1.0)
        dxdt[nn:(2*nn), :] = s
        dxdt[(2*nn):(3*nn), :] = inv_tauo * (f - v ** (inv_alpha))
        dxdt[3*nn:, :] = (inv_tauo) * ((f * (1.0 - (1.0 - E0) ** (1.0 / f)) / E0) -
                                       (v ** (inv_alpha)) * (q / v))

        return dxdt

    def integrate_fmri(self, yin, y, t):
        '''
        Integrate Balloon model

        Parameters
        ----------
        yin : array [2*nn, ns]
            r and v time series, r is used as input
        y : array [4*nn, ns]
            state, update in place
        t : float
            time

        Returns
        -------
        yb : array [nn, ns]
            BOLD signal

        '''

        V0 = self.V0
        K1 = self.K1
        K2 = self.K2
        K3 = self.K3

        nn = self.num_nodes
        self.heunDeterministic(yin, y, t)
        yb = V0 * (K1 * (1.0 - y[(3*nn):, :]) + K2 * (1.0 - y[(3*nn):,
                   :] / y[(2*nn):(3*nn), :]) + K3 * (1.0 - y[(2*nn):(3*nn), :]))
        return yb

    def heunDeterministic(self, yin, y, t):
        '''Heun scheme for bold model'''

        dt = self.dt_bold
        k1 = self.f_fmri(yin, y, t)
        tmp = y + dt * k1
        k2 = self.f_fmri(yin, tmp, t + dt)
        y += 0.5 * dt * (k1 + k2)

    def heunStochastic(self, y, t, dt):
        '''Heun scheme to integrate MPR model with noise.'''

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

        k2 = self.f_mpr(tmp, t+dt)
        y += 0.5 * dt * (k1 + k2)
        y[:nn, :] += dW_r
        y[:nn, :] = (y[:nn, :] > 0) * y[:nn, :]  # set zero if negative
        y[nn:, :] += dW_v

    def eulerStochastic(self, y, t, dt):
        '''Euler scheme'''

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
        y += y + dt * k1
        y[:nn, :] += dW_r
        y[:nn, :] = (y[:nn, :] > 0) * y[:nn, :]  # set zero if negative
        y[nn:, :] += dW_v

    def bgStochastic(self, y, t, dt):
        '''Bogacki Shampine scheme'''

        xp = self.xp
        nn = self.num_nodes
        ns = self.num_sim

        if not self.same_noise_per_sim:
            dW_r = self.sigma_r * xp.random.randn(nn, ns)
            dW_v = self.sigma_v * xp.random.randn(nn, ns)
        else:
            dW_r = self.sigma_r * xp.random.randn(nn, 1)
            dW_v = self.sigma_v * xp.random.randn(nn, 1)

        f = self.f_mpr

        k1 = f(y, t)
        k2 = f(y + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = f(y + 0.75 * dt * k2, t + 0.75 * dt)
        y += dt / 9.0 * (2.0 * k1 + 3.0 * k2 + 4.0 * k3)

        y[:nn, :] += dW_r
        y[:nn, :] = (y[:nn] > 0) * y[:nn]  # set zero if negative
        y[nn:, :] += dW_v

    def set_integrator(self):
        if self.method == "euler":
            return self.eulerStochastic
        elif self.method == "heun":
            return self.heunStochastic
        elif self.method == "bg":
            return self.bgStochastic
        else:
            raise ValueError("Invalid method: " + self.method)

    def sync_(self, engine="gpu"):
        if engine == "gpu":
            cp.cuda.Stream.null.synchronize()
        else:
            pass

    def run(self, verbose=True):

        self.prepare_input()
        integrator = self.set_integrator()
        dt = self.dt
        xp = self.xp
        ns = self.num_sim
        nn = self.num_nodes
        dec = self.decimate
        engine = self.engine
        rs = self.record_step
        t_cut = self.t_transition
        n_steps = np.ceil(self.t_end / dt).astype(int)
        i_cut = np.ceil(t_cut / dt).astype(int)

        y0_fmri = xp.zeros((4 * nn, ns)).astype(self.dtype)
        y0_fmri[nn:, :] = 1.0
        y0 = copy(self.initial_state)

        rv_d = np.array([]).astype("f")
        fmri_t = []
        rv_t = []

        if self.RECORD_TS:
            _nt = np.ceil((n_steps - i_cut)/rs).astype(int)
            rv_d = np.zeros((_nt, 2*nn, ns), dtype="f")

        _nt = np.ceil((n_steps)/(rs*dec)).astype(int)
        fmri_d = np.zeros((_nt, nn, ns), dtype="f")

        ii = 0  # index for rv_d
        jj = 0  # index for fmri_D

        for it in tqdm.trange(n_steps, disable=not verbose, desc="Integrating"):

            t = it * dt
            integrator(y0, t, dt)
            self.sync_(engine)  # !TODO: check if this is necessary

            if (it % rs) == 0:  # and (it >= i_cut):

                if self.RECORD_TS and (it >= i_cut):
                    rv_d[ii, :, :] = get_(y0, engine, "f")
                    rv_t.append(t)
                    ii += 1

                fmri_i = self.integrate_fmri(y0, y0_fmri, t)
                self.sync_(engine)

            if it % (dec*rs) == 0:
                fmri_d[jj, :, :] = get_(fmri_i, engine, "f")
                jj += 1
                fmri_t.append(t)
        fmri_d = np.asarray(fmri_d).astype("f")
        fmri_t = np.asarray(fmri_t).astype("f")
        fmri_d = fmri_d[fmri_t >= t_cut, :, :]
        fmri_t = fmri_t[fmri_t >= t_cut] * 10.0
        rv_t = np.asarray(rv_t).astype("f") * 10.0

        return {
            "rv_t": rv_t,
            "rv_d": rv_d,
            "fmri_d": fmri_d,
            "fmri_t": fmri_t,
        }


def set_initial_state(nn, ns, engine, seed=None, same_initial_state=False, dtype=float):
    '''
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
    '''

    if seed is not None:
        np.random.seed(seed)

    if same_initial_state:
        y0 = np.random.rand(2*nn)
        y0 = repmat_vec(y0, ns, engine)
    else:
        y0 = np.random.rand(2*nn, ns)
        y0 = move_data(y0, engine)

    y0[:nn, :] = y0[:nn, :] * 1.5
    y0[nn:, :] = y0[nn:, :] * 4 - 2

    return y0.astype(dtype)
