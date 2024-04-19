import tqdm
import cupy as cp
from copy import copy
from vbi.models.cupy.utils import *


class GHB_sde:
    """
    Generic Hopf model cupy implementation

    Parameters
    ----------
    par: dict
        Dictionary of parameters
        - 'G': Global coupling
        - 'dt': Time step

    """

    epsilon = 0.5
    itaus = 1.25
    itauf = 2.5
    itauo = 1.02040816327
    ialpha = 5.
    E0 = 0.4
    V0 = 4.
    K1 = 2.77264
    K2 = 0.572
    K3 = -0.43

    def __init__(self, par: dict = {}) -> None:

        self.valid_params = list(self.get_default_parameters().keys())
        self.check_parameters(par)
        self.par_ = self.get_default_parameters()
        self.par_.update(par)

        for item in self.par_.items():
            setattr(self, *item)

        self.xp = get_module(self.engine)
        if self.seed is not None:
            self.xp.random.seed(self.seed)

    def __call__(self):
        print("GHB model")
        return self.par_

    def __str__(self):
        print("GHB model")
        print("-" * 50)
        for item in self.par_items():
            name = item[0]
            value = item[1]
            print(f"{name} : {value}")
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

    def check_parameters(self, par):
        for key in par.keys():
            assert key in self.valid_params, "Invalid parameter: " + key

    def get_default_parameters(self):
        par = {
            "G": 25.0,
            "t_cut": 0,
            "dt": 0.01,
            "eta": None,
            "num_sim": 1,
            "sigma": 0.1,
            "seed": None,
            "decimate": 1,
            "omega": None,
            "t_end": 10.0,
            "engine": "cpu",
            "weights": None,
            "dtype": "float",
            "method": "euler",
            "output": "output",
            "initial_state": None,
            "same_initial_state": False,
        }
        return par

    def prepare_input(self):
        self.G = self.xp.array(self.G, dtype=self.dtype)
        assert self.weights is not None, "weights not provided"
        self.weights = self.xp.array(self.weights, dtype=self.dtype)
        self.weights = self.weights.reshape(self.weights.shape+(1,))
        self.weights = move_data(self.weights, self.engine)
        self.nn = self.num_nodes = self.weights.shape[0]


        if self.initial_state is None:
            self.set_initial_state()
        else:
            self.initial_state = move_data(
                self.initial_state, self.engine)


        self.eta = prepare_vec(self.eta, self.num_sim, self.engine, self.dtype)
        self.omega = prepare_vec(self.omega, self.num_sim, self.engine, self.dtype)

    def f_sys(self, x0, t):

        G = self.G
        xp = self.xp
        nn = self.nn
        eta = self.eta
        x = x0[:nn, :]
        y = x0[nn:, :]
        ns = self.num_sim
        sc = self.weights
        omega = self.omega

        gx = xp.sum(sc * (x - x[:, None]), axis=1)
        gy = xp.sum(sc * (y - y[:, None]), axis=1)
        dxdt = xp.zeros((2 * nn, ns)).astype(self.dtype)

        dxdt[:nn, :] = x * (eta - x * x - y * y) - omega * y + G * gx
        dxdt[nn:, :] = y * (eta - x * x - y * y) + omega * x + G * gy

        return dxdt

    def f_fmri(self, xin, x, t):

        E0 = self.E0
        xp = self.xp
        nn = self.num_nodes
        ns = self.num_sim
        itauf = self.itauf
        itauo = self.itauo
        itaus = self.itaus
        ialpha = self.ialpha

        dxdt = xp.zeros((4 * nn, ns)).astype(self.dtype)
        s = x[:nn, :]
        f = x[nn : 2 * nn, :]
        v = x[2 * nn : 3 * nn, :]
        q = x[3 * nn :, :]

        dxdt[:nn, :] = xin[:nn, :] - itaus * s - itauf * (f - 1.0)
        dxdt[nn : (2 * nn), :] = s
        dxdt[(2 * nn) : (3 * nn), :] = itauo * (f - v ** (ialpha))
        dxdt[3 * nn :, :] = (itauo) * (
            (f * (1.0 - (1.0 - E0) ** (1.0 / f)) / E0) - (v ** (ialpha)) * (q / v)
        )

        return dxdt

    def heun_sde_step(self, x0, t):

        xp = self.xp
        dt = self.dt
        dx = self.f_sys(x0, t) * dt
        dW = self.sigma * xp.random.normal(0, 1, size=x0.shape) * xp.sqrt(dt)
        x1 = x0 + dx + dW
        dx1 = self.f_sys(x1, t + dt) * dt
        return x0 + 0.5 * (dx + dx1) + dW

    def heun_ode_step(self, yin, y, t):

        dt = self.dt
        dy = self.f_fmri(yin, y, t) * dt
        y1 = y + dy
        dy1 = self.f_fmri(yin, y1, t + dt) * dt
        return y + 0.5 * (dy + dy1)

    def intg_fmri(self, yin, y, t):
        """
        Integrate one step of Balloon model

        Parameters
        ----------
        yin: array
            input
        y: array [4*nn, ns]
            state
        t : float
            current time

        Returns
        -------
        bold: array [nn, ns]
            BOLD signal
        y: array [4*nn, ns]
            updated state

        """

        V0 = self.V0
        K1 = self.K1
        K2 = self.K2
        K3 = self.K3

        nn = self.num_nodes
        y = self.heun_ode_step(yin, y, t)
        bold = V0 * (
            K1 * (1.0 - y[(3 * nn) :, :])
            + K2 * (1.0 - y[(3 * nn) :, :] / y[(2 * nn) : (3 * nn), :])
            + K3 * (1.0 - y[(2 * nn) : (3 * nn), :])
        )

        return bold, y

    def sync(self, engine="gpu"):
        if engine == "gpu":
            cp.cuda.Stream.null.synchronize()
        else:
            pass

    def run(self, x0=None, verbose=True):
        """
        run ghb model
        """
        self.prepare_input()
        dt = self.dt
        xp = self.xp
        ns = self.num_sim
        nn = self.num_nodes
        dec = self.decimate
        engine = self.engine
        t_cut = self.t_cut
        n_steps = np.ceil(self.t_end / dt).astype(int)

        y0_state = xp.zeros((4 * nn, ns)).astype(self.dtype)
        y0_state[nn:, :] = 1.0
        y0 = copy(self.initial_state)
        bold = np.zeros((nn, n_steps // dec, ns)).astype(np.float32)

        for it in tqdm.trange(n_steps, disable=not verbose, desc="Integrating"):
            y0 = self.heun_sde_step(y0, it * dt)
            bold_, y0_state = self.intg_fmri(y0, y0_state, it * dt)
            self.sync(engine)
            if it % dec == 0:
                bold[:, it // dec, :] = bold_.get() if engine == "gpu" else bold_

        t = np.arange(0, self.t_end, dec * dt).astype(np.float32)
        bold = bold[:, t > t_cut, :]
        t_bold = t[t > t_cut]
        return {"t": t_bold, "bold": bold}


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

    return y0.astype(dtype)
