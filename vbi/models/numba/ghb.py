import warnings
import numpy as np
from numba import njit
from numba.experimental import jitclass
from numba.core.errors import NumbaPerformanceWarning
from numba import float64, int64

warnings.simplefilter("ignore", category=NumbaPerformanceWarning)


@njit(nogil=True)
def run(P, times):

    G = P.G
    dt = P.dt
    eta = P.eta
    SC = P.weights
    sigma = P.sigma
    omega = P.omega
    tcut = P.tcut
    decimate = P.decimate
    init_state = P.init_state

    epsilon = 0.5
    itaus = 1.25
    itauf = 2.5
    itauo = 1.02040816327
    ialpha = 5.0
    Eo = 0.4
    V0 = 4.0
    k1 = 2.77264
    k2 = 0.572
    k3 = -0.43

    nt = times.shape[0]
    nn = SC.shape[0]

    # state variables
    n_buffer = np.int64(np.floor(nt / decimate) + 1)
    x = np.zeros(nn)
    bold = np.zeros((nn, n_buffer))
    y = np.zeros(nn)
    z = np.array([0.0] * nn + [1.0] * 3 * nn)
    # act = np.zeros((nn, nt))

    # initial conditions (similar value for all regions)
    x_init, y_init = init_state[:nn], init_state[nn:]
    x[:] = x_init
    y[:] = y_init

    for i in range(nn):
        bold[i, 0] = V0 * (
            k1
            - k1 * z[3 * nn + i]
            + k2
            - k2 * (z[3 * nn + i] / z[2 * nn + i])
            + k3
            - k3 * z[2 * nn + i]
        )

    ii = 0 # counter for decimation
    for it in range(nt - 1):
        for i in range(nn):
            gx, gy = 0.0, 0.0
            for j in range(nn):
                gx = gx + SC[i, j] * (x[j] - x[i])
                gy = gy + SC[i, j] * (y[j] - y[i])
            dx = (
                (x[i] * (eta[i] - (x[i] * x[i]) - (y[i] * y[i])))
                - (omega[i] * y[i])
                + (G * gx)
            )
            dy = (
                (y[i] * (eta[i] - (x[i] * x[i]) - (y[i] * y[i])))
                + (omega[i] * x[i])
                + (G * gy)
            )
            dz0 = epsilon * x[i] - itaus * z[i] - itauf * (z[nn + i] - 1)
            dz1 = z[i]
            dz2 = itauo * (z[nn + i] - z[2 * nn + i] ** ialpha)
            dz3 = itauo * (
                z[nn + i] * (1 - (1 - Eo) ** (1 / z[nn + i])) / Eo
                - (z[2 * nn + i] ** ialpha) * z[3 * nn + i] / z[2 * nn + i]
            )

            x[i] = x[i] + dt * dx + np.sqrt(dt) * sigma * np.random.randn()
            y[i] = y[i] + dt * dy + np.sqrt(dt) * sigma * np.random.randn()

            z[i] = z[i] + dt * dz0
            z[nn + i] = z[nn + i] + dt * dz1
            z[2 * nn + i] = z[2 * nn + i] + dt * dz2
            z[3 * nn + i] = z[3 * nn + i] + dt * dz3
            if (it%decimate == 0):
                bold[i, ii + 1] = V0 * (
                    k1
                    - k1 * z[3 * nn + i]
                    + k2
                    - k2 * (z[3 * nn + i] / z[2 * nn + i])
                    + k3
                    - k3 * z[2 * nn + i]
                )
        if (it%decimate == 0):
            ii += 1
    bold = bold[:, times[::decimate]>tcut]
    t_bold = times[times[::decimate]>tcut]
    return t_bold, bold


class GHB_sde(object):
    def __init__(self, par: dict = {}) -> None:
        self.valid_par = [par_spec[i][0] for i in range(len(par_spec))]
        self.check_parameters(par)
        self.P = self.get_par_obj(par)

    def get_par_obj(self, par: dict):
        if "init_state" in par.keys():
            par["init_state"] = np.array(par["init_state"])
        if "weights" in par.keys():
            par["weights"] = np.array(par["weights"])
        return ParGHB(**par)

    def __str__(self) -> str:
        print("GHB model")
        for key in self.valid_par:
            print(f"{key}: {getattr(self.P, key)}")
        return ""

    def check_parameters(self, par: dict) -> None:
        for key in par.keys():
            if key not in self.valid_par:
                raise ValueError(f"Invalid parameter: {key}")

    def set_initial_state(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        assert self.P.weights is not None
        return np.random.uniform(0, 1, 2 * self.P.weights.shape[0])

    def check_input(self):
        assert self.P.weights is not None
        assert self.P.weights.shape[0] == self.P.weights.shape[1]
        assert self.P.eta is not None
        assert self.P.omega is not None
        assert self.P.weights.shape[0] == self.P.eta.shape[0]

    def run(self, par={}, tspan=None, x0=None, verbose=True):
        if x0 is None:
            self.seed = self.P.seed if self.P.seed > 0 else None
            self.P.init_state = self.set_initial_state(seed=self.seed)
        else:
            self.P.init_state = x0

        if tspan is None:
            times = np.arange(0, self.P.tend, self.P.dt)
        else:
            times = np.arange(tspan[0], tspan[1], self.P.dt)

        if par:
            self.check_parameters(par)
            for key in par.keys():
                setattr(self.P, key, par[key])

        self.check_input()
        t, b = run(self.P, times)
        return {'t': t, 'bold': b}


par_spec = [
    ("G", float64),
    ("dt", float64),
    ("seed", int64),
    ("tend", float64),
    ("tcut", float64),
    ("sigma", float64),
    ("eta", float64[:]),
    ("decimate", int64),
    ("omega", float64[:]),
    ("weights", float64[:, :]),
    ("init_state", float64[:]),
]


@jitclass(par_spec)
class ParGHB:
    def __init__(
        self,
        G=1.0,
        dt=0.001,
        sigma=0.1,
        tend=10.0,
        tcut=0.0,
        eta=np.array([]),
        init_state=np.array([]),
        omega=np.array([]),
        weights=np.array([[], []]),
        decimate=1,
    ):
        self.G = G
        self.dt = dt
        self.seed = -1
        self.eta = eta
        self.tend = tend
        self.tcut = tcut
        self.sigma = sigma
        self.omega = omega
        self.weights = weights
        self.decimate = decimate
        self.init_state = init_state
