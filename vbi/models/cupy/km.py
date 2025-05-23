import tqdm
import numpy as np
from vbi.models.cupy.utils import *


class KM_sde:
    
    """
    Kuramoto model with noise (stochastic differential equation)
    
    Parameters
    ----------
    G: float
        global coupling strength
    dt: float
        time step
    noise_amp: float
        noise amplitude
    weights: array
        weighted connection matrix
    omega: array
        natural angular frequency
    seed: int
        fix random seed for initial state
    t_cut: float
        transition time
    t_end: float
        end time
    decimate: int
        decimate the output time series
    output: str
        output directory
    initial_state: array
        initial state
    engine: str
        cpu or gpu
    type: type
        data type for calculations, default is np.float32
    alpha: array # TODO not implemented
        frustration matrix 
    num_sim: int
        number of simulations
    method: str
        integration method, default is heun
    same_initial_state: bool
        use the same initial state for all simulations, default is False    
    
    """

    def __init__(self, par={}) -> None:

        self._par = self.get_default_parameters()
        self.valid_parameters = list(self._par.keys())
        self.check_parameters(par)
        self._par.update(par)

        for item in self._par.items():
            setattr(self, item[0], item[1])

        self.xp = get_module(self.engine)
        self.ns = self.num_sim
        self.nn = self.num_nodes = self.weights.shape[0]

        if self.seed is not None:
            self.xp.random.seed(self.seed)

        if self.initial_state is None:
            self.INITIAL_STATE_SET = False

    def set_initial_state(self):
        self.INITIAL_STATE_SET = True
        self.initial_state = set_initial_state(
            self.nn, self.ns, self.xp, self.seed, self.same_initial_state)

    def __str__(self) -> str:
        print (f"Kuramoto model with noise (sde), {self.engine} implementation.")
        print ("----------------")
        for item in self._par.items():
            print (f"{item[0]} = {item[1]}")
        return ""

    def __call__(self):
        print(
            f"Kuramoto model with noise (sde), {self.engine} implementation.")
        return self._par

    def get_default_parameters(self):

        return {
            "G": 1.0,                        # global coupling strength
            "dt": 0.01,                      # time step
            "noise_amp": 0.1,                # noise amplitude
            "weights": None,                 # weighted connection matrix
            "omega": None,                   # natural angular frequency
            "seed": None,                    # fix random seed for initial state
            "t_start": 0.0,                  # initial time
            "t_cut": 0.0,                    # transition time
            "t_end": 100.0,                  # end time
            "output": "output",              # output directory
            "initial_state": None,           # initial state
            "engine": "cpu",                 # cpu or gpu
            "type": np.float32,              # output times series data type
            "alpha": None,                   # frustration matrix # TODO not implemented
            "num_sim": 1,                    # number of simulations
            "method": "heun",                # integration method
            "same_initial_state": False,     # use the same initial state for all simulations
            "decimate": 1,                   # decimate the output time series

        }

    def check_parameters(self, par):
        for key in par.keys():
            if key not in self.valid_parameters:
                raise ValueError(f"Invalid parameter: {key}")

    def prepare_input(self):

        assert(self.weights is not None), "weights must be provided"
        assert (self.omega is not None), "omega must be provided"

        self.G = self.xp.array(self.G)
        self.weights = self.xp.array(self.weights).T #! Directed network
        self.weights = self.weights.reshape(self.weights.shape+(1,))
        self.weights = move_data(self.weights, self.engine)

        if self.omega.ndim == 1:
            self.omega = repmat_vec(self.omega, self.ns, self.engine)

    def f_sys(self, x, t):
        return self.omega + self.G * self.xp.sum(self.weights * self.xp.sin(x - x[:, None]), axis=1)

    def euler(self, x, t):
        ''' Euler's method integration'''
        coef = self.xp.sqrt(self.dt)
        dW = self.xp.random.normal(0, self.noise_amp, size=x.shape)
        return x + self.dt * self.f_sys(x, t) + coef * dW

    def heun(self, x, t):
        ''' Heun's method integration'''
        coef = self.xp.sqrt(self.dt)
        dW = self.xp.random.normal(0, self.noise_amp, size=x.shape)
        k1 = self.f_sys(x, t) * self.dt
        tmp = x + k1 + coef * dW
        k2 = self.f_sys(tmp, t + self.dt) * self.dt
        return x + 0.5 * (k1 + k2) + coef * dW

    def integrate(self, t, verbose=True):
        ''' Integrate the model'''
        x = self.initial_state
        xs = []
        integrator = self.euler if self.method == "euler" else self.heun
        n_transition = int(self.t_cut /
                           self.dt) if self.t_cut > 0 else 1

        for it in tqdm.tqdm(range(1, len(t)), disable=not verbose, desc="Integrating"):
            x = integrator(x, t[it])
            if (it >= n_transition) and (it % self.decimate == 0):
                if self.engine == "gpu":
                    xs.append(x.get())
                else:
                    xs.append(x)
        xs = np.asarray(xs).astype(self.type)
        t = t[n_transition:]

        return {"t": t, "x": xs}
    
    def step(self, x, t):
        ''' Step function for the model'''
        if self.method == "euler":
            return self.euler(x, t)
        elif self.method == "heun":
            return self.heun(x, t)
        else:
            raise ValueError(f"Invalid method: {self.method}")

    def run(self, x0=None, verbose=True):
        '''
        run the model

        Parameters
        ----------
        par: dict
            parameters
        x0: array
            initial state
        verbose: bool
            print progress bar

        Returns
        -------
        dict
            x: array [n_timesteps, n_regions, n_sim]
                time series data
            t: array
                time points [n_timepoints]

        '''

        if x0 is not None:
            self.initial_state = x0
            self.INITIAL_STATE_SET = True
        else:
            self.set_initial_state()
            if verbose:
                print("Initial state set randomly.")
        # self.check_parameters(par)
        # for key in par.keys():
        #     setattr(self, key, par[key]['value'])
        self.prepare_input()
        t = self.xp.arange(self.t_initial, self.t_end, self.dt)
        data = self.integrate(t, verbose=verbose)

        data['t'] = data['t'].get() if self.engine == "gpu" else data['t']
        return data


def set_initial_state(nn, ns=1, engine="cpu", seed=None, same_initial_state=False, dtype=float):
    '''
    set initial state

    Parameters
    ----------

    nn: int
        number of nodes
    ns: int
        number of states
    engine: str
        cpu or gpu
    seed: int
        set random seed if not None
    same_initial_state: bool
        use the same initial state for all simulations

    Returns
    -------
    x: array [nn, ns]
        initial state

    '''


    if seed is not None:
        np.random.seed(seed)
    if same_initial_state:
        x0 = np.random.uniform(0, 2*np.pi, size=nn)
        x0 = repmat_vec(x0, ns, engine)
    else:
        x0 = np.random.uniform(0, 2*np.pi, size=(nn, ns))
        x0 = move_data(x0, engine)

    return x0.astype(dtype)
