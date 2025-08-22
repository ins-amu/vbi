import warnings
import numpy as np
from numba import njit
from numba.experimental import jitclass
from numba import float64, boolean, int64
from numba.extending import register_jitable
from numba.core.errors import NumbaPerformanceWarning

warnings.simplefilter("ignore", category=NumbaPerformanceWarning)

# ---------------------------------------------------------------
# Helper utilities (broadcasting, seeding, initial state)
# ---------------------------------------------------------------

@register_jitable
def set_seed_compat(x):
    np.random.seed(x)


@register_jitable
def _as_1d_array_like(x, nn):
    """Broadcast scalar to 1D array of length nn if needed."""
    x_arr = np.array(x) if not isinstance(x, np.ndarray) else x
    if x_arr.ndim == 0:
        return np.ones(nn) * float(x_arr)
    if x_arr.ndim == 1 and x_arr.shape[0] == nn:
        return x_arr.astype(np.float64)
    raise ValueError("Parameter must be scalar or 1D array of length nn")


@njit
def set_initial_state_jr(nn, seed=-1):
    """Initial state for JR: stack 6*n vectors.
    Mirrors ranges similar to the CuPy reference implementation.
    """
    if seed is not None and seed >= 0:
        set_seed_compat(seed)

    y0 = np.random.uniform(-1.0, 1.0, nn)      # x
    y1 = np.random.uniform(-500.0, 500.0, nn)  # y
    y2 = np.random.uniform(-50.0, 50.0, nn)    # z
    y3 = np.random.uniform(-6.0, 6.0, nn)      # x'
    y4 = np.random.uniform(-20.0, 20.0, nn)    # y'
    y5 = np.random.uniform(-500.0, 500.0, nn)  # z'

    y = np.zeros(6 * nn)
    y[:nn] = y0
    y[nn:2*nn] = y1
    y[2*nn:3*nn] = y2
    y[3*nn:4*nn] = y3
    y[4*nn:5*nn] = y4
    y[5*nn:6*nn] = y5
    return y


# ---------------------------------------------------------------
# JR parameters as a jitclass (Numba-friendly container)
# ---------------------------------------------------------------

jr_spec = [
    ("G", float64),
    ("A", float64),
    ("B", float64),
    ("a", float64),
    ("b", float64),
    ("v0", float64),
    ("vmax", float64),
    ("r", float64),
    ("mu", float64),
    ("noise_amp", float64),
    ("dt", float64),
    ("t_cut", float64),
    ("t_end", float64),
    ("decimate", int64),
    ("nn", int64),
    ("seed", int64),
    ("sigma", float64),  # sqrt(dt) * noise_amp
    # arrays
    ("weights", float64[:, :]),
    ("C0", float64[:]),
    ("C1", float64[:]),
    ("C2", float64[:]),
    ("C3", float64[:]),
    ("initial_state", float64[:]),
]


@jitclass(jr_spec)
class ParJR:
    def __init__(
        self,
        weights,
        G=1.0,
        A=3.25,
        B=22.0,
        a=0.1,
        b=0.05,
        v0=6.0,
        vmax=0.005,
        r=0.56,
        mu=0.24,
        noise_amp=0.01,
        dt=0.01,
        t_cut=0.0,
        t_end=1000.0,
        decimate=1,
        C0=135.0,
        C1=0.8*135.0,
        C2=0.25*135.0,
        C3=0.25*135.0,
        seed=-1
    ):
        self.weights = weights
        self.nn = len(weights)

        self.G = G
        self.A = A
        self.B = B
        self.a = a
        self.b = b
        self.v0 = v0
        self.vmax = vmax
        self.r = r
        self.mu = mu
        self.noise_amp = noise_amp
        self.dt = dt
        self.t_cut = t_cut
        self.t_end = t_end
        self.decimate = decimate
        self.seed = seed

        # C arrays are now passed pre-processed from outside
        self.C0 = C0
        self.C1 = C1
        self.C2 = C2
        self.C3 = C3

        self.sigma = np.sqrt(dt) * noise_amp
        self.initial_state = np.zeros(6 * self.nn)  # set by caller later


# ---------------------------------------------------------------
# JR model equations + integrator (Numba-jitted)
# ---------------------------------------------------------------

@register_jitable
def S_sigmoid(x, vmax, r, v0):
    """Numerically stable sigmoid function to avoid overflow."""
    z = r * (v0 - x)
    # Clip z to avoid overflow: exp(700) is near overflow limit
    z_clipped = np.clip(z, -700, 700)
    return vmax / (1.0 + np.exp(z_clipped))


@njit
def f_jr(x, t, P):
    nn = P.nn

    # Unpack state
    x0 = x[0*nn:1*nn]  # x
    y0 = x[1*nn:2*nn]  # y
    z0 = x[2*nn:3*nn]  # z
    xp = x[3*nn:4*nn]  # x'
    yp = x[4*nn:5*nn]  # y'
    zp = x[5*nn:6*nn]  # z'

    # Precompute constants
    Aa = P.A * P.a
    Bb = P.B * P.b
    aa = P.a * P.a
    bb = P.b * P.b

    # Coupling term: weights @ (y - z)
    couplings = S_sigmoid(P.weights.dot(y0 - z0), P.vmax, P.r, P.v0)

    # Allocate derivative
    dxdt = np.zeros_like(x)

    # Dynamics
    dxdt[0*nn:1*nn] = xp
    dxdt[1*nn:2*nn] = yp
    dxdt[2*nn:3*nn] = zp

    dxdt[3*nn:4*nn] = Aa * S_sigmoid(y0 - z0, P.vmax, P.r, P.v0) - 2.0 * P.a * xp - aa * x0
    dxdt[4*nn:5*nn] = (
        Aa * (P.mu + P.C1 * S_sigmoid(P.C0 * x0, P.vmax, P.r, P.v0) + P.G * couplings)
        - 2.0 * P.a * yp - aa * y0
    )
    dxdt[5*nn:6*nn] = Bb * P.C3 * S_sigmoid(P.C2 * x0, P.vmax, P.r, P.v0) - 2.0 * P.b * zp - bb * z0

    return dxdt


@njit
def heun_sde(x, t, P):
    nn = P.nn
    dt = P.dt

    # Stochastic drive on the y' block, sigma already includes sqrt(dt)
    dW = P.sigma * np.random.randn(nn)

    k1 = f_jr(x, t, P)
    x1 = x + dt * k1
    x1[4*nn:5*nn] += dW

    k2 = f_jr(x1, t + dt, P)
    x = x + 0.5 * dt * (k1 + k2)
    x[4*nn:5*nn] += dW

    return x


# ---------------------------------------------------------------
# Top-level integrate and driver class (mirrors mpr.py style)
# ---------------------------------------------------------------

def integrate(P):
    nn = P.nn
    dt = P.dt
    dec = P.decimate

    # Ensure initial state is defined
    x = P.initial_state.copy()

    nt = int(P.t_end / dt)
    tspan = np.linspace(0.0, (nt - 1) * dt, nt)

    # Cut & decimate bookkeeping
    i_cut = int(np.searchsorted(tspan, P.t_cut, side='left'))
    n_keep = (nt - i_cut + (dec - 1)) // dec

    # Output: y - z 
    ts = np.zeros(n_keep, dtype=np.float32)
    ys = np.zeros((n_keep, nn), dtype=np.float32)

    k = 0
    for i in range(nt):
        t = tspan[i]
        x = heun_sde(x, t, P)

        if i >= i_cut and ((i - i_cut) % dec == 0):
            ts[k] = t
            y0 = x[1*nn:2*nn]
            z0 = x[2*nn:3*nn]
            ys[k, :] = (y0 - z0).astype(np.float32)
            k += 1
            if k >= n_keep:
                break

    return {"t": ts, "x": ys}


class JR_sde:
    """
    Numba implementation of the Jansen-Rit neural mass model.
    
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
        * - `a`
          - Inverse time constant of the excitatory postsynaptic potential (1/a = time constant).
          - 0.1 (time constant: 10.0)
        * - `b`
          - Inverse time constant of the inhibitory postsynaptic potential (1/b = time constant).
          - 0.05 (time constant: 20.0)
        * - `C0`
          - Average number of synapses between pyramidal cells and excitatory interneurons. If array-like, it should be of length `nn` (number of nodes).
          - 135.0
        * - `C1`
          - Average number of synapses between excitatory interneurons and pyramidal cells. If array-like, it should be of length `nn`.
          - 0.8 * 135.0
        * - `C2`
          - Average number of synapses between pyramidal cells and inhibitory interneurons. If array-like, it should be of length `nn`.
          - 0.25 * 135.0
        * - `C3`
          - Average number of synapses between inhibitory interneurons and pyramidal cells. If array-like, it should be of length `nn`.
          - 0.25 * 135.0
        * - `vmax`
          - Maximum firing rate of the sigmoid function.
          - 0.005
        * - `v0`
          - Potential at half of maximum firing rate (inflection point of sigmoid).
          - 6.0
        * - `r`
          - Slope of sigmoid function at `v0`.
          - 0.56
        * - `G`
          - Global coupling strength scaling the network connections.
          - 1.0
        * - `mu`
          - Mean input to the excitatory population (external drive).
          - 0.24
        * - `noise_amp`
          - Amplitude of the stochastic noise applied to the excitatory population.
          - 0.01
        * - `weights`
          - Structural connectivity matrix of shape (`nn`, `nn`). Must be provided.
          - None
        * - `dt`
          - Integration time step.
          - 0.01
        * - `t_end`
          - End time of simulation.
          - 1000.0
        * - `t_cut`
          - Time from which to start collecting output (burn-in period).
          - 0.0
        * - `decimate`
          - Decimation factor for the output time series (every `decimate`-th point is saved).
          - 1
        * - `seed`
          - Random seed for reproducible simulations. If -1 or None, no seeding is applied.
          - -1
        * - `initial_state`
          - Initial state vector of shape (6*nn,). If None, random initial conditions are generated.
          - None

    Usage example (single simulation):
        >>> import numpy as np
        >>> from vbi.models.numba.jansen_rit import JR_sde
        >>> W = np.eye(2)  # 2-node demo connectivity
        >>> jr = JR_sde({"weights": W, "dt": 0.01, "t_end": 200.0, "t_cut": 100.0, "decimate": 1})
        >>> out = jr.run()
        >>> t, x = out["t"], out["x"]  # x has shape (n_step, nn)

    Notes
    -----
    The Jansen-Rit model describes the dynamics of a cortical column with three neural populations:
    - Pyramidal cells (main excitatory population)
    - Excitatory interneurons
    - Inhibitory interneurons
    
    The model equations are integrated using the Heun stochastic integration scheme.
    The output represents the difference between excitatory and inhibitory postsynaptic potentials (y - z),
    which corresponds to the local field potential that can be measured experimentally.
    """

    def __init__(self, par_jr: dict):
        """
        Initialize the Jansen-Rit model.

        Parameters
        ----------
        par_jr : dict
            Dictionary containing model parameters. See class documentation for available parameters.
            The 'weights' parameter is required and must be a square connectivity matrix.
        """
        # Validate weights early and create parameter jitclass
        if "weights" not in par_jr or par_jr["weights"] is None:
            raise ValueError("'weights' must be provided (square connectivity matrix)")

        W = np.array(par_jr["weights"], dtype=np.float64)
        if W.ndim != 2 or W.shape[0] != W.shape[1]:
            raise ValueError("'weights' must be a square 2D array")

        nn = len(W)
        
        # Pre-process C parameters before passing to jitclass
        params = dict(par_jr)
        params["weights"] = W
        
        # Handle C parameters - broadcast them here outside jitclass
        for c_name in ["C0", "C1", "C2", "C3"]:
            if c_name in params:
                c_val = params[c_name] 
                params[c_name] = _as_1d_array_like(c_val, nn)
            else:
                # Set defaults
                if c_name == "C0":
                    params[c_name] = _as_1d_array_like(135.0, nn)
                elif c_name == "C1":
                    params[c_name] = _as_1d_array_like(0.8*135.0, nn)
                elif c_name == "C2":
                    params[c_name] = _as_1d_array_like(0.25*135.0, nn)
                elif c_name == "C3":
                    params[c_name] = _as_1d_array_like(0.25*135.0, nn)
        
        # Create jitclass instance
        self.P = ParJR(**params)

        # Seed handling
        self.seed = int(self.P.seed)
        if self.seed >= 0:
            np.random.seed(self.seed)

        # Ensure initial state
        if "initial_state" in par_jr and par_jr["initial_state"] is not None:
            x0 = np.array(par_jr["initial_state"], dtype=np.float64)
            if x0.shape[0] != 6 * self.P.nn:
                raise ValueError("initial_state must have length 6*nn")
            self.P.initial_state = x0
        else:
            self.P.initial_state = set_initial_state_jr(self.P.nn, self.seed)

        self._checked = False

    def __str__(self) -> str:
        """
        Return a string representation of the model parameters.
        
        Returns
        -------
        str
            Formatted string showing all model parameters and their values.
        """
        print("Jansen-Rit Model (Numba)")
        print("------------------------")
        
        # Model parameters
        print(f"G = {self.P.G}")
        print(f"A = {self.P.A}")
        print(f"B = {self.P.B}")
        print(f"a = {self.P.a}")
        print(f"b = {self.P.b}")
        print(f"v0 = {self.P.v0}")
        print(f"vmax = {self.P.vmax}")
        print(f"r = {self.P.r}")
        print(f"mu = {self.P.mu}")
        print(f"noise_amp = {self.P.noise_amp}")
        
        # Connectivity parameters
        print(f"C0 = {self.P.C0}")
        print(f"C1 = {self.P.C1}")
        print(f"C2 = {self.P.C2}")
        print(f"C3 = {self.P.C3}")
        
        # Simulation parameters
        print(f"dt = {self.P.dt}")
        print(f"t_end = {self.P.t_end}")
        print(f"t_cut = {self.P.t_cut}")
        print(f"decimate = {self.P.decimate}")
        print(f"nn = {self.P.nn}")
        print(f"seed = {self.P.seed}")
        print(f"sigma = {self.P.sigma}")
        print(f"weights shape = {self.P.weights.shape}")
        
        return ""

    def check_input(self):
        """
        Validate model parameters.

        Raises
        ------
        ValueError
            If any parameter values are invalid (e.g., t_cut >= t_end, 
            decimate < 1, or dimension mismatches).
        """
        if self.P.t_cut >= self.P.t_end:
            raise ValueError("t_cut must be less than t_end")
        if self.P.decimate < 1:
            raise ValueError("decimate must be >= 1")
        if self.P.nn != self.P.weights.shape[0]:
            raise ValueError("nn != weights.shape[0]")
        self._checked = True

    def set_initial_state(self, seed: int = None):
        """
        Set random initial state for the simulation.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducible initial conditions. 
            If None, uses the seed specified during initialization.
        """
        seed_ = self.seed if seed is None else seed
        self.P.initial_state = set_initial_state_jr(self.P.nn, seed_)

    def run(self, par: dict = None, x0: np.ndarray = None):
        """
        Run the Jansen-Rit simulation.

        Parameters
        ----------
        par : dict, optional
            Dictionary of parameters to update for this simulation run.
            Any parameter from the class documentation can be updated.
        x0 : np.ndarray, optional
            Initial state vector of shape (6*nn,). If None, uses the 
            initial state set during initialization or by set_initial_state().

        Returns
        -------
        dict
            Dictionary containing simulation results:
            - 't': np.ndarray of shape (n_steps,) - time points
            - 'x': np.ndarray of shape (n_steps, nn) - simulated time series (y - z)
                   representing local field potentials
        """
        # Optionally update parameters on the jitclass (Numba allows setattr)
        if par:
            for k, v in par.items():
                if k == "weights":
                    W = np.array(v, dtype=np.float64)
                    if W.ndim != 2 or W.shape[0] != W.shape[1]:
                        raise ValueError("'weights' must be a square 2D array")
                    setattr(self.P, "weights", W)
                    setattr(self.P, "nn", len(W))
                elif k in ("C0", "C1", "C2", "C3"):
                    arr = _as_1d_array_like(v, self.P.nn)
                    setattr(self.P, k, arr)
                elif hasattr(self.P, k):
                    setattr(self.P, k, v)
                else:
                    raise ValueError(f"Invalid parameter: {k}")

        # Optionally replace initial state
        if x0 is not None:
            x0 = np.array(x0, dtype=np.float64)
            if x0.shape[0] != 6 * self.P.nn:
                raise ValueError("initial_state must have length 6*nn")
            self.P.initial_state = x0

        if not self._checked:
            self.check_input()

        return integrate(self.P)


# Alias for consistency with naming convention
JR_sde_numba = JR_sde
