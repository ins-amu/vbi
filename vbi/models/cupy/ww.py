import os
import tqdm
import logging
import numpy as np
from copy import copy
from vbi.models.cupy.utils import *
from vbi.models.cupy.bold import Bold
from vbi.models.cupy.base import BaseCupyModel
from typing import List, Dict

try:
    import cupy as cp
except ImportError:
    logging.warning("Cupy is not installed. Using Numpy instead.")


class WW_sde(BaseCupyModel):
    """
    Wong-Wang neural mass including Excitatory and Inhibitory populations.

    This model explicitly simulates both excitatory and inhibitory neural populations
    with distinct synaptic gating variables, firing rate dynamics, and network coupling.
    The model is based on the original Wong-Wang decision-making framework and has been
    extended for whole-brain network simulations.

    Main reference:
        [original] Wong, K. F., & Wang, X. J. (2006). A recurrent network mechanism
        of time integration in perceptual decisions. Journal of Neuroscience, 26(4),
        1314-1328.

    Additional references:
        [reduced] Deco, G., Ponce-Alvarez, A., Mantini, D., Romani, G. L., Hagmann,
        P., & Corbetta, M. (2013). Resting-state functional connectivity emerges
        from structurally and dynamically shaped slow linear fluctuations. Journal
        of Neuroscience, 33(27), 11239-11252.

        [original] Deco, G., Ponce-Alvarez, A., Hagmann, P., Romani, G. L., Mantini,
        D., & Corbetta, M. (2014). How local excitation-inhibition ratio impacts the
        whole brain dynamics. Journal of Neuroscience, 34(23), 7886-7898.

    .. list-table:: Parameters
        :widths: 25 50 25
        :header-rows: 1

        * - Name
          - Explanation
          - Default Value
        * - `a_exc`
          - Excitatory population gain parameter for firing rate transfer function (n/C).
          - 310
        * - `a_inh`
          - Inhibitory population gain parameter for firing rate transfer function (nC^-1).
          - 0.615
        * - `b_exc`
          - Excitatory population threshold parameter for firing rate transfer function (Hz).
          - 125
        * - `b_inh`
          - Inhibitory population threshold parameter for firing rate transfer function (Hz).
          - 177
        * - `d_exc`
          - Excitatory population saturation parameter for firing rate transfer function (s).
          - 0.16
        * - `d_inh`
          - Inhibitory population saturation parameter for firing rate transfer function (s).
          - 0.087
        * - `tau_exc`
          - Excitatory synaptic time constant (ms).
          - 100.0
        * - `tau_inh`
          - Inhibitory synaptic time constant (ms).
          - 10.0
        * - `gamma_exc`
          - Excitatory kinetic parameter (ms^-1).
          - 0.641/1000.0
        * - `gamma_inh`
          - Inhibitory kinetic parameter (ms^-1).
          - 1.0/1000.0
        * - `W_exc`
          - Excitatory population local weight for external input.
          - 1.0
        * - `W_inh`
          - Inhibitory population local weight for external input.
          - 0.7
        * - `ext_current`
          - External current input (nA). If array-like, it should be of shape (`num_nodes`, `num_sim`).
          - 0.382
        * - `J_NMDA`
          - NMDA synaptic coupling strength (nA).
          - 0.15
        * - `J_I`
          - Inhibitory synaptic coupling strength (nA).
          - 1.0
        * - `w_plus`
          - Local excitatory recurrence strength.
          - 1.4
        * - `lambda_inh_exc`
          - Long-range feedforward inhibition switch (0=off, 1=on).
          - 0.0
        * - `G_exc`
          - Global excitatory coupling strength. If array-like, it should be of length `num_sim`.
          - 0.0
        * - `G_inh`
          - Global inhibitory coupling strength. If array-like, it should be of length `num_sim`.
          - 0.0
        * - `t_end`
          - End time of simulation (ms).
          - 1000.0
        * - `t_cut`
          - Time to cut off initial transient (ms).
          - 0.0
        * - `dt`
          - Time step for integration (ms).
          - 0.1
        * - `tr`
          - Repetition time for BOLD signal (ms).
          - 300.0
        * - `s_decimate`
          - Decimation factor for recording gating variables S.
          - 1
        * - `sigma`
          - Noise amplitude for stochastic integration.
          - 0.0
        * - `weights`
          - Structural connectivity matrix of shape (`num_nodes`, `num_nodes`).
          - None
        * - `num_sim`
          - Number of parallel simulations.
          - 1
        * - `nn`
          - Number of brain regions/nodes.
          - 1
        * - `engine`
          - Computation engine: "cpu" or "gpu".
          - "cpu"
        * - `dtype`
          - Data type: "float32" or "float".
          - "float32"
        * - `seed`
          - Random seed for reproducibility.
          - None
        * - `output`
          - Output directory for results.
          - "output"
        * - `initial_state`
          - Initial state of the system of shape (2*`num_nodes`, `num_sim`).
          - None
        * - `same_initial_state`
          - If True, all simulations have the same initial state.
          - False
        * - `same_noise_per_sim`
          - If True, all simulations have the same noise realization.
          - False
        * - `RECORD_S`
          - If True, record synaptic gating variables S.
          - False
        * - `RECORD_BOLD`
          - If True, record BOLD signal using Balloon-Windkessel model.
          - True

    """

    def __init__(self, par: Dict = {}, Bpar: Dict = {}) -> None:
        
        super().__init__()
        
        self._par = self.get_default_parameters()
        self.valid_parameters = list(self._par.keys())
        self.valid_params = self.valid_parameters
        self.check_parameters(par)
        self._par.update(par)
        
        self.par_ = self._par

        for item in self._par.items():
            setattr(self, item[0], item[1])

        self.B = Bold(Bpar)

        self.xp = get_module(self.engine)
        if self.seed is not None:
            self.xp.random.seed(self.seed)

        os.makedirs(self.output, exist_ok=True)

    def __call__(self):
        print("Wong-Wang model.")
        return self._par

    def __str__(self) -> str:
        return self._format_parameters_table("Wong-Wang (CuPy)")
    
    def get_parameter_descriptions(self):
        """Get parameter descriptions and types."""
        return {
            "a_exc": ("Excitatory population gain parameter (n/C)", "float"),
            "a_inh": ("Inhibitory population gain parameter (nC^-1)", "float"),
            "b_exc": ("Excitatory population threshold parameter (Hz)", "float"),
            "b_inh": ("Inhibitory population threshold parameter (Hz)", "float"),
            "d_exc": ("Excitatory population saturation parameter (s)", "float"),
            "d_inh": ("Inhibitory population saturation parameter (s)", "float"),
            "tau_exc": ("Excitatory synaptic time constant (ms)", "float"),
            "tau_inh": ("Inhibitory synaptic time constant (ms)", "float"),
            "gamma_exc": ("Excitatory kinetic parameter (ms^-1)", "float"),
            "gamma_inh": ("Inhibitory kinetic parameter (ms^-1)", "float"),
            "W_exc": ("Excitatory population local weight for external input", "float"),
            "W_inh": ("Inhibitory population local weight for external input", "float"),
            "ext_current": ("External current input (nA)", "float"),
            "J_NMDA": ("NMDA synaptic coupling strength (nA)", "float"),
            "J_I": ("Inhibitory synaptic coupling strength (nA)", "float"),
            "w_plus": ("Local excitatory recurrence strength", "float"),
            "lambda_inh_exc": ("Long-range feedforward inhibition switch (0=off, 1=on)", "float"),
            "G_exc": ("Global excitatory coupling strength", "float"),
            "G_inh": ("Global inhibitory coupling strength", "float"),
            "t_end": ("End time of simulation (ms)", "float"),
            "t_cut": ("Time to cut off initial transient (ms)", "float"),
            "dt": ("Time step for integration (ms)", "float"),
            "tr": ("Repetition time for BOLD signal (ms)", "float"),
            "s_decimate": ("Decimation factor for recording gating variables S", "int"),
            "sigma": ("Noise amplitude for stochastic integration", "float"),
            "weights": ("Structural connectivity matrix (nn x nn)", "ndarray"),
            "num_sim": ("Number of parallel simulations", "int"),
            "nn": ("Number of brain regions/nodes", "int"),
            "engine": ("Computation engine: cpu or gpu", "str"),
            "dtype": ("Data type: float32 or float", "str"),
            "seed": ("Random seed for reproducibility", "int"),
            "output": ("Output directory for results", "str"),
            "initial_state": ("Initial state (2*num_nodes x num_sim)", "ndarray"),
            "same_initial_state": ("If True, all sims have same initial state", "bool"),
            "same_noise_per_sim": ("If True, all sims have same noise realization", "bool"),
            "RECORD_S": ("If True, record synaptic gating variables S", "bool"),
            "RECORD_BOLD": ("If True, record BOLD signal", "bool"),
        }

    def set_initial_state(self):
        return set_initial_state(
            self.nn,
            self.num_sim,
            self.engine,
            self.seed,
            self.same_initial_state,
            self.dtype,
        )

    def get_default_parameters(self) -> Dict:
        """Get default parameters for the Wong-Wang full model."""

        par = {
            # Excitatory parameters
            "a_exc": 310,  # n/C
            "a_inh": 0.615,  # nC^-1
            "b_exc": 125,  # Hz
            "b_inh": 177,  # Hz
            "d_exc": 0.16,  # s
            "d_inh": 0.087,  # ms
            "tau_exc": 100.0,  # ms
            "tau_inh": 10.0,  # ms
            "gamma_exc": 0.641 / 1000.0,
            "gamma_inh": 1.0 / 1000.0,  # ms
            "W_exc": 1.0,
            "W_inh": 0.7,
            "ext_current": 0.382,  # nA external current
            "J_NMDA": 0.15,  # nA
            "J_I": 1.0,  # nA
            "w_plus": 1.4,
            "lambda_inh_exc": 0.0,  # logn-range feedforward inhibition is considered =1, otherwise =0
            # other parameters
            "t_end": 1000.0,  # end time of simulation in ms
            "t_cut": 0.0,  # time to cut off initial transient in ms
            "dt": 0.1,  # time step for integration in ms
            "G_exc": 0.0,  # global excitatory coupling strength
            "G_inh": 0.0,  # global inhibitory coupling strength
            "weights": None,  # connectivity matrix (nn x nn)
            "tr": 300.0,  # repetition time in ms for BOLD
            "s_decimate": 1,  # decimation factor for recording gating variables S
            "same_noise_per_sim": False,  # if True, same noise is used for all simulations
            "sigma": 0.0,  # noise strength
            "num_sim": 1,  # number of simulations
            "nn": 1,  # number of nodes
            "engine": "cpu",  # computation engine (cpu or gpu)
            "seed": None,  # random seed
            "output": "output",  # output directory
            "dtype": "float32",  # data type (float or float32)
            "initial_state": None,  # initial state
            "same_initial_state": False,  # if True, same initial state for all simulations
            "RECORD_S": False,  # if True, record gating variables S
            "RECORD_BOLD": True,  # if True, record BOLD signal
        }
        return par

    def check_parameters(self, par):
        for key in par.keys():
            if key not in self.valid_parameters:
                raise ValueError(f"Invalid parameter {key:s} provided.")

    def prepare_input(self):
        self.G_exc = self.xp.array(self.G_exc, dtype=self.dtype)

        self.ext_current = prepare_vec_2d(
            self.ext_current, self.nn, self.num_sim, self.engine, self.dtype
        )
        self.sigma = self.xp.array(self.sigma, dtype=self.dtype)
        assert self.weights is not None, "Weights must be provided."
        self.weights = self.xp.array(self.weights, dtype=self.dtype)
        self.nn = self.num_nodes = self.weights.shape[0]

        if self.initial_state is None:
            self.set_initial_state()

    def get_firing_rate(self, current: float, is_exc: bool = True):
        """Calculate firing rate based on input current"""
        if is_exc:
            a, b, d = self.a_exc, self.b_exc, self.d_exc
        else:
            a, b, d = self.a_inh, self.b_inh, self.d_inh

        return (a * current - b) / (1.0 - np.exp(-d * (a * current - b)))

    def f_ww(self, S, t=None):
        """Wong-Wang neural mass model equations."""

        xp = self.xp
        nn = self.nn
        ns = self.num_sim
        weights = self.weights
        S_exc, S_inh = S[: self.nn, :], S[self.nn :, :]

        network_exc_exc = weights @ S_exc
        if self.lambda_inh_exc > 0:
            network_inh_exc = weights @ S_inh
        else:
            network_inh_exc = 0.0

        current_exc = (
            self.W_exc * self.ext_current
            + self.w_plus * self.J_NMDA * S_exc
            + self.G_exc * self.J_NMDA * network_exc_exc
            - self.J_I * S_inh
        )

        current_inh = (
            self.W_inh * self.ext_current
            + self.J_NMDA * S_inh
            - S_inh
            + self.G_inh * self.J_NMDA * network_inh_exc
        )

        r_exc = self.get_firing_rate(current_exc, is_exc=True)
        r_inh = self.get_firing_rate(current_inh, is_exc=False)
        dSdt = xp.zeros((2 * nn, ns)).astype(self.dtype)

        # exc
        dSdt[:nn, :] = (-S_exc / self.tau_exc) + (1.0 - S_exc) * self.gamma_exc * r_exc
        # inh
        dSdt[nn:, :] = (-S_inh / self.tau_inh) + self.gamma_inh * r_inh

        return dSdt

    def heunStochastic(self, y, t, dt):

        xp = self.xp
        nn = self.nn
        ns = self.num_sim

        if not self.same_noise_per_sim:
            dW = self.sigma * xp.random.randn(2 * nn, ns) * xp.sqrt(dt)
        else:
            dW = self.sigma * xp.random.randn(2 * nn, 1) * xp.sqrt(dt)
        k1 = self.f_ww(y, t)
        y_ = y + dt * k1 + dW
        k2 = self.f_ww(y_, t + dt)
        y = y + dt * 0.5 * (k1 + k2) + dW

        return y

    def do_step(self, S, t, dt):
        """run one step of the model"""
        S = self.heunStochastic(S, t, dt)
        return S

    def do_bold_step(self, r_in, s, f, ftilde, vtilde, qtilde, v, q, dt, P):
        """
        Step the BOLD model forward in time.
        """
        return self.Bold.do_bold_step(r_in, s, f, ftilde, vtilde, qtilde, v, q, dt, P)

    def run(self, x0=None, tspan: np.ndarray = None, verbose=True):
        """Run the Wong-Wang model simulation."""

        self.prepare_input()
        if x0 is None:
            x0 = copy(self.set_initial_state())
        else:
            x0 = copy(self.x0)

        if tspan is None:
            t = np.arange(0.0, self.t_end, self.dt)
        else:
            t = tspan

        dt = self.dt
        t_cut = self.t_cut
        dt_bold = dt / 1000.0  # BOLD time step in seconds

        tr = self.tr
        xp = self.xp
        nn = self.nn
        ns = self.num_sim
        engine = self.engine
        s_decimate = self.s_decimate
        bold_decimate = int(np.round(tr / dt))
        s_curr = copy(x0)
        valid_points = np.sum(t > t_cut)
        s_buffer_size = valid_points // s_decimate
        # b_buffer_size = int(np.ceil(len(t)/ bold_decimate))
        t_buffer = np.zeros((s_buffer_size), dtype=np.float32)
        n_steps = len(t)

        B = self.B
        B.allocate_memory(xp, nn, ns, n_steps, bold_decimate, self.dtype)
        S_exc = np.array([])

        if self.RECORD_S:
            S_exc = np.zeros((s_buffer_size, nn, ns), dtype=np.float32)

        buffer_idx = 0
        for i in tqdm.trange(len(t), disable=not verbose, desc="Integrating"):
            t_curr = i * dt

            s_curr = self.do_step(s_curr, t_curr, dt)

            if (t_curr > t_cut) and (i % s_decimate == 0):

                if buffer_idx < s_buffer_size:
                    t_buffer[buffer_idx] = t_curr

                    if self.RECORD_S:
                        S_exc[buffer_idx] = get_(s_curr[:nn, :], engine, "f")

                    buffer_idx += 1

            if self.RECORD_BOLD:
                B.do_bold_step(s_curr[:nn, :], dt_bold)

                if (i % bold_decimate == 0) and ((i // bold_decimate) < B.vv.shape[0]):
                    B.vv[i // bold_decimate] = get_(B.v[1], engine, "f")
                    B.qq[i // bold_decimate] = get_(B.q[1], engine, "f")

        if self.RECORD_BOLD:
            # Calculate indices for t_cut
            bold_t = np.linspace(0, self.t_end - dt * bold_decimate, len(B.vv))
            valid_indices = np.where(bold_t > self.t_cut)[0]
            if len(valid_indices) > 0:
                start_idx = valid_indices[0]
                bold_d = B.vo * (
                    B.k1 * (1 - B.qq[start_idx:])
                    + B.k2 * (1 - B.qq[start_idx:] / B.vv[start_idx:])
                    + B.k3 * (1 - B.vv[start_idx:])
                )
                bold_t = bold_t[start_idx:]
            else:
                bold_d = np.array([])
                bold_t = np.array([])

        return {
            "S": S_exc,
            "t": t_buffer,
            "bold_t": bold_t,
            "bold_d": bold_d,
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
        y0 = np.random.rand(2 * nn) * 0.1
        y0 = repmat_vec(y0, ns, engine)
    else:
        y0 = np.random.rand(2 * nn, ns) * 0.1
        y0 = move_data(y0, engine)

    return y0.astype(dtype)
