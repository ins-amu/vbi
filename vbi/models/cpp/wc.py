import numpy as np

from vbi.models.cpp.base import BaseModel

try:
    from vbi.models.cpp._src.wc_ode import WC_ode as _WC_ode
except ImportError as e:
    print(
        f"Could not import modules: {e}, probably C++ code is not compiled or properly linked."
    )


################################## Wilson-Cowan ode ###########################
###############################################################################


class WC_ode(BaseModel):
    r"""
    Wilson-Cowan neural mass model.
    
    This model supports heterogeneous parameters across brain regions.
    Parameters marked as "scalar|vector" in the parameter descriptions can be 
    specified as either single values (applied to all regions) or arrays 
    (one value per region).

    **References**:

    .. [WC_1972] Wilson, H.R. and Cowan, J.D. *Excitatory and inhibitory
        interactions in localized populations of model neurons*, Biophysical
        journal, 12: 1-24, 1972.
    .. [WC_1973] Wilson, H.R. and Cowan, J.D  *A Mathematical Theory of the
        Functional Dynamics of Cortical and Thalamic Nervous Tissue*
    .. [D_2011] Daffertshofer, A. and van Wijk, B. *On the influence of
        amplitude on the connectivity between phases*
        Frontiers in Neuroinformatics, July, 2011

    Used Eqns 11 and 12 from [WC_1972]_ in ``rhs``.  P and Q represent external
    inputs, which when exploring the phase portrait of the local model are set
    to constant values. However in the case of a full network, P and Q are the
    entry point to our long range and local couplings, that is, the  activity
    from all other nodes is the external input to the local population [WC_1973]_, [D_2011]_ .

    The default parameters are taken from figure 4 of [WC_1972]_, pag. 10.

    """

    def __init__(self, par={}) -> None:
        import warnings

        super().__init__()
        
        # Backward compatibility mapping
        param_mapping = {
            'g_e': 'G_exc',
            'g_i': 'G_inh'
        }
        
        # Apply backward compatibility
        for old_name, new_name in param_mapping.items():
            if old_name in par:
                warnings.warn(f"Parameter '{old_name}' is deprecated. Use '{new_name}' instead.", 
                            DeprecationWarning, stacklevel=2)
                par[new_name] = par.pop(old_name)
        
        self.valid_params = list(self.get_default_parameters().keys())
        self.check_parameters(par)
        self._par = self.get_default_parameters()
        self._par.update(par)

        for item in self._par.items():
            name = item[0]
            value = item[1]
            setattr(self, name, value)

        if self.seed is not None:
            np.random.seed(self.seed)

        self.N = self.num_nodes = np.asarray(self.weights).shape[0]

    def get_parameter_descriptions(self):
        """
        Get descriptions and types for Wilson-Cowan model parameters.
        
        Returns
        -------
        dict
            Dictionary mapping parameter names to (description, type) tuples.
        """
        return {
            "c_ee": ("Excitatory to excitatory coupling", "scalar"),
            "c_ei": ("Excitatory to inhibitory coupling", "scalar"),
            "c_ie": ("Inhibitory to excitatory coupling", "scalar"),
            "c_ii": ("Inhibitory to inhibitory coupling", "scalar"),
            "tau_e": ("Excitatory time constant", "scalar"),
            "tau_i": ("Inhibitory time constant", "scalar"),
            "a_e": ("Excitatory gain parameter", "scalar"),
            "a_i": ("Inhibitory gain parameter", "scalar"),
            "b_e": ("Excitatory threshold parameter", "scalar"),
            "b_i": ("Inhibitory threshold parameter", "scalar"),
            "c_e": ("Excitatory sigmoid slope", "scalar"),
            "c_i": ("Inhibitory sigmoid slope", "scalar"),
            "theta_e": ("Excitatory threshold offset", "scalar"),
            "theta_i": ("Inhibitory threshold offset", "scalar"),
            "r_e": ("Excitatory refractory parameter", "scalar"),
            "r_i": ("Inhibitory refractory parameter", "scalar"),
            "k_e": ("Excitatory recovery parameter", "scalar"),
            "k_i": ("Inhibitory recovery parameter", "scalar"),
            "alpha_e": ("Excitatory scaling parameter", "scalar"),
            "alpha_i": ("Inhibitory scaling parameter", "scalar"),
            "P": ("External input to excitatory population", "scalar|vector"),
            "Q": ("External input to inhibitory population", "scalar|vector"),
            "G_exc": ("Excitatory global coupling", "scalar"),
            "G_inh": ("Inhibitory global coupling", "scalar"),
            "method": ("Integration method", "string"),
            "weights": ("Structural connectivity matrix", "matrix"),
            "seed": ("Random seed for reproducibility", "-"),
            "t_end": ("End time of simulation", "scalar"),
            "t_cut": ("Transition time to cut", "scalar"),
            "dt": ("Integration time step", "scalar"),
            "noise_seed": ("Seed for noise generation", "bool"),
            "output": ("Output directory", "string"),
        }

    def __str__(self) -> str:
        return self._format_parameters_table()

    def get_default_parameters(self):
        par = {
            "c_ee": 16.0,
            "c_ei": 12.0,
            "c_ie": 15.0,
            "c_ii": 3.0,
            "tau_e": 8.0,
            "tau_i": 8.0,
            "a_e": 1.3,
            "a_i": 2.0,
            "b_e": 4.0,
            "b_i": 3.7,
            "c_e": 1.0,
            "c_i": 1.0,
            "theta_e": 0.0,
            "theta_i": 0.0,
            "r_e": 1.0,
            "r_i": 1.0,
            "k_e": 0.994,
            "k_i": 0.999,
            "alpha_e": 1.0,
            "alpha_i": 1.0,
            "P": 1.25,
            "Q": 0.0,
            "G_exc": 0.0,
            "G_inh": 0.0,
            "method": "heun",
            "weights": None,
            "seed": None,
            "t_end": 300.0,
            "t_cut": 0.0,
            "dt": 0.01,
            "noise_seed": False,
            "output": "output",
        }
        return par

    def set_initial_state(self, seed=None):

        if seed is not None:
            np.random.seed(seed)
        self.initial_state = np.random.rand(2 * self.num_nodes)

    def prepare_input(self):
        self.noise_seed = int(self.noise_seed)
        self.t_end = float(self.t_end)
        self.t_cut = float(self.t_cut)
        self.dt = float(self.dt)
        self.P = check_sequence(self.P, self.num_nodes)
        self.Q = check_sequence(self.Q, self.num_nodes)
        self.c_ee = float(self.c_ee)
        self.c_ei = float(self.c_ei)
        self.c_ie = float(self.c_ie)
        self.c_ii = float(self.c_ii)
        self.tau_e = float(self.tau_e)
        self.tau_i = float(self.tau_i)
        self.a_e = float(self.a_e)
        self.a_i = float(self.a_i)
        self.b_e = float(self.b_e)
        self.b_i = float(self.b_i)
        self.c_e = float(self.c_e)
        self.c_i = float(self.c_i)
        self.theta_e = float(self.theta_e)
        self.theta_i = float(self.theta_i)
        self.r_e = float(self.r_e)
        self.r_i = float(self.r_i)
        self.k_e = float(self.k_e)
        self.k_i = float(self.k_i)
        self.alpha_e = float(self.alpha_e)
        self.alpha_i = float(self.alpha_i)
        self.G_exc = float(self.G_exc)
        self.G_inh = float(self.G_inh)
        self.method = str(self.method)
        self.weights = np.asarray(self.weights)

    def run(self, par: dict = {}, x0: np.ndarray = None, verbose: bool = False):
        """
        Integrate the system of equations for the Wilson-Cowan model.

        Parameters
        ----------
        par : dict
            Dictionary with parameters of the model.
        x0 : array-like
            Initial state of the system.
        verbose : bool
            If True, print the integration progress.

        """

        if x0 is None:
            self.set_initial_state()
            if verbose:
                print("Initial state set by default.")
        else:
            self.initial_state = x0

        for key in par.keys():
            if key not in self.valid_params:
                raise ValueError(f"Invalid parameter: {key}")
            setattr(self, key, par[key]["value"])

        self.prepare_input()

        obj = _WC_ode(
            self.N,
            self.dt,
            self.P,
            self.Q,
            self.initial_state,
            self.weights,
            self.t_end,
            self.t_cut,
            self.c_ee,
            self.c_ei,
            self.c_ie,
            self.c_ii,
            self.tau_e,
            self.tau_i,
            self.a_e,
            self.a_i,
            self.b_e,
            self.b_i,
            self.c_e,
            self.c_i,
            self.theta_e,
            self.theta_i,
            self.r_e,
            self.r_i,
            self.k_e,
            self.k_i,
            self.alpha_e,
            self.alpha_i,
            self.G_exc,
            self.G_inh,
            self.noise_seed,
        )

        if self.method == "euler":
            obj.eulerIntegrate()
        elif self.method == "heun":
            obj.heunIntegrate()
        elif self.method == "rk4":
            obj.rk4Integrate()

        t = np.asarray(obj.get_times())
        x = np.asarray(obj.get_states()).T

        del obj
        return {"t": t, "x": x}


def check_sequence(x, n):
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
