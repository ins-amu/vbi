import numpy as np

try:
    from vbi.models.cpp._src.wc_ode import WC_ode as _WC_ode
except ImportError as e:
    print(f"Could not import modules: {e}, probably C++ code is not compiled or properly linked.")


################################## Wilson-Cowan ode ###########################
###############################################################################

class WC_ode(object):
    r"""
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

        self.valid_params = self.get_default_parameters().keys()
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

    def __str__(self) -> str:
        print("Wilson-Cowan model.")
        print("--------------------")
        for item in self._par.items():
            print(f"{item[0]}, : , {item[1]}")
        return ""

    def __call__(self):
        print("Wilson-Cowan model.")
        return self._par

    def check_parameters(self, par):
        for key in par.keys():
            if key not in self.valid_params:
                raise ValueError(f"Invalid parameter: {key}")

    def get_default_parameters(self):
        par = {
            'c_ee': 16.0,
            'c_ei': 12.0,
            'c_ie': 15.0,
            'c_ii': 3.0,
            'tau_e': 8.0,
            'tau_i': 8.0,
            'a_e': 1.3,
            'a_i': 2.0,
            'b_e': 4.0,
            'b_i': 3.7,
            'c_e': 1.0,
            'c_i': 1.0,
            'theta_e': 0.0,
            'theta_i': 0.0,
            'r_e': 1.0,
            'r_i': 1.0,
            'k_e': 0.994,
            'k_i': 0.999,
            'alpha_e': 1.0,
            'alpha_i': 1.0,
            'P': 1.25,
            'Q': 0.0,
            'g_e': 0.0,
            'g_i': 0.0,
            "method": "heun",
            "weights": None,
            'seed': None,
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
        self.initial_state = np.random.rand(2*self.num_nodes)

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
        self.g_e = float(self.g_e)
        self.g_i = float(self.g_i)
        self.method = str(self.method)
        self.weights = np.asarray(self.weights)


    def run(self, par={}, x0=None, verbose=False):

        '''
        Integrate the system of equations for the Wilson-Cowan model.

        Parameters
        ----------
        par : dict
            Dictionary with parameters of the model.
        x0 : array-like
            Initial state of the system.
        verbose : bool
            If True, print the integration progress.

        '''

        if x0 is None:
            self.set_initial_state()
            if verbose:
                print("Initial state set by default.")
        else:
            self.initial_state = x0

        for key in par.keys():
            if key not in self.valid_params:
                raise ValueError(f"Invalid parameter: {key}")
            setattr(self, key, par[key]['value'])

        self.prepare_input()

        obj = _WC_ode(
            self.N, self.dt, self.P, self.Q, self.initial_state, self.weights,
            self.t_end, self.t_cut, self.c_ee, self.c_ei, self.c_ie, self.c_ii,
            self.tau_e, self.tau_i, self.a_e, self.a_i, self.b_e, self.b_i,
            self.c_e, self.c_i, self.theta_e, self.theta_i, self.r_e, self.r_i,
            self.k_e, self.k_i, self.alpha_e, self.alpha_i, self.g_e, self.g_i,
            self.noise_seed
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
    '''
    check if x is a scalar or a sequence of length n

    parameters
    ----------
    x: scalar or sequence of length n
    n: number of nodes

    returns
    -------
    x: sequence of length n
    '''
    if isinstance(x, (np.ndarray, list, tuple)):
        assert (len(x) == n), f" variable must be a sequence of length {n}"
        return x
    else:
        return x * np.ones(n)
