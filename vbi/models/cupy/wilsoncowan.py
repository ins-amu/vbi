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

class WC_sde:
    r"""
    Wilson-Cowan model of neural population dynamics.

    **References**:

    .. [WC_1972] Wilson, H.R. and Cowan, J.D. *Excitatory and inhibitory
        interactions in localized populations of model neurons*, Biophysical
        journal, 12: 1-24, 1972.
    .. [WC_1973] Wilson, H.R. and Cowan, J.D  *A Mathematical Theory of the
        Functional Dynamics of Cortical and Thalamic Nervous Tissue*

    .. [D_2011] Daffertshofer, A. and van Wijk, B. *On the influence of
        amplitude on the connectivity between phases*
        Frontiers in Neuroinformatics, July, 2011

    """
    
    def __init__(self, par: dict = {}) -> None:
        
        self._par = self.get_default_parameters()
        self.valid_parameters = list(self._par.keys())
        self.check_parameters(par)
        self._par.update(par)

        for item in self._par.items():
            name = item[0]
            value = item[1]
            setattr(self, name, value)
        
        self.xp = get_module(self.engine)
        if self.seed is not None:
            self.xp.random.seed(self.seed)

        os.makedirs(self.output, exist_ok=True)
        self.update_dependent_parameters()
        self.PREPARE_INPUT = False
        
    def __call__(self):
        print("Wilson-Cowan model of neural population dynamics")
        return self._par
    
    def __str__(self) -> str:
        print("Wilson-Cowan model of neural population dynamics")
        print("----------------")
        for item in self._par.items():
            name = item[0]
            value = item[1]
            print(f"{name} = {value}")
        return ""
    
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
            'P': 0.0,          # external input to excitatory population
            'Q': 0.0,           # external input to inhibitory population
            'g_e': 0.0,         # coupling excitatory
            'g_i': 0.0,         # coupling inhibitory
            "method": "heun",   # integration method
            "weights": None,    # connectivity matrix
            'seed': None,       # random seed
            "t_end": 300.0,     # end time
            "t_cut": 0.0,       # cut time
            "dt": 0.01,         # time step
            "noise_amp": 0.0,   # noise
            "output": "output", # output directory
            "num_sim": 1,
            "engine": "cpu",
            "same_initial_state": False,
            "dtype": "float",
            "RECORD_EI": "E",
            "initial_state": None,
            "decimate": 1,
            "shift_sigmoid": False,
        }

        return par
    
    
    def update_dependent_parameters(self):
        
        self.inv_tau_e = 1.0 / self.tau_e
        self.inv_tau_i = 1.0 / self.tau_i

    def check_parameters(self, par: dict) -> None:
        for key in par.keys():
            if key not in self.valid_parameters:
                raise ValueError(f"Invalid parameter: {key} provided.")
                

    def set_initial_state(self):
        self.x0 = set_initial_state(
            self.nn,
            self.num_sim,
            self.engine,
            self.seed,
            self.same_initial_state,
            self.dtype,
        )
    
    def prepare_input(self):
        '''
        Prepare input parameters, check dimensions and convert to cupy array if needed. Some parameters 
        can be scalars, vectors or 2D arrays. 2D arrays parameters are heterogeneous accross nodes and 
        simulations, but 1D arrays parameters are homogeneous accross nodes and heterogeneous accross 
        simulations.
        
        vector parameters: ns
        scalar parameters: 1
        matrix parameters: nn x ns
        '''
        assert self.weights is not None, "weights must be provided"
        self.g_e = self.xp.array(self.g_e)
        self.g_i = self.xp.array(self.g_i)
        
        for i in ["P", "Q", "c_ee", "c_ei", "c_ie", "c_ii", "tau_e", "tau_i"]:
            setattr(self, i, prepare_vec(getattr(self, i), self.num_sim, self.engine, self.dtype))
            
        self.weights = self.xp.array(self.weights)
        self.weights = move_data(self.weights, self.engine)
        self.num_nodes = self.nn = self.weights.shape[0]
        self.PREPARE_INPUT = True
    
    def derivative(self, x, t):
        """ 
        Derivative of the Wilson-Cowan model
        """
        
        nn = self.nn 
        E = x[:nn, :]
        I = x[nn:, :]
        dxdt = self.xp.zeros((2*nn, self.num_sim), dtype=self.dtype)
        lc_e = lc_i = 0.0
        
        if (self.g_e > 0.0).any():
            lc_e = self.g_e * (self.weights @ E)
        if (self.g_i > 0.0).any():
            lc_i = self.g_i * (self.weights @ I)
            
        x_e = self.alpha_e * (self.c_ee * E - self.c_ei * I + self.P - self.theta_e + lc_e)
        x_i = self.alpha_i * (self.c_ie * E - self.c_ii * I + self.Q - self.theta_i + lc_i)
        s_e = self.sigmoid(x_e, self.a_e, self.b_e, self.c_e)
        s_i = self.sigmoid(x_i, self.a_i, self.b_i, self.c_i)
        dxdt[:nn, :] = self.inv_tau_e * (-E + (self.k_e - self.r_e * E) * s_e)
        dxdt[nn:, :] = self.inv_tau_i * (-I + (self.k_i - self.r_i * I) * s_i)
        
        return dxdt
    
    def sigmoid(self, x, a, b, c):
        '''
        Sigmoid function
        '''
        
        if self.shift_sigmoid:
            return c * (1.0 / (1.0 + self.xp.exp(-a * (x - b))) - 1.0 / (1.0 + self.xp.exp(-a * -b)))
        else:
            return c / (1.0 + self.xp.exp(-a * (x - b)))
    
    def euler_maruyama(self, x, t):
        '''
        Euler-Maruyama method
        '''
        
        dw = self.xp.random.normal(size=x.shape)
        coeff = self.noise_amp * self.xp.sqrt(self.dt)
        
        return x + self.dt * self.derivative(x, t) + coeff * dw
    
    def heunStochastic(self, x, t):
        '''
        Heun method
        '''
        
        coeff = self.noise_amp * self.xp.sqrt(self.dt)
        dw = self.xp.random.normal(size=x.shape)
        
        k1 = self.derivative(x, t)        
        x_predictor = x + self.dt * k1 + coeff * dw
        k2 = self.derivative(x_predictor, t + self.dt)
        
        return x + self.dt * (k1 + k2) / 2.0 + coeff * dw
    
    def run(self, x0=None, tspan=None, verbose=True):
        '''
        Run the Wilson-Cowan model
        #TODO: optimize memory usage
        '''
        
        self.prepare_input()
        
        if x0 is None:
            self.set_initial_state()
        else:
            self.x0 = x0
            
        if tspan is None:
            t = np.arange(0.0, self.t_end, self.dt)
        else:
            t = tspan
        
        nn = self.nn
        t_cut = self.t_cut
        decimate = self.decimate
        RECORD_EI = self.RECORD_EI.lower()       
        
        valid_points = np.sum(t > t_cut)
        buffer_size = valid_points // decimate
        t_buffer = np.zeros((buffer_size), dtype=np.float32)
        E = I = None
        
        if "e" in RECORD_EI:
            E = np.zeros((buffer_size, self.nn, self.num_sim), dtype=np.float32)
        
        if "i" in RECORD_EI:
            I = np.zeros((buffer_size, self.nn, self.num_sim), dtype=np.float32)
            
        
        buffer_idx = 0
        for i in tqdm.trange(len(t), disable=not verbose, desc="Integrating"):
            t_curr = i * self.dt
            
            self.x0 = self.heunStochastic(self.x0, t_curr)
            
            if (t_curr > t_cut) and (i % decimate == 0):
                if buffer_idx < buffer_size:
                    t_buffer[buffer_idx] = t_curr
                    
                    if "e" in RECORD_EI:
                        E[buffer_idx] = get_(self.x0[:nn, :], self.engine, "f")
                        
                    if "i" in RECORD_EI:
                        I[buffer_idx] = get_(self.x0[nn:, :], self.engine, "f")
                        
                    buffer_idx += 1
            
        return {"t": t_buffer, "E": E, "I": I}
    

    def do_step_EI(self, x, t, method="heunStochastic"):
        '''
        Do a single step of the Wilson-Cowan model
        '''
        if not self.PREPARE_INPUT:
            self.prepare_input()
            
        if method == "heunStochastic":
            x = self.heunStochastic(x, t)
        elif method == "euler_maruyama":
            x = self.euler_maruyama(x, t)
        else:
            raise ValueError(f"Invalid method: {method}")
        
        return x
        
        
def set_initial_state(nn, ns, engine, seed=None, same_initial_state=False, dtype=float):
    """
    Set initial state for the Wilson-Cowan model

    Parameters
    ----------
    nn : int
        number of nodes
    ns : int
        number of simulations
    engine : str
        cpu or gpu
    seed : int
        random seed
    dtype : str
        float: float64
        f    : float32
    """
    
    if seed is not None:
        np.random.seed(seed)

    if same_initial_state:
        y0 = np.random.rand(2*nn)
        y0 = repmat_vec(y0, ns, engine)
    else:
        y0 = np.random.rand(2*nn, ns)
        y0 = move_data(y0, engine)

    return y0.astype(dtype)