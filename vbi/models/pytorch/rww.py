"""
Reduced Wong-Wang Mean Field Model with Hemodynamic Response
A structured implementation for parallel simulation on PyTorch/CUDA
"""

import csv
import math
import time
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional, Union, Dict, Any
from vbi.models.pytorch.utils import check_and_expand_param

import numpy as np
import torch


# ============================================================================
# Parameter Classes
# ============================================================================

@dataclass
class BOLD_par:
    """Parameters for the Balloon-Windkessel hemodynamic model"""
    beta: float = 0.65
    gamma: float = 0.41
    tau: float = 0.98
    alpha: float = 0.33
    p_constant: float = 0.34
    v_0: float = 0.02
    k_1: float = None  # Will be computed
    k_2: float = None  # Will be computed
    k_3: float = 0.53
    
    def __post_init__(self):
        if self.k_1 is None:
            self.k_1 = 4.3 * 28.265 * 3 * 0.0331 * self.p_constant
        if self.k_2 is None:
            self.k_2 = 0.47 * 110 * 0.0331 * self.p_constant


@dataclass
class Simulation_par:
    """Parameters for simulation settings"""
    t_cut: float = 120.0      # dropped time in seconds (60*2)
    t_end: float = 14.4 * 60  # Total simulation time in seconds (including t_cut)
    dt: float = 0.01          # Time step for integration
    tr: float = 0.72          # BOLD sampling time (TR)
    warmup_steps: int = 1000  # Number of warm-up steps
    
    @property
    def t_total(self) -> float:
        """Total simulation time"""
        return self.t_end  # t_end now represents total time, not additional time
    
    @property
    def n_samples(self) -> int:
        """Number of time samples"""
        return int(self.t_total / self.dt) + 1
    
# ============================================================================
# Main Model Class
# ============================================================================

class RWW_sde:
    """
    Wong-Wang Mean Field Model with Balloon-Windkessel Hemodynamic Response
    
    This class implements the Wong-Wang neural mass model coupled with
    the Balloon-Windkessel hemodynamic model for simulating BOLD signals.
    Optimized for parallel simulation on CUDA.
    
    Parameters can be provided as:
    - Scalars: Applied to all nodes and sets
    - 1D arrays: Applied per-node or per-set
    - 2D arrays: Applied per-node per-set
    
    Parameters
    ----------
    weights : np.ndarray or torch.Tensor
        Structural connectivity matrix [n_nodes x n_nodes]
    w : float, array, or tensor, optional
        Recurrent strength (default: 0.9)
    I_ext : float, array, or tensor, optional
        External input (default: 0.382)
    G : float, array, or tensor, optional
        Global coupling (default: 0.0)
    sigma : float, array, or tensor, optional
        Noise amplitude (default: 0.02)
    J : float, array, or tensor, optional
        Synaptic coupling (default: 0.2609)
    a : float, array, or tensor, optional
        Firing rate slope (default: 270)
    b : float, array, or tensor, optional
        Firing rate threshold (default: 108)
    d : float, array, or tensor, optional
        Firing rate scaling (default: 0.154)
    tau_s : float, array, or tensor, optional
        Synaptic time constant (default: 0.1)
    gamma : float, array, or tensor, optional
        Synaptic gating (default: 0.641)
    n_sim : int, optional
        Number of parameter sets (default: 1)
    hemodynamic_params : BOLD_par, optional
        Hemodynamic parameters
    simulation_params : Simulation_par, optional
        Simulation settings
    device : str, optional
        Computation device: 'cpu', 'gpu', or 'cuda' (default: 'cpu')
        Note: 'gpu' and 'cuda' are equivalent and both use CUDA
    RECORD_BOLD : bool, optional
        Whether to record BOLD signals (default: True)
    integration_method : str, optional
        Integration method for neural ODE ('euler' or 'heun', default: 'euler')
    seed : int, optional
        Random seed for reproducibility (default: None, no fixed seed)
        
    """
    
    # Valid parameter names
    VALID_PARAMS = [
        'w', 'I_ext', 'G', 'sigma', 'J', 'a', 'b', 'd', 'tau_s', 'gamma',
        'n_sim', 'hemodynamic_params', 'simulation_params', 'device',
        'RECORD_BOLD', 'weights', 't_cut', 't_end', 'dt', 'tr', 'warmup_steps',
        'integration_method', 'seed'
    ]
    
    # Default parameter values
    DEFAULTS = {
        # Neural model parameters
        'w': 0.9,                       # Recurrent strength (Numba default)
        'I_ext': 0.382,                    # External input (matches Numba's I_ext)
        'G': 0.0,                       # Global coupling
        'sigma': 0.02,                  # Noise amplitude
        'J': 0.2609,                    # Synaptic coupling
        'a': 270.0,                     # Firing rate slope
        'b': 108.0,                     # Firing rate threshold
        'd': 0.154,                     # Firing rate scaling
        'tau_s': 0.1,                   # Synaptic time constant
        'gamma': 0.641,                 # Synaptic gating
        'n_sim': 1,                     # Number of parameter sets
        'device': 'cuda',               # Computation device
        "weights": None,                # Must be provided
        'RECORD_BOLD': True,            # Whether to record BOLD signals
        'integration_method': 'euler',  # Integration method: 'euler' or 'heun'
        'seed': None,                   # Random seed for reproducibility (None = no fixed seed)
        # Simulation parameters
        't_cut': 120.0,      # Warm-up time in seconds
        't_end': 14.4 * 60,  # Simulation time in seconds
        'dt': 0.01,          # Time step for integration
        'tr': 0.72,          # BOLD sampling time (TR)
        'warmup_steps': 1000,  # Number of warm-up steps
    }
    
    @classmethod
    def get_default_parameters(cls) -> dict:
        """
        Get default parameters for the model
        
        Returns
        -------
        dict
            Dictionary containing all default parameters
            
        Examples
        --------
        >>> defaults = WW_sde.get_default_parameters()
        >>> defaults['G']
        0.0
        """
        return cls.DEFAULTS.copy()
    
    def __init__(self, par:dict={}):
        """
        Initialize the Wong-Wang model with parameter validation
        
        Args:
            **kwargs: Model parameters (see class docstring)
        """
        
        self._check_parameters(par)
        
        # Merge user parameters with defaults
        _par = self.DEFAULTS.copy()
        _par.update(par)
        
        # Set basic attributes
        # Handle device parameter: accept 'cpu', 'gpu', or 'cuda'
        self.device = _par.get('device', self.DEFAULTS['device']).lower()
        
        if self.device in ['gpu', 'cuda']:
            self.device = 'cuda'
            self.device = 'cuda'  # Normalize to 'cuda'
        elif self.device == 'cpu':
            self.device = 'cpu'
        else:
            raise ValueError(
                f"Invalid device '{self.device}'. "
                f"Must be 'cpu', 'gpu', or 'cuda'."
            )

        self.n_sim = _par.get('n_sim', self.DEFAULTS['n_sim'])
        
        # --- Handle weights ---
        if 'weights' not in _par:
            raise ValueError("weights parameter is required")
        weights = _par['weights']
        if isinstance(weights, np.ndarray):
            weights = torch.from_numpy(weights).float()
        if weights.dim() != 2 or weights.shape[0] != weights.shape[1]:
            raise ValueError(f"Invalid weights shape: {weights.shape}. Must be [n_nodes, n_nodes].")
        
        self.weights = weights.to(self.device)
        self.n_nodes = self.weights.shape[0]
        
        # Process and validate neural model parameters
        self._process_parameters(_par)
        
        # Set hemodynamic parameters
        self.hemo_params = _par.get('hemodynamic_params', BOLD_par())
        
        # Handle simulation parameters - either as a full object or individual parameters from dict
        if 'simulation_params' in _par and isinstance(_par['simulation_params'], Simulation_par):
            self.sim_params = _par['simulation_params']
        else:
            # Create Simulation_par with values from dictionary (already includes defaults)
            sim_kwargs = {}
            for param in ['t_cut', 't_end', 'dt', 'tr', 'warmup_steps']:
                sim_kwargs[param] = _par[param]
            self.sim_params = Simulation_par(**sim_kwargs)

        self.RECORD_BOLD = _par.get('RECORD_BOLD', self.DEFAULTS['RECORD_BOLD'])
        
        # Set integration method
        self.integration_method = _par.get('integration_method', self.DEFAULTS['integration_method']).lower()
        if self.integration_method not in ['euler', 'heun']:
            raise ValueError(f"Invalid integration_method: {self.integration_method}. Must be 'euler' or 'heun'.")
        
        # Set random seed
        self.seed = _par.get('seed', self.DEFAULTS['seed'])

    def _check_parameters(self, params: dict):
        """Validate that all provided parameters are recognized"""
        for key in params.keys():
            if key not in self.VALID_PARAMS:
                raise ValueError(
                    f"Invalid parameter: {key}\n"
                    f"Valid parameters: {', '.join(self.VALID_PARAMS)}"
                )
        
    def _process_parameters(self, params: dict):
        """Process and validate all neural model parameters"""
        param_names = ['w', 'I_ext', 'G', 'sigma', 'J', 'a', 'b', 'd', 'tau_s', 'gamma']
        
        for name in param_names:
            value = params[name]  # Already merged with defaults
            processed = check_and_expand_param(
                value, name, self.n_nodes, self.n_sim, self.device
            )
            setattr(self, name, processed)
    
    def update_params(self, **kwargs):
        """
        Update model parameters
        
        Args:
            **kwargs: Parameters to update (w, I_ext, G, sigma, J, a, b, d, tau_s, gamma, 
                     simulation params like t_cut, t_end, dt, tr, warmup_steps)
            
        Examples
        --------
        >>> model.update_params(G=0.6, J=0.28)
        >>> model.update_params(w=np.ones(68) * 1.1)
        >>> model.update_params(t_cut=100.0, dt=0.005)
        """
        self._check_parameters(kwargs)
        
        # Update n_sim if provided
        if 'n_sim' in kwargs:
            self.n_sim = kwargs['n_sim']
        
        # Update neural model parameters
        param_names = ['w', 'I_ext', 'G', 'sigma', 'J', 'a', 'b', 'd', 'tau_s', 'gamma']
        for name in param_names:
            if name in kwargs:
                processed = check_and_expand_param(
                    kwargs[name], name, self.n_nodes, self.n_sim, self.device
                )
                setattr(self, name, processed)
        
        # Update hemodynamic parameters
        if 'hemodynamic_params' in kwargs:
            self.hemo_params = kwargs['hemodynamic_params']
        
        # Update simulation parameters - handle both object and individual parameters
        sim_param_names = ['t_cut', 't_end', 'dt', 'tr', 'warmup_steps']
        if 'simulation_params' in kwargs and isinstance(kwargs['simulation_params'], Simulation_par):
            self.sim_params = kwargs['simulation_params']
        elif any(param in kwargs for param in sim_param_names):
            # Update individual simulation parameters
            sim_kwargs = {
                't_cut': self.sim_params.t_cut,
                't_end': self.sim_params.t_end,
                'dt': self.sim_params.dt,
                'tr': self.sim_params.tr,
                'warmup_steps': self.sim_params.warmup_steps,
            }
            # Override with provided values
            for param in sim_param_names:
                if param in kwargs:
                    sim_kwargs[param] = kwargs[param]
            self.sim_params = Simulation_par(**sim_kwargs)
        
        # Update RECORD_BOLD
        if 'RECORD_BOLD' in kwargs:
            self.RECORD_BOLD = kwargs['RECORD_BOLD']
        
        # Update integration method
        if 'integration_method' in kwargs:
            self.integration_method = kwargs['integration_method'].lower()
            if self.integration_method not in ['euler', 'heun']:
                raise ValueError(f"Invalid integration_method: {self.integration_method}. Must be 'euler' or 'heun'.")
        
        # Update random seed
        if 'seed' in kwargs:
            self.seed = kwargs['seed']
    
    def __str__(self) -> str:
        """Return string representation of model with parameter table"""
        
        # Parameter definitions
        param_definitions = {
            'w': 'Recurrent strength',
            'I_ext': 'External input',
            'G': 'Global coupling',
            'sigma': 'Noise amplitude',
            'J': 'Synaptic coupling',
            'a': 'Firing rate slope',
            'b': 'Firing rate threshold',
            'd': 'Firing rate scaling',
            'tau_s': 'Synaptic time constant',
            'gamma': 'Synaptic gating',
        }
        
        lines = [
            "=" * 90,
            "Reduced Wong-Wang Neural Mass Model (PyTorch)",
            "=" * 90,
            f"Network: {self.n_nodes} nodes, {self.n_sim} parameter sets",
            f"Device: {self.device}",
            f"Integration method: {self.integration_method}",
            "",
            "Neural Model Parameters:",
            "-" * 90,
            f"{'Parameter':<12} | {'Definition':<30} | {'Value/Shape':<40}",
            "-" * 90,
        ]
        
        # Display neural model parameters
        for name in ['w', 'I_ext', 'G', 'sigma', 'J', 'a', 'b', 'd', 'tau_s', 'gamma']:
            param = getattr(self, name)
            definition = param_definitions[name]
            
            # Check if parameter is iterable (has shape) or scalar
            if hasattr(param, 'shape') and param.shape != torch.Size([1, 1]):
                # Iterable: show shape
                value_str = f"shape {list(param.shape)}"
            else:
                # Scalar: show value
                value_str = f"{param[0,0].item():.6g}"
            
            lines.append(f"{name:<12} | {definition:<30} | {value_str:<40}")
        
        # Display weights information
        weights_shape_str = f"shape {list(self.weights.shape)}"
        lines.extend([
            "-" * 90,
            f"{'weights':<12} | {'Structural connectivity':<30} | {weights_shape_str:<40}",
            "",
        ])
        
        # Display simulation settings
        lines.extend([
            "Simulation Settings:",
            "-" * 90,
            f"{'Parameter':<12} | {'Definition':<30} | {'Value':<40}",
            "-" * 90,
            f"{'t_cut':<12} | {'Warm-up time (dropped)':<30} | {self.sim_params.t_cut} s",
            f"{'t_end':<12} | {'Total simulation time':<30} | {self.sim_params.t_end} s",
            f"{'t_record':<12} | {'Recording time (after drop)':<30} | {self.sim_params.t_end - self.sim_params.t_cut} s",
            f"{'dt':<12} | {'Integration time step':<30} | {self.sim_params.dt} s",
            f"{'tr':<12} | {'BOLD sampling time (TR)':<30} | {self.sim_params.tr} s",
            f"{'warmup_steps':<12} | {'Number of warm-up steps':<30} | {self.sim_params.warmup_steps}",
            f"{'RECORD_BOLD':<12} | {'Record BOLD signals':<30} | {self.RECORD_BOLD}",
            "",
        ])
        
        # Display hemodynamic parameters
        lines.extend([
            "Hemodynamic Parameters (Balloon-Windkessel):",
            "-" * 90,
            f"{'Parameter':<12} | {'Definition':<30} | {'Value':<40}",
            "-" * 90,
            f"{'beta':<12} | {'Rate constant (flow/volume)':<30} | {self.hemo_params.beta:.6g}",
            f"{'gamma':<12} | {'Rate constant (elimination)':<30} | {self.hemo_params.gamma:.6g}",
            f"{'tau':<12} | {'Hemodynamic transit time':<30} | {self.hemo_params.tau:.6g}",
            f"{'alpha':<12} | {'Grubb exponent':<30} | {self.hemo_params.alpha:.6g}",
            f"{'p_constant':<12} | {'Resting oxygen extraction':<30} | {self.hemo_params.p_constant:.6g}",
            f"{'v_0':<12} | {'Resting blood volume fraction':<30} | {self.hemo_params.v_0:.6g}",
            f"{'k_1':<12} | {'Signal coefficient':<30} | {self.hemo_params.k_1:.6g}",
            f"{'k_2':<12} | {'Signal coefficient':<30} | {self.hemo_params.k_2:.6g}",
            f"{'k_3':<12} | {'Signal coefficient':<30} | {self.hemo_params.k_3:.6g}",
            "=" * 90,
        ])
        
        return "\n".join(lines)
    
    def bold_ode(self, y_t: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
        """Balloon-Windkessel hemodynamic model ODE"""
        hp = self.hemo_params
        n_nodes, n_sim = y_t.shape
        
        dF = torch.zeros((n_nodes, n_sim, 4), device=self.device)
        
        dF[:, :, 0] = y_t - hp.beta * F[:, :, 0] - hp.gamma * (F[:, :, 1] - 1)
        dF[:, :, 1] = F[:, :, 0]
        dF[:, :, 2] = (1 / hp.tau) * (F[:, :, 1] - F[:, :, 2] ** (1 / hp.alpha))
        dF[:, :, 3] = (1 / hp.tau) * (
            F[:, :, 1] / hp.p_constant * (1 - (1 - hp.p_constant) ** (1 / F[:, :, 1]))
            - F[:, :, 3] / F[:, :, 2] * F[:, :, 2] ** (1 / hp.alpha)
        )
        
        return dF
    
    def neural_ode(self, y_t: torch.Tensor) -> torch.Tensor:
        """Wong-Wang neural mass model ODE"""
        # Total input
        x = (self.J * self.w * y_t + 
             self.J * self.G.repeat(self.n_nodes, 1) * 
             torch.mm(self.weights, y_t) + 
             self.I_ext)
        
        # Population firing rate (sigmoid transfer function)
        H = (self.a * x - self.b) / (
            1 - torch.exp(-self.d * (self.a * x - self.b))
        )
        
        # Synaptic dynamics
        dy = (-1 / self.tau_s) * y_t + self.gamma * (1 - y_t) * H
        
        return dy
    
    def integrate_neural_step(self, y_t: torch.Tensor, w_coef: torch.Tensor, 
                             n_dup: int, n_num: int) -> torch.Tensor:
        """
        Integrate neural ODE for one time step using selected method
        
        Args:
            y_t: Current state [n_nodes x n_set]
            w_coef: Noise coefficient (sigma) [n_nodes x n_set]
            n_dup: Number of duplicates
            n_num: Number of parameter sets
            
        Returns:
            Updated state after one time step
        """
        sp = self.sim_params
        
        # Generate noise for this step
        noise_raw = math.sqrt(sp.dt) * torch.randn(n_dup, self.n_nodes, device=self.device)
        noise = w_coef * noise_raw.repeat(1, 1, n_num).contiguous().view(-1, self.n_nodes).T
        
        if self.integration_method == 'euler':
            # Euler method: y(t+dt) = y(t) + f(y(t))*dt + noise
            d_y = self.neural_ode(y_t)
            y_new = y_t + d_y * sp.dt + noise
            
        elif self.integration_method == 'heun':
            # Heun's method (predictor-corrector / RK2)
            # Predictor: y_pred = y(t) + f(y(t))*dt + noise
            d_y1 = self.neural_ode(y_t)
            y_pred = y_t + d_y1 * sp.dt + noise
            
            # Corrector: y(t+dt) = y(t) + 0.5*(f(y(t)) + f(y_pred))*dt + noise
            d_y2 = self.neural_ode(y_pred)
            y_new = y_t + 0.5 * (d_y1 + d_y2) * sp.dt + noise
        
        return y_new

    
    def run(self, 
            n_dup: int = 1, 
            verbose: bool = True,
            record_neural: bool = False,
            neural_subsample: int = 5) -> Dict[str, torch.Tensor]:
        """
        Run the Wong-Wang simulation
        
        Args:
            n_dup: Number of duplicate simulations per parameter set
            verbose: Whether to print timing information
            record_neural: Whether to record neural activity over time
            neural_subsample: Subsample neural activity every N steps (default: 10)
            
        Returns:
            Dictionary containing:
            - 'bold_d': BOLD signal [n_nodes x (n_sim*n_dup) x n_timepoints]
            - 'bold_t': Time points for BOLD signal (in seconds)
            - 'S': Neural activity [n_nodes x (n_sim*n_dup) x n_subsampled] (if record_neural=True)
            - 't': Time points for neural activity (if record_neural=True)

        Examples
        --------
        >>> result = model.run(n_dup=5)
        >>> bold = result['bold_d']
        >>> times = result['bold_t']
        
        >>> # With neural recording
        >>> result = model.run(n_dup=5, record_neural=True, neural_subsample=100)
        >>> neural = result['S']  # Much less frequent sampling
        """
        # Set random seed if provided
        if self.seed is not None:
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.seed)
                torch.cuda.manual_seed_all(self.seed)
            np.random.seed(self.seed)
        
        sp = self.sim_params
        hp = self.hemo_params
        
        # Expand parameters for duplicates
        n_num = self.n_sim
        n_set = n_dup * n_num
        
        def repeat_param(p):
            """
            Duplicate parameters for multiple runs.
            
            Input shapes after check_and_expand_param:
            - [1, 1]: scalar for all nodes and sets → stays [1, 1]
            - [1, n_sim]: per-set values → becomes [1, n_sim*n_dup] 
            - [n_nodes, 1]: per-node values → stays [n_nodes, 1]
            - [n_nodes, n_sim]: per-node-per-set → becomes [n_nodes, n_sim*n_dup]
            
            For duplication, each original set should appear n_dup times consecutively.
            """
            if p.shape == torch.Size([1, 1]):
                # Scalar: no need to change, it applies to all
                return p
            elif p.shape[0] == 1 and p.shape[1] > 1:
                # Shape [1, n_sim] → [1, n_sim*n_dup]
                # Each set value should be repeated n_dup times
                return p.repeat_interleave(n_dup, dim=1)
            elif p.shape[0] > 1 and p.shape[1] == 1:
                # Shape [n_nodes, 1]: no need to change, same for all sets
                return p
            else:
                # Shape [n_nodes, n_sim] → [n_nodes, n_sim*n_dup]
                # Each column (set) should be repeated n_dup times
                return p.repeat_interleave(n_dup, dim=1)

        # Repeat all parameters for duplicates
        w_exp = repeat_param(self.w)
        I_ext_exp = repeat_param(self.I_ext)
        G_exp = repeat_param(self.G)
        sigma_exp = repeat_param(self.sigma)
        J_exp = repeat_param(self.J)
        a_exp = repeat_param(self.a)
        b_exp = repeat_param(self.b)
        d_exp = repeat_param(self.d)
        tau_s_exp = repeat_param(self.tau_s)
        gamma_exp = repeat_param(self.gamma)
        
        # Store expanded parameters temporarily
        orig_params = {
            'w': self.w, 'I_ext': self.I_ext, 'G': self.G, 'sigma': self.sigma,
            'J': self.J, 'a': self.a, 'b': self.b, 'd': self.d,
            'tau_s': self.tau_s, 'gamma': self.gamma
        }
        
        self.w = w_exp
        self.I_ext = I_ext_exp
        self.G = G_exp
        self.sigma = sigma_exp
        self.J = J_exp
        self.a = a_exp
        self.b = b_exp
        self.d = d_exp
        self.tau_s = tau_s_exp
        self.gamma = gamma_exp
        
        # Initialize neural activity
        y_t = torch.full((self.n_nodes, n_set), 0.001, device=self.device)
        
        # Initialize hemodynamic activity
        f_mat = torch.ones((self.n_nodes, n_set, 4), device=self.device)
        f_mat[:, :, 0] = 0
        
        # Prepare noise generation parameters
        w_coef = self.sigma / math.sqrt(0.001)  # ← Extra scaling factor compatible with Kong 2021 NattureComm!
        if w_coef.shape[0] == 1:
            w_coef = w_coef.repeat(self.n_nodes, 1)
        
        n_samples = sp.n_samples
        
        # Calculate when to start recording (skip warmup period)
        warmup_bold_steps = int(sp.t_cut / sp.tr)  # BOLD steps to skip
        warmup_neural_steps = int(sp.t_cut / sp.dt / neural_subsample) if record_neural else 0
        
        # Initialize storage for post-warmup data only
        tr_steps = int(sp.tr / sp.dt)  # Steps per BOLD sample
        total_bold_samples = n_samples // tr_steps  # Total BOLD samples in simulation
        n_bold_samples = total_bold_samples - warmup_bold_steps
        y_bold = torch.zeros((self.n_nodes, n_set, n_bold_samples), device=self.device)
        
        # Initialize neural activity storage (if requested)
        if record_neural:
            total_neural_samples = n_samples // neural_subsample
            n_neural_samples = total_neural_samples - warmup_neural_steps  
            y_neural = torch.zeros((self.n_nodes, n_set, n_neural_samples), device=self.device)
            neural_count = 0
        
        # Warm-up phase
        start_time = time.time()
        for i in tqdm(range(sp.warmup_steps), disable=not verbose, desc="Warm-up ..."):
            y_t = self.integrate_neural_step(y_t, w_coef, n_dup, n_num)
        
        # Main simulation loop
        count = 0
        warmup_time_steps = int(sp.t_cut / sp.dt)  # Neural timesteps to skip for warmup
        
        for i in tqdm(range(n_samples), disable=not verbose, desc="Simulating ..."):
            # Neural dynamics
            y_t = self.integrate_neural_step(y_t, w_coef, n_dup, n_num)
            
            # Only record neural activity after warmup period
            if record_neural and i >= warmup_time_steps and (i + 1) % neural_subsample == 0:
                y_neural[:, :, neural_count] = y_t
                neural_count += 1
            
            # Hemodynamic dynamics
            if self.RECORD_BOLD:
                d_f = self.bold_ode(y_t, f_mat)
                f_mat = f_mat + d_f * sp.dt
                
                # BOLD signal calculation (at sampling intervals)
                if (i + 1) % (sp.tr / sp.dt) == 0:
                    current_bold_step = (i + 1) // int(sp.tr / sp.dt) - 1  # 0-based BOLD step
                    
                    # Always calculate BOLD signal to maintain hemodynamic state
                    z_t, f_t, v_t, q_t = torch.chunk(f_mat, 4, dim=2)
                    y_bold_temp = (
                        100 / hp.p_constant * hp.v_0 *
                        (hp.k_1 * (1 - q_t) + hp.k_2 * (1 - q_t / v_t) + hp.k_3 * (1 - v_t))
                    )
                    
                    # Only store if past warmup period and within bounds
                    if current_bold_step >= warmup_bold_steps and count < n_bold_samples:
                        y_bold[:, :, count] = y_bold_temp[:, :, 0]
                        count += 1
        
        elapsed = time.time() - start_time
        if verbose:
            print(f"BOLD simulation completed in {elapsed:.2f} seconds")
        
        # Restore original parameters
        for name, val in orig_params.items():
            setattr(self, name, val)
        
        # Create time vector for BOLD (already excludes warmup)
        bold_t = torch.arange(0, y_bold.shape[2]) * sp.tr
        
        # Prepare return dictionary  
        result = {
            'bold_d': y_bold,
            'bold_t': bold_t,
        }
        
        # Add neural data if recorded (already excludes warmup)
        if record_neural:
            # Create time vector for neural activity
            neural_t = torch.arange(0, y_neural.shape[2]) * sp.dt * neural_subsample
            
            result['S'] = y_neural
            result['t'] = neural_t
        
        return result

