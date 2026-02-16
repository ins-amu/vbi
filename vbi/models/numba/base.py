"""
Base class for VBI Numba models providing a unified interface for parameter management.

This base class provides a consistent API across all Numba-based neural mass models,
similar to the BaseModel class used for C++ models. However, it accounts for the
specific architecture of Numba models which use jitclass for efficient parameter storage.
"""
from typing import Dict, List, Any
import numpy as np


class BaseNumbaModel:
    """
    Abstract base class for all VBI Numba models.
    
    This class provides a unified interface for model parameter management,
    ensuring consistency across different Numba model implementations.
    
    Unlike the C++ BaseModel, Numba models typically use a jitclass (e.g., ParVEP, ParJR)
    for storing parameters that will be used in JIT-compiled functions. This base class
    provides a Python-level interface while the actual parameters may be stored in the
    jitclass instance (typically self.P or self.par).
    
    Attributes
    ----------
    P : jitclass instance
        Internal jitclass instance storing all model parameters for Numba compilation.
    valid_params : list
        List of valid parameter names for this model.
    """
    
    def __init__(self):
        """Initialize the base Numba model."""
        self.P = None  # Will be set by subclass to jitclass instance
        self.valid_params = []
        self._checked = False
    
    def get_default_parameters(self) -> Dict[str, Any]:
        """
        Get the default parameters for the model.
        
        Returns
        -------
        dict
            Dictionary containing default parameter values.
            
        Examples
        --------
        >>> model = JR_sde({"weights": np.eye(2)})
        >>> defaults = model.get_default_parameters()
        >>> print(defaults['G'])
        1.0
        """
        raise NotImplementedError("Subclass must implement get_default_parameters()")
    
    def get_parameter_descriptions(self) -> Dict[str, str]:
        """
        Get descriptions for all model parameters.
        
        Returns
        -------
        dict
            Dictionary mapping parameter names to their descriptions.
            Can also return tuples of (description, type) where type is one of:
            'scalar', 'vector', 'matrix', 'string', 'int', 'bool'.
            
        Examples
        --------
        >>> model = JR_sde({"weights": np.eye(2)})
        >>> descriptions = model.get_parameter_descriptions()
        >>> print(descriptions['G'])
        ('Global coupling strength', 'scalar')
        """
        raise NotImplementedError("Subclass must implement get_parameter_descriptions()")
    
    def check_parameters(self, par: Dict[str, Any]) -> None:
        """
        Validate that all provided parameters are valid for this model.
        
        Parameters
        ----------
        par : dict
            Dictionary of parameters to validate.
            
        Raises
        ------
        ValueError
            If any parameter name is not in the valid_params list.
            
        Examples
        --------
        >>> model = JR_sde({"weights": np.eye(2)})
        >>> model.check_parameters({'G': 2.0})  # Valid
        >>> model.check_parameters({'invalid_param': 1.0})  # Raises ValueError
        """
        invalid_params = []
        for key in par.keys():
            if key not in self.valid_params:
                invalid_params.append(key)
        
        if invalid_params:
            print(f"Invalid parameter(s): {', '.join(invalid_params)}")
            self.print_valid_parameters()
            raise ValueError(f"Invalid parameter(s): {', '.join(invalid_params)}")
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get all model parameters as a dictionary.
        
        This method can be overridden by subclasses that have a _par_to_dict()
        helper method. By default, it reads parameters directly from self.P.
        
        Returns
        -------
        dict
            Dictionary containing all current model parameters from the jitclass.
            
        Examples
        --------
        >>> model = JR_sde({"weights": np.eye(2), "G": 2.0})
        >>> params = model.get_parameters()
        >>> print(params['G'])
        2.0
        """
        if self.P is None:
            return {}
        
        # If subclass has _par_to_dict, use it
        if hasattr(self, '_par_to_dict'):
            return self._par_to_dict()
        
        # Default fallback: read directly from self.P
        params = {}
        for param_name in self.valid_params:
            if hasattr(self.P, param_name):
                value = getattr(self.P, param_name)
                # Convert numpy arrays for consistency
                if isinstance(value, np.ndarray):
                    params[param_name] = np.array(value)
                else:
                    params[param_name] = value
        
        return params
    
    def get_parameter(self, name: str) -> Any:
        """
        Get the value of a specific parameter.
        
        Parameters
        ----------
        name : str
            Name of the parameter to retrieve.
            
        Returns
        -------
        Any
            Value of the requested parameter.
            
        Raises
        ------
        AttributeError
            If parameter name does not exist.
        ValueError
            If the parameter object (jitclass) has not been initialized.
            
        Notes
        -----
        This method allows reading any parameter that exists in the parameter
        object (self.P), including derived parameters like 'nn' (number of nodes)
        even though they are not in valid_params (user-settable parameters).
            
        Examples
        --------
        >>> model = JR_sde({"weights": np.eye(2)})
        >>> g_value = model.get_parameter('G')
        >>> print(g_value)
        1.0
        >>> nn_value = model.get_parameter('nn')  # Derived parameter
        >>> print(nn_value)
        2
        """
        if self.P is None:
            raise ValueError("Parameter object not initialized. Run __init__ first.")
        
        if not hasattr(self.P, name):
            raise AttributeError(f"Parameter '{name}' not found in parameter object.")
        
        return getattr(self.P, name)
    
    def list_parameters(self) -> List[str]:
        """
        Get a list of all valid parameter names for this model.
        
        Returns
        -------
        list
            List of valid parameter names.
            
        Examples
        --------
        >>> model = JR_sde({"weights": np.eye(2)})
        >>> params = model.list_parameters()
        >>> print('G' in params)
        True
        """
        return list(self.valid_params)
    
    def print_valid_parameters(self):
        """
        Print all valid parameter names for this model.
        
        This is a helper method for debugging and documentation purposes.
        
        Examples
        --------
        >>> model = JR_sde({"weights": np.eye(2)})
        >>> model.print_valid_parameters()
        Valid parameters:
        G, A, B, a, b, v0, vmax, r, mu, noise_amp, ...
        """
        print("Valid parameters:")
        print(", ".join(self.valid_params))
    
    def run(self, par: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        """
        Run the model simulation.
        
        Parameters
        ----------
        par : dict, optional
            Dictionary of parameters to update before running.
        **kwargs
            Additional keyword arguments specific to the model.
            
        Returns
        -------
        dict
            Dictionary containing simulation results (typically 't' for time
            and state variables like 'x', 'E', 'I', etc.).
            
        Examples
        --------
        >>> model = JR_sde({"weights": np.eye(2), "dt": 0.01, "t_end": 100.0})
        >>> result = model.run()
        >>> t, x = result['t'], result['x']
        """
        raise NotImplementedError("Subclass must implement run()")
    
    def __call__(self) -> Dict[str, Any]:
        """
        Return model parameters when instance is called.
        
        Returns
        -------
        dict
            Dictionary of all model parameters.
            
        Examples
        --------
        >>> model = JR_sde({"weights": np.eye(2)})
        >>> params = model()
        >>> print(params['G'])
        1.0
        """
        return self.get_parameters()
    
    def _format_value(self, value):
        """
        Format a parameter value for display.
        
        Parameters
        ----------
        value : Any
            The parameter value to format.
            
        Returns
        -------
        str
            Formatted string representation of the value.
        """
        if hasattr(value, 'shape') and value is not None:
            # Array-like object
            try:
                shape_tuple = tuple(value.shape)
                if len(shape_tuple) == 0:
                    # 0-d array (scalar stored as array)
                    return f"{value}"
                else:
                    return f"shape {shape_tuple}"
            except:
                return str(type(value).__name__)
        elif isinstance(value, (list, tuple)) and len(value) > 0:
            # List/tuple
            if len(value) <= 3:
                return f"{value}"
            else:
                return f"[{len(value)} items]"
        elif isinstance(value, dict):
            return f"dict({len(value)})"
        elif isinstance(value, (int, float, np.integer, np.floating)):
            return f"{value}"
        elif isinstance(value, str):
            return f'"{value}"'
        elif isinstance(value, bool):
            return f"{value}"
        else:
            # Other types
            return str(value)
    
    def _format_parameters_table(self, model_name: str = None) -> str:
        """
        Format model parameters as a table with names, descriptions, values, and types.
        
        Parameters
        ----------
        model_name : str, optional
            Custom name to display for the model. If None, uses self.__class__.__name__.
        
        Returns
        -------
        str
            Formatted table string with 4 columns:
            - Parameter: parameter name
            - Description: what the parameter does
            - Value: current value or shape
            - Type: scalar | vector | matrix | string | bool | int | -
        """
        param_info = self.get_parameter_descriptions()
        current_params = self.get_parameters()
        
        # Use provided model_name or default to class name
        display_name = model_name if model_name is not None else self.__class__.__name__
        
        lines = [
            "=" * 110,
            f"{display_name}",
            "=" * 110,
            "",
            "Model Parameters:",
            "-" * 110,
            f"{'Parameter':<15} | {'Description':<40} | {'Value/Shape':<30} | {'Type':<15}",
            "-" * 110,
        ]
        
        for name in sorted(self.valid_params):
            if name in current_params:
                current_value = current_params[name]
                
                # Get description and type from param_info
                if isinstance(param_info.get(name), tuple):
                    description, param_type = param_info[name]
                else:
                    description = param_info.get(name, "No description")
                    param_type = "-"
                
                # Format current value for display
                current_str = self._format_value(current_value)
                
                # Truncate long strings
                if len(description) > 40:
                    description = description[:37] + "..."
                if len(current_str) > 30:
                    current_str = current_str[:27] + "..."
                
                lines.append(f"{name:<15} | {description:<40} | {current_str:<30} | {param_type:<15}")
        
        lines.append("=" * 110)
        return "\n".join(lines)
    
    def __str__(self) -> str:
        """
        Return string representation of the model with parameter table.
        
        Returns
        -------
        str
            Formatted string with model information and parameters table.
        """
        return self._format_parameters_table()
    
    def __repr__(self) -> str:
        """
        Return detailed string representation of the model.
        
        Returns
        -------
        str
            String representation showing class name and number of parameters.
        """
        return f"{self.__class__.__name__}(n_params={len(self.valid_params)})"
