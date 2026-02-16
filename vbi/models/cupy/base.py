"""
Base class for VBI CuPy models providing a unified interface for parameter management.

This base class provides a consistent API across all CuPy-based neural mass models,
ensuring consistency in parameter access, documentation, and display.
"""
from typing import Dict, List, Any
import numpy as np


class BaseCupyModel:
    """
    Abstract base class for all VBI CuPy models.
    
    This class provides a unified interface for model parameter management,
    ensuring consistency across different CuPy model implementations.
    
    CuPy models typically store parameters directly as instance attributes
    (e.g., self.G, self.dt, self.sigma) and may also maintain a par_ dict.
    This base class provides a Python-level interface for accessing and
    documenting these parameters.
    
    Attributes
    ----------
    par_ : dict
        Dictionary storing all model parameters.
    valid_params : list
        List of valid parameter names for this model.
    """
    
    def __init__(self):
        """Initialize the base CuPy model."""
        self.par_ = {}
        self.valid_params = []
    
    def get_default_parameters(self) -> Dict[str, Any]:
        """
        Get the default parameters for the model.
        
        Returns
        -------
        dict
            Dictionary containing default parameter values.
            
        Examples
        --------
        >>> model = GHB_sde({"G": 25.0})
        >>> defaults = model.get_default_parameters()
        >>> print(defaults['G'])
        25.0
        """
        raise NotImplementedError("Subclass must implement get_default_parameters()")
    
    def get_parameter_descriptions(self) -> Dict[str, tuple]:
        """
        Get descriptions for all model parameters.
        
        Returns
        -------
        dict
            Dictionary mapping parameter names to (description, type) tuples.
            Type should be one of: 'float', 'int', 'str', 'bool', 'ndarray'.
            
        Examples
        --------
        >>> model = GHB_sde({"G": 25.0})
        >>> descriptions = model.get_parameter_descriptions()
        >>> print(descriptions['G'])
        ('Global coupling strength', 'float')
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
        AssertionError
            If any parameter name is not in the valid_params list.
            
        Examples
        --------
        >>> model = GHB_sde({"G": 25.0})
        >>> model.check_parameters({'G': 30.0})  # Valid
        >>> model.check_parameters({'invalid_param': 1.0})  # Raises AssertionError
        """
        for key in par.keys():
            assert key in self.valid_params, f"Invalid parameter: {key}"
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get all model parameters as a dictionary.
        
        Returns
        -------
        dict
            Dictionary containing all current model parameters.
            
        Examples
        --------
        >>> model = GHB_sde({"G": 25.0, "dt": 0.01})
        >>> params = model.get_parameters()
        >>> print(params['G'])
        25.0
        """
        if hasattr(self, 'par_') and self.par_:
            # Return a copy to prevent external modification
            return dict(self.par_)
        
        # Fallback: construct from valid_params
        params = {}
        for param_name in self.valid_params:
            if hasattr(self, param_name):
                value = getattr(self, param_name)
                # Convert cupy arrays to numpy for display/serialization
                try:
                    import cupy as cp
                    if isinstance(value, cp.ndarray):
                        value = cp.asnumpy(value)
                except ImportError:
                    pass
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
            
        Examples
        --------
        >>> model = GHB_sde({"G": 25.0})
        >>> g_value = model.get_parameter('G')
        >>> print(g_value)
        25.0
        """
        if hasattr(self, 'par_') and name in self.par_:
            return self.par_[name]
        
        if hasattr(self, name):
            value = getattr(self, name)
            # Convert cupy arrays to numpy for display
            try:
                import cupy as cp
                if isinstance(value, cp.ndarray):
                    return cp.asnumpy(value)
            except ImportError:
                pass
            return value
        
        raise AttributeError(f"Parameter '{name}' not found.")
    
    def list_parameters(self) -> List[str]:
        """
        Get a list of all valid parameter names for this model.
        
        Returns
        -------
        list
            List of valid parameter names.
            
        Examples
        --------
        >>> model = GHB_sde({"G": 25.0})
        >>> params = model.list_parameters()
        >>> print('G' in params)
        True
        """
        return list(self.valid_params)
    
    def _format_value(self, value: Any) -> str:
        """
        Format a parameter value for display in the table.
        
        Parameters
        ----------
        value : Any
            Parameter value to format.
            
        Returns
        -------
        str
            Formatted string representation.
        """
        if value is None:
            return "None"
        elif isinstance(value, np.ndarray):
            if value.size == 1:
                return f"{value.item()}"
            else:
                return f"shape {value.shape}"
        elif isinstance(value, (list, tuple)) and len(value) > 3:
            return f"length {len(value)}"
        elif isinstance(value, (int, float, np.integer, np.floating)):
            return f"{value}"
        elif isinstance(value, str):
            return f'"{value}"'
        elif isinstance(value, bool):
            return f"{value}"
        else:
            # Check for cupy arrays
            try:
                import cupy as cp
                if isinstance(value, cp.ndarray):
                    if value.size == 1:
                        return f"{cp.asnumpy(value).item()}"
                    else:
                        return f"shape {value.shape}"
            except ImportError:
                pass
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
            - Type: float | int | str | bool | ndarray | -
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
