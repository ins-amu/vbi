"""
Base class for VBI models providing a unified interface for parameter management.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any


class BaseModel(ABC):
    """
    Abstract base class for all VBI models.
    
    This class provides a unified interface for model parameter management,
    ensuring consistency across different model implementations.
    
    Attributes
    ----------
    _par : dict
        Internal dictionary storing all model parameters.
    valid_params : list
        List of valid parameter names for this model.
    """
    
    def __init__(self):
        """Initialize the base model."""
        self._par = {}
        self.valid_params = []
    
    @abstractmethod
    def get_default_parameters(self) -> Dict[str, Any]:
        """
        Get the default parameters for the model.
        
        Returns
        -------
        dict
            Dictionary containing default parameter values.
        """
        pass
    
    @abstractmethod
    def get_parameter_descriptions(self) -> Dict[str, str]:
        """
        Get descriptions for all model parameters.
        
        Returns
        -------
        dict
            Dictionary mapping parameter names to their descriptions.
        """
        pass
    
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
        """
        for key in par.keys():
            if key not in self.valid_params:
                raise ValueError(f"Invalid parameter: {key}")
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get all model parameters as a dictionary.
        
        Returns
        -------
        dict
            Dictionary containing all current model parameters.
            
        Examples
        --------
        >>> model = VEP_sde()
        >>> params = model.get_parameters()
        >>> print(params['G'])
        1.0
        """
        return self._par.copy()
    
    def set_parameters(self, par: Dict[str, Any]) -> None:
        """
        Update model parameters.
        
        Parameters
        ----------
        par : dict
            Dictionary of parameters to update. Only valid parameters
            will be updated.
            
        Raises
        ------
        ValueError
            If any parameter name is invalid.
            
        Examples
        --------
        >>> model = VEP_sde()
        >>> model.set_parameters({'G': 2.0, 'tau': 15.0})
        """
        self.check_parameters(par)
        self._par.update(par)
        
        # Update attributes
        for name, value in par.items():
            setattr(self, name, value)
    
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
        KeyError
            If parameter name does not exist.
            
        Examples
        --------
        >>> model = VEP_sde()
        >>> g_value = model.get_parameter('G')
        >>> print(g_value)
        1.0
        """
        if name not in self._par:
            raise KeyError(f"Parameter '{name}' not found in model.")
        return self._par[name]
    
    def list_parameters(self) -> List[str]:
        """
        Get a list of all valid parameter names for this model.
        
        Returns
        -------
        list
            List of valid parameter names.
            
        Examples
        --------
        >>> model = VEP_sde()
        >>> params = model.list_parameters()
        >>> print('G' in params)
        True
        """
        return list(self.valid_params)
    
    @abstractmethod
    def run(self, par: Dict[str, Any] = {}, **kwargs) -> Dict[str, Any]:
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
            and state variables).
        """
        pass
    
    def __call__(self) -> Dict[str, Any]:
        """
        Return model parameters when instance is called.
        
        Returns
        -------
        dict
            Dictionary of all model parameters.
        """
        return self._par
    
    def _format_value(self, value):
        """Format a parameter value for display"""
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
        else:
            # Scalar or other type
            return f"{value}"
    
    def _format_parameters_table(self) -> str:
        """
        Format model parameters as a table with names, descriptions, values, and types.
        
        Returns
        -------
        str
            Formatted table string with 4 columns:
            - Parameter: parameter name
            - Description: what the parameter does  
            - Value: current value or shape
            - Type: scalar | vector | matrix | string | bool | -
        """
        param_info = self.get_parameter_descriptions()
        
        lines = [
            "=" * 110,
            f"{self.__class__.__name__}",
            "=" * 110,
            "",
            "Model Parameters:",
            "-" * 110,
            f"{'Parameter':<15} | {'Description':<40} | {'Value/Shape':<30} | {'Type':<15}",
            "-" * 110,
        ]
        
        for name in sorted(self.valid_params):
            if name in self._par:
                # Get current value (may have been converted)
                try:
                    current_value = getattr(self, name)
                except AttributeError:
                    current_value = self._par[name]
                
                # Get description and type from param_info
                if isinstance(param_info.get(name), tuple):
                    description, param_type = param_info[name]
                else:
                    description = param_info.get(name, "No description")
                    param_type = "-"
                
                # Format current value for display
                current_str = self._format_value(current_value)
                
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
