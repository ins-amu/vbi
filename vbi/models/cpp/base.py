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
