import numpy as np
from vbi.utils import print_valid_parameters


def check_vec_size_1d(x, nn):
    """
    Return a 1D vector of size nn, broadcasting scalar if needed.

    This utility function ensures that parameter inputs are properly
    formatted as arrays of the correct size for multi-region simulations.

    Parameters
    ----------
    x : scalar or array-like
        Input value(s) to be broadcast or validated
    nn : int
        Required array size (number of brain regions)

    Returns
    -------
    np.ndarray
        Array of shape (nn,) with input values properly broadcast
    """
    x = (
        np.array(x, dtype=np.float64)
        if np.ndim(x) > 0
        else np.array([x], dtype=np.float64)
    )
    return np.ones(nn, dtype=np.float64) * x if x.size != nn else x.astype(np.float64)


def check_parameters(par: dict, valid_params: list, model_spec: list) -> None:
    """
    Validate that all provided parameters are recognized.

    Parameters
    ----------
    par : dict
        Dictionary of parameters to validate.

    Raises
    ------
    ValueError
        If any parameter name is not recognized.
    """
    for key in par.keys():
        if key not in valid_params:
            print(f"Invalid parameter: {key}")
            print_valid_parameters(model_spec)
            raise ValueError(f"Invalid parameter: {key}")
