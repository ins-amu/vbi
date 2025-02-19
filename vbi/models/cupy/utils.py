import numpy as np

try:
    import cupy as cp
except:
    cp = None


def get_module(engine="gpu"):
    """
    Switches the computational engine between GPU and CPU.

    Parameters
    ----------
    engine : str, optional
        The computational engine to use. Can be either "gpu" or "cpu". 
        Default is "gpu".

    Returns
    -------
    module
        The appropriate array module based on the specified engine. 
        If "gpu", returns the CuPy module. If "cpu", returns the NumPy module.

    Raises
    ------
    ValueError
        - If the specified engine is not "gpu" or "cpu".
        - If CuPy is not installed.
    """
    
    if engine == "gpu":
        if cp is None:
            raise ValueError("CuPy is not installed.")
        else:
            return cp.get_array_module(cp.array([1]))
    else:
        return np
        # return cp.get_array_module(np.array([1]))


def tohost(x):
    '''
    move data to cpu if it is on gpu

    Parameters
    ----------
    x: array
        data

    Returns
    -------
    array
        data moved to cpu
    '''
    if cp is not None and isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    return x


def todevice(x):
    '''
    move data to gpu

    Parameters
    ----------
    x: array
        data

    Returns
    -------
    array
        data moved to gpu

    '''
    return cp.asarray(x)


def move_data(x, engine):
    if engine == "cpu":
        return tohost(x)
    elif engine == "gpu":
        return todevice(x)


def repmat_vec(vec, ns, engine):
    '''
    repeat vector ns times

    Parameters
    ----------
    vec: array 1d
        vector to be repeated
    ns: int
        number of repetitions
    engine: str
        cpu or gpu

    Returns
    -------
    vec: array [len(vec), n_sim]
        repeated vector

    '''
    vec = np.tile(vec, (ns, 1)).T
    vec = move_data(vec, engine)
    return vec


def is_seq(x):
    '''
    check if x is a sequence

    Parameters
    ----------
    x: any
        variable to be checked

    Returns
    -------
    bool
        True if x is a sequence

    '''
    return hasattr(x, '__iter__')


def prepare_vec(x, ns, engine, dtype="float"):
    '''
    check and prepare vector dimension and type

    Parameters
    ----------
    x: array 1d
        vector to be prepared, if x is a scalar, only the type is changed
    ns: int
        number of simulations
    engine: str
        cpu or gpu

    Returns
    -------
    x: array [len(x), n_sim]
        prepared vector

    '''
    xp = get_module(engine)

    if not is_seq(x):
        return eval(f"{dtype}({x})")
    else:
        x = np.array(x)
        if x.ndim == 1:
            x = repmat_vec(x, ns, engine)
        elif x.ndim == 2:
            assert(x.shape[1] == ns), "second dimension of x must be equal to ns"
            x = move_data(x, engine)
        else:
            raise ValueError("x.ndim must be 1 or 2")
    return x.astype(dtype)


def get_(x, engine="cpu", dtype="f"):
    """
    Parameters
    ----------
    x : array-like
        The input array to be converted.
    engine : str, optional
        The computation engine to use. If "gpu", the array is transferred from GPU to CPU. Defaults to "cpu".
    dtype : str, optional
        The desired data type for the output array. Defaults to "f".

    Returns
    -------
    array-like
        The converted array with the specified data type.
    
    """
    
    if engine == "gpu":
        return x.get().astype(dtype)
    else:
        return x.astype(dtype)
