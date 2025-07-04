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

# write a function to detexct where is x on cpu or gpu

def is_on_cpu(x):
    '''
    Check if input is on CPU (i.e., not a CuPy array)

    Parameters
    ----------
    x: any
        Input to check

    Returns
    -------
    bool
        True if input is not a CuPy array (i.e., is on CPU), False otherwise
    '''
    if cp is None:
        return None
    return not isinstance(x, cp.ndarray)

def is_on_gpu(x):
    '''
    Check if input is on GPU (i.e., is a CuPy array)

    Parameters
    ----------
    x: any
        Input to check

    Returns
    -------
    bool or None
        True if input is a CuPy array (i.e., is on GPU)
        False if input is not a CuPy array
        None if CuPy is not installed
    '''
    if cp is None:
        return None
    return isinstance(x, cp.ndarray)



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

# def is_seq(x):
#     """Check if x is a sequence (list, tuple, array) but not a string"""
#     return hasattr(x, '__len__') and not isinstance(x, (str, bytes))


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
    dtype: str or numpy.dtype
        data type to convert to

    Returns
    -------
    x: array [len(x), n_sim]
        prepared vector

    '''
    xp = get_module(engine)
    if not is_seq(x):
        # Convert dtype string to numpy dtype
        if isinstance(dtype, str):
            if dtype == "float":
                dtype = np.float64
            elif dtype == "float32":
                dtype = np.float32
            elif dtype == "float64":
                dtype = np.float64
            else:
                raise ValueError(f"Unsupported dtype: {dtype}")
        x = xp.array(x, dtype=dtype)
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

def prepare_vec_1d(x, ns, engine, dtype="float"):
    '''
    check and prepare vector dimension and type
    '''
    if not is_seq(x):
        # Convert dtype string to numpy dtype
        if isinstance(dtype, str):
            if dtype == "float":
                dtype = np.float64
            elif dtype == "float32":
                dtype = np.float32
            elif dtype == "float64":
                dtype = np.float64
            else:
                raise ValueError(f"Unsupported dtype: {dtype}")
        x = np.array(x, dtype=dtype)
    else:
        assert(x.ndim == 1), "x must be a 1d array"
        assert(x.shape[0] == ns), "x must have the same number of elements as ns"
    x = move_data(x, engine)
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


def dtype_convert(dtype):
    """
    Convert a string representation of a data type to a numpy dtype.

    Parameters
    ----------
    dtype : str
        The string representation of the data type (e.g., "float", "float32", "float64").

    Returns
    -------
    numpy.dtype
        The corresponding numpy dtype.
    
    Raises
    ------
    ValueError
        If the input string does not match any known data type.
    """
    
    if dtype == "float":
        return np.float64
    elif dtype == "f":
        return np.float32
    elif dtype == "float32":
        return np.float32
    elif dtype == "float64":
        return np.float64
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def prepare_vec_2d(x, nn, ns, engine, dtype="float"):
    ''' 
    if x is scalar pass 
    if x is 1d array, shape should be (ns,)
    if x is 2d array, shape should be (nn, ns)
    '''
    import numpy as np
    
    # Determine target numpy dtype
    try:
        target_dtype = dtype_convert(dtype)
    except ValueError:
        target_dtype = np.dtype(dtype)
    
    if not is_seq(x):
        # x is scalar - return
        return x
    
    # Convert x to numpy array if it isn't already
    x = np.asarray(x)
    
    if x.ndim == 1:
        # 1D array - should have shape (ns,)
        if x.shape[0] != ns:
            raise ValueError(f"1D array should have shape ({ns},), got {x.shape}")
        
        # Change dtype
        x = x.astype(target_dtype)
        
        # Move to appropriate device
        x = move_data(x, engine)
        
    elif x.ndim == 2:
        # 2D array - should have shape (nn, ns)
        if x.shape != (nn, ns):
            raise ValueError(f"2D array should have shape ({nn}, {ns}), got {x.shape}")
        
        # Change dtype
        x = x.astype(target_dtype)
        
        # Move to appropriate device
        x = move_data(x, engine)
        
    else:
        raise ValueError(f"Array should be 1D or 2D, got {x.ndim}D")
    
    return x
