import numpy as np
from numpy.matlib import repmat

try:
    import cupy as cp
except:
    cp = None


def get_module(engine="gpu"):
    '''
    to switch engine between gpu and cpu
    '''
    if engine == "gpu":
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
    vec = repmat(vec, ns, 1).T
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
    if engine == "gpu":
        return x.get().astype(dtype)
    else:
        return x.astype(dtype)
