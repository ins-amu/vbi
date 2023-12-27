import numpy as np

try:
    import cupy as cp
except:
    pass


def get_module(engine="gpu"):
    '''
    to switch engine between gpu and cpu
    '''
    if engine == "gpu":
        return cp.get_array_module(cp.array([1]))
    else:
        return cp.get_array_module(np.array([1]))


def tohost(x):
    '''
    move data to cpu

    Parameters
    ----------
    x: array
        data

    Returns
    -------
    array
        data moved to cpu
    '''
    return cp.asnumpy(x)


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
