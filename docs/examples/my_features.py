import numpy as np 

def mean(ts, arg1, arg2):
    '''
    This is a custom feature function. It takes a time series as input and returns
    the mean of each region.

    Parameters
    ----------
    ts : numpy array
        Time series of shape (n_region, n_timepoint)
    arg1 : float
        Argument 1, dummy argument for illustration
    arg2 : float
        Argument 2, dummy argument for illustration

    Returns
    -------
    mean_ts : numpy array
        Mean of each region
    labels : list of strings
        List of labels of the features

    '''

    if not isinstance(ts, np.ndarray):
        ts = np.array(ts)

    if ts.ndim == 1:
        ts = ts.reshape(1, -1)
    n_region, n_timepoint = ts.shape
    
    # compute the mean of each region
    mean_ts = ts.mean(axis=1)
    labels = [f'm_{i}' for i in range(n_region)]
    
    return mean_ts, labels

def std(ts, arg1, arg2):
    '''
    This is a custom feature function. It takes a time series as input and returns
    the mean of each region.

    Parameters
    ----------
    ts : numpy array
        Time series of shape (n_region, n_timepoint)
    arg1 : float
        Argument 1, dummy argument for illustration
    arg2 : float
        Argument 2, dummy argument for illustration
    
    Returns
    -------
    std_ts : numpy array
        Standard deviation of each region
    labels : list of strings
        List of labels of the features

    '''

    if not isinstance(ts, np.ndarray):
        ts = np.array(ts)

    if ts.ndim == 1:
        ts = ts.reshape(1, -1)
    n_region, n_timepoint = ts.shape
    
    # compute the mean of each region
    std_ts = ts.std(axis=1)
    labels = [f's_{i}' for i in range(n_region)]
    
    return std_ts, labels
