import os
import time
# import tqdm
# import torch
# import logging
# from os import stat
import numpy as np
import pandas as pd
# import pylab as plt
# from numba import jit
# import collections.abc
# import multiprocessing as mp
from os.path import join
import matplotlib.pyplot as plt


def count_depth(ls):
    '''
    count the depth of a list

    '''
    if isinstance(ls, (list, tuple)):
        return 1 + max(count_depth(item) for item in ls)
    else:
        return 0


def prepare_input(ts, dtype=np.float32):
    '''
    prepare input format

    Parameters
    ----------
    ts : array-like or list
        Input from which the features are extracted
    Returns
    -------
    ts: nd-array
        formatted input

    '''

    if isinstance(ts, np.ndarray):
        if ts.ndim == 3:
            pass
        elif ts.ndim == 2:
            ts = ts[:, np.newaxis, :] # n_region = 1 
        else:
            ts = ts[np.newaxis, np.newaxis, :] # n_region , n_trial = 1

    elif isinstance(ts, (list, tuple)):
        if isinstance(ts[0], np.ndarray):
            if ts[0].ndim == 2:
                ts = np.array(ts, dtype=dtype)
            elif ts[0].ndim == 1:
                ts = np.array(ts, dtype=dtype)
                ts = ts[:, np.newaxis, :] # n_region = 1
            else:
                ts = np.array(ts, dtype=dtype)[np.newaxis, np.newaxis, :]
        else:
            if isinstance(ts[0], (list, tuple)):
                depth = count_depth(ts)
                if depth == 3:
                    ts = np.asarray(ts)
                elif depth == 2:
                    ts = np.array(ts)
                    ts = ts[:, np.newaxis, :] # n_region = 1
                else:
                    ts = np.array(ts)[np.newaxis, np.newaxis, :] # n_region , n_trial = 1

    # if ts is dataframe
    elif isinstance(ts, pd.DataFrame):
        # assume that the dataframe is in the form of
        # columns: time series
        # rows: time
        ts = ts.values.T
        ts = ts[:, np.newaxis, :] # n_region = 1

    return ts


def check_input(ts):

    if not isinstance(ts, np.ndarray):
        ts = np.array(ts)
    if ts.ndim == 1:
        ts = ts.reshape(1, -1)
    if ts.size == 0:
        return False, 1

    if np.isnan(ts).any() or np.isinf(ts).any():
        n = ts.shape[0]
        return False, n
    return True, ts


def make_mask(n, indices):
    '''
    make a mask matrix with given indices

    Parameters
    ----------
    n : int
        size of the mask matrix
    indices : list
        indices of the mask matrix

    Returns
    -------
    mask : numpy.ndarray
        mask matrix
    '''
    # check validity of indices
    if not isinstance(indices, (list, tuple, np.ndarray)):
        raise ValueError('indices must be a list, tuple, or numpy array.')
    if not all(isinstance(i, int) for i in indices):
        raise ValueError('indices must be a list of integers.')
    if not all(i < n for i in indices):
        raise ValueError('indices must be smaller than n.')

    mask = np.zeros((n, n))
    mask[np.ix_(indices, indices)] = 1

    return mask


def get_intrah_mask(n_nodes):
    '''
    Get a mask for intrahemispheric connections.

    Inputs
    ------------
    n_nodes: int
        number of total nodes that constitute the data.

    Outputs
    ------------
    mask_intrah: 2d array
        mask for intrahemispheric connections.
    '''
    row_idx = np.arange(n_nodes)
    idx1 = np.ix_(row_idx[:n_nodes//2], row_idx[:n_nodes//2])
    idx2 = np.ix_(row_idx[n_nodes//2:], row_idx[n_nodes//2:])
    # build on a zeros mask
    mask_intrah = np.zeros((n_nodes, n_nodes))
    mask_intrah[idx1] = 1
    mask_intrah[idx2] = 1
    return mask_intrah


def get_interh_mask(n_nodes):
    '''
    Get a mask for interhemispheric connections.

    Inputs
    ------------
    n_nodes: int
        number of total nodes that constitute the data.

    Outputs
    ------------
    mask_interh: 2d array
        mask for interhemispheric connections.
    '''
    row_idx = np.arange(n_nodes//2)
    col_idx1 = np.where(np.eye(n_nodes, k=-n_nodes//2))[0]
    col_idx2 = np.where(np.eye(n_nodes, k=n_nodes//2))[0]
    idx1 = np.ix_(row_idx, col_idx1)
    idx2 = np.ix_(row_idx+n_nodes//2, col_idx2)
    # build on a zeros mask
    mask_interh = np.zeros((n_nodes, n_nodes))
    mask_interh[idx1] = 1
    mask_interh[idx2] = 1
    return mask_interh


def get_masks(n_nodes, networks):
    '''
    Get a dictionary of masks based on the requested networks.

    Parameters
    ------------
    n_nodes: int
        number of total nodes that constitute the data.
    networks: list of str
        list of networks to be included in the dictionary.
        'full': full-network connections
        'intrah': intrahemispheric connections
        'interh': interhemispheric connections
        to get a custom mask with specific indices
        refere to `hbt.utility.make_mask(n, indices)`.

    Outputs
    ------------
    masks: dict
        dictionary of masks based on the requested networks.
    '''
    masks = {}
    valid_networks = ['full', 'intrah', 'interh']

    for i, ntw in enumerate(networks):
        if ntw not in valid_networks:
            raise ValueError(
                f"Invalid network: {ntw}. Please choose from {valid_networks}.")
        if ntw == 'full':
            masks[ntw] = np.ones((n_nodes, n_nodes))
        elif ntw == 'intrah':
            masks[ntw] = get_intrah_mask(n_nodes)
        elif ntw == 'interh':
            masks[ntw] = get_interh_mask(n_nodes)

    return masks


def is_sequence(arg):
    '''
    Check if the input is a sequence (list, tuple, np.ndarray, etc.)

    Parameters
    ----------
    arg : any
        input to be checked.

    Returns
    -------
    bool
        True if the input is a sequence, False otherwise.

    '''
    return isinstance(arg, (list, tuple, np.ndarray))


def set_k_diagonals(A, k=0, value=0):
    '''
    set k diagonals of the given matrix to given value.

    Parameters
    ----------
    A : numpy.ndarray
        input matrix.
    k : int
        number of diagonals to be set. The default is 0.
        Notice that the main diagonal is 0.
    value : int, optional
        value to be set. The default is 0.
    '''

    if not isinstance(A, np.ndarray):
        A = np.array(A)
    if A.ndim != 2:
        raise ValueError('A must be a 2d array.')
    if not isinstance(k, int):
        raise ValueError('k must be an integer.')
    if not isinstance(value, (int, float)):
        raise ValueError('value must be a number.')
    if k >= A.shape[0]:
        raise ValueError('k must be smaller than the size of A.')

    n = A.shape[0]

    for i in range(-k, k+1):
        a1 = np.diag(np.random.randint(1, 2, n - abs(i)), i)
        idx = np.where(a1)
        A[idx] = value
    return A


def if_symmetric(A, tol=1e-8):
    '''
    Check if the input matrix is symmetric.

    Parameters
    ----------
    A : numpy.ndarray
        input matrix.
    tol : float, optional
        tolerance for checking symmetry. The default is 1e-8.

    Returns
    -------
    bool
        True if the input matrix is symmetric, False otherwise.

    '''
    if not isinstance(A, np.ndarray):
        A = np.array(A)
    if A.ndim != 2:
        raise ValueError('A must be a 2d array.')

    return np.allclose(A, A.T, atol=tol)
