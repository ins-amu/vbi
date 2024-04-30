import torch 
import logging
import numpy as np
import pandas as pd
from torch import Tensor
from typing import Union, List


def count_depth(ls):
    """
    count the depth of a list

    """
    if isinstance(ls, (list, tuple)):
        return 1 + max(count_depth(item) for item in ls)
    else:
        return 0


def prepare_input(ts, dtype=np.float32):
    """
    prepare input format

    Parameters
    ----------
    ts : array-like or list
        Input from which the features are extracted
    Returns
    -------
    ts: nd-array
        formatted input

    """
    n_trial = 0

    if isinstance(ts, np.ndarray):
        if ts.ndim == 3:
            pass
        elif ts.ndim == 2:
            ts = ts[:, np.newaxis, :]  # n_region = 1
        else:
            ts = ts[np.newaxis, np.newaxis, :]  # n_region , n_trial = 1

    elif isinstance(ts, (list, tuple)):
        if isinstance(ts[0], np.ndarray):
            if ts[0].ndim == 2:
                ts = np.array(ts, dtype=dtype)
            elif ts[0].ndim == 1:
                ts = np.array(ts, dtype=dtype)
                ts = ts[:, np.newaxis, :]  # n_region = 1
            else:
                ts = np.array(ts, dtype=dtype)[np.newaxis, np.newaxis, :]
        else:
            if isinstance(ts[0], (list, tuple)):
                depth = count_depth(ts)
                if depth == 3:
                    ts = np.asarray(ts)
                elif depth == 2:
                    ts = np.array(ts)
                    ts = ts[:, np.newaxis, :]  # n_region = 1
                else:
                    ts = np.array(ts)[
                        np.newaxis, np.newaxis, :
                    ]  # n_region , n_trial = 1

    # if ts is dataframe
    elif isinstance(ts, pd.DataFrame):
        # assume that the dataframe is in the form of
        # columns: time series
        # rows: time
        ts = ts.values.T
        ts = ts[:, np.newaxis, :]  # n_region = 1

    return ts, n_trial


def prepare_input_ts(ts, indices: List[int] = None):

    if not isinstance(ts, np.ndarray):
        ts = np.array(ts)
    if indices is None:
        indices = np.arange(ts.shape[0], dtype=np.int32)

    # check indices validity
    if not isinstance(indices, (list, tuple, np.ndarray)):
        raise ValueError("indices must be a list, tuple, or numpy array.")
    if not all(isinstance(i, (int, np.int64, np.int32, np.int16)) for i in indices):
        raise ValueError("indices must be a list of integers.")
    if not all(i < ts.shape[0] for i in indices):
        raise ValueError("indices must be smaller than the number of time series.")

    ts = ts[indices]

    if ts.ndim == 1:
        ts = ts.reshape(1, -1)

    if ts.size == 0:
        return False, ts

    if np.isnan(ts).any() or np.isinf(ts).any():
        return False, ts
    return True, ts


def make_mask(n, indices):
    """
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
    """
    # check validity of indices
    if not isinstance(indices, (list, tuple, np.ndarray)):
        raise ValueError("indices must be a list, tuple, or numpy array.")
    if not all(isinstance(i, (int, np.int64, np.int32, np.int16)) for i in indices):
        raise ValueError("indices must be a list of integers.")
    if not all(i < n for i in indices):
        raise ValueError("indices must be smaller than n.")

    mask = np.zeros((n, n), dtype=np.int64)
    mask[np.ix_(indices, indices)] = 1
    mask = mask - np.diag(np.diag(mask))

    return mask


def get_intrah_mask(n_nodes):
    """
    Get a mask for intrahemispheric connections.

    Parameters
    ----------
    n_nodes: int
        number of total nodes that constitute the data.

    Returns
    -------
    mask_intrah: 2d array
        mask for intrahemispheric connections.
    """
    row_idx = np.arange(n_nodes)
    idx1 = np.ix_(row_idx[: n_nodes // 2], row_idx[: n_nodes // 2])
    idx2 = np.ix_(row_idx[n_nodes // 2 :], row_idx[n_nodes // 2 :])
    # build on a zeros mask
    mask_intrah = np.zeros((n_nodes, n_nodes))
    mask_intrah[idx1] = 1
    mask_intrah[idx2] = 1
    return mask_intrah


def get_interh_mask(n_nodes):
    """
    Get a mask for interhemispheric connections.

    Parameters
    ----------
    n_nodes: int
        number of total nodes that constitute the data.

    Returns
    -------
    mask_interh: 2d array
        mask for interhemispheric connections.
    """
    row_idx = np.arange(n_nodes // 2)
    col_idx1 = np.where(np.eye(n_nodes, k=-n_nodes // 2))[0]
    col_idx2 = np.where(np.eye(n_nodes, k=n_nodes // 2))[0]
    idx1 = np.ix_(row_idx, col_idx1)
    idx2 = np.ix_(row_idx + n_nodes // 2, col_idx2)
    # build on a zeros mask
    mask_interh = np.zeros((n_nodes, n_nodes))
    mask_interh[idx1] = 1
    mask_interh[idx2] = 1
    return mask_interh


def get_masks(n_nodes, networks):
    """
    Get a dictionary of masks based on the requested networks.

    Parameters
    ----------
    n_nodes: int
        number of total nodes that constitute the data.
    networks: list of str
        list of networks to be included in the dictionary.
        'full': full-network connections
        'intrah': intrahemispheric connections
        'interh': interhemispheric connections
        to get a custom mask with specific indices
        refere to `hbt.utility.make_mask(n, indices)`.

    Returns
    -------
    masks: dict
        dictionary of masks based on the requested networks.
    """
    masks = {}
    valid_networks = ["full", "intrah", "interh"]
    # check if networks are valid
    if not is_sequence(networks):
        networks = [networks]

    for i, ntw in enumerate(networks):
        if ntw not in valid_networks:
            raise ValueError(
                f"Invalid network: {ntw}. Please choose from {valid_networks}."
            )
        if ntw == "full":
            masks[ntw] = np.ones((n_nodes, n_nodes))
        elif ntw == "intrah":
            masks[ntw] = get_intrah_mask(n_nodes)
        elif ntw == "interh":
            masks[ntw] = get_interh_mask(n_nodes)

    return masks


def is_sequence(arg):
    """
    Check if the input is a sequence (list, tuple, np.ndarray, etc.)

    Parameters
    ----------
    arg : any
        input to be checked.

    Returns
    -------
    bool
        True if the input is a sequence, False otherwise.

    """
    return isinstance(arg, (list, tuple, np.ndarray))


def set_k_diagonals(A, k=0, value=0):
    """
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
    """

    if not isinstance(A, np.ndarray):
        A = np.array(A)
    if A.ndim != 2:
        raise ValueError("A must be a 2d array.")
    if not isinstance(k, int):
        raise ValueError("k must be an integer.")
    if not isinstance(value, (int, float)):
        raise ValueError("value must be a number.")
    if k >= A.shape[0]:
        raise ValueError("k must be smaller than the size of A.")

    n = A.shape[0]

    for i in range(-k, k + 1):
        a1 = np.diag(np.random.randint(1, 2, n - abs(i)), i)
        idx = np.where(a1)
        A[idx] = value
    return A


def if_symmetric(A, tol=1e-8):
    """
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

    """
    if not isinstance(A, np.ndarray):
        A = np.array(A)
    if A.ndim != 2:
        raise ValueError("A must be a 2d array.")

    return np.allclose(A, A.T, atol=tol)


def scipy_iir_filter_data(
    x, sfreq, l_freq, h_freq, l_trans_bandwidth=None, h_trans_bandwidth=None, **kwargs
):
    """
    Custom, scipy based filtering function with basic butterworth filter.
    #comes from neurolib

    Parameters
    ----------
    x : np.ndarray
        data to be filtered, time is the last axis
    sfreq : float
        sampling frequency of the data in Hz
    l_freq : float|None
        frequency below which to filter the data in Hz
    h_freq : float|None
        frequency above which to filter the data in Hz
    l_trans_bandwidth : keeping for compatibility with mne
    h_trans_bandwidth : keeping for compatibility with mne
    **kwargs : possible keywords to `scipy.signal.butter`:

    Returns
    -------
    np.ndarray
        filtered data

    """

    from scipy.signal import butter, sosfiltfilt

    nyq = 0.5 * sfreq
    if l_freq is not None:
        low = l_freq / nyq
        if h_freq is not None:
            # so we have band filter
            high = h_freq / nyq
            if l_freq < h_freq:
                btype = "bandpass"
            elif l_freq > h_freq:
                btype = "bandstop"
            Wn = [low, high]
        elif h_freq is None:
            # so we have a high-pass filter
            Wn = low
            btype = "highpass"
    elif l_freq is None:
        # we have a low-pass
        high = h_freq / nyq
        Wn = high
        btype = "lowpass"
    # get butter coeffs
    sos = butter(N=kwargs.pop("order", 8), Wn=Wn, btype=btype, output="sos")
    return sosfiltfilt(sos, x, axis=-1)


def filter(
    ts: np.ndarray,
    fs: float,
    low_freq: float,
    high_freq: float,
    l_trans_bandwidth: str = "auto",
    h_trans_bandwidth: str = "auto",
    **kwargs,
):
    """
    Filter data. Can be:
        - low-pass (low_freq is None, high_freq is not None),
        - high-pass (high_freq is None, low_freq is not None),
        - band-pass (l_freq < h_freq),
        - band-stop (l_freq > h_freq) filter type

    Parameters
    ----------
    ts: np.ndarray
        Time series data
    low_freq : float|None
        frequency below which to filter the data.
    high_freq : float|None
        frequency above which to filter the data.
    l_trans_bandwidth : float|str
        transition band width for low frequency
    h_trans_bandwidth : float|str
        transition band width for high frequency
    inplace : bool
        whether to do the operation in place or return
    kwargs : possible keywords to mne.filter.create_filter:
        filter_length="auto",
        method="fir",
        iir_params=None
        phase="zero",
        fir_window="hamming",
        fir_design="firwin"

    Returns
    -------
    np.ndarray
        filtered data
    """

    try:
        from mne.filter import filter_data

    except ImportError:
        logging.warning(
            "`mne` module not found, falling back to basic scipy's function"
        )
        filter_data = scipy_iir_filter_data

    filtered = filter_data(
        ts,  # times has to be the last axis
        sfreq=fs,
        l_freq=low_freq,
        h_freq=high_freq,
        l_trans_bandwidth=l_trans_bandwidth,
        h_trans_bandwidth=h_trans_bandwidth,
        **kwargs,
    )
    return filtered




def posterior_shrinkage(
    prior_samples: Union[Tensor, np.ndarray], post_samples: Union[Tensor, np.ndarray]
) -> Tensor:
    """
    Calculate the posterior shrinkage, quantifying how much
    the posterior distribution contracts from the initial
    prior distribution.
    References:
    https://arxiv.org/abs/1803.08393

    Parameters
    ----------
    prior_samples : array_like or torch.Tensor [n_samples, n_params]
        Samples from the prior distribution.
    post_samples : array-like or torch.Tensor [n_samples, n_params]
        Samples from the posterior distribution.

    Returns
    -------
    shrinkage : torch.Tensor [n_params]
        The posterior shrinkage.
    """

    if len(prior_samples) == 0 or len(post_samples) == 0:
        raise ValueError("Input samples are empty")

    if not isinstance(prior_samples, torch.Tensor):
        prior_samples = torch.tensor(prior_samples, dtype=torch.float32)
    if not isinstance(post_samples, torch.Tensor):
        post_samples = torch.tensor(post_samples, dtype=torch.float32)

    if prior_samples.ndim == 1:
        prior_samples = prior_samples[:, None]
    if post_samples.ndim == 1:
        post_samples = post_samples[:, None]

    prior_std = torch.std(prior_samples, dim=0)
    post_std = torch.std(post_samples, dim=0)

    return 1 - (post_std / prior_std) ** 2


def posterior_zscore(
    true_theta: Union[Tensor, np.array, float], post_samples: Union[Tensor, np.array]
):
    """
    Calculate the posterior z-score, quantifying how much the posterior
    distribution of a parameter encompasses its true value.
    References:
    https://arxiv.org/abs/1803.08393

    Parameters
    ----------
    true_theta : float, array-like or torch.Tensor [n_params]
        The true value of the parameters.
    post_samples : array-like or torch.Tensor [n_samples, n_params]
        Samples from the posterior distributions.

    Returns
    -------
    z : Tensor [n_params]
        The z-score of the posterior distributions.
    """

    if len(post_samples) == 0:
        raise ValueError("Input samples are empty")

    if not isinstance(true_theta, torch.Tensor):
        true_theta = torch.tensor(true_theta, dtype=torch.float32)
    if not isinstance(post_samples, torch.Tensor):
        post_samples = torch.tensor(post_samples, dtype=torch.float32)

    true_theta = np.atleast_1d(true_theta)
    if post_samples.ndim == 1:
        post_samples = post_samples[:, None]

    post_mean = torch.mean(post_samples, dim=0)
    post_std = torch.std(post_samples, dim=0)

    return torch.abs((post_mean - true_theta) / post_std)