import numpy as np
import scipy.signal
from numpy import linalg as LA
from scipy.signal import hilbert
from sklearn.decomposition import PCA
from scipy.stats import moment, skew, kurtosis
from vbi.feature_extraction.features_utils import *


@set_domain("domain", "temporal")
@set_domain("tag", "audio")
def abs_energy(ts):
    """Computes the absolute energy of the time serie.

    Feature computational cost: 1

    Parameters
    ----------
    ts : nd-arrays [n_regions x n_samples]
        Input from which the area under the curve is computed

    Returns
    -------
    list of float
        Absolute energy

    """

    if isinstance(ts, (list, tuple)):
        ts = np.array(ts)

    if ts.ndim == 1:
        ts = ts.reshape(-1, 1)
    return np.sum(np.abs(ts) ** 2, axis=1).tolist()


@set_domain("domain", "statistical")
def calc_var(ts):
    """Computes variance of the time serie.

    Feature computational cost: 1

    Parameters
    ----------
    ts : nd-array [n_regions x n_samples]
       Input from which var is computed

    Returns
    -------
    list
        Variance result

    """

    if isinstance(ts, (list, tuple)):
        ts = np.array(ts)
    if ts.ndim == 1:
        ts = ts.reshape(-1, 1)

    return np.var(ts, axis=1).tolist()


@set_domain("domain", "statistical")
def calc_std(ts):
    """Computes standard deviation of the time serie.

    Feature computational cost: 1

    Parameters
    ----------
    ts : nd-array [n_regions x n_samples]
       Input from which std is computed

    Returns
    -------
    list
        Standard deviation result

    """

    if isinstance(ts, (list, tuple)):
        ts = np.array(ts)
    if ts.ndim == 1:
        ts = ts.reshape(-1, 1)

    return np.std(ts, axis=1).tolist()


def auc(ts, fs):
    """Computes the area under the curve of the signal computed with trapezoid rule.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-arrays [n_regions x n_samples]
        Input from which the area under the curve is computed
    fs : int
        Sampling Frequency
    Returns
    -------
    list of float
        The area under the curve value

    """
    if isinstance(ts, (list, tuple)):
        ts = np.array(ts)

    t = np.arange(ts.shape[1]) / fs
    return np.sum(0.5 * np.diff(t) * np.abs(np.array(ts[:, :-1]) + np.array(ts[:, 1:])), axis=1)


def fc_sum(x):
    '''
    Calculate the sum of functional connectivity (FC)

    Parameters
    ----------

    x: np.ndarray [n_regions, n_samples]
       input signal 

    Returns
    -------
    result: float
        sum of functional connectivity
    '''
    fc = np.corrcoef(x)

    return np.sum(np.abs(fc)) - np.trace(np.abs(fc))


def fc_stat(x, 
            k=0, 
            PCA_n_components=3, 
            positive=False, 
            demean=False,  
            method="corr", **kwargs):
    '''
    extract features from functional connectivity (FC)

    Parameters
    ----------

    x: np.ndarray [n_regions, n_samples]
        input array
    k: int
        kth diagonal of FC matrix
    PCA_n_components: int
        number of components for PCA
    demean: bool
        if True, ignore mean value of fc elements distribution
    positive: bool
        if True, ignore negative values of fc elements
    
    Returns
    -------
    stats: np.ndarray (1d)
        feature values
    '''
    indices = kwargs.get('indices', None)
    if indices is not None:
        mask = make_mask(x.shape[0], indices)

    def funcs(x, demean=False):
        if demean:
            vec = np.zeros(3)
            vec[0] = np.std(x)
            vec[1] = skew(x)
            vec[2] = kurtosis(x)

        else:
            vec = np.zeros(7)
            vec[0] = np.sum(x)
            vec[1] = np.max(x)
            vec[2] = np.min(x)
            vec[3] = np.mean(x)
            vec[4] = np.std(x)
            vec[5] = skew(x)
            vec[6] = kurtosis(x)
        return vec

    if method == "corr":
        FC = np.corrcoef(x)
    elif method == "plv":
        FC = ... # TODO use mne_connectivity.spectral_connectivity
    elif method == "pli":
        FC = ... # TODO use mne_connectivity.spectral_connectivity
    else:
        raise ValueError("method must be one of 'corr', 'plv', 'pli'")

    if indices is not None:
        FC = FC * mask 

    if positive:
        FC = FC * (FC > 0)

    off_diag_sum_FC = np.sum(np.abs(FC)) - np.trace(np.abs(FC))

    FC_TRIU = np.triu(FC, k=k)
    eigen_vals_FC, _ = LA.eig(FC)
    pca = PCA(n_components=PCA_n_components)
    PCA_FC = pca.fit_transform(FC)

    Upper_FC = []
    Lower_FC = []
    for i in range(0, len(FC)):
        Upper_FC.extend(FC[i][i+1:])
        Lower_FC.extend(FC[i][0:i])

    q = np.quantile(FC, [0.05, 0.25, 0.5, 0.75, 0.95])
    stat_vec = np.array([])
    stat_vec = np.append(stat_vec, q)
    stat_vec = np.append(stat_vec, funcs(Upper_FC, demean))
    stat_vec = np.append(stat_vec, funcs(Lower_FC, demean))
    stat_vec = np.append(stat_vec, funcs(PCA_FC.reshape(-1), demean))
    stat_vec = np.append(stat_vec, funcs(FC_TRIU.reshape(-1), demean))
    stat_vec = np.append(stat_vec, funcs(np.real(eigen_vals_FC[:-1]), demean))

    # keep this the last element
    stat_vec = np.append(stat_vec, [off_diag_sum_FC])

    return stat_vec.tolist()

def coactivation_degree(ts, modality='noncor'):
    '''
    calculate coactivation degree (CAD)

    Parameters
    ----------
    ts: np.ndarray [n_regions, n_samples]
        input array
    modality: str

    
    '''
    nn, nt = ts.shape
    ts = stats.zscore(ts, axis=1)
    if modality == 'cor':
        global_signal = stats.zscore(np.mean(ts, axis=1))

    M = np.zeros((nn, nt))
    for i in range(nn):
        if modality != 'cor':
            global_signal = np.mean(np.delete(ts, i, axis=0), axis=0)
        M[i] = ts[i, :]*global_signal
    return M.tolist()

def coactivation_phase(ts):
    '''
    calculate the coactivation phase (CAP)

    Parameters
    ----------
    ts: np.ndarray [n_regions, n_samples]
        input array

    Returns
    -------
    CAP: list
    '''

    if isinstance(ts, (list, tuple)):
        ts = np.array(ts)
    if ts.ndim == 1:
        ts = ts.reshape(-1, 1)

    ts = stats.zscore(ts, axis=1)

    # phase global
    GS = np.mean(ts, axis=0)
    Phase = np.unwrap(np.angle(hilbert(GS)))
    Phase = (Phase + np.pi) % (2 * np.pi) - np.pi

    # phase regional
    phase_i = np.unwrap(np.angle(hilbert(ts, axis=1)), axis=1)
    phase_i = (phase_i + np.pi) % (2 * np.pi) - np.pi
    MSphase = np.mean(Phase - phase_i, axis=1)

    return MSphase.tolist()


def burstiness(x):
        '''
        calculate the burstiness statistic
        [from hctsa-py]

        Parameters
        ----------
        x: np.ndarray [n_regions, n_samples]
            input array

        Returns
        -------
        B: list of floats
            burstiness statistic

        References
        ----------

        - Goh and Barabasi, 'Burstiness and memory in complex systems' Europhys. Lett.
        81, 48002 (2008).
        '''

        
        if x.mean() == 0:
            return np.nan
        
        r = np.std(x, axis=1) / np.mean(x, axis=1)
        B = (r - 1) / (r + 1)

        return B