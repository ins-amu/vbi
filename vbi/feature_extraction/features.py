import vbi
import numpy as np
import scipy.signal
from scipy.signal import hilbert
from scipy.stats import moment, skew, kurtosis
import scipy.stats as stats
from vbi.feature_extraction.utility import prepare_input_ts
from vbi.feature_extraction.features_utils import (
    km_order,
    get_fc,
    get_fcd,
    matrix_stat,
    compute_time,
    init_jvm,
    nat2bit,
    kde,
    gaussian,
    calc_fft,
    wavelet,
    state_duration,
    seizure_onset_indicator,
    max_frequency,
    max_psd,
    spectral_distance,
    fundamental_frequency,
    spectral_centroid,
    spectral_variation,
    spectral_kurtosis,
    median_frequency
)

from typing import List, Tuple, Dict

try:
    import ssm
except:
    pass

try:
    import jpype as jp
except:
    pass


def abs_energy(ts: np.ndarray, indices: List[int] = None, verbose=False):
    """Computes the absolute energy of the time serie.

    >>> abs_energy([1, 2, 3, 4, 5])
    (array([55]), ['abs_energy_0'])

    Parameters
    ----------
    ts : nd-arrays [n_regions x n_samples]
        Input from which the area under the curve is computed
    indices: list of int
        Indices of the time series to compute the feature

    Returns
    -------
    values: list of float
        Absolute energy
    labels: list of str
        Labels of the features

    """

    info, ts = prepare_input_ts(ts, indices)
    if not info:
        if verbose:
            print("Error in abs_energy")
        return [np.nan], [f"abs_energy_{0}"]
    else:
        values = np.sum(np.abs(ts) ** 2, axis=1)
        labels = [f"abs_energy_{i}" for i in range(len(values))]

    return values, labels


def average_power(ts: np.ndarray, fs: float = 1.0, indices: List[int] = None, verbose=False):
    """Computes the average power of the time serie.

    >>> average_power([1, 2, 3, 4, 5], 1)
    (array([13.75]), ['average_power_0'])

    Parameters
    ----------
    ts : nd-arrays [n_regions x n_samples]
        Input from which the area under the curve is computed
    fs : float
        Sampling frequency
    indices: list of int
        Indices of the time series to compute the feature

    Returns
    -------
    values: list of float
        Average power
    labels: list of str
        Labels of the features

    """

    info, ts = prepare_input_ts(ts, indices)
    if not info:
        if verbose:
            print("Error in average_power")
        return [np.nan], [f"average_power_{0}"]
    else:

        times = compute_time(ts[0], fs)
        values = np.sum(ts**2, axis=1) / (times[-1] - times[0])
        labels = [f"average_power_{i}" for i in range(len(values))]
        return values, labels


def auc(
    ts: np.ndarray, dx: float = None, x: np.ndarray = None, indices: List[int] = None, verbose=False
):
    """Computes the area under the curve of the signal computed with trapezoid rule.

    >>> auc(np.array([[1, 2, 3], [4, 5, 6]]), None, np.array([0, 1, 2]))
    (array([ 4., 10.]), ['auc_0', 'auc_1'])

    Parameters
    ----------
    ts : nd-arrays [n_regions x n_samples]
        Input from which the area under the curve is computed
    dx: float
        Spacing between values
    x: array_like, optional
        x values of the time series
    indices: list of int
        Indices of the time series to compute the feature

    Returns
    -------
    list of float
        The area under the curve value
    labels: list of str
        Labels of the features

    """

    info, ts = prepare_input_ts(ts, indices)
    if not info:
        if verbose:
            print("Error in auc")
        return [np.nan], ["auc_0"]

    if dx is None:
        dx = 1
    values = np.trapz(ts, x=x, dx=dx, axis=1)
    labels = [f"auc_{i}" for i in range(len(values))]

    return values, labels


def auc_lim(
    ts: np.ndarray,
    dx: float = None,
    x: np.ndarray = None,
    xlim: List[Tuple[float, float]] = None,
    indices: List[int] = None,
    verbose=False
):
    """
    Compute the area under the curve for a given time series within a given limit

    >>> auc_lim(np.array([[1, 2, 3], [4, 5, 6]]), None, None, [(0, 1), (1, 2)])
    ([1.5, 4.5, 2.5, 5.5], ['auc_lim_0', 'auc_lim_1', 'auc_lim_2', 'auc_lim_3'])

    Parameters
    ----------
    ts : nd-arrays [n_regions x n_samples]
        Input from which the area under the curve is computed
    dx: float
        Spacing between values
    x: array_like
        x values of the time series
    xlim: list of tuples
        The limits of the time series
    indices: list of int
        Indices of the time series to compute the feature

    Returns
    -------
    list of float
        The area under the curve value
    labels: list of str
        Labels of the features

    """

    info, ts = prepare_input_ts(ts, indices)
    if not info:
        if verbose:
            print("Error in auc_lim")
        return [np.nan], ["auc_lim_0"]

    if x is None:
        x = np.arange(0, ts.shape[1])
    else:
        x = np.array(x)
    assert x.shape[0] == ts.shape[1], "x and ts must have the same length"

    if xlim is None:
        xlim = [(x[0], x[-1])]

    if not isinstance(xlim[0], (list, tuple)):
        xlim = [xlim]

    values = []
    for i, (xmin, xmax) in enumerate(xlim):
        idx = np.where((x >= xmin) & (x <= xmax))[0]
        if len(idx) == 0:
            continue
        values.extend(np.trapz(ts[:, idx], x=x[idx], dx=dx, axis=1))
    labels = [f"auc_lim_{i}" for i in range(len(values))]

    return values, labels


def calc_var(ts: np.ndarray, indices: List[int] = None, verbose=False):
    """Computes variance of the time series.

    >>> calc_var(np.array([[1, 2, 3], [4, 5, 6]]))
    (array([0.66666667, 0.66666667]), ['var_0', 'var_1'])

    Parameters
    ----------
    ts : nd-array [n_regions x n_samples]
       Input from which var is computed
    indices: list of int
        Indices of the time series to compute the feature

    Returns
    -------
    values: array-like
        variance of the time series
    labels: array-like
        labels of the features

    """
    info, ts = prepare_input_ts(ts, indices)
    if not info:
        if verbose:
            print("Error in calc_var")
        return [np.nan], ["var_0"]

    values = np.var(ts, axis=1)
    labels = [f"var_{i}" for i in range(len(values))]

    return values, labels


def calc_std(ts: np.ndarray, indices: List[int] = None, verbose=False):
    """Computes standard deviation of the time serie.

    >>> calc_std(np.array([[1, 2, 3], [4, 5, 6]]))
    (array([0.81649658, 0.81649658]), ['std_0', 'std_1'])

    Parameters
    ----------
    ts : nd-array [n_regions x n_samples]
       Input from which std is computed
    indices: list of int
        Indices of the time series to compute the feature

    Returns
    -------
    values: array-like
        std of the time series
    labels: array-like
        labels of the features
    """

    info, ts = prepare_input_ts(ts, indices)
    if not info:
        if verbose:
            print("Error in calc_std")
        return [np.nan], [f"std_{0}"]
    else:
        values = np.std(ts, axis=1)
        labels = [f"std_{i}" for i in range(len(values))]
        return values, labels


def calc_mean(ts: np.ndarray, indices: List[int] = None, verbose=False):
    """Computes median of the time serie.

    >>> calc_mean(np.array([[1, 2, 3], [4, 5, 6]]))
    (array([2., 5.]), ['mean_0', 'mean_1'])

    Parameters
    ----------
    ts : nd-array [n_regions x n_samples]
       Input from which median is computed
    indices: list of int
        Indices of the time series to compute the feature

    Returns
    -------
    values: array-like
        mean of the time series
    labels: array-like
        labels of the features

    """

    info, ts = prepare_input_ts(ts, indices)
    if not info:
        if verbose:
            print("Error in calc_mean")
        return [np.nan], [f"mean_{0}"]
    else:
        values = np.mean(ts, axis=1)
        labels = [f"mean_{i}" for i in range(len(values))]
        return values, labels


def calc_centroid(ts: np.ndarray, fs: float, indices: List[int] = None, verbose=False):
    """Computes the centroid along the time axis.

    Parameters
    ----------
    signal : nd-array
        Input from which centroid is computed
    fs: int
        Signal sampling frequency
    indices: list of int
        Indices of the time series to compute the feature

    Returns
    -------
    float
        Temporal centroid

    """
    info, ts = prepare_input_ts(ts, indices)
    if not info:
        if verbose:
            print("Error in calc_centroid")
        return [np.nan], [f"centroid_{0}"]
    else:
        tol = 1e-10
        r, c = ts.shape
        centroid = np.zeros(r)
        time = compute_time(ts[0], fs)
        energy = ts**2
        t_energy = np.dot(time, energy.T)
        energy_sum = np.sum(energy, axis=1)
        ind_nonzero = (np.abs(energy_sum) > tol) | (np.abs(t_energy) > tol)
        centroid[ind_nonzero] = t_energy[ind_nonzero] / energy_sum[ind_nonzero]
        labels = [f"centroid_{i}" for i in range(len(centroid))]

        return centroid, labels


def calc_kurtosis(ts: np.ndarray, indices: List[int] = None, verbose=False):
    """
    Computes the kurtosis of the time series.

    Parameters
    ----------
    ts : nd-array [n_regions x n_samples]
       Input from which kurtosis is computed
    indices: list of int
        Indices of the time series to compute the feature

    Returns
    -------
    values: array-like
        kurtosis of the time series
    labels: array-like
        labels of the features

    """

    info, ts = prepare_input_ts(ts, indices)
    if not info:
        if verbose:
            print("Error in calc_kurtosis")
        return [np.nan], [f"kurtosis_{0}"]
    else:
        values = kurtosis(ts, axis=1)
        labels = [f"kurtosis_{i}" for i in range(len(values))]
        return values, labels


def calc_skewness(ts: np.ndarray, indices: List[int] = None, verbose=False):
    """
    Computes the skewness of the time series.

    Parameters
    ----------
    ts : nd-array [n_regions x n_samples]
       Input from which skewness is computed

    Returns
    -------
    values: array-like
        skewness of the time series
    labels: array-like
        labels of the features

    """

    info, n = prepare_input_ts(ts, indices)
    if not info:
        if verbose:
            print("Error in calc_skewness")
        return [np.nan], [f"skewness_{0}"]
    else:
        values = skew(ts, axis=1)
        labels = [f"skewness_{i}" for i in range(len(values))]
        return values, labels


def calc_max(ts: np.ndarray, indices: List[int] = None, verbose=False):
    """
    Computes the maximum of the time series.

    Parameters
    ----------
    ts : nd-array [n_regions x n_samples]
       Input from which maximum is computed

    Returns
    -------
    values: array-like
        maximum of the time series
    labels: array-like
        labels of the features

    """

    info, ts = prepare_input_ts(ts, indices)
    if not info:
        if verbose:
            print("Error in calc_max")
        return [np.nan], [f"max_{0}"]
    else:
        values = np.max(ts, axis=1)
        labels = [f"max_{i}" for i in range(len(values))]
        return values, labels


def calc_min(ts: np.ndarray, indices: List[int] = None, verbose=False):
    """
    Computes the minimum of the time series.

    Parameters
    ----------
    ts : nd-array [n_regions x n_samples]
       Input from which minimum is computed
    indices: list of int
        Indices of the time series to compute the feature

    Returns
    -------
    values: array-like
        minimum of the time series
    labels: array-like
        labels of the features

    """

    info, ts = prepare_input_ts(ts, indices)
    if not info:
        if verbose:
            print("Error in calc_min")
        return [np.nan], [f"min_{0}"]
    else:
        values = np.min(ts, axis=1)
        labels = [f"min_{i}" for i in range(len(values))]
        return values, labels


def calc_median(ts: np.ndarray, indices: List[int] = None, verbose=False):
    """
    Computes the median of the time series.

    Parameters
    ----------
    ts : nd-array [n_regions x n_samples]
       Input from which median is computed
    indices: list of int
        Indices of the time series to compute the feature

    Returns
    -------
    values: array-like
        median of the time series
    labels: array-like
        labels of the features
    """

    info, ts = prepare_input_ts(ts, indices)
    if not info:
        if verbose:
            print("Error in calc_median")
        return [np.nan], [f"median_{0}"]
    else:
        values = np.median(ts, axis=1)
        labels = [f"median_{i}" for i in range(len(values))]
        return values, labels


def mean_abs_dev(ts: np.ndarray, indices: List[int] = None, verbose=False):
    """
    Computes the mean absolute deviation of the time series.

    Parameters
    ----------
    ts : nd-array [n_regions x n_samples]
       Input from which mean absolute deviation is computed

    Returns
    -------
    values: array-like
        mean absolute deviation of the time series
    labels: array-like
        labels of the features

    """

    info, ts = prepare_input_ts(ts, indices)
    if not info:
        if verbose:
            print("Error in mean_abs_dev")
        return [np.nan], [f"mean_abs_dev_{0}"]
    else:
        values = np.mean(np.abs(ts - np.mean(ts, axis=1, keepdims=True)), axis=1)
        labels = [f"mean_abs_dev_{i}" for i in range(len(values))]
        return values, labels


def median_abs_dev(ts: np.ndarray, indices: List[int] = None, verbose=False):
    """
    Computes the median absolute deviation of the time series.

    Parameters
    ----------
    ts : nd-array [n_regions x n_samples]
       Input from which median absolute deviation is computed
    indices: list of int
        Indices of the time series to compute the feature

    Returns
    -------
    values: array-like
        median absolute deviation of the time series
    labels: array-like
        labels of the features

    """

    info, ts = prepare_input_ts(ts, indices)
    if not info:
        if verbose:
            print("Error in median_abs_dev")
        return [np.nan], [f"median_abs_dev_{0}"]
    else:
        values = np.median(np.abs(ts - np.median(ts, axis=1, keepdims=True)), axis=1)
        labels = [f"median_abs_dev_{i}" for i in range(len(values))]
        return values, labels


def rms(ts: np.ndarray, indices: List[int] = None, verbose=False):
    """
    Computes the root mean square of the time series.

    Parameters
    ----------
    ts : nd-array [n_regions x n_samples]
       Input from which root mean square is computed
    indices: list of int
        Indices of the time series to compute the feature

    Returns
    -------
    values: array-like
        root mean square of the time series
    labels: array-like
        labels of the features

    """

    info, ts = prepare_input_ts(ts, indices)
    if not info:
        if verbose:
            print("Error in rms")
        return [np.nan], [f"rms_{0}"]
    else:
        values = np.sqrt(np.mean(ts**2, axis=1))
        labels = [f"rms_{i}" for i in range(len(values))]
        return values, labels


def interq_range(ts: np.ndarray, indices: List[int] = None, verbose=False):
    """
    Computes the interquartile range of the time series.

    Parameters
    ----------
    ts : nd-array [n_regions x n_samples]
       Input from which interquartile range is computed
    indices: list of int
        Indices of the time series to compute the feature

    Returns
    -------
    values: array-like
        interquartile range of the time series
    labels: array-like
        labels of the features

    """

    info, ts = prepare_input_ts(ts, indices)
    if not info:
        if verbose:
            print("Error in interq_range")
        return [np.nan], [f"interq_range_{0}"]
    else:
        values = np.subtract(*np.percentile(ts, [75, 25], axis=1))
        labels = [f"interq_range_{i}" for i in range(len(values))]
        return values, labels


def zero_crossing(ts: np.ndarray, indices: List[int] = None, verbose=False):
    """
    Computes the number of zero crossings of the time series.

    Parameters
    ----------
    ts : nd-array [n_regions x n_samples]
       Input from which number of zero crossings is computed
    indices: list of int
        Indices of the time series to compute the feature

    Returns
    -------
    values: array-like
        number of zero crossings of the time series
    labels: array-like
        labels of the features

    """
    info, ts = prepare_input_ts(ts, indices)
    if not info:
        if verbose:
            print("Error in zero_crossing")
        return [np.nan], [f"zero_crossing_{0}"]
    else:
        values = np.array([np.sum(np.diff(np.sign(y_i)) != 0) for y_i in ts], dtype=int)
        labels = [f"zero_crossing_{i}" for i in range(len(values))]
        return values, labels


def seizure_onset(ts: np.ndarray, 
                  threshold: float = 0.02,
                  indices: List[int] = None, verbose=False):
    '''
    Computes the seizure onset of the time series.
    
    Parameters
    ----------
    ts : nd-array [n_regions x n_samples]
       Input from which number of zero crossings is computed
    indices: list of int
        Indices of the time series to compute the feature
    
    Returns
    -------
    values: array-like
        index of the onset of the seizures in the time series, zero if no onset in each region
    labels: array-like
        labels of the features
    '''
    
    info, ts = prepare_input_ts(ts, indices)
    if not info:
        if verbose:
            print("Error in zero_crossing")
        return [np.nan], [f"seizure_onset_{0}"]
    else:
        values = seizure_onset_indicator(ts, threshold)
        labels = [f"seizure_onset_{i}" for i in range(len(values))]
        return values, labels
    
    
    

# def calc_ress(
#     ts: np.ndarray, percentile: Union[int, float] = 95, indices: List[int] = None
# ):
#     """
#     Calculates Residual Sum of Squares (RSS) with given percentile

#     Parameters
#     ----------
#     ts : nd-array [n_regions x n_samples]
#          Input time seris
#     percentile : float
#             Percentile of RSS
#     indices: list of int
#         Indices of the time series to compute the feature

#     Returns
#     -------
#     values: array-like
#         RSS of the time series
#     labels: array-like
#         labels of the features
#     """

#     info, ts = prepare_input_ts(ts, indices)
#     if not info:
#         return [np.nan], [f"ress_{0}"]
#     else:
#         nn, nt = ts.shape
#         rss = np.zeros(nt)
#         for t in range(nt):
#             z = np.power(np.outer(ts[:, t], ts[:, t]), 2)
#             rss[t] = np.sqrt(np.einsum("ij->", z))
#         return np.percentile(rss, percentile), ["ress"]


def kop(ts: np.ndarray, indices: List[int] = None, verbose=False, extract_phase=False):
    """
    Calculate the Kuramoto order parameter (KOP)

    """

    info, ts = prepare_input_ts(ts, indices)
    if not info:
        if verbose:
            print("Error in kop")
        return [np.nan], ["kop"]
    else:
        if extract_phase:
            analytic_signal = hilbert(ts, axis=1)
            amplitude_envelope = np.abs(analytic_signal)
            instantaneous_phase = np.unwrap(np.angle(analytic_signal))
            R = km_order(instantaneous_phase, indices=indices, avg=True)
        else:
            R = km_order(ts, indices=indices, avg=True)
        return R, ["kop"]


def calc_moments(
    ts: np.ndarray, indices: List[int] = None, orders: List[int] = [2, 3, 4, 5, 6], verbose=False
):
    """
    Computes the moments of the time series.

    Parameters
    ----------
    ts : nd-array [n_regions x n_samples]
       Input from which moments are computed
    orders: list
        List of orders of the moments

    Returns
    -------
    values: array-like
        moments of the time series
    labels: array-like
        labels of the features

    """

    info, ts = prepare_input_ts(ts, indices)
    if not info:
        if verbose:
            print("Error in calc_moments")
        return [np.nan], ["moments"]
    else:
        labels = []
        values = np.array([])
        for i in orders:
            v = moment(ts, moment=i, axis=1)
            values = np.append(values, v)
            labels.extend([f"moments_{i}_{j}" for j in range(len(v))])

        return values, labels


def calc_envelope(
    ts: np.ndarray,
    indices: List[int] = None,
    features: List[str] = ["mean", "std", "median", "max", "min"],
    verbose=False,
):
    """
    calculate some statistics on envelope of the time series using hilbert transform
    """
    
    from numpy import mean, std, median, max, min

    info, ts = prepare_input_ts(ts, indices)
    if not info:
        if verbose:
            print("Error in calc_envelope")
        return [np.nan], ["envelope"]
    else:
        analytic_signal = hilbert(ts, axis=1)
        amplitude_envelope = np.abs(analytic_signal)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))

        labels = []
        values = np.array([])

        for f in features:
            v = np.append(values, eval(f"{f}(amplitude_envelope, axis=1)"))
            l = [f"env_amp_{f}_{j}" for j in range(len(v))]
            values = np.append(values, v)
            labels.extend(l)

        for f in features:
            v = eval(f"{f}(instantaneous_phase, axis=1)")
            l = [f"env_ph_{f}_{j}" for j in range(len(v))]
            values = np.append(values, v)
            labels.extend(l)

        return values, labels


def fc_sum(x: np.ndarray, positive=False, masks: Dict[str, np.ndarray] = None, verbose=False):
    """
    Calculate the sum of functional connectivity (FC)

    Parameters
    ----------
    ts : nd-array [n_regions x n_samples]
       Input from which var is computed

    Returns
    -------
    result: float
        sum of functional connectivity
    """

    label = "fc_sum"

    info, ts = prepare_input_ts(x)
    if not info:
        if verbose:
            print("Error in fc_sum")
        return [np.nan], [label]
    if ts.shape[0] < 2:
        return [np.nan], [label]
    nn = ts.shape[0]

    if masks is None:
        masks = {"full": np.ones((nn, nn))}

    for key in masks.keys():
        assert (
            masks[key].shape[0] == nn
        ), "mask size must be equal to the number of regions"

    fc = np.corrcoef(x)
    if positive:
        fc = fc * (fc > 0)

    values = np.array([])
    for key in masks.keys():
        mask = masks[key]
        fc = fc * mask
        v = np.sum(np.abs(fc)) - np.trace(np.abs(fc))
        values = np.append(values, v)
    labels = [f"{label}_{key}" for key in masks.keys()]

    return values, labels


def fc_stat(
    ts: np.ndarray,
    k: int = 0,
    positive: bool = False,
    eigenvalues: bool = True,
    pca_num_components: int = 3,
    fc_function: str = "corrcoef",
    masks: Dict[str, np.ndarray] = None,
    quantiles: List[float] = [0.05, 0.25, 0.5, 0.75, 0.95],
    features: List[str] = ["sum", "max", "min", "mean", "std", "skew", "kurtosis"],
    verbose=False,
):
    """
    extract features from functional connectivity (FC)

    Parameters
    ----------

    ts: np.ndarray [n_regions, n_samples]
        input array
    k: int
        to remove up to kth diagonal of FC matrix
    pca_num_components: int
        number of components for PCA
    positive: bool
        if True, ignore negative values of fc elements
    masks: dict
        dictionary of masks
    features: list of str
        list of features to be extracted
    quantiles: list of float
        list of quantiles, set 0 to ignore
    eigenvalues: bool
        if True, extract features from eigenvalues
    fc_function: str
        functional connectivity function: 'corrcoef' or 'cov'

    Returns
    -------
    stats: np.ndarray (1d)
        feature values
    labels: list of str
        feature labels
    """

    info, ts = prepare_input_ts(ts)
    if not info:
        if verbose:
            print("Error in fc_")
        return [np.nan], ["fc_0"]

    nn = ts.shape[0]
    if nn < 2:
        return [np.nan], ["fc_0"]

    if masks is None:
        masks = {"full": np.ones((nn, nn))}

    for key in masks.keys():
        assert (
            masks[key].shape[0] == nn
        ), "mask size must be equal to the number of regions"

    Values = []
    Labels = []

    fc = get_fc(ts, masks=masks, fc_fucntion=fc_function, positive=positive)

    for key in fc.keys():
        values, labels = matrix_stat(
            fc[key],
            k=k,
            features=features,
            quantiles=quantiles,
            eigenvalues=eigenvalues,
            pca_num_components=pca_num_components,
        )
        labels = [f"fc_{key}_{label}" for label in labels]
        Values.extend(values)
        Labels.extend(labels)

    return Values, Labels


def fc_homotopic(
    ts: np.ndarray, average: bool = False, positive: bool = True, fc_function="corrcoef", verbose=False
):
    """
    Calculate the homotopic connectivity vector of a given brain activity

    Parameters
    ----------
    bold: array_like [nn, nt]
        The brain activity to be analyzed.
    averag: bool
        If True, the average homotopic connectivity is returned.
        Otherwise, the homotopic connectivity vector is returned.
    positive: bool
        If True, only positive correlations are considered.

    Returns
    -------
    values : array_like [n_nodes]
        The homotopic correlation vector.
    labels : list of str
        The labels of the homotopic correlation vector.

    Negative correlations may be artificially induced when using global signal regression
    in functional imaging pre-processing (Fox et al., 2009; Murphy et al., 2009; Murphy and Fox, 2017).
    Therefore, results on negative weights should be interpreted with caution and should be understood
    as complementary information underpinning the findings based on positive connections
    """
    
    from numpy import corrcoef, cov

    info, ts = prepare_input_ts(ts)
    if not info:
        if verbose:
            print("Error in fc_homotopic")
        return [np.nan], ["fc_homotopic"]

    nn, nt = ts.shape
    if nn < 2:
        return [np.nan], ["fc_homotopic"]

    NHALF = int(nn // 2)
    fc = eval(fc_function)(ts)

    if positive:
        fc = fc * (fc > 0)
    fc = fc - np.diag(np.diag(fc))  # not necessary for hfc
    hfc = np.diag(fc, k=NHALF)
    if average:
        return [np.mean(hfc)], ["fc_homotopic_avg"]
    else:
        values = hfc.squeeze()
        labels = [f"fc_homotopic_{i}" for i in range(len(values))]
        return values, labels


def coactivation_degree(ts: np.ndarray, modality="noncor"):
    """
    calculate coactivation degree (CAD) #!TODO not tested

    Parameters
    ----------
    ts: np.ndarray [n_regions, n_samples]
        input array
    modality: str

    Returns
    -------
    values: array-like
        coactivation degree
    labels: array-like
        labels of the features


    """
    nn, nt = ts.shape
    ts = stats.zscore(ts, axis=1)
    if modality == "cor":
        global_signal = stats.zscore(np.mean(ts, axis=1))

    M = np.zeros((nn, nt))
    for i in range(nn):
        if modality != "cor":
            global_signal = np.mean(np.delete(ts, i, axis=0), axis=0)
        M[i] = ts[i, :] * global_signal
    return M.tolist()


def coactivation_phase(ts):
    """
    calculate the coactivation phase (CAP) #!TODO not tested

    Parameters
    ----------
    ts: np.ndarray [n_regions, n_samples]
        input array

    Returns
    -------
    CAP: list
    """

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


def burstiness(ts: np.ndarray, indices: List[int] = None, verbose=False):
    """
    calculate the burstiness statistic
    - Goh and Barabasi, 'Burstiness and memory in complex systems' Europhys. Lett.
    81, 48002 (2008).
    [from hctsa-py]

    Parameters
    ----------
    x: np.ndarray [n_regions, n_samples]
        input array
    indices: list of int
        Indices of the time series to compute the feature

    Returns
    -------
    B: list of floats
        burstiness statistic
    """

    info, ts = prepare_input_ts(ts, indices)
    if not info:
        if verbose:
            print("Error in burstiness")
        return [np.nan], ["burstiness"]

    if ts.mean() == 0:
        return [0], ["burstiness"]

    r = np.std(ts, axis=1) / np.mean(ts, axis=1)
    B = (r - 1) / (r + 1)
    labels = [f"burstiness_{i}" for i in range(len(B))]

    return B, labels


def fcd_stat(
    ts,
    TR=1,
    win_len=30,
    masks=None,
    positive=False,
    eigenvalues=True,
    pca_num_components=3,
    quantiles=[0.05, 0.25, 0.5, 0.75, 0.95],
    features=["sum", "max", "min", "mean", "std", "skew", "kurtosis"],
    k=None,
    verbose=False,
):
    
    from numpy import sum, max, min, mean, std
    from scipy.stats import skew, kurtosis
    
    info, ts = prepare_input_ts(ts)
    if not info:
        if verbose:
            print("Error in fcd_stat")
        return [np.nan], ["fcd_stat_0"]

    Values = []
    Labels = []
    k = k if k is not None else int(win_len / TR)
    fcd = get_fcd(ts=ts, TR=TR, win_len=win_len, positive=positive, masks=masks)
    for key in fcd.keys():
        values, labels = matrix_stat(
            fcd[key],
            k=k,
            features=features,
            quantiles=quantiles,
            eigenvalues=eigenvalues,
            pca_num_components=pca_num_components,
        )
        labels = [f"fcd_{key}_{label}" for label in labels]
        Values.extend(values)
        Labels.extend(labels)

    return Values, Labels


def calc_mi(
    ts: np.ndarray,
    k: int = 4,
    time_diff: int = 1,
    num_threads: int = 1,
    source_indices: List[int] = None,
    target_indices: List[int] = None,
    mode: str = "pairwise",
    verbose=False,
    **kwargs,
):
    """
    calculate the mutual information between time series
    based on the Kraskov method #!TODO bug in multiprocessing

    Parameters
    ----------
    ts: np.ndarray [n_regions, n_samples]
        input array
    k: int
        kth nearest neighbor
    time_diff: int
        time difference between time series
    num_threads: int
        number of threads
    source_indices: list or np.ndarray
        indices of source time series, if None, all time series are used
    target_indices: list or np.ndarray
        indices of target time series, if None, all time series are used
    mode: str
        "pairwise" or "all", if "pairwise", source_indices and target_indices must have the same length

    Returns
    -------
    MI: list of floats
        mutual information
    labels: list of str
        labels of the features
    """

    num_surrogates = kwargs.get("num_surrogates", 0)

    if not isinstance(ts, np.ndarray):
        ts = np.array(ts)
    if ts.ndim == 1:
        assert False, "ts must be a 2d array"

    init_jvm()
    calcClass = jp.JPackage(
        "infodynamics.measures.continuous.kraskov"
    ).MutualInfoCalculatorMultiVariateKraskov2
    calc = calcClass()
    calc.setProperty("k", str(int(k)))
    calc.setProperty("NUM_THREADS", str(int(num_threads)))
    calc.setProperty("TIME_DIFF", str(int(time_diff)))
    calc.initialise()
    calc.startAddObservations()

    if source_indices is None:
        source_indices = np.arange(ts.shape[0])
    if target_indices is None:
        target_indices = np.arange(ts.shape[0])

    ts = ts.tolist()
    if mode == "all":
        for i in source_indices:
            for j in target_indices:
                calc.addObservations(ts[i], ts[j])

    elif mode == "pairwise":
        assert len(source_indices) == len(target_indices)
        for i, j in zip(source_indices, target_indices):
            calc.addObservations(ts[i], ts[j])
    calc.finaliseAddObservations()
    MI = calc.computeAverageLocalOfObservations()

    if num_surrogates > 0:
        NullDist = calc.computeSignificance(num_surrogates)
        NullMean = NullDist.getMeanOfDistribution()
        MI = MI - NullMean if (MI >= NullMean) else 0.0

    MI = nat2bit(MI)
    MI = MI if MI >= 0 else 0.0
    label = "mi"

    return [MI], [label]


def calc_te(
    ts: np.ndarray,
    k: int = 4,
    delay: int = 1,
    num_threads: int = 1,
    source_indices: List[int] = None,
    target_indices: List[int] = None,
    mode: str = "pairwise",
    verbose=False,
    **kwargs,
):
    """
    calculate the transfer entropy between time series based on the Kraskov method.

    Parameters
    ----------
    ts: np.ndarray [n_regions, n_samples]
        input array
    num_threads: int
        number of threads
    source_indices: list or np.ndarray
        indices of source time series, if None, all time series are used
    target_indices: list or np.ndarray
        indices of target time series, if None, all time series are used
    mode: str
        "pairwise" or "all", if "pairwise", source_indices and target_indices must have the same length


    Returns
    -------
    TE: list of floats
        transfer entropy
    """

    num_surrogates = kwargs.get("num_surrogates", 0)

    info, ts = prepare_input_ts(ts)
    if not info:
        return [np.nan], ["te"]

    if ts.shape[0] == 1:
        assert False, "ts must have more than one time series"

    init_jvm()
    calcClass = jp.JPackage(
        "infodynamics.measures.continuous.kraskov"
    ).TransferEntropyCalculatorKraskov
    calc = calcClass()
    calc.setProperty("NUM_THREADS", str(int(num_threads)))
    calc.setProperty("DELAY", str(int(delay)))
    calc.setProperty("AUTO_EMBED_RAGWITZ_NUM_NNS", "4")
    calc.setProperty("k", str(int(k)))
    calc.initialise()
    calc.startAddObservations()

    if source_indices is None:
        source_indices = np.arange(ts.shape[0])
    if target_indices is None:
        target_indices = np.arange(ts.shape[0])

    ts = ts.tolist()
    if mode == "all":
        for i in source_indices:
            for j in target_indices:
                calc.addObservations(ts[i], ts[j])

    elif mode == "pairwise":
        assert len(source_indices) == len(target_indices)
        for i, j in zip(source_indices, target_indices):
            calc.addObservations(ts[i], ts[j])
    calc.finaliseAddObservations()
    te = calc.computeAverageLocalOfObservations()

    if num_surrogates > 0:
        NullDist = calc.computeSignificance(num_surrogates)
        NullMean = NullDist.getMeanOfDistribution()
        # NullStd = NullDist.getStdOfDistribution()
        te = te - NullMean if (te >= NullMean) else 0.0
    te = te if te >= 0 else 0.0
    label = "te"

    return [te], [label]


def calc_entropy(ts: np.ndarray, average: bool = False, verbose=False):
    """
    calculate entropy of time series
    """

    if not isinstance(ts, np.ndarray):
        ts = np.array(ts)
    if ts.ndim == 1:
        ts = ts.reshape(1, -1)
    n = ts.shape[0]
    labels = [f"entropy_{i}" for i in range(n)]

    if ts.size == 0:
        return np.nan, labels
    if np.isnan(ts).any() or np.isinf(ts).any():
        n = ts.shape[0]
        return [np.nan] * n, labels

    init_jvm()

    calcClass = jp.JPackage(
        "infodynamics.measures.continuous.kozachenko"
    ).EntropyCalculatorMultiVariateKozachenko
    calc = calcClass()

    values = []
    if not average:
        for i in range(n):
            calc.initialise()
            calc.setObservations(ts[i, :])
            value = nat2bit(calc.computeAverageLocalOfObservations())
            values.append(value)
    else:
        calc.initialise()
        ts = ts.squeeze().flatten().tolist()
        calc.setObservations(ts)
        values = nat2bit(calc.computeAverageLocalOfObservations())
        labels = "entropy"

    return values, labels


def calc_entropy_bin(ts: np.ndarray, prob: str = "standard", average: bool = False, verbose=False):
    """Computes the entropy of the signal using the Shannon Entropy.

    Description in Article:
    Regularities Unseen, Randomness Observed: Levels of Entropy Convergence
    Authors: Crutchfield J. Feldman David

    Parameters
    ----------
    signal : nd-array
        Input from which entropy is computed
    prob : string
        Probability function (kde or gaussian functions are available)

    Returns
    -------
    values: float or array-like
        The normalized entropy value
    labels: string or array-like
        The label of the feature

    """

    def one_dim(x):
        if prob == "standard":
            value, counts = np.unique(ts, return_counts=True)
            p = counts / counts.sum()
        elif prob == "kde":
            p = kde(ts)
        elif prob == "gauss":
            p = gaussian(ts)

        if np.sum(p) == 0:
            return 0.0

        # Handling zero probability values
        p = p[np.where(p != 0)]

        # If probability all in one value, there is no entropy
        if np.log2(len(ts)) == 1:
            return 0.0
        elif np.sum(p * np.log2(p)) / np.log2(len(ts)) == 0:
            return 0.0
        else:
            return -np.sum(p * np.log2(p)) / np.log2(len(ts))

    info, ts = prepare_input_ts(ts)
    if not info:
        return [np.nan], [f"entropy_bin_{0}"]
    else:
        r, c = ts.shape
        values = np.zeros(r)
        for i in range(r):
            values[i] = one_dim(ts[i])
        if average:
            values = np.mean(values)
            labels = "entropy_bin"
        else:
            labels = [f"entropy_bin_{i}" for i in range(len(values))]
        return values, labels


def spectrum_stats(
    ts: np.ndarray,
    fs: float,
    method: str = "fft",
    nperseg: int = None,
    verbose=False,
    indices: List[int] = None,
    average=False,
    features: List[str] = [
        "spectral_distance",
        "fundamental_frequency",
        "max_frequency",
        "max_psd"
        "median_frequency",
        "spectral_centroid",
        "spectral_kurtosis",
        "spectral_variation",
    ],
):
    """
    compute some statistics of the power spectrum of the time series.

    Parameters
    ----------
    ts : nd-array [n_regions x n_samples]
       Input from which power spectrum statistics are computed
    fs : float
        Sampling frequency
    method : str
        Method to compute the power spectrum. Can be 'welch' or 'fft'
    indices: list of int
        indices of the regions to be used

    Returns
    -------
    values: array-like
        power spectrum statistics of the time series
    labels: array-like
        labels of the features
    """

    info, ts = prepare_input_ts(ts, indices)
    if not info:
        return [np.nan], [f"spectrum_stats_{0}"]
    else:
        ts = ts - ts.mean(axis=1, keepdims=True)

        if method == "welch":
            if nperseg is None:
                nperseg = ts.shape[1] // 2
            freq, psd = scipy.signal.welch(ts, fs=fs, axis=1, nperseg=nperseg)
        elif method == "fft":
            freq, psd = calc_fft(ts, fs)
        else:
            raise ValueError("method must be one of 'welch', 'fft'")
        
        if average:
            psd = np.mean(psd, axis=0).reshape(1, -1)

        values = np.array([])
        labels = []

        for f in features:
            
            
            v, l = eval(f)(freq, psd)
            values = np.append(values, v)
            labels = labels + l

    return values, labels


def spectrum_auc(
    ts, fs, method="fft", bands=None, nperseg=None, average=False, indices=None, verbose=False
):
    """
    calculate the area under the curve of the power spectrum of the time series over given frequency bands.

    Parameters
    ----------
    ts : nd-array [n_regions x n_samples]
        Input time series
    fs : float
        Sampling frequency
    method : str
        Method to compute the power spectrum. Can be 'welch' or 'fft'
    bands : list of tuples
        Frequency bands
    nperseg: int
        Length of each segment. default is half of the time series
    avg: bool
        averaging psd over all regions
    indices: list of int
        indices of the regions to be used

    Returns
    -------
    values: array-like
        area under the curve of the power spectrum of the time series
    labels: array-like
        labels of the features

    """

    info, ts = prepare_input_ts(ts)
    if not info:
        return [np.nan], [f"spectrum_auc_{0}"]
    else:
        ts = ts - ts.mean(axis=1, keepdims=True)
        # r, c = ts.shape

        if indices is None:
            indices = np.arange(ts.shape[0])
        else:
            indices = np.array(indices, dtype=int)
            ts = ts[indices, :]
            if len(indices) == 1:
                ts = ts.reshape(1, -1)

        if method == "welch":
            if nperseg is None:
                nperseg = ts.shape[1] // 2
            freq, psd = scipy.signal.welch(ts, fs=fs, axis=1, nperseg=nperseg)
        elif method == "fft":
            freq, psd = calc_fft(ts, fs)
        else:
            raise ValueError("method must be one of 'welch', 'fft'")

        if bands is None:
            bands = [(0, 4), (4, 8), (8, 12), (12, 30), (30, 70)]

        if average:
            psd = np.mean(psd, axis=0).reshape(1, -1)

        values = []
        for i, band in enumerate(bands):
            idx = (freq >= band[0]) & (freq < band[1])
            if np.sum(idx) == 0:
                continue
            psd_band = psd[:, idx]
            values.append(np.trapz(psd_band, axis=1))

        if len(values) > 0:
            values = np.concatenate(values)
            labels = [f"spectrum_auc_{i}" for i in range(len(values))]
        if len(values) == 0:
            values = [np.nan]
            labels = ["spectrum_auc"]

        return values, labels


def spectrum_moments(
    ts,
    fs,
    method="fft",
    nperseg=None,
    avg=False,
    moments=[2, 3, 4, 5, 6],
    normalize=False,
    indices=None,
    average=False,
    verbose=False,
):
    """
    Computes the moments of power spectrum

    Parameters
    ----------
    ts : nd-array [n_regions x n_samples]
       Input from which power spectrum statistics are computed
    fs : float
        Sampling frequency
    method : str
        Method to compute the power spectrum. Can be 'welch' or 'fft'
    nperseg: int
        ...
    avg: bool
        averaging over all regions
    nm: list of int
        moments orders

    Returns
    -------
    values: array-like
        power spectrum statistics of the time series
    labels: array-like
        labels of the features
    """

    info, n = prepare_input_ts(ts)
    if not info:
        return [np.nan] * n, [f"spectrum_moment_{i}" for i in range(n)]
    else:
        ts = n
        ts = ts - ts.mean(axis=1, keepdims=True)
        # r, c = ts.shape
        if indices is None:
            indices = np.arange(ts.shape[0])
        else:
            indices = np.array(indices, dtype=int)
            ts = ts[indices, :]
            if len(indices) == 1:
                ts = ts.reshape(1, -1)

        if method == "welch":
            if nperseg is None:
                nperseg = ts.shape[1] // 2
            freq, psd = scipy.signal.welch(ts, fs=fs, axis=1, nperseg=nperseg)
        elif method == "fft":
            freq, psd = calc_fft(ts, fs)
        else:
            raise ValueError("method must be one of 'welch', 'fft'")

        Values = np.array([])
        Labels = []
        if normalize:
            psd = psd / np.max(psd, axis=1, keepdims=True)

        if avg:
            psd = np.mean(psd, axis=0)

        for i in moments:
            _m = moment(psd, i, axis=1)
            if not average:
                Values = np.append(Values, _m)
                Labels = Labels + [f"spectrum_moment_{i}_{j}" for j in range(len(_m))]
            else:
                Values = np.append(Values, np.mean(_m))
                Labels = Labels + [f"spectrum_moment_{i}"]

    return Values, Labels


def psd_raw(
    ts,
    fs,
    bands=[(0, 4), (4, 8), (8, 12), (12, 30), (30, 70)],
    df=None,
    method="fft",
    nperseg=None,
    average=False,
    normalize=False,
    normalize_to: float = None,  # normalize to given value in Hz
    indices=None,
    verbose=False,
):
    """
    Calculate frequency spectrum and return with specified frequency resolution.

    Parameters
    ----------

    ts : nd-array [n_regions x n_samples]
        Input time series
    fs : float
        Sampling frequency
    bands : list of tuples
        Frequency bands
    df : float
        Frequency resolution, default is fs / n_samples
    method : str
        Method to compute the power spectrum. Can be 'welch' or 'fft'
    nperseg: int
        Length of each segment. default is half of the time series
    avg: bool
        averaging psd over all regions
    normalize: bool
        normalize the psd by the maximum value
    normalize_to: float
        normalize the psd to the given frequency value
    indices: list of int
        indices of the regions to be used

    Returns
    -------
    psd: array-like
        power spectrum density

    """

    info, ts = prepare_input_ts(ts)
    if not info:
        return [np.nan], [f"spectrum_moment_{0}"]
    else:
        ts = ts - ts.mean(axis=1, keepdims=True)
        # r, c = ts.shape
        if indices is None:
            indices = np.arange(ts.shape[0])
        else:
            indices = np.array(indices, dtype=int)
            ts = ts[indices, :]
            if len(indices) == 1:
                ts = ts.reshape(1, -1)

        if method == "welch":
            if nperseg is None:
                nperseg = ts.shape[1] // 2
            freq, psd = scipy.signal.welch(ts, fs=fs, axis=1, nperseg=nperseg)
        elif method == "fft":
            freq, psd = calc_fft(ts, fs)
        else:
            raise ValueError("method must be one of 'welch', 'fft'")

        if average:
            psd = np.mean(psd, axis=0).reshape(1, -1)

        if normalize and (normalize_to is not None):
            raise ValueError("normalize and normalize_to cannot be used together")

        if normalize_to is not None:
            # check if the value is in the frequency range
            if normalize_to < 0 or normalize_to > fs / 2:
                raise ValueError("normalize_to must be in the range of 0 to fs/2")

            # find index of the frequency closest to the given value
            idx = np.argmin(np.abs(freq - normalize_to))
            psd = psd / psd[:, idx].reshape(-1, 1)
        elif normalize:
            psd = psd / np.max(psd, axis=1, keepdims=True)

        if df is None:
            df = fs / ts.shape[1]
        fr_intp = np.arange(0, fs / 2, df)
        psd_intp = np.apply_along_axis(
            lambda row: np.interp(fr_intp, freq, row), axis=1, arr=psd
        )

        psd_bands = np.array([])
        for i in range(len(bands)):
            idx = (fr_intp >= bands[i][0]) & (fr_intp < bands[i][1])
            if np.sum(idx) == 0:
                continue
            psd_bands = np.append(psd_bands, psd_intp[:, idx].flatten())

        psd_bands = psd_bands.astype(float)
        labels = [f"psd_{i}" for i in range(len(psd_bands))]

        return psd_bands, labels


def wavelet_abs_mean_1d(ts, function=None, widths=np.arange(1, 10), verbose=False):
    """Computes CWT absolute mean value of each wavelet scale.

    Parameters
    ----------
    ts : nd-array
        Input from which CWT is computed
    function :  wavelet function
        Default: scipy.signal.ricker
    widths :  nd-array
        Widths to use for transformation
        Default: np.arange(1,10)

    Returns
    -------
    tuple
        CWT absolute mean value

    """
    if function is None:
        function = scipy.signal.ricker
    
    return tuple(np.abs(np.mean(wavelet(ts, function, widths), axis=1)))


def wavelet_abs_mean(ts, function=None, widths=np.arange(1, 10), verbose=False):
    '''
    """Computes CWT absolute mean value of each wavelet scale.

    Parameters
    ----------
    ts : nd-array [n_regions x n_samples]
        Input from which CWT is computed
    function :  wavelet function
        Default: scipy.signal.ricker
    widths :  nd-array
        Widths to use for transformation
        Default: np.arange(1,10)

    Returns
    -------
    values: array-like
        CWT absolute mean value of the time series
    labels: array-like
        labels of the features
    '''
    
    if function is None:
        function = scipy.signal.ricker

    info, n = prepare_input_ts(ts)
    if not info:
        return [np.nan] * n, [f"wavelet_abs_mean_{i}" for i in range(n)]
    else:
        ts = n
        r, _ = ts.shape
        values = np.zeros((r, len(widths)))
        for i in range(r):
            values[i] = wavelet_abs_mean_1d(ts[i], function, widths)

        values = values.flatten()
        labels = [
            f"wavelet_abs_mean_n{i}_w{j}"
            for i in range(len(values))
            for j in range(len(widths))
        ]
        return values, labels


def wavelet_std(ts, function=None, widths=np.arange(1, 10), verbose=False):
    """
    Computes CWT std value of each wavelet scale.

    Parameters
    ----------
    ts : nd-array [n_regions x n_samples]
        Input from which CWT is computed
    function :  wavelet function
        Default: scipy.signal.ricker
    widths :  nd-array
        Widths to use for transformation
        Default: np.arange(1,10)

    Returns
    -------
    values: array-like
        CWT std value of the time series
    labels: array-like
        labels of the features

    """
    
    if function is None:
        function = scipy.signal.ricker

    info, n = prepare_input_ts(ts)
    if not info:
        return [np.nan] * n, [f"wavelet_std_{i}" for i in range(n)]
    else:
        ts = n
        r, _ = ts.shape
        values = np.zeros((r, len(widths)))
        for i in range(r):
            values[i] = np.std(wavelet(ts[i], function, widths), axis=1)

        values = values.flatten()
        labels = [
            f"wavelet_std_n{i}_w{j}"
            for i in range(len(values))
            for j in range(len(widths))
        ]
        return values, labels


def wavelet_energy_1d(ts, function=None, widths=np.arange(1, 10), verbose=False):
    """Computes CWT energy of each wavelet scale.

    Implementation details:
    https://stackoverflow.com/questions/37659422/energy-for-1-d-wavelet-in-python

    Feature computational cost: 2

    Parameters
    ----------
    signal : nd-array
        Input from which CWT is computed
    function :  wavelet function
        Default: scipy.signal.ricker
    widths :  nd-array
        Widths to use for transformation
        Default: np.arange(1,10)

    Returns
    -------
    tuple
        CWT energy

    """
    if function is None:
        function = scipy.signal.ricker
    cwt = wavelet(ts, function, widths)
    energy = np.sqrt(np.sum(cwt**2, axis=1) / np.shape(cwt)[1])

    return tuple(energy)


def wavelet_energy(ts, function=None, widths=np.arange(1, 10), verbose=False):
    """
    Computes CWT energy of each wavelet scale.

    Parameters
    ----------
    ts : nd-array [n_regions x n_samples]
        Input from which CWT is computed
    function :  wavelet function
        Default: scipy.signal.ricker
    widths :  nd-array
        Widths to use for transformation
        Default: np.arange(1,10)

    Returns
    -------
    values: array-like
        CWT energy of the time series
    labels: array-like
        labels of the features

    """
    if function is None:
        function = scipy.signal.ricker

    info, n = prepare_input_ts(ts)
    if not info:
        return [np.nan] * n, [f"wavelet_energy_{i}" for i in range(n)]
    else:
        ts = n
        r, _ = ts.shape
        values = np.zeros((r, len(widths)))
        for i in range(r):
            values[i] = wavelet_energy_1d(ts[i], function, widths)

        values = values.flatten()
        labels = [
            f"wavelet_energy_n{i}_w{j}"
            for i in range(len(values))
            for j in range(len(widths))
        ]
        return values, labels


# -----------------------------------------------------------------------------


def hmm_stat(
    ts,
    node_indices=None,
    n_states=4,
    subname="",
    n_iter=100,
    seed=None,
    observations="gaussian",
    method="em",
    tcut=5,
    bins=10,
    verbose=False,
):
    """
    Calculate the state duration of the HMM.

    Parameters
    ----------
    ts : nd-array [n_regions x n_samples]
        Input from which HMM is computed
    node_indices : list
        List of node indices to be used for HMM
    n_states : int
        Number of states
    subname : str
        subname for the labels
    n_iter : int
        Number of iterations
    seed : int
        Random seed
    observations : str
        Observation distribution
    method : str
        Method to fit the HMM
    t_cut : int
        maximum duration of a state, default is 5

    Returns
    -------
    stat_vec : array-like
        HMM features
    labels : array-like
        labels of the features

    """

    if seed is not None:
        np.random.seed(seed)

    info, ts = prepare_input_ts(ts, indices)
    if not info:
        return [np.nan], [f"hmm_dur"]
    else:

        obs = ts[node_indices, :].T
        nt, obs_dim = obs.shape
        model = ssm.HMM(n_states, obs_dim, observations=observations)
        model_lls = model.fit(obs, method=method, num_iters=n_iter, verbose=0)
        hmm_z = model.most_likely_states(obs)
        # emmision_hmm_z, emmision_hmm_y = model.sample(nt) #!TODO: check if need to be used
        # hmm_x = model.smooth(obs)
        # upper = np.triu_indices(n_states, 0)
        trans_mat = (model.transitions.transition_matrix).flatten()  # [upper]

        stat_duration = state_duration(hmm_z, n_states, avg=True, tcut=tcut, bins=bins)
        labels = [f"hmm{subname}_dur_{i}" for i in range(len(stat_duration))]
        labels += [f"hmm{subname}_trans_{i}" for i in range(len(trans_mat))]
        stat_vec = np.concatenate([stat_duration, trans_mat])

        return stat_vec, labels


def catch22(
    ts,
    indices: List[int] = None,
    catch24=False,
    verbose=False,
    features=[
        "DN_HistogramMode_5",
        "DN_HistogramMode_10",
        "CO_f1ecac",
        "CO_FirstMin_ac",
        "CO_HistogramAMI_even_2_5",
        "CO_trev_1_num",
        "MD_hrv_classic_pnn40",
        "SB_BinaryStats_mean_longstretch1",
        "SB_TransitionMatrix_3ac_sumdiagcov",
        "PD_PeriodicityWang_th0_01",
        "CO_Embed2_Dist_tau_d_expfit_meandiff",
        "IN_AutoMutualInfoStats_40_gaussian_fmmi",
        "FC_LocalSimple_mean1_tauresrat",
        "DN_OutlierInclude_p_001_mdrmd",
        "DN_OutlierInclude_n_001_mdrmd",
        "SP_Summaries_welch_rect_area_5_1",
        "SB_BinaryStats_diff_longstretch0",
        "SB_MotifThree_quantile_hh",
        "SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1",
        "SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1",
        "SP_Summaries_welch_rect_centroid",
        "FC_LocalSimple_mean3_stderr",
    ],
):
    """
    Calculate the Catch22 features.

    Parameters
    ----------
    ts : nd-array [n_regions x n_samples]
        Input from which Catch22 features are computed
    node_indices : list
        List of node indices to be used for Catch22
    catch24 : bool
        If True, calculate mean and std of the features

    Returns
    -------
    values : array-like
        feature values
    labels : array-like
        labels of the features

    """
    try:
        import catch22_C
    except ImportError:
        raise ImportError(
            "pycatch22 is not installed. Please install it using `pip install pycatch22`"
        )
        
    if catch24:
        features = features.copy()
        features.append('DN_Mean')
        features.append('DN_Spread_Std')
        
    def get_features(x, features):
        out = []
        for f in features:
            f_fun = getattr(catch22_C, f)
            out.append(f_fun(list(x)))
        return out

    info, ts = prepare_input_ts(ts, indices)
    if not info:
       return [np.nan], [f"catch22"]
 
    else:
        nn = ts.shape[0]
        nf = 22 if not catch24 else 24
        values = np.zeros((nn, nf))
        for i in range(nn):
            v = get_features(ts[i], features)
            values[i] = v
        
        values = values.flatten()
        labels =  features * nn

        return values, labels