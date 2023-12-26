import numpy as np
import scipy.signal
from numpy import linalg as LA
from scipy.signal import hilbert
from vbi.feature_extraction.features_utils import *
from sklearn.decomposition import PCA
from scipy.stats import moment, skew, kurtosis
from vbi.feature_extraction.utility import *


try:
    import ssm
except:
    pass


########################### TEMPORAL ####################################


def abs_energy(ts):
    """Computes the absolute energy of the time serie.

    Feature computational cost: 1

    Parameters
    ----------
    ts : nd-arrays [n_regions x n_samples]
        Input from which the area under the curve is computed

    Returns
    -------
    values: list of float
        Absolute energy
    labels: list of str
        Labels of the features

    """

    info, n = check_input(ts)
    if not info:
        return [np.nan]*n, [f"abs_energy_{i}" for i in range(n)]
    else:
        ts = n
        values = np.sum(np.abs(ts) ** 2, axis=1)
        labels = [f"abs_energy_{i}" for i in range(len(values))]

    return values, labels


def average_power(ts, fs):
    """Computes the average power of the time serie.

    Feature computational cost: 1

    Parameters
    ----------
    ts : nd-arrays [n_regions x n_samples]
        Input from which the area under the curve is computed

    Returns
    -------
    values: list of float
        Average power
    labels: list of str
        Labels of the features

    """

    info, n = check_input(ts)
    if not info:
        return [np.nan]*n, [f"average_power_{i}" for i in range(n)]
    else:
        ts = n
        times = compute_time(ts[0], fs)
        values = np.sum(ts ** 2, axis=1) / (times[-1] - times[0])
        labels = [f"average_power_{i}" for i in range(len(values))]
        return values, labels


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
    if ts.ndim == 1:
        ts = ts.reshape(1, -1)

    if ts.size == 0:
        return [], []

    # t = np.arange(ts.shape[1]) / fs
    values = np.trapz(np.abs(ts), dx=1/fs, axis=1)
    labels = [f"auc_{i}" for i in range(len(values))]
    return values, labels


########################### STATISTICAL ####################################


def calc_var(ts):
    """Computes variance of the time series.

    Feature computational cost: 1

    Parameters
    ----------
    ts : nd-array [n_regions x n_samples]
       Input from which var is computed

    Returns
    -------
    values: array-like
        variance of the time series
    labels: array-like
        labels of the features

    """

    if not isinstance(ts, np.ndarray):
        ts = np.array(ts)
    if ts.ndim == 1:
        ts = ts.reshape(1, -1)

    if ts.size == 0:
        return [np.nan], ["variance_0"]

    values = np.var(ts, axis=1)
    labels = [f"variance_{i}" for i in range(len(values))]

    return values, labels


def calc_std(ts):
    """Computes standard deviation of the time serie.

    Parameters
    ----------
    ts : nd-array [n_regions x n_samples]
       Input from which std is computed

    Returns
    -------
    values: array-like
        std of the time series
    labels: array-like
        labels of the features
    """

    info, n = check_input(ts)
    if not info:
        return [np.nan] * n, [f"std_{i}" for i in range(n)]
    else:
        ts = n
        values = np.std(ts, axis=1)
        labels = [f"std_{i}" for i in range(len(values))]
        return values, labels


def calc_mean(ts):
    """Computes median of the time serie.

    Parameters
    ----------
    ts : nd-array [n_regions x n_samples]
       Input from which median is computed

    Returns
    -------
    values: array-like
        mean of the time series
    labels: array-like
        labels of the features

    """

    info, n = check_input(ts)
    if not info:
        return [np.nan]*n, [f"mean_{i}" for i in range(n)]
    else:
        ts = n
        values = np.mean(ts, axis=1)
        labels = [f"mean_{i}" for i in range(len(values))]
        return values, labels


def calc_centroid(ts, fs):
    """Computes the centroid along the time axis.

    Parameters
    ----------
    signal : nd-array
        Input from which centroid is computed
    fs: int
        Signal sampling frequency

    Returns
    -------
    float
        Temporal centroid

    """
    info, n = check_input(ts)
    if not info:
        return [np.nan]*n, [f"centroid_{i}" for i in range(n)]
    else:
        ts = n
        tol = 1e-10
        r, c = ts.shape
        centroid = np.zeros(r)
        time = compute_time(ts[0], fs)
        energy = ts ** 2
        t_energy = np.dot(time, energy.T)
        energy_sum = np.sum(energy, axis=1)
        ind_nonzero = (np.abs(energy_sum) > tol) | (np.abs(t_energy) > tol)
        centroid[ind_nonzero] = t_energy[ind_nonzero] / energy_sum[ind_nonzero]
        labels = [f"centroid_{i}" for i in range(len(centroid))]

        return centroid, labels


def calc_kurtosis(ts):
    """ 
    Computes the kurtosis of the time series.

    Parameters
    ----------
    ts : nd-array [n_regions x n_samples]
       Input from which kurtosis is computed

    Returns
    -------
    values: array-like
        kurtosis of the time series
    labels: array-like
        labels of the features

    """

    info, n = check_input(ts)
    if not info:
        return [np.nan]*n, [f"kurtosis_{i}" for i in range(n)]
    else:
        ts = n
        values = kurtosis(ts, axis=1)
        labels = [f"kurtosis_{i}" for i in range(len(values))]
        return values, labels


def calc_skewness(ts):
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

    info, n = check_input(ts)
    if not info:
        return [np.nan]*n, [f"skewness_{i}" for i in range(n)]
    else:
        ts = n
        values = skew(ts, axis=1)
        labels = [f"skewness_{i}" for i in range(len(values))]
        return values, labels


def calc_max(ts):
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

    info, n = check_input(ts)
    if not info:
        return [np.nan]*n, [f"max_{i}" for i in range(n)]
    else:
        ts = n
        values = np.max(ts, axis=1)
        labels = [f"max_{i}" for i in range(len(values))]
        return values, labels


def calc_min(ts):
    """ 
    Computes the minimum of the time series.

    Parameters
    ----------
    ts : nd-array [n_regions x n_samples]
       Input from which minimum is computed

    Returns
    -------
    values: array-like
        minimum of the time series
    labels: array-like
        labels of the features

    """

    info, n = check_input(ts)
    if not info:
        return [np.nan]*n, [f"min_{i}" for i in range(n)]
    else:
        ts = n
        values = np.min(ts, axis=1)
        labels = [f"min_{i}" for i in range(len(values))]
        return values, labels


def calc_median(ts):
    """ 
    Computes the median of the time series.

    Parameters
    ----------
    ts : nd-array [n_regions x n_samples]
       Input from which median is computed

    Returns
    -------
    values: array-like
        median of the time series
    labels: array-like
        labels of the features

    """

    info, n = check_input(ts)
    if not info:
        return [np.nan]*n, [f"median_{i}" for i in range(n)]
    else:
        ts = n
        values = np.median(ts, axis=1)
        labels = [f"median_{i}" for i in range(len(values))]
        return values, labels


def mean_abs_deviation(ts):
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

    info, n = check_input(ts)
    if not info:
        return [np.nan]*n, [f"mean_abs_deviation_{i}" for i in range(n)]
    else:
        ts = n
        values = np.mean(
            np.abs(ts - np.mean(ts, axis=1, keepdims=True)), axis=1)
        labels = [f"mean_abs_deviation_{i}" for i in range(len(values))]
        return values, labels


def median_abs_deviation(ts):
    """ 
    Computes the median absolute deviation of the time series.

    Parameters
    ----------
    ts : nd-array [n_regions x n_samples]
       Input from which median absolute deviation is computed

    Returns
    -------
    values: array-like
        median absolute deviation of the time series
    labels: array-like
        labels of the features

    """

    info, n = check_input(ts)
    if not info:
        return [np.nan]*n, [f"median_abs_deviation_{i}" for i in range(n)]
    else:
        ts = n
        values = np.median(
            np.abs(ts - np.median(ts, axis=1, keepdims=True)), axis=1)
        labels = [f"median_abs_deviation_{i}" for i in range(len(values))]
        return values, labels


def rms(ts):
    """ 
    Computes the root mean square of the time series.

    Parameters
    ----------
    ts : nd-array [n_regions x n_samples]
       Input from which root mean square is computed

    Returns
    -------
    values: array-like
        root mean square of the time series
    labels: array-like
        labels of the features

    """

    info, n = check_input(ts)
    if not info:
        return [np.nan]*n, [f"rms_{i}" for i in range(n)]
    else:
        ts = n
        values = np.sqrt(np.mean(ts ** 2, axis=1))
        labels = [f"rms_{i}" for i in range(len(values))]
        return values, labels


def interq_range(ts):
    """ 
    Computes the interquartile range of the time series.

    Parameters
    ----------
    ts : nd-array [n_regions x n_samples]
       Input from which interquartile range is computed

    Returns
    -------
    values: array-like
        interquartile range of the time series
    labels: array-like
        labels of the features

    """

    info, n = check_input(ts)
    if not info:
        return [np.nan]*n, [f"interq_range_{i}" for i in range(n)]
    else:
        ts = n
        values = np.subtract(*np.percentile(ts, [75, 25], axis=1))
        labels = [f"interq_range_{i}" for i in range(len(values))]
        return values, labels


def zero_crossing(ts):
    """ 
    Computes the number of zero crossings of the time series.

    Parameters
    ----------
    ts : nd-array [n_regions x n_samples]
       Input from which number of zero crossings is computed

    Returns
    -------
    values: array-like
        number of zero crossings of the time series
    labels: array-like
        labels of the features

    """
    info, n = check_input(ts)
    if not info:
        return [np.nan]*n, [f"zero_crossing_{i}" for i in range(n)]
    else:
        ts = n
        values = np.array([np.sum(np.diff(np.sign(y_i)) != 0)
                          for y_i in ts], dtype=int)
        labels = [f"zero_crossing_{i}" for i in range(len(values))]
        return values, labels


def calc_rss(ts, percentile=95):
    """
    Calculates RSS with given percentile

    Parameters
    ----------
    ts : nd-array [n_regions x n_samples]
         Input time seris
    percentile : float
            Percentile of RSS
    """

    info, n = check_input(ts)
    if not info:
        return [np.nan]*n, [f"rss_{i}" for i in range(n)]
    else:
        ts = n
        nn, n_samples = ts.shape
        rss = np.zeros(n_samples)
        for t in range(n_samples):
            z = np.power(np.outer(ts[:, t], ts[:, t]), 2)
            rss[t] = np.sqrt(np.einsum('ij->', z))
        return np.percentile(rss, percentile), ['RSS th']
    

def kop(ts, indices=None):
    '''
    Calculate the Kuramoto order parameter (KOP)
    
    '''

    info, n = check_input(ts)
    if not info:
        return [np.nan], ["kop"]
    else:
        ts = n 
        R = km_order(ts, indices=indices, avg=True)
        return R, ['kop']

    

########################### CONNECTIVITY ####################################


def fc_sum(x, positive=False):
    '''
    Calculate the sum of functional connectivity (FC)

    Parameters
    ----------
    ts : nd-array [n_regions x n_samples]
       Input from which var is computed

    Returns
    -------
    result: float
        sum of functional connectivity
    '''

    label = "fc_sum"

    if isinstance(x, np.ndarray):
        if x.shape[1] < 2:
            return 0.0, label

    info, n = check_input(x)
    if not info:
        return np.nan, label

    fc = np.corrcoef(x)
    if positive:
        fc = fc * (fc > 0)
    value = np.sum(np.abs(fc)) - np.trace(np.abs(fc))

    return value, label


def fc_stat(x,
            k=0,
            PCA_n_components=3,
            positive=False,
            demean=False,
            method="corr",
            masks=None):
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
    method: str
        method to calculate FC, "corr" is available for now
        #!TODO: add "plv" and "pli"

    Returns
    -------
    stats: np.ndarray (1d)
        feature values
    '''

    if not isinstance(x, np.ndarray):
        x = np.array(x)
    if x.ndim == 1:
        return np.nan, "fc_0"
    if x.size == 0:
        return np.nan, "fc_0"
    if x.shape[1] == 1:
        return 0.0, "fc_0"

    if masks is None:
        masks = {"full": np.ones((x.shape[0], x.shape[0]))}

    if method == "corr":
        FC = np.corrcoef(x)
    # elif method == "plv":
    #     FC = ...  # TODO use mne_connectivity.spectral_connectivity
    # elif method == "pli":
    #     FC = ...  # TODO use mne_connectivity.spectral_connectivity
    else:
        raise ValueError("method must be one of 'corr'. ")

    Values = []
    Labels = []
    fc = get_fc(x, masks=masks, positive=positive)
    for key in fc.keys():
        values, labels = matrix_stat(fc[key],
                                     demean=demean,
                                     k=k,
                                     PCA_n_components=PCA_n_components)
        labels = [f"fc_{key}_{label}" for label in labels]
        Values.extend(values)
        Labels.extend(labels)

    return Values, Labels


def fc_homotopic(x, avg=False, positive=True):
    '''
    Calculate the homotopic connectivity vector of a given brain activity

    Parameters
    ----------
    bold: array_like [nn, nt]
        The brain activity to be analyzed.
    avg: bool
        If True, the average homotopic connectivity is returned. 
        Otherwise, the homotopic connectivity vector is returned.
    positive: bool
        If True, only positive correlations are considered.

    Returns
    -------
    Homotopic_FC_vector : array_like [n_nodes]
        The homotopic correlation vector.

    Negative correlations may be artificially induced when using global signal regression
    in functional imaging pre-processing (Fox et al., 2009; Murphy et al., 2009; Murphy and Fox, 2017).
    Therefore, results on negative weights should be interpreted with caution and should be understood
    as complementary information underpinning the findings based on positive connections
    '''

    if isinstance(x, (list, tuple)):
        x = np.array(x)
    assert (x.ndim == 2)
    nn, nt = x.shape
    assert (nn > 1)

    NHALF = int(nn//2)
    rsFC = np.corrcoef(x)
    if positive:
        rsFC = rsFC * (rsFC > 0)
    rsFC = rsFC - np.diag(np.diag(rsFC))  # not necessary for hfc
    hfc = np.diag(rsFC, k=NHALF)
    if avg:
        return [np.mean(hfc)], ["fc_homotopic_avg"]
    else:
        values = hfc.tolist()
        labels = [f"fc_homotopic_{i}" for i in range(len(values))]
        return values, labels

###################### FCD METRICS ##########################################


def fluidity(x,
             TR=1.0,
             window_length=30,
             positive=True,  # ! TODO: check if need to be False
             masks={},
             get_FCDs=False
             ):
    """
    calculates FCD and subsequently fluidity metrics for different networks.
    Input

    Parameters
    ----------

    x: np.ndarray [n_regions, n_samples]
        input array

    ---------------------------
    x: array_like [nn, nt]
        The brain activity to be analyzed.
    TR: float
        time per scanning.

    window_length: int
    positive: bool
        If True, only positive correlations are considered, in consistence with Lavagna et al. 2023, Stumme et al., 2020.
    networks: dictionary
        network name : array of nodes, e.g. np.arange(200) for a whole brain network with 200 nodes
            or string e.g. "ihemi" (only acceptable for now) for interhemispherical connections.

    Return
    ----------------------------
    fluidity metrics: list
        [fluidity_median, fluidity_mean, fluidity_var, fluidity_kurtosis] for each different key of the networks dictionary.
    fluidity labels: list
        [label_med, label_mean, label_var, label_kur] for each different key of the networks dictionary.
    """
    ts = x.T
    nt, nn = ts.shape

    mask_full = np.ones((nn, nn))
    effective_masks = {}
    # initialize FCD variables
    FCDs = []
    FCDs_ut = []
    # calculate windowed FC ### calculate FCD ###
    windowed_data = np.lib.stride_tricks.sliding_window_view(
        ts, (int(window_length/TR), nn), axis=(0, 1)).squeeze()
    n_windows = windowed_data.shape[0]
    fc_stream = np.asarray(
        [np.corrcoef(windowed_data[i, :, :], rowvar=False) for i in range(n_windows)])
    if positive == True:
        # mask out the negative correlations by multiplying with 1 the positive
        # and with 0 the negative correlations
        fc_stream *= fc_stream > 0
    if len(masks) == 0:  # ! TODO check getting length of dictionary
        masks["full"] = mask_full
    for j, key in enumerate(masks.keys()):
        mask = masks[key]
        # get the upper triangle of the given mask
        mask *= np.triu(mask_full, k=1)
        nonzero_idx = np.nonzero(mask)
        fc_stream_masked = fc_stream[:, nonzero_idx[0], nonzero_idx[1]]
        fcd = np.corrcoef(fc_stream_masked, rowvar=True)
        if get_FCDs == True:
            FCDs.append(fcd)
        else:
            ut_idx = np.triu_indices_from(fcd, k=int(window_length/TR))
            FCDs_ut.append(fcd[ut_idx[0], ut_idx[1]])
    if get_FCDs == True:
        return FCDs, [f"{key}fcd" for key in masks.keys()]
    else:
        fluidity_metrics = []
        fluidity_labels = []
        for j, key in enumerate(masks.keys()):
            ut_fcd = FCDs_ut[j]

            # var(ut_fcd)
            fluidity_var = np.var(ut_fcd)
            label_var = f"{key}varfcd"

            # mean(ut_fcd)
            fluidity_mean = np.mean(ut_fcd)
            label_mean = f"{key}meanfcd"

            # median(ut_fcd)
            fluidity_median = np.median(ut_fcd)
            label_med = f"{key}medfcd"

            # median(ut_fcd)
            fluidity_kurtosis = kurtosis(ut_fcd)
            label_kur = f"{key}kurfcd"
            fluidity_metrics += [fluidity_median,
                                 fluidity_mean, fluidity_var, fluidity_kurtosis]
            fluidity_labels += [label_med, label_mean, label_var, label_kur]

        return fluidity_metrics, fluidity_labels


############################# CO-FLUCTUATIONS METRICS ################################

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


########################### OTHER ####################################


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


def fcd_stat(ts,
             win_len=30,
             TR=1,
             demean=False,
             positive=False,
             PCA_n_components=3,
             masks=None):

    Values = []
    Labels = []

    k = int(win_len/TR)
    fcd = get_fcd(ts=ts, TR=TR, win_len=win_len,
                  positive=positive, masks=masks)
    for key in fcd.keys():
        values, labels = matrix_stat(fcd[key],
                                     demean=demean,
                                     k=k,
                                     PCA_n_components=PCA_n_components)
        labels = [f"fcd_{key}_{label}" for label in labels]
        Values.extend(values)
        Labels.extend(labels)

    return Values, Labels


############################# Information Theory ##############################

def calc_mi(ts,
            k=4,
            time_diff=1,
            num_threads=1,
            source_indices=None,
            target_indices=None,
            mode="pairwise",
            **kwargs):
    '''
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
    '''

    num_surrogates = kwargs.get("num_surrogates", 0)

    if not isinstance(ts, np.ndarray):
        ts = np.array(ts)
    if ts.ndim == 1:
        assert (False), "ts must be a 2d array"

    init_jvm()
    calcClass = jp.JPackage(
        "infodynamics.measures.continuous.kraskov").MutualInfoCalculatorMultiVariateKraskov2
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
        assert (len(source_indices) == len(target_indices))
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

    return MI, label


def calc_te(ts,
            k=4,
            delay=1,
            num_threads=1,
            source_indices=None,
            target_indices=None,
            mode="pairwise",
            **kwargs):
    '''
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
    '''

    num_surrogates = kwargs.get("num_surrogates", 0)

    if not isinstance(ts, np.ndarray):
        ts = np.array(ts)
    if ts.ndim == 1:
        assert (False), "ts must be a 2d array"

    init_jvm()
    calcClass = jp.JPackage(
        "infodynamics.measures.continuous.kraskov").TransferEntropyCalculatorKraskov
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
        assert (len(source_indices) == len(target_indices))
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

    return te, label


def calc_entropy(ts,
                 average=False,
                 **kwargs):
    '''
    calculate entropy of time series
    '''

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
        return [np.nan]*n, labels

    init_jvm()

    calcClass = jp.JPackage(
        "infodynamics.measures.continuous.kozachenko").EntropyCalculatorMultiVariateKozachenko
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


def calc_entropy_bin(ts, prob="standard", average=False):
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

    info, n = check_input(ts)
    if not info:
        return [np.nan]*n, [f"entropy_bin_{i}" for i in range(n)]
    else:
        ts = n
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


############################# Spectral ########################################


def spectrum_stats(ts, fs, method='fft'):
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

    Returns
    -------
    values: array-like
        power spectrum statistics of the time series
    labels: array-like
        labels of the features
    """

    info, n = check_input(ts)
    if not info:
        return [np.nan]*n, [f"spectrum_stats_{i}" for i in range(n)]
    else:
        ts = n
        ts = ts - ts.mean(axis=1, keepdims=True)
        # r, c = ts.shape

        if method == 'welch':
            freq, psd = scipy.signal.welch(ts, fs=fs, axis=1)
        elif method == 'fft':
            freq, psd = calc_fft(ts, fs)
        else:
            raise ValueError("method must be one of 'welch', 'fft'")

        Values = np.array([])
        Labels = []

        # spectral distance
        val, lab = spectral_distance(psd)
        Values = np.append(Values, val)
        Labels = Labels + lab

        # fundamental_frequency
        val, lab = fundamental_frequency(freq, psd)
        Values = np.append(Values, val)
        Labels = Labels + lab

        # max frequency
        val, lab = max_frequency(freq, psd)
        Values = np.append(Values, val)
        Labels = Labels + lab

        # median frequency
        val, lab = median_frequency(freq, psd)
        Values = np.append(Values, val)
        Labels = Labels + lab

        # spectral centroid
        val, lab = spectral_centroid(freq, psd)
        Values = np.append(Values, val)
        Labels = Labels + lab

        # spectral kurtosis
        val, lab = spectral_kurtosis(freq, psd)
        Values = np.append(Values, val)
        Labels = Labels + lab

        # spectral variation
        val, lab = spectral_variation(psd)
        Values = np.append(Values, val)
        Labels = Labels + lab

    return Values, Labels


def wavelet_abs_mean_1d(ts, function=scipy.signal.ricker, widths=np.arange(1, 10)):
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
    return tuple(np.abs(np.mean(wavelet(ts, function, widths), axis=1)))


def wavelet_abs_mean(ts, function=scipy.signal.ricker, widths=np.arange(1, 10)):
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

    info, n = check_input(ts)
    if not info:
        return [np.nan]*n, [f"wavelet_abs_mean_{i}" for i in range(n)]
    else:
        ts = n
        r, _ = ts.shape
        values = np.zeros((r, len(widths)))
        for i in range(r):
            values[i] = wavelet_abs_mean_1d(ts[i], function, widths)

        values = values.flatten()
        labels = [f"wavelet_abs_mean_n{i}_w{j}"
                  for i in range(len(values))
                  for j in range(len(widths))]
        return values, labels


def wavelet_std(ts, function=scipy.signal.ricker, widths=np.arange(1, 10)):
    '''
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

    '''

    info, n = check_input(ts)
    if not info:
        return [np.nan]*n, [f"wavelet_std_{i}" for i in range(n)]
    else:
        ts = n
        r, _ = ts.shape
        values = np.zeros((r, len(widths)))
        for i in range(r):
            values[i] = np.std(wavelet(ts[i], function, widths), axis=1)

        values = values.flatten()
        labels = [f"wavelet_std_n{i}_w{j}"
                  for i in range(len(values))
                  for j in range(len(widths))]
        return values, labels


def wavelet_energy_1d(ts, function=scipy.signal.ricker, widths=np.arange(1, 10)):
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
    cwt = wavelet(ts, function, widths)
    energy = np.sqrt(np.sum(cwt ** 2, axis=1) / np.shape(cwt)[1])

    return tuple(energy)


def wavelet_energy(ts, function=scipy.signal.ricker, widths=np.arange(1, 10)):
    '''
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

    '''

    info, n = check_input(ts)
    if not info:
        return [np.nan]*n, [f"wavelet_energy_{i}" for i in range(n)]
    else:
        ts = n
        r, _ = ts.shape
        values = np.zeros((r, len(widths)))
        for i in range(r):
            values[i] = wavelet_energy_1d(ts[i], function, widths)

        values = values.flatten()
        labels = [f"wavelet_energy_n{i}_w{j}"
                  for i in range(len(values))
                  for j in range(len(widths))]
        return values, labels


def state_duration(hmm_z, n_states, avg=True):
    ''' 
    Measure the duration of each state 

    Parameters
    ----------
    hmm_z : nd-array [n_samples]
        The most likely states for each time point
    n_states : int
        The number of states
    avg : bool
        If True, the average duration of each state is returned.
        Otherwise, the duration of each state is returned.

    Returns
    -------
    stat_vec : array-like
        The duration of each state

    '''
    infered_state = hmm_z.astype(int)
    inferred_state_list, inferred_durations = ssm.util.rle(infered_state)

    inf_durs_stacked = []
    for s in range(n_states):
        inf_durs_stacked.append(inferred_durations[inferred_state_list == s])

    stat_vec = []
    for s in range(n_states):
        value, count = np.unique(inf_durs_stacked[s], return_counts=True)
        _dur = []
        for v in range(1, n_states+1):
            if v in value:
                _dur.append(count[np.where(value == v)][0])
            else:
                _dur.append(0)
        stat_vec.append(_dur)

    stat_vec = np.array(stat_vec, dtype=int)
    if avg:
        return stat_vec.mean(axis=0)
    else:
        return stat_vec.flatten()
# -----------------------------------------------------------------------------


def hmm_stat(ts,
             node_indices=None,
             n_states=4,
             subname="",
             n_iter=100,
             seed=None,
             observations="gaussian",
             method="em"
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

    Returns
    -------
    stat_vec : array-like
        HMM features
    labels : array-like
        labels of the features

    """

    if seed is not None:
        np.random.seed(seed)

    info, n = check_input(ts)
    if not info:
        return [np.nan]*n, [f"hmm_dur_{i}" for i in range(n)]
    else:
        ts = n

        if node_indices is None:
            node_indices = np.arange(ts.shape[0])

        obs = ts[node_indices, :].T
        nt, obs_dim = obs.shape
        model = ssm.HMM(n_states, obs_dim, observations=observations)
        model_lls = model.fit(obs, method=method, num_iters=n_iter, verbose=0)
        hmm_z = model.most_likely_states(obs)
        # emmision_hmm_z, emmision_hmm_y = model.sample(nt) #!TODO: check if need to be used
        # hmm_x = model.smooth(obs)
        # upper = np.triu_indices(n_states, 0)
        # transition_mat = model.transitions.transition_matrix[upper]

        stat_duration = state_duration(hmm_z, n_states, avg=True)
        labels = [f"hmm{subname}_dur_{i}" for i in range(len(stat_duration))]
        stat_vec = stat_duration

        return stat_vec, labels
