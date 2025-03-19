import vbi
import scipy
import torch
import numpy as np
from os.path import join
from typing import Union
from copy import deepcopy
import scipy.stats as stats
from numpy import linalg as LA
from sklearn.decomposition import PCA
from scipy.signal import butter, detrend, filtfilt, hilbert
from vbi.feature_extraction.features_settings import load_json
from vbi.feature_extraction.utility import *

try:
    import jpype as jp
    import ssm
except:
    # logging.warning("jpype not imported.")
    pass


def slice_features(x: Union[np.ndarray, torch.Tensor], feature_names: list, info: dict):
    """
    Slice features using given feature list

    Parameters
    ----------
    x: array-like
    features: list of strings
        list of features
    info: dict
        features's colum indices in x

    Returns
    -------
    x_sliced: array-like
        sliced features
    """
    if isinstance(x, (list, tuple)):
        x = np.array(x)

    if x.ndim == 1:
        x = x.reshape(1, -1)

    is_tensor = isinstance(x, torch.Tensor)
    if is_tensor:
        x_sliced = torch.Tensor([])
    else:
        x_sliced = np.array([])

    if len(feature_names) == 0:
        return x_sliced

    for f_name in feature_names:
        if f_name in info:
            coli, colf = info[f_name]["index"][0], info[f_name]["index"][1]
            if is_tensor:
                x_sliced = torch.cat((x_sliced, x[:, coli:colf]), dim=1)
            else:
                if x_sliced.size == 0:
                    x_sliced = x[:, coli:colf]
                else:
                    x_sliced = np.concatenate((x_sliced, x[:, coli:colf]), axis=1)
        else:
            raise ValueError(f"{f_name} not in info")

    return x_sliced


def preprocess(ts, fs=None, preprocess_dict={}, **kwargs):
    """
    Preprocess time series data

    Parameters
    ----------
    ts : nd-array [n_regions, n_timepoints]
        Input from which the features are extracted
    fs : int
        Sampling frequency, set to 1 if not used
    preprocess_dict : dictionary
        Dictionary of preprocessing options
    **kwargs : dict
        Additional arguments


    """

    if not preprocess_dict:
        preprocess_dict = load_json(
            vbi.__path__[0] + "/feature_extraction/preprocess.json"
        )

    if preprocess_dict["zscores"]["use"] == "yes":
        ts = stats.zscore(ts, axis=1)
    if preprocess_dict["offset"]["use"] == "yes":
        value = preprocess_dict["offset"]["parameters"]["value"]
        ts = ts[:, value:]

    if preprocess_dict["demean"]["use"] == "yes":
        ts = ts - np.mean(ts, axis=1)[:, None]

    if preprocess_dict["detrend"]["use"] == "yes":
        ts = detrend(ts, axis=1)

    if preprocess_dict["filter"]["use"] == "yes":
        low_cut = preprocess_dict["filter"]["parameters"]["low"]
        high_cut = preprocess_dict["filter"]["parameters"]["high"]
        order = preprocess_dict["filter"]["parameters"]["order"]
        TR = 1.0 / fs
        ts = band_pass_filter(ts, k=order, TR=TR, low_cut=low_cut, high_cut=high_cut)

    if preprocess_dict["remove_strong_artefacts"]["use"] == "yes":
        ts = remove_strong_artefacts(ts)

    return ts


def band_pass_filter(ts, low_cut=0.02, high_cut=0.1, TR=2.0, order=2):
    """
    apply band pass filter to given time series

    Parameters
    ----------
    ts : numpy.ndarray [n_regions, n_timepoints]
        Input signal
    low_cut : float, optional
        Low cut frequency. The default is 0.02.
    high_cut : float, optional
        High cut frequency. The default is 0.1.
    TR : float, optional
        Sampling interval. The default is 2.0 second.

    returns
    -------
    ts_filt : numpy.ndarray
        filtered signal


    """

    assert np.isnan(ts).any() == False

    fnq = 1.0 / (2.0 * TR)  # Nyquist frequency
    Wn = [low_cut / fnq, high_cut / fnq]
    bfilt, afilt = butter(order, Wn, btype="band")
    return filtfilt(bfilt, afilt, ts, axis=1)


def remove_strong_artefacts(ts, threshold=3.0):

    if isinstance(ts, (list, tuple)):
        ts = np.array(ts)

    if ts.ndim == 1:
        ts = ts.reshape(1, -1)

    nn = ts.shape[0]

    for i in range(nn):
        x_ = ts[i, :]
        std_dev = threshold * np.std(x_)
        x_[x_ > std_dev] = std_dev
        x_[x_ < -std_dev] = -std_dev
        ts[i, :] = x_
    return ts


def get_fc(ts, masks=None, positive=False, fc_fucntion="corrcoef"):
    """
    calculate the functional connectivity matrix

    Parameters
    ----------
    ts : numpy.ndarray [n_regions, n_timepoints]
        Input signal

    Returns
    -------
    FC : numpy.ndarray
        functional connectivity matrix
    """

    from numpy import corrcoef, cov

    n_noes = ts.shape[0]
    if masks is None:
        masks = {"full": np.ones((n_noes, n_noes))}

    FCs = {}
    FC = eval(fc_fucntion)(ts)
    for _, key in enumerate(masks.keys()):
        mask = masks[key]
        fc = deepcopy(FC)
        if positive:
            fc = fc * (fc > 0)
        fc = fc * mask
        fc = fc - np.diag(np.diagonal(fc))
        FCs[key] = fc

    return FCs


def get_fcd(
    ts,
    TR=1,
    win_len=30,
    positive=False,
    masks=None,
    #!TODO: add overlap
):
    """
    Compute dynamic functional connectivity.

    Parameters
    ----------

    ts: numpy.ndarray [n_regions, n_timepoints]
        Input signal
    win_len: int
        sliding window length in samples, default is 30
    TR: int
        repetition time. It refers to the amount of time that
        passes between consecutive acquired brain volumes during
        functional magnetic resonance imaging (fMRI) scans.
    positive: bool
        if True, only positive values of FC are considered.
        default is False
    masks: dict
        dictionary of masks to compute FCD on.
        default is None, which means that FCD is computed on the full matrix.
        see also `hbt.utility.make_mask` and `hbt.utility.get_masks`.

    Returns
    -------
        FCD: ndarray
            matrix of functional connectivity dynamics
    """
    if not isinstance(ts, np.ndarray):
        ts = np.array(ts)

    ts = ts.T
    n_samples, n_nodes = ts.shape
    # check if lenght of the time series is enough
    if n_samples < 2 * win_len:
        raise ValueError(
            f"get_fcd: Length of the time series should be at least 2 times of win_len. n_samples: {n_samples}, win_len: {win_len}"
        )

    mask_full = np.ones((n_nodes, n_nodes))
    if masks is None:
        masks = {"full": mask_full}

    windowed_data = np.lib.stride_tricks.sliding_window_view(
        ts, (int(win_len / TR), n_nodes), axis=(0, 1)
    ).squeeze()
    n_windows = windowed_data.shape[0]
    fc_stream = np.asarray(
        [np.corrcoef(windowed_data[i, :, :], rowvar=False) for i in range(n_windows)]
    )

    if positive:
        fc_stream *= fc_stream > 0

    FCDs = {}
    for _, key in enumerate(masks.keys()):
        mask = masks[key].astype(np.float64)
        mask *= np.triu(mask_full, k=1)
        nonzero_idx = np.nonzero(mask)
        fc_stream_masked = fc_stream[:, nonzero_idx[0], nonzero_idx[1]]
        fcd = np.corrcoef(fc_stream_masked, rowvar=True)
        FCDs[key] = fcd

    return FCDs


def get_fcd2(ts, wwidth=30, maxNwindows=200, olap=0.94, indices=[], verbose=False):
    """
    Functional Connectivity Dynamics from the given of time series

    Parameters
    ----------
    data: np.ndarray (2d)
        time series in rows [n_nodes, n_samples]
    opt: dict
        parameters

    Returns
    -------
    FCD: np.ndarray (2d)
        functional connectivity dynamics matrix

    """

    assert olap <= 1 and olap >= 0, "olap must be between 0 and 1"

    all_corr_matrix = []
    nt = len(ts[0]) # number of time points/ samples

    try:
        Nwindows = min(
            ((nt - wwidth * olap) // (wwidth * (1 - olap)), maxNwindows)
        )
        shift = int((nt - wwidth) // (Nwindows - 1))
        if Nwindows == maxNwindows:
            wwidth = int(shift // (1 - olap))

        indx_start = range(0, (nt - wwidth + 1), shift)
        indx_stop = range(wwidth, (1 + nt), shift)

        nnodes = ts.shape[0]

        for j1, j2 in zip(indx_start, indx_stop):
            aux_s = ts[:, j1:j2]
            corr_mat = np.corrcoef(aux_s)
            all_corr_matrix.append(corr_mat)

        corr_vectors = np.array(
            [allPm[np.tril_indices(nnodes, k=-1)] for allPm in all_corr_matrix]
        )
        CV_centered = corr_vectors - np.mean(corr_vectors, -1)[:, None]

        return np.corrcoef(CV_centered)

    except Exception as e:
        if verbose:
            print(e)
        return np.array([np.nan])


def set_attribute(key, value):
    def decorate_func(func):
        setattr(func, key, value)
        return func

    return decorate_func


def compute_time(signal, fs):
    """Creates the signal correspondent time array.

    Parameters
    ----------
    signal: nd-array
        Input from which the time is computed.
    fs: int
        Sampling Frequency

    Returns
    -------
    time : float list
        Signal time

    """

    return np.arange(0, len(signal)) / fs


def calculate_plv(data):
    n_channels, n_samples = data.shape

    analytic_signal = hilbert(data)
    phase_angles = np.angle(analytic_signal)
    plv_matrix = np.zeros((n_channels, n_channels))

    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            plv = np.abs(np.mean(np.exp(1j * (phase_angles[i] - phase_angles[j]))))
            plv_matrix[i, j] = plv
            plv_matrix[j, i] = plv

    return plv_matrix


def calc_fft(signal, fs):
    """This functions computes the fft of a signal.

    Parameters
    ----------
    signal : nd-array
        The input signal from which fft is computed
    fs : int
        Sampling frequency

    Returns
    -------
    f: nd-array
        Frequency values (xx axis)
    fmag: nd-array
        Amplitude of the frequency values (yy axis)

    """

    fmag = np.abs(np.fft.fft(signal))
    f = np.linspace(0, fs // 2, len(signal) // 2)

    return f[: len(signal) // 2].copy(), fmag[: len(signal) // 2].copy()


def filterbank(signal, fs, pre_emphasis=0.97, nfft=512, nfilt=40):
    """Computes the MEL-spaced filterbank.

    It provides the information about the power in each frequency band.

    Implementation details and description on:
    https://www.kaggle.com/ilyamich/mfcc-implementation-and-tutorial
    https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html#fnref:1

    Parameters
    ----------
    signal : nd-array
        Input from which filterbank is computed
    fs : int
        Sampling frequency
    pre_emphasis : float
        Pre-emphasis coefficient for pre-emphasis filter application
    nfft : int
        Number of points of fft
    nfilt : int
        Number of filters

    Returns
    -------
    nd-array
        MEL-spaced filterbank

    """

    # Signal is already a window from the original signal, so no frame is needed.
    # According to the references it is needed the application of a window function such as
    # hann window. However if the signal windows don't have overlap, we will lose information,
    # as the application of a hann window will overshadow the windows signal edges.

    # pre-emphasis filter to amplify the high frequencies

    emphasized_signal = np.append(
        np.array(signal)[0], np.array(signal[1:]) - pre_emphasis * np.array(signal[:-1])
    )

    # Fourier transform and Power spectrum
    mag_frames = np.absolute(
        np.fft.rfft(emphasized_signal, nfft)
    )  # Magnitude of the FFT

    pow_frames = (1.0 / nfft) * (mag_frames**2)  # Power Spectrum

    low_freq_mel = 0
    high_freq_mel = 2595 * np.log10(1 + (fs / 2) / 700)  # Convert Hz to Mel
    # Equally spaced in Mel scale
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)  # Convert Mel to Hz
    filter_bin = np.floor((nfft + 1) * hz_points / fs)

    fbank = np.zeros((nfilt, int(np.floor(nfft / 2 + 1))))
    for m in range(1, nfilt + 1):

        f_m_minus = int(filter_bin[m - 1])  # left
        f_m = int(filter_bin[m])  # center
        f_m_plus = int(filter_bin[m + 1])  # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - filter_bin[m - 1]) / (
                filter_bin[m] - filter_bin[m - 1]
            )
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (filter_bin[m + 1] - k) / (
                filter_bin[m + 1] - filter_bin[m]
            )

    # Area Normalization
    # If we don't normalize the noise will increase with frequency because of the filter width.
    enorm = 2.0 / (hz_points[2 : nfilt + 2] - hz_points[:nfilt])
    fbank *= enorm[:, np.newaxis]

    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(
        filter_banks == 0, np.finfo(float).eps, filter_banks
    )  # Numerical Stability
    filter_banks = 20 * np.log10(filter_banks)  # dB

    return filter_banks


def autocorr_norm(signal):
    """Computes the autocorrelation.

    Implementation details and description in:
    https://ccrma.stanford.edu/~orchi/Documents/speaker_recognition_report.pdf

    Parameters
    ----------
    signal : nd-array
        Input from linear prediction coefficients are computed

    Returns
    -------
    nd-array
        Autocorrelation result

    """

    variance = np.var(signal)
    signal = np.copy(signal - signal.mean())
    r = scipy.signal.correlate(signal, signal)[-len(signal) :]

    if (signal == 0).all():
        return np.zeros(len(signal))

    acf = r / variance / len(signal)

    return acf


def create_symmetric_matrix(acf, order=11):
    """Computes a symmetric matrix.

    Implementation details and description in:
    https://ccrma.stanford.edu/~orchi/Documents/speaker_recognition_report.pdf

    Parameters
    ----------
    acf : nd-array
        Input from which a symmetric matrix is computed
    order : int
        Order

    Returns
    -------
    nd-array
        Symmetric Matrix

    """

    smatrix = np.empty((order, order))
    xx = np.arange(order)
    j = np.tile(xx, order)
    i = np.repeat(xx, order)
    smatrix[i, j] = acf[np.abs(i - j)]

    return smatrix


def lpc(signal, n_coeff=12):
    """Computes the linear prediction coefficients.

    Implementation details and description in:
    https://ccrma.stanford.edu/~orchi/Documents/speaker_recognition_report.pdf

    Parameters
    ----------
    signal : nd-array
        Input from linear prediction coefficients are computed
    n_coeff : int
        Number of coefficients

    Returns
    -------
    nd-array
        Linear prediction coefficients

    """

    if signal.ndim > 1:
        raise ValueError("Only 1 dimensional arrays are valid")
    if n_coeff > signal.size:
        raise ValueError("Input signal must have a length >= n_coeff")

    # Calculate the order based on the number of coefficients
    order = n_coeff - 1

    # Calculate LPC with Yule-Walker
    acf = np.correlate(signal, signal, "full")

    r = np.zeros(order + 1, "float32")
    # Assuring that works for all type of input lengths
    nx = np.min([order + 1, len(signal)])
    r[:nx] = acf[len(signal) - 1 : len(signal) + order]

    smatrix = create_symmetric_matrix(r[:-1], order)

    if np.sum(smatrix) == 0:
        return tuple(np.zeros(order + 1))

    lpc_coeffs = np.dot(np.linalg.inv(smatrix), -r[1:])

    return tuple(np.concatenate(([1.0], lpc_coeffs)))


def create_xx(features):
    """Computes the range of features amplitude for the probability density function calculus.

    Parameters
    ----------
    features : nd-array
        Input features

    Returns
    -------
    nd-array
        range of features amplitude

    """

    features_ = np.copy(features)

    if max(features_) < 0:
        max_f = -max(features_)
        min_f = min(features_)
    else:
        min_f = min(features_)
        max_f = max(features_)

    if min(features_) == max(features_):
        xx = np.linspace(min_f, min_f + 10, len(features_))
    else:
        xx = np.linspace(min_f, max_f, len(features_))

    return xx


def kde(features):
    """Computes the probability density function of the input signal
       using a Gaussian KDE (Kernel Density Estimate)

    Parameters
    ----------
    features : nd-array
        Input from which probability density function is computed

    Returns
    -------
    nd-array
        probability density values

    """
    features_ = np.copy(features)
    xx = create_xx(features_)

    if min(features_) == max(features_):
        noise = np.random.randn(len(features_)) * 0.0001
        features_ = np.copy(features_ + noise)

    kernel = scipy.stats.gaussian_kde(features_, bw_method="silverman")

    return np.array(kernel(xx) / np.sum(kernel(xx)))


def gaussian(features):
    """Computes the probability density function of the input signal using a Gaussian function

    Parameters
    ----------
    features : nd-array
        Input from which probability density function is computed
    Returns
    -------
    nd-array
        probability density values

    """

    features_ = np.copy(features)

    xx = create_xx(features_)
    std_value = np.std(features_)
    mean_value = np.mean(features_)

    if std_value == 0:
        return 0.0
    pdf_gauss = scipy.stats.norm.pdf(xx, mean_value, std_value)

    return np.array(pdf_gauss / np.sum(pdf_gauss))


def calc_ecdf(signal):
    """Computes the ECDF of the signal.
     ECDF is the empirical cumulative distribution function.

    Parameters
    ----------
    signal : nd-array
        Input from which ECDF is computed
    Returns
    -------
    nd-array
      Sorted signal and computed ECDF.

    """
    return np.sort(signal), np.arange(1, len(signal) + 1) / len(signal)


def matrix_stat(
    A: np.ndarray,
    k: int = 1,
    eigenvalues: bool = True,
    pca_num_components: int = 3,
    quantiles: List[float] = [0.05, 0.25, 0.5, 0.75, 0.95],
    features: List[str] = ["sum", "max", "min", "mean", "std", "skew", "kurtosis"],
):
    """
    calculate statistics of the given matrix

    Parameters
    ----------
    A: np.ndarray (2d)
        input matrix
    k: int
        upper triangular matrix offset
    pca_num_components: int
        number of components to keep for PCA, set to 0 if not used
    features: list
        list of features to compute
    quantiles: list
        list of quantiles to compute, set to [] or None if not used

    Returns
    -------
    values: np.ndarray (1d)
        feature values
    labels: list
        feature labels

    """
    from numpy import sum, max, min, mean, std
    from scipy.stats import skew, kurtosis

    off_diag_sum_A = np.sum(np.abs(A)) - np.trace(np.abs(A))

    ut_idx = np.triu_indices_from(A, k=k)
    A_ut = A[ut_idx[0], ut_idx[1]]

    values = []
    labels = []
    if quantiles:
        q = np.quantile(A, quantiles)
        values.extend(q.tolist())
        labels.extend([f"quantile_{i}" for i in quantiles])

    if pca_num_components:
        try: 
            pca = PCA(n_components=pca_num_components)
            pca_a = pca.fit_transform(A)
        except:
            return [np.nan], ["pca_error"]
        
        for f in features:
            v = eval(f)(pca_a.reshape(-1))
            values.append(v)
            labels.append(f"pca_{f}")

    if eigenvalues:
        eigen_vals_A, _ = LA.eig(A)
        for f in features:
            v = eval(f)(np.real(eigen_vals_A[:-1]))
            values.append(v)
            labels.append(f"eig_{f}")

    for f in features:
        v = eval(f)(A_ut)
        values.append(v)
        labels.append(f"ut_{f}")

    values.append(off_diag_sum_A)
    labels.append("sum")

    return values, labels


def report_cfg(cfg: dict):
    """
    report the features in provided config file
    """

    print("Selected features:")
    print("------------------")

    for d in cfg:
        if d == "features_path":
            continue
        else:
            if cfg[d]:
                print("■ Domain:", d)
            for f in cfg[d]:
                print(" ▢ Function: ", f)
                print("   ▫ description: ", cfg[d][f]["description"])
                print("   ▫ function   : ", cfg[d][f]["function"])
                print("   ▫ parameters : ", cfg[d][f]["parameters"])
                print("   ▫ tag        : ", cfg[d][f]["tag"])
                print("   ▫ use        : ", cfg[d][f]["use"])


def get_jar_location():

    jar_file_name = "infodynamics.jar"
    jar_location = join(vbi.__file__, "feature_extraction")
    jar_location = jar_location.replace("__init__.py", "")
    jar_location = join(jar_location, jar_file_name)

    return jar_location


def init_jvm():

    jar_location = get_jar_location()

    if jp.isJVMStarted():
        return
    else:
        jp.startJVM(jp.getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jar_location)


def nat2bit(x):
    """
    convert nats to bits
    """
    return x * 1.4426950408889634


def compute_time(ts, fs):
    """Creates the signal correspondent time array.

    Parameters
    ----------
    signal: nd-array
        Input from which the time is computed.
    fs: int
        Sampling Frequency

    Returns
    -------
    time : float list
        Signal time

    """

    return np.arange(0, len(ts)) / fs


def calc_fft(ts, fs):
    """This functions computes the fft of a signal.

    Parameters
    ----------
    signal : nd-array [n_regions, n_timepoints]
        The input signal from which fft is computed
    fs : float
        Sampling frequency

    Returns
    -------
    f: nd-array
        Frequency values (xx axis)
    fmag: nd-array [n_regions, n_freqs]
        Amplitude of the frequency values (yy axis)

    """

    fmag = np.abs(np.fft.rfft(ts, axis=1))
    f = np.fft.rfftfreq(len(ts[0]), d=1 / fs)

    return f, fmag


def fundamental_frequency(f, fmag):
    """Computes fundamental frequency of the signal.

    The fundamental frequency integer multiple best explain
    the content of the signal spectrum.

    Feature computational cost: 1

    Parameters
    ----------
    ts : nd-array [n_regions x n_samples]
        Input from which fundamental frequency is computed
    fs : float
        Sampling frequency

    Returns
    -------
    f0: array of floats
       Predominant frequency of the signals

    """

    def one_dim(f, fmag):
        bp = scipy.signal.find_peaks(fmag, height=max(fmag) * 0.3)[0]

        # Condition for offset removal, since the offset generates a peak at frequency zero
        bp = bp[bp != 0]
        if not list(bp):
            f0 = 0
        else:
            # f0 is the minimum big peak frequency
            f0 = f[min(bp)]

        return f0

    r, c = fmag.shape
    f0 = np.zeros(r)
    for i in range(r):
        f0[i] = one_dim(f, fmag[i])
    labels = [f"fundamental_frequency_{i}" for i in range(len(f0))]
    return f0, labels


def spectral_distance(freq, fmag):
    """Computes the signal spectral distance.

    Distance of the signal's cumulative sum of the FFT elements to
    the respective linear regression.

    Parameters
    ----------
    fmag: nd-array [n_regions x n_freqs]
        power spectrum of the signal

    Returns
    -------
    values: array-like
        spectral distances
    labels: array-like
        labels of the features

    """

    r, c = fmag.shape
    values = np.zeros(r)
    cum_fmag = np.cumsum(fmag, axis=1)

    for i in range(r):
        points_y = np.linspace(0, cum_fmag[i], c)
        values[i] = np.sum(points_y - cum_fmag[i]) / c
    labels = [f"spectral_distance_{i}" for i in range(r)]
    return values, labels


def max_frequency(f, psd):
    """
    Computes the maximum frequency of the signals.
    
    parameters
    ----------
    f: nd-array
        frequency values
    psd: nd-array [n_regions x n_freqs]
        power spectral density of the signal
        
    Returns
    -------
    values: array-like
        maximum frequencies

    """
    if not isinstance(f, np.ndarray):
        f = np.array(f)
    if not isinstance(psd, np.ndarray):
        psd = np.array(psd)
    if psd.ndim == 1:
        psd = psd.reshape(1, -1)
        
    nn, nt = psd.shape
    fmax = np.zeros(nn)
    ind_max = np.argmax(psd, axis=1)
    fmax = f[ind_max]
    
    
    labels = [f"max_frequency_{i}" for i in range(len(fmax))]
    return fmax, labels

def max_psd(f, psd):
    """
    Computes the maximum power spectral density of the signals.
    
    Parameters
    ----------
    f: nd-array
        frequency values
    psd: nd-array [n_regions x n_freqs]
        power spectral density of the signal
        
    Returns
    -------
    values: array-like
        maximum power spectral densities
    """
    nn, nt = psd.shape
    if not isinstance(psd, np.ndarray):
        psd = np.array(psd)
    if psd.ndim == 1:
        psd = psd.reshape(1, -1)
        
    pmax = np.max(psd, axis=1)    
    labels = [f"max_psd_{i}" for i in range(len(pmax))]
    return pmax, labels


def median_frequency(f, fmag):
    """
    Computes the median frequency of the signals.

    """

    def one_d(cum_fmag):

        try:
            ind_mag = np.where(cum_fmag > cum_fmag[-1] * 0.5)[0][0]
        except:
            ind_mag = np.argmax(cum_fmag)
        return f[ind_mag]

    cum_fmag = np.cumsum(fmag, axis=1)
    # use map to apply one_d to each row of cum_fmag
    fmed = np.array(list(map(one_d, cum_fmag)))
    labels = [f"median_frequency_{i}" for i in range(len(fmed))]
    return fmed, labels


def spectral_centroid(f, fmag):
    """
    Calculate the spectral centroid of the signals.
    The Timbre Toolbox: Extracting audio descriptors from musicalsignals
    Authors Peeters G., Giordano B., Misdariis P., McAdams S.

    Parameters
    ----------
    f: nd-array
        frequency values
    fmag: nd-array [n_regions x n_freqs]
        power spectrum of the signal

    Returns
    -------
    values: array-like
        spectral centroids
    labels: array-like
        labels of the features

    """

    def one_d(f, fmag):
        if not np.sum(fmag):
            return 0
        else:
            return np.sum(f * fmag) / np.sum(fmag)

    # use map to apply one_d to each row of fmag
    values = np.array(list(map(one_d, f, fmag)))
    labels = [f"spectral_centroid_{i}" for i in range(len(values))]
    return values, labels


def spectral_kurtosis(f, fmag):
    """
    Measure the flatness of the power spectrum of the signals.
    The Timbre Toolbox: Extracting audio descriptors from musicalsignals
    Authors Peeters G., Giordano B., Misdariis P., McAdams S.

    Parameters
    ----------
    f: nd-array
        frequency values
    fmag: nd-array [n_regions x n_freqs]
        power spectrum of the signal

    Returns
    -------
    values: array-like
        spectral kurtosis
    labels: array-like
        labels of the features

    """

    spread = spectral_spread(f, fmag)[0]
    centroid = spectral_centroid(f, fmag)[0]
    values = np.zeros(len(spread))
    for i in range(len(spread)):
        if spread[i] == 0:
            values[i] = 0
        else:
            spect_kurt = ((f - centroid[i]) ** 4) * (fmag / np.sum(fmag))
            values[i] = np.sum(spect_kurt) / (spread[i] ** 4)
    labels = [f"spectral_kurtosis_{i}" for i in range(len(values))]

    return values, labels


def spectral_spread(f, fmag):
    """Measures the spread of the spectrum around its mean value.

    Description and formula in Article:
    The Timbre Toolbox: Extracting audio descriptors from musicalsignals
    Authors Peeters G., Giordano B., Misdariis P., McAdams S.

    Feature computational cost: 2

    Parameters
    ----------
    signal : nd-array
        Signal from which spectral spread is computed.
    fs : float
        Sampling frequency

    Returns
    -------
    float
        Spectral Spread

    """
    n = fmag.shape[0]
    centroid = spectral_centroid(f, fmag)[0]
    values = np.zeros(n)
    for i in range(n):
        if not np.sum(fmag[i]):
            values[i] = 0
        else:
            values[i] = (
                np.dot(((f - centroid[i]) ** 2), (fmag[i] / np.sum(fmag[i]))) ** 0.5
            )

    return values, [f"spectral_spread_{i}" for i in range(len(values))]


def spectral_variation(freq, fmag):
    """
    Computes the amount of variation of the spectrum along time.
    Spectral variation is computed from the normalized cross-correlation between two consecutive amplitude spectra.

    Description and formula in Article:
    The Timbre Toolbox: Extracting audio descriptors from musicalsignals
    Authors Peeters G., Giordano B., Misdariis P., McAdams S.
    """

    def one_d(sum1, sum2, sum3):

        if not sum2 or not sum3:
            return 1
        else:
            return 1 - (sum1 / ((sum2**0.5) * (sum3**0.5)))

    sum1 = np.sum(fmag[:, :-1] * fmag[:, 1:], axis=1)
    sum2 = np.sum(fmag[:, 1:] ** 2, axis=1)
    sum3 = np.sum(fmag[:, :-1] ** 2, axis=1)
    sums = np.array([sum1, sum2, sum3]).T

    n = fmag.shape[0]
    values = np.array(list(map(lambda x: one_d(*x), sums)))
    labels = [f"spectral_variation_{i}" for i in range(len(values))]
    return values, labels


def wavelet(signal, function=None, widths=np.arange(1, 10)):
    """Computes CWT (continuous wavelet transform) of the signal.

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
    nd-array
        The result of the CWT along the time axis
        matrix with size (len(widths),len(signal))

    """
    
    if function is None:
        function = scipy.signal.ricker

    if isinstance(function, str):
        function = eval(function)

    if isinstance(widths, str):
        widths = eval(widths)

    cwt = scipy.signal.cwt(signal, function, widths)

    return cwt


def km_order(ts, indices=None, avg=True):
    """
    Calculate the (local) Kuramoto order parameter (KOP) of the given time series

    Parameters
    ----------
    ts: np.ndarray (2d) [n_regions, n_timepoints]
        input array
    indices: list
        list of indices of the regions of interest
    avg: bool
        if True, average the KOP across time

    Returns
    -------
    values: np.ndarray (1d) or float
        feature values

    """

    if not isinstance(ts, np.ndarray):
        ts = np.array(ts)

    if ts.ndim == 1:
        raise ValueError("Input array must be 2d")

    if indices is None:
        indices = np.arange(ts.shape[0], dtype=int)

    if max(indices) >= ts.shape[0]:
        raise ValueError("Invalid indices")

    if not all(isinstance(i, (int, np.int64)) for i in indices):
        raise ValueError("Indices must be integers")

    if len(indices) < 2:
        raise ValueError("At least two indices are required")

    ts = ts[indices, :]

    nn, nt = ts.shape
    r = np.abs(np.sum(np.exp(1j * ts), axis=0) / nn)
    if avg:
        return np.mean(r)
    else:
        return r


def normalize_signal(ts, method="zscore"):
    """
    Normalize the input time series

    Parameters
    ----------
    ts: np.ndarray (2d) [n_regions, n_timepoints]
        input array
    method: str
        normalization method
    index: int
        index of the times point to normalize with respect to
        x = x / x[:, index]

    Returns
    -------
    ts: np.ndarray (2d) [n_regions, n_timepoints]
        normalized array

    """

    if not isinstance(ts, np.ndarray):
        ts = np.array(ts)
    if ts.ndim == 1:
        ts = ts.reshape(1, -1)

    if method == "zscore":
        ts = stats.zscore(ts, axis=1)

    elif method == "minmax":
        ts = (ts - np.min(ts, axis=1)[:, None]) / (
            np.max(ts, axis=1) - np.min(ts, axis=1)
        )[:, None]

    elif method == "mean":
        ts = (ts - np.mean(ts, axis=1)[:, None]) / np.std(ts, axis=1)[:, None]

    elif method == "max":
        ts = ts / np.max(ts, axis=1)[:, None]

    elif method == "none":
        pass

    else:
        raise ValueError("Invalid method")

    return ts


def state_duration(
    hmm_z: np.ndarray, n_states: int, avg: bool = True, tcut: int = 5, bins: int = 10
):
    """
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
    t_cut : int
        maximum duration of a state, default is 5
    bins : int
        number of bins for the histogram, default is 10

    Returns
    -------
    stat_vec : array-like
        The duration of each state

    """

    infered_state = hmm_z.astype(int)
    inferred_state_list, inffered_dur = ssm.util.rle(infered_state)

    inferred_dur_stack = []
    for s in range(n_states):
        inferred_dur_stack.append(inffered_dur[inferred_state_list == s])

    V = []
    for i in range(n_states):
        v, _ = np.histogram(inferred_dur_stack[i], bins=bins, range=(0, tcut))
        V.append(v)
    V = np.array(V)

    if avg:
        return V.mean(axis=0)
    else:
        return V.flatten()


# not used in the code
def set_attribute(key, value):
    def decorate_func(func):
        setattr(func, key, value)
        return func

    return decorate_func



def seizure_onset_indicator(ts: np.ndarray, thr:float=0.02):
    '''
    return the index of the onset of seizures
    '''
    
    if not isinstance(ts, np.ndarray):
        ts = np.array(ts)
    
    if ts.ndim == 1:
        ts = ts.reshape(1, -1)
    
    df = np.diff(ts, axis=1)
    onset_idx = np.argmax(df, axis=1)
    onset_amp = np.max(df, axis=1)
    onset_idx = np.where(onset_amp < thr, 0, onset_idx)
    # onset_amp = np.where(onset_amp < thr, 0, onset_amp)
    return onset_idx
