import vbi
import scipy
import torch
import numpy as np
from typing import Union
import scipy.stats as stats
from scipy.signal import butter, detrend, filtfilt, hilbert
from vbi.feature_extraction.features_settings import load_json


def slice_features(x:Union[np.ndarray,torch.Tensor], feature_names:list, info:dict):
    """
    Slice features from a feature list
    
    Parameters
    ----------
    x: array-like
    features: list of strings 
        list of features
    info: dict 
        features's colum indices in x 

    Returns
    -------
    x_: array-like
        sliced features
    """
    if isinstance(x, (list, tuple)):
        x = np.array(x)

    if x.ndim == 1:
        x = x.reshape(1, -1)

    is_tensor = isinstance(x, torch.Tensor)
    if is_tensor:
        x_ = torch.Tensor([])
    else:
        x_ = np.array([])

    if len(feature_names) == 0:
        return x_

    for f_name in feature_names:
        if f_name in info:
            coli, colf = info[f_name]['index'][0], info[f_name]['index'][1]
            if is_tensor:
                x_ = torch.cat((x_, x[:, coli:colf]), dim=1)
            else:
                if x_.size == 0:
                    x_ = x[:, coli:colf]
                else:
                    x_ = np.concatenate((x_, x[:, coli:colf]), axis=1)
        else:
            raise ValueError(f"{f_name} not in info")

    return x_



def preprocess(ts, fs=None, preprocess_dict={}, **kwargs):
    '''
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


    '''

    if not preprocess_dict:
        preprocess_dict = load_json(
            vbi.__path__[0] + '/feature_extraction/preprocess.json')

    if preprocess_dict['zscores']['use'] == 'yes':
        ts = stats.zscore(ts, axis=1)
    if preprocess_dict['offset']['use'] == 'yes':
        value = preprocess_dict['offset']['parameters']['value']
        ts = ts[:, value:]

    if preprocess_dict['demean']['use'] == 'yes':
        ts = ts - np.mean(ts, axis=1)[:, None]

    if preprocess_dict['detrend']['use'] == 'yes':
        ts = detrend(ts, axis=1)

    if preprocess_dict['filter']['use'] == 'yes':
        low_cut = preprocess_dict['filter']['parameters']['low']
        high_cut = preprocess_dict['filter']['parameters']['high']
        order = preprocess_dict['filter']['parameters']['order']
        TR = 1.0/fs
        ts = band_pass_filter(ts,
                              k=order,
                              TR=TR,
                              low_cut=low_cut,
                              high_cut=high_cut)

    if preprocess_dict['remove_strong_artefacts']['use'] == 'yes':
        ts = remove_strong_artefacts(ts)

    return ts


def band_pass_filter(ts, low_cut=0.02, high_cut=0.1, TR=2.0, order=2):
    '''
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


    '''

    assert (np.isnan(ts).any() == False)

    fnq = 1./(2.0*TR)              # Nyquist frequency
    Wn = [low_cut/fnq, high_cut/fnq]
    bfilt, afilt = butter(order, Wn, btype='band')
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
    mask = np.zeros((n, n))
    mask[np.ix_(indices, indices)] = 1
    return mask


def get_fc(ts):
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
    FC = np.corrcoef(ts)
    FC = FC * (FC > 0)
    FC = FC - np.diag(np.diagonal(FC))
    return FC


def get_fcd(ts, win_len=30, win_sp=1, indices=[]):
    """
    Compute dynamic functional connectivity.

    Parameters:
    ----------

    ts: numpy.ndarray [n_regions, n_timepoints]
        Input signal
    win_len: int
        sliding window length in samples, default is 30
    win_sp: int
        sliding window step in samples, default is 1
    indices : array_like
        indices of regions to be masked

    Returns:
    -------
        FCD: ndarray
            matrix of functional connectivity dynamics    
    """
    if isinstance(ts, (list, tuple)):
        ts = np.array(ts)

    if len(indices) > 1:
        nn = ts.shape[0]
        mask = np.zeros((nn, nn))
        mask[np.ix_(indices, indices)] = 1


    ts = ts.T
    n_samples, n_nodes = ts.shape
    # returns the indices for upper triangle
    fc_triu_ids = np.triu_indices(n_nodes, 1)
    n_fcd = len(fc_triu_ids[0])
    fc_stack = []
    speed_stack = []
    fc_prev = []

    for t0 in range(0, ts.shape[0]-win_len, win_sp):
        t1 = t0+win_len
        fc = np.corrcoef(ts[t0:t1, :].T)
        if len(indices) > 1:
            fc = fc * mask # fc*(fc > 0)*(mask)
        fc = fc[fc_triu_ids]
        fc_stack.append(fc)
        if t0 > 0:
            corr_fcd = np.corrcoef([fc, fc_prev])[0, 1]
            speed_fcd = 1-corr_fcd
            speed_stack.append(speed_fcd)
            fc_prev = fc
        else:
            fc_prev = fc

    fcs = np.array(fc_stack)
    fcd = np.corrcoef(fcs)
    return fcd


def get_fcd2(ts, wwidth, maxNwindows, olap, indices=[], verbose=False):
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

 
    assert(olap <= 1 and olap >= 0), 'olap must be between 0 and 1'

    all_corr_matrix = []
    lenseries = len(ts[0])

    try:
        Nwindows = min(((lenseries-wwidth*olap) //
                        (wwidth*(1-olap)), maxNwindows))
        shift = int((lenseries-wwidth)//(Nwindows-1))
        if Nwindows == maxNwindows:
            wwidth = int(shift//(1-olap))

        indx_start = range(0, (lenseries-wwidth+1), shift)
        indx_stop = range(wwidth, (1+lenseries), shift)

        nnodes = ts.shape[0]

        for j1, j2 in zip(indx_start, indx_stop):
            aux_s = ts[:, j1:j2]
            corr_mat = np.corrcoef(aux_s)                
            all_corr_matrix.append(corr_mat)

        corr_vectors = np.array([allPm[np.tril_indices(nnodes, k=-1)]
                                for allPm in all_corr_matrix])
        CV_centered = corr_vectors - np.mean(corr_vectors, -1)[:, None]

        return np.corrcoef(CV_centered)

    except Exception as e:
        if verbose:
            print(e)
        return np.array([np.nan])


def set_domain(key, value):
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

    return np.arange(0, len(signal))/fs


def calc_fft(signal, fs):
    """ This functions computes the fft of a signal.

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

    return f[:len(signal) // 2].copy(), fmag[:len(signal) // 2].copy()


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

    emphasized_signal = np.append(np.array(signal)[0], np.array(
        signal[1:]) - pre_emphasis * np.array(signal[:-1]))

    # Fourier transform and Power spectrum
    mag_frames = np.absolute(np.fft.rfft(
        emphasized_signal, nfft))  # Magnitude of the FFT

    pow_frames = ((1.0 / nfft) * (mag_frames ** 2))  # Power Spectrum

    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (fs / 2) / 700))  # Convert Hz to Mel
    # Equally spaced in Mel scale
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)
    hz_points = (700 * (10 ** (mel_points / 2595) - 1))  # Convert Mel to Hz
    filter_bin = np.floor((nfft + 1) * hz_points / fs)

    fbank = np.zeros((nfilt, int(np.floor(nfft / 2 + 1))))
    for m in range(1, nfilt + 1):

        f_m_minus = int(filter_bin[m - 1])  # left
        f_m = int(filter_bin[m])  # center
        f_m_plus = int(filter_bin[m + 1])  # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - filter_bin[m - 1]) / \
                (filter_bin[m] - filter_bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (filter_bin[m + 1] - k) / \
                (filter_bin[m + 1] - filter_bin[m])

    # Area Normalization
    # If we don't normalize the noise will increase with frequency because of the filter width.
    enorm = 2.0 / (hz_points[2:nfilt + 2] - hz_points[:nfilt])
    fbank *= enorm[:, np.newaxis]

    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(
        float).eps, filter_banks)  # Numerical Stability
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
    r = scipy.signal.correlate(signal, signal)[-len(signal):]

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
    acf = np.correlate(signal, signal, 'full')

    r = np.zeros(order+1, 'float32')
    # Assuring that works for all type of input lengths
    nx = np.min([order+1, len(signal)])
    r[:nx] = acf[len(signal)-1:len(signal)+order]

    smatrix = create_symmetric_matrix(r[:-1], order)

    if np.sum(smatrix) == 0:
        return tuple(np.zeros(order+1))

    lpc_coeffs = np.dot(np.linalg.inv(smatrix), -r[1:])

    return tuple(np.concatenate(([1.], lpc_coeffs)))


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
        max_f = - max(features_)
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

    kernel = scipy.stats.gaussian_kde(features_, bw_method='silverman')

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


def wavelet(signal, function=scipy.signal.ricker, widths=np.arange(1, 10)):
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

    if isinstance(function, str):
        function = eval(function)

    if isinstance(widths, str):
        widths = eval(widths)

    cwt = scipy.signal.cwt(signal, function, widths)

    return cwt


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
    return np.sort(signal), np.arange(1, len(signal)+1)/len(signal)
