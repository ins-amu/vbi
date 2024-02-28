import os
import time
import torch
import numpy as np
from os.path import join
from scipy.stats import gaussian_kde
from sbi.analysis.plot import _get_default_opts, _update, ensure_numpy


def timer(func):
    """
    decorator to measure elapsed time

    Parameters
    -----------
    func: function
        function to be decorated
    """

    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        display_time(end - start, message="{:s}".format(func.__name__))
        return result

    return wrapper


def display_time(time, message=""):
    """
    display elapsed time in hours, minutes, seconds

    Parameters
    -----------
    time: float
        elaspsed time in seconds
    """

    hour = int(time / 3600)
    minute = (int(time % 3600)) // 60
    second = time - (3600.0 * hour + 60.0 * minute)
    print(
        "{:s} Done in {:d} hours {:d} minutes {:09.6f} seconds".format(
            message, hour, minute, second
        )
    )


class LoadSample(object):
    def __init__(self, nn=84) -> None:

        self.root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.nn = nn

    def get_weights(self, normalize=True):
        nn = self.nn
        SC_name = join(
            self.root_dir, "vbi/dataset", f"connectivity_{nn}", "weights.txt"
        )
        SC = np.loadtxt(SC_name)
        np.fill_diagonal(SC, 0.0)
        if normalize:
            SC /= SC.max()
        SC[SC < 0] = 0.0
        return SC

    def get_lengths(self):
        nn = self.nn
        tract_lenghts_name = join(
            self.root_dir, "dataset", f"connectivity_{nn}", "tract_lengths.txt"
        )
        tract_lengths = np.loadtxt(tract_lenghts_name)
        return tract_lengths

    def get_bold(self):
        nn = self.nn
        bold_name = join(
            self.root_dir, "vbi", "dataset", f"connectivity_{nn}", "Bold.npz"
        )
        bold = np.load(bold_name)["Bold"]
        return bold.T


def get_limits(samples, limits=None):

    if type(samples) != list:
        samples = ensure_numpy(samples)
        samples = [samples]
    else:
        for i, sample_pack in enumerate(samples):
            samples[i] = ensure_numpy(samples[i])

    # Dimensionality of the problem.
    dim = samples[0].shape[1]

    if limits == [] or limits is None:
        limits = []
        for d in range(dim):
            min = +np.inf
            max = -np.inf
            for sample in samples:
                min_ = sample[:, d].min()
                min = min_ if min_ < min else min
                max_ = sample[:, d].max()
                max = max_ if max_ > max else max
            limits.append([min, max])
    else:
        if len(limits) == 1:
            limits = [limits[0] for _ in range(dim)]
        else:
            limits = limits
    limits = torch.as_tensor(limits)

    return limits


def posterior_peaks(samples, return_dict=False, **kwargs):

    opts = _get_default_opts()
    opts = _update(opts, kwargs)

    limits = get_limits(samples)
    samples = samples.numpy()
    n, dim = samples.shape

    try:
        labels = opts["labels"]
    except:
        labels = range(dim)

    peaks = {}
    if labels is None:
        labels = range(dim)
    for i in range(dim):
        peaks[labels[i]] = 0

    for row in range(dim):
        density = gaussian_kde(samples[:, row], bw_method=opts["kde_diag"]["bw_method"])
        xs = np.linspace(limits[row, 0], limits[row, 1], opts["kde_diag"]["bins"])
        ys = density(xs)

        # y, x = np.histogram(samples[:, row], bins=bins)
        peaks[labels[row]] = xs[ys.argmax()]

    if return_dict:
        return peaks
    else:
        return list(peaks.values())
