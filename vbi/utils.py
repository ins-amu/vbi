import os
import time
import torch
import numpy as np
from os.path import join
from scipy.stats import gaussian_kde
from sbi.analysis.plot import _get_default_opts, _update, ensure_numpy

from torch import Tensor
from typing import Union


import re
try :
    import nbformat
    from nbconvert import PythonExporter
except:
    pass


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
            self.root_dir, "vbi/dataset", f"connectivity_{nn}", "tract_lengths.txt"
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


def p2j(modulePath):
    """convert python script to jupyter notebook"""
    os.system(f"p2j -o {modulePath}")


def j2p(notebookPath, modulePath=None):
    """
    convert a jupyter notebook to a python module

    >>> j2p("sample.ipynb", "sample.py")

    """

    with open(notebookPath) as fh:
        nb = nbformat.reads(fh.read(), nbformat.NO_CONVERT)

    exporter = PythonExporter()
    source, meta = exporter.from_notebook_node(nb)

    # remove lines start with `# In[` from source
    source = re.sub(r"^# In\[[0-9 ]*\]:\n", "", source, flags=re.MULTILINE)

    # replace more that 1 empty lines with 1 empty line
    source = re.sub(r"\n{2,}", "\n\n", source)

    if modulePath is None:
        modulePath = notebookPath.replace(".ipynb", ".py")

    with open(modulePath, "w+") as fh:
        fh.writelines(source)


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


def set_diag(A: np.ndarray, k: int = 0, value: float = 0.0):
    """
    set k diagonals of the given matrix to given value.
    
    Parameters
    ----------
    A: np.ndarray
        matrix
    k: int
        number of diagonals
    value: float
        value to be set
    
    Returns
    -------
    A: np.ndarray
        matrix with k diagonals set to value
    
    """

    assert len(A.shape) == 2
    n = A.shape[0]
    assert k < n
    for i in range(-k, k + 1):
        a1 = np.diag(np.random.randint(1, 2, n - abs(i)), i)
        idx = np.where(a1)
        A[idx] = value
    return A


def test_imports():
    """ 
    Import some of required dependencies and print their versions if available 
    and warn if not available.
    
    The required dependencies are:
    - vbi
    - numpy
    - scipy
    - matplotlib
    - sbi
    - torch [also check if GPU is available]
    - cupy [also check if GPU is available]
    
    """
    
    try:
        import vbi
        print(f"vbi: {vbi.__version__}")
    except ImportError:
        print("vbi not found")
    
    try:
        import numpy
        print(f"numpy: {numpy.__version__}")
    except ImportError:
        print("numpy not found")
        
    try:
        import scipy
        print(f"scipy: {scipy.__version__}")
    except ImportError:
        print("scipy not found")
        
    try:
        import matplotlib
        print(f"matplotlib: {matplotlib.__version__}")
    except ImportError:
        print("matplotlib not found")
        
    try:
        import sbi
        print(f"sbi: {sbi.__version__}")
    except ImportError:
        print("sbi not found")
        
    try:
        import torch
        print(f"torch: {torch.__version__}")
        print(f"GPU available: {torch.cuda.is_available()=}")
    except ImportError:
        print("torch not found")
        
    try:
        import cupy
        print(f"cupy: {cupy.__version__}")
        print(f"GPU available: {cupy.cuda.is_available()=}")
    except ImportError:
        print("cupy not found")
        
        
    
    