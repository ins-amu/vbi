import os
import time
import numpy as np

from rich import box
from rich.table import Table
from rich.console import Console

from os.path import join
from scipy.stats import gaussian_kde
from typing import Union

# Optional imports
from vbi.optional_deps import torch, require_optional, optional_import

import re
try :
    import nbformat
    import nbformat
    from nbconvert import PythonExporter
except:
    pass

try:
    from sbi.analysis.plot import _get_default_opts, _update, ensure_numpy
except ImportError:
    pass


def timer(func):
    """
    Decorator to measure elapsed time.

    Parameters
    ----------
    func : function
        Function to be decorated.

    Returns
    -------
    function
        Wrapped function that measures execution time.
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
    Display elapsed time in hours, minutes, seconds.

    Parameters
    ----------
    time : float
        Elapsed time in seconds.
    message : str, optional
        Optional message to display with the time. Default is empty string.
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
    """
    Utility class for loading sample datasets and connectivity matrices.
    
    This class provides convenient methods to load structural connectivity matrices,
    tract lengths, and BOLD signal data from the VBI dataset directory.
    
    Parameters
    ----------
    nn : int, optional
        Number of nodes/regions in the connectivity matrix. Default is 84.
        Supported values are typically 84 and 88.
    """
    def __init__(self, nn=84) -> None:
        """
        Initialize the LoadSample utility.
        
        Parameters
        ----------
        nn : int, optional
            Number of nodes/regions in the connectivity matrix. Default is 84.
        """
        self.root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.nn = nn

    def get_weights(self, normalize=True):
        """
        Load structural connectivity weights matrix.
        
        Parameters
        ----------
        normalize : bool, optional
            Whether to normalize the weights by the maximum value. Default is True.
            
        Returns
        -------
        np.ndarray
            Structural connectivity matrix of shape (nn, nn) with diagonal set to 0.
            Values are non-negative after removing negative entries.
        """
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
        """
        Load tract lengths matrix.
        
        Returns
        -------
        np.ndarray
            Tract lengths matrix of shape (nn, nn) containing the physical 
            distances between brain regions.
        """
        nn = self.nn
        tract_lenghts_name = join(
            self.root_dir, "vbi/dataset", f"connectivity_{nn}", "tract_lengths.txt"
        )
        tract_lengths = np.loadtxt(tract_lenghts_name)
        return tract_lengths

    def get_bold(self):
        """
        Load BOLD signal data.
        
        Returns
        -------
        np.ndarray
            BOLD signal data matrix of shape (nn, n_timepoints) containing
            the empirical BOLD time series for each brain region.
        """
        nn = self.nn
        bold_name = join(
            self.root_dir, "vbi", "dataset", f"connectivity_{nn}", "Bold.npz"
        )
        bold = np.load(bold_name)["Bold"]
        return bold.T


def get_limits(samples, limits=None):
    """
    Calculate or validate parameter limits for samples.
    
    This function computes the min/max limits for each parameter dimension
    across one or more sample arrays, or validates provided limits.
    
    Parameters
    ----------
    samples : np.ndarray or list of np.ndarray
        Sample array(s) of shape (n_samples, n_params) or list of such arrays.
        If PyTorch tensors, they will be converted to numpy arrays.
    limits : list or None, optional
        Predefined limits as [[min1, max1], [min2, max2], ...] for each parameter.
        If None or empty list, limits are computed from the data.
        If single limit pair provided, it will be broadcast to all parameters.
        
    Returns
    -------
    torch.Tensor
        Tensor of shape (n_params, 2) containing [min, max] for each parameter.
    """

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
    """
    Find the peaks (modes) of a posterior distribution using kernel density estimation.
    
    This function estimates the probability density of the posterior samples
    and identifies the locations of peak density for each parameter dimension.
    
    Parameters
    ----------
    samples : np.ndarray or torch.Tensor
        Posterior samples of shape (n_samples, n_params).
        If torch.Tensor, it will be converted to numpy array.
    return_dict : bool, optional
        If True, returns results as a dictionary with parameter labels as keys.
        If False, returns a simple list of peak values. Default is False.
    **kwargs
        Additional keyword arguments passed to the plotting/analysis functions.
        These may include 'labels' for parameter names.
        
    Returns
    -------
    list or dict
        If return_dict=False: List of peak values for each parameter.
        If return_dict=True: Dictionary with parameter labels as keys and 
        peak values as values.
    """

    opts = _get_default_opts()
    opts = _update(opts, kwargs)

    limits = get_limits(samples)
    samples = ensure_numpy(samples)
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
    prior_samples: "Union[torch.Tensor, np.ndarray]", 
    post_samples: "Union[torch.Tensor, np.ndarray]"
) -> "torch.Tensor":
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
        
    Raises
    ------
    ImportError
        If PyTorch is not installed.
    """
    
    if torch is None:
        raise ImportError(
            "PyTorch is required for posterior_shrinkage function. "
            "Please install PyTorch with: pip install torch"
        )

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
    true_theta: "Union[torch.Tensor, np.ndarray, float]", 
    post_samples: "Union[torch.Tensor, np.ndarray]"
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
    z : torch.Tensor [n_params]
        The z-score of the posterior distributions.
        
    Raises
    ------
    ImportError
        If PyTorch is not installed.
    """
    
    if torch is None:
        raise ImportError(
            "PyTorch is required for posterior_zscore function. "
            "Please install PyTorch with: pip install torch"
        )

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
    """Check required dependencies, print versions, and warn if unavailable."""
    console = Console()
    table = Table(title="Dependency Check", box=box.SIMPLE_HEAVY)
    table.add_column("Package", style="bold cyan")
    table.add_column("Version", style="bold green")
    table.add_column("Status", style="bold yellow")
    
    dependencies = [
        ("vbi", "vbi"),
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("matplotlib", "matplotlib"),
        ("sbi", "sbi"),
        ("torch", "torch"),
        ("cupy", "cupy")
    ]
    
    for name, module in dependencies:
        try:
            pkg = __import__(module)
            version = pkg.__version__
            status = "✅ Available"
        except ImportError:
            version = "-"
            status = "❌ Not Found"
        
        table.add_row(name, version, status)
    
    console.print(table)
    
    # Additional GPU checks
    try:
        import torch
        console.print(f"[bold blue]Torch GPU available:[/bold blue] {torch.cuda.is_available()}")
        console.print(f"[bold blue]Torch device count:[/bold blue] {torch.cuda.device_count()}")
        console.print(f"[bold blue]Torch CUDA version:[/bold blue] {torch.version.cuda}")  # Display CUDA version used by PyTorch
    except ImportError:
        pass

    try:
        import cupy
        console.print(f"[bold blue]CuPy GPU available:[/bold blue] {cupy.cuda.is_available()}")
        console.print(f"[bold blue]CuPy device count:[/bold blue] {cupy.cuda.runtime.getDeviceCount()}")
        info = get_cuda_info()
        if isinstance(info, dict):
            print(f"CUDA Version: {info['cuda_version']}")
            print(f"Device Name: {info['device_name']}")
            print(f"Total Memory: {info['total_memory']:.2f} GB")
            print(f"Compute Capability: {info['compute_capability']}")

    except ImportError:
        pass

    
    
def get_cuda_info():
    """
    Get CUDA version and device information using CuPy.
    
    Returns:
        dict: Dictionary containing CUDA version and device information
    """
    import cupy as cp
    
    try:
        # Get CUDA version
        cuda_version = cp.cuda.runtime.runtimeGetVersion()
        major = cuda_version // 1000
        minor = (cuda_version % 1000) // 10
        
        # Get device info
        device = cp.cuda.runtime.getDeviceProperties(0)
        
        return {
            'cuda_version': f"{major}.{minor}",
            'device_name': device['name'].decode(),
            'total_memory': device['totalGlobalMem'] / (1024**3),  # Convert to GB
            'compute_capability': f"{device['major']}.{device['minor']}"
        }
    except ImportError:
        return "CuPy is not installed"
    except Exception as e:
        return f"Error getting CUDA information: {str(e)}"



# def tests():
#     from vbi.tests.test_suite import tests
#     tests()
    