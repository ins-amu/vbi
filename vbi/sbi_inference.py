# Check for required dependencies with informative error messages
try:
    import torch
except ImportError as e:
    raise ImportError(
        "PyTorch is required for inference functionality but is not available. "
        "You may have installed VBI with the light version (pip install vbi) "
        "which excludes heavy inference dependencies to reduce installation size. "
        "To enable inference capabilities, install with: pip install vbi[inference] "
        "or add PyTorch manually: pip install torch"
    ) from e

try:
    from sbi.inference import SNPE, SNLE, SNRE
    from sbi.utils.user_input_checks import process_prior
except ImportError as e:
    raise ImportError(
        "SBI (Simulation-Based Inference) is required for inference functionality but is not available. "
        "You may have installed VBI with the light version (pip install vbi) "
        "which excludes heavy inference dependencies to reduce installation size. "
        "To enable inference capabilities, install with: pip install vbi[inference] "
        "or add SBI manually: pip install sbi"
    ) from e

from vbi.utils import *


class Inference(object):
    """
    Main inference class for simulation-based inference (SBI) using the sbi library.
    
    This class provides methods for training neural posterior/likelihood estimators
    and sampling from posterior distributions using various SBI methods including
    SNPE (Sequential Neural Posterior Estimation), SNLE (Sequential Neural Likelihood
    Estimation), and SNRE (Sequential Neural Ratio Estimation).
    
    The class acts as a wrapper around the sbi library, providing a convenient
    interface for training and inference tasks in the VBI (Variational Bayesian
    Inference) framework.
    """
    def __init__(self) -> None:
        """
        Initialize the Inference object.
        
        No parameters are required for initialization. The object serves as a
        container for static and instance methods for training and sampling.
        """
        pass

    @timer
    def train(
        self,
        theta,
        x,
        prior,
        num_threads=1,
        method="SNPE",
        device="cpu",
        density_estimator="maf",
    ):
        '''
        train the inference model
        
        Parameters
        ----------
        theta: torch.tensor float32 (n, d)
            parameter samples, where n is the number of samples and d is the dimension of the parameter space
        x: torch.tensor float32 (n, d)
            feature samples, where n is the number of samples and d is the dimension of the feature space
        prior: sbi.utils object 
            prior distribution object
        num_threads: int
            number of threads to use for training, for multi-threading support, default is 1
        method: str
            inference method to use, one of "SNPE", "SNLE", "SNRE", default is "SNPE"
        device: str
            device to use for training, one of "cpu", "cuda", default is "cpu"
        density_estimator: str
            density estimator to use, one of "maf", "nsf", default is "maf"
        Returns
        -------
        posterior: sbi.utils object
            posterior distribution object trained on the given data
            
        '''

        torch.set_num_threads(num_threads)

        if len(x.shape) == 1:
            x = x[:, None]
        if len(theta.shape) == 1:
            theta = theta[:, None]

        if method == "SNPE":
            inference = SNPE(
                prior=prior, density_estimator=density_estimator, device=device
            )
        elif method == "SNLE":
            inference = SNLE(
                prior=prior, density_estimator=density_estimator, device=device
            )
        elif method == "SNRE":
            inference = SNRE(
                prior=prior, density_estimator=density_estimator, device=device
            )
        else:
            raise ValueError("Invalid method: " + method)

        inference = inference.append_simulations(theta, x)
        estimator_ = inference.train()
        posterior = inference.build_posterior(estimator_)

        return posterior

    @staticmethod
    def sample_prior(prior, n, seed=None):
        """
        Sample parameter values from the prior distribution.

        Parameters
        ----------
        prior : sbi.utils.BoxUniform or similar
            Prior distribution object from the sbi library. Must be compatible
            with sbi's process_prior function.
        n : int
            Number of samples to draw from the prior distribution.
        seed : int, optional
            Random seed for reproducible sampling. If None, no seed is set.

        Returns
        -------
        torch.Tensor
            Tensor of shape (n, n_params) containing parameter samples drawn
            from the prior distribution.
        """
        if seed is not None:
            torch.manual_seed(seed)

        prior, _, _ = process_prior(prior)
        theta = prior.sample((n,))
        return theta

    @staticmethod
    def sample_posterior(xo, num_samples, posterior):
        """
        Sample parameter values from the trained posterior distribution.

        Parameters
        ----------
        xo : torch.Tensor
            Observed data point of shape (1, n_features) or (n_features,).
            This is the target observation for which we want to infer parameters.
        num_samples : int
            Number of posterior samples to draw.
        posterior : sbi posterior object
            Trained posterior distribution object returned by the train() method.
            Must have a sample() method compatible with sbi posteriors.

        Returns
        -------
        torch.Tensor
            Tensor of shape (num_samples, n_params) containing parameter samples
            drawn from the posterior distribution conditioned on the observation xo.
        """
        if not isinstance(xo, torch.Tensor):
            xo = torch.tensor(xo, dtype=torch.float32)
        if len(xo.shape) == 1:
            xo = xo[None, :]

        samples = posterior.sample((num_samples,), x=xo)
        return samples
