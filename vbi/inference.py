import torch
from vbi.utils import *
from sbi.inference import SNPE, SNLE, SNRE
from sbi.utils.user_input_checks import process_prior


class Inference(object):
    def __init__(self) -> None:
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
        sample from prior distribution

        Parameters
        ----------
        prior: ?
            prior distribution
        n: int
            number of samples

        Returns
        -------

        """
        if seed is not None:
            torch.manual_seed(seed)

        prior, _, _ = process_prior(prior)
        theta = prior.sample((n,))
        return theta

    @staticmethod
    def sample_posterior(xo, num_samples, posterior):
        """
        sample from the posterior using the given observation point.

        Parameters
        ----------
        x0: torch.tensor float32 (1, d)
            observation point
        num_samples: int
            number of samples
        posterior: ?
            posterior object

        Returns
        -------
        samples: torch.tensor float32 (num_samples, d)
            samples from the posterior

        """

        if not isinstance(xo, torch.Tensor):
            xo = torch.tensor(xo, dtype=torch.float32)
        if len(xo.shape) == 1:
            xo = xo[None, :]

        samples = posterior.sample((num_samples,), x=xo)
        return samples
