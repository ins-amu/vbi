import torch
from vbi.utils import *
from sbi.inference import SNPE, SNLE, SNRE
from sbi.utils.user_input_checks import process_prior

class Inference(object):
    def __init__(self) -> None:
        pass

    @timer
    def train(self,
              theta,
              x,
              prior,
              num_threads=1,
              method="SNPE",
              device="cpu",
              density_estimator="maf"
              ):

        torch.set_num_threads(num_threads)

        if (len(x.shape) == 1):
            x = x[:, None]
        if (len(theta.shape) == 1):
            theta = theta[:, None]

        if method == "SNPE":
            inference = SNPE(
                prior=prior, density_estimator=density_estimator, device=device)
        elif method == "SNLE":
            inference = SNLE(
                prior=prior, density_estimator=density_estimator, device=device)
        elif method == "SNRE":
            inference = SNRE(
                prior=prior, density_estimator=density_estimator, device=device)
        else:
            raise ValueError("Invalid method: " + method)

        inference = inference.append_simulations(theta, x)
        estimator_ = inference.train()
        posterior = inference.build_posterior(estimator_)

        return posterior

    @staticmethod
    def sample_prior(prior, n, seed=None):
        '''
        sample from prior distribution

        Parameters
        ----------
        prior: ?
            prior distribution
        n: int
            number of samples

        Returns
        -------

        '''
        if seed is not None:
            torch.manual_seed(seed)
            
        prior, _, _ = process_prior(prior)
        theta = prior.sample((n,))
        return theta

    @staticmethod
    def sample_posterior(xo,
                         num_samples,
                         posterior):
        '''
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

        '''

        if not isinstance(xo, torch.Tensor):
            xo = torch.tensor(xo, dtype=torch.float32)
        if len(xo.shape) == 1:
            xo = xo[None, :]

        samples = posterior.sample((num_samples,), x=xo)
        return samples
