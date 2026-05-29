"""Learned embedding / summary networks for vbi.inference."""
from __future__ import annotations

import autograd.numpy as anp


def _tanh(z):
    """
    tanh dispatched to the right backend based on the array type.

    Inside jax.jit/jax.grad the intermediate arrays are JAX tracers
    (type.__module__ starts with 'jax').  autograd.grad produces autograd
    Boxes (type.__module__ == 'autograd.tracer').  Everything else is plain
    numpy and uses autograd.numpy.tanh (= numpy.tanh outside grad).
    """
    mod = type(z).__module__
    if mod.startswith("jax"):
        import jax.numpy as jnp
        return jnp.tanh(z)
    return anp.tanh(z)


class EmbeddingNet:
    """
    Learnable MLP that compresses high-dimensional observations to a
    low-dimensional summary before conditioning the density estimator.

    Weights are included in the estimator's weight dict and trained
    end-to-end jointly with the flow / mixture.

    Parameters
    ----------
    input_dim    : int               Raw feature / observation dimensionality.
    output_dim   : int               Embedding (summary) dimensionality.
    hidden_sizes : tuple[int, ...]   Hidden layer widths.

    Examples
    --------
    >>> from vbi.inference import SNPE, BoxUniform, EmbeddingNet
    >>> emb = EmbeddingNet(input_dim=50, output_dim=8)
    >>> inf = SNPE(prior=prior, density_estimator='maf', embedding_net=emb)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_sizes: tuple[int, ...] = (64,),
    ):
        self.input_dim   = input_dim
        self.output_dim  = output_dim
        self.hidden_sizes = hidden_sizes

    def init_weights(self, rng) -> dict:
        """Return initial weight dict; all keys are prefixed with 'emb_'."""
        w    = {}
        in_d = self.input_dim
        for i, h in enumerate(self.hidden_sizes):
            scale = (2.0 / in_d) ** 0.5
            w[f"emb_W{i}"] = (rng.randn(in_d, h) * scale).astype("f")
            w[f"emb_b{i}"] = anp.zeros(h, "f")
            in_d = h
        scale = (2.0 / in_d) ** 0.5
        w["emb_Wout"] = (rng.randn(in_d, self.output_dim) * scale).astype("f")
        w["emb_bout"] = anp.zeros(self.output_dim, "f")
        return w

    def forward(self, weights: dict, x):
        """
        Apply the embedding MLP.

        Parameters
        ----------
        weights : dict   Must contain 'emb_W0', 'emb_b0', ..., 'emb_Wout', 'emb_bout'.
        x       : array  (n, input_dim)

        Returns
        -------
        ndarray  (n, output_dim)
        """
        mod = type(x).__module__
        if mod.startswith("jax"):
            import jax.numpy as jnp
            h = jnp.asarray(x, dtype="f")
        else:
            h = anp.asarray(x, dtype="f")
        for i in range(len(self.hidden_sizes)):
            h = _tanh(h @ weights[f"emb_W{i}"] + weights[f"emb_b{i}"])
        return h @ weights["emb_Wout"] + weights["emb_bout"]

    def __repr__(self):
        return (f"EmbeddingNet(input_dim={self.input_dim}, "
                f"output_dim={self.output_dim}, "
                f"hidden_sizes={self.hidden_sizes})")
