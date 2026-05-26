"""Learned embedding / summary networks for vbi.inference."""
from __future__ import annotations

import autograd.numpy as anp


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

    def forward(self, weights: dict, x) -> anp.ndarray:
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
        h = anp.asarray(x, dtype="f")
        for i in range(len(self.hidden_sizes)):
            h = anp.tanh(anp.dot(h, weights[f"emb_W{i}"]) + weights[f"emb_b{i}"])
        return anp.dot(h, weights["emb_Wout"]) + weights["emb_bout"]

    def __repr__(self):
        return (f"EmbeddingNet(input_dim={self.input_dim}, "
                f"output_dim={self.output_dim}, "
                f"hidden_sizes={self.hidden_sizes})")
