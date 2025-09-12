"""
A module for conditional density estimation using Mixture Density
Networks (MDNs) and Masked Autoregressive Flows (MAFs).

Original Source:
    This file is based on code from the tvbl project:
    https://github.com/maedoc/tvbl/blob/main/content/cde.py

    Original Author: Marmaduke Woodman (maedoc)

Notes:
    - Adapted and modified for use in the VBI (Virtual Brain Inference) project
    - May include additional features, modifications, or optimizations

"""

import abc
import math
from dataclasses import dataclass, field

import autograd.numpy as anp
from autograd import grad
from autograd.scipy.special import logsumexp
from sklearn.datasets import make_moons
from scipy.stats import t
from tqdm.auto import trange

# =============================================================================
# == Base Class for Conditional Density Estimators
# =============================================================================


class ConditionalDensityEstimator(abc.ABC):
    """
    Abstract base class for conditional density estimators.

    This class provides a unified training interface using the Adam optimizer
    and standardizes the API for training, sampling, and log-probability
    evaluation.

    Parameters
    ----------
    param_dim : int, optional
        Dimensionality of the target variable (parameters to be estimated).
        If None, will be inferred from training data.
    feature_dim : int, optional
        Dimensionality of the conditional variable (features).
        If None, will be inferred from training data.
    """

    def __init__(self, param_dim: int = None, feature_dim: int = None):
        self.param_dim = param_dim
        self.feature_dim = feature_dim
        self._dims_inferred = False
        self.weights = None
        self.loss_history = []

    def _infer_dimensions(self, params: anp.ndarray, features: anp.ndarray):
        """
        Infer parameter and feature dimensions from training data.

        Parameters
        ----------
        params : anp.ndarray
            Parameter array of shape (N, param_dim)
        features : anp.ndarray
            Feature array of shape (N, feature_dim)
        """
        # Convert to arrays and ensure 2D
        params = anp.asarray(params)
        features = anp.asarray(features)

        if params.ndim == 1:
            params = params.reshape(-1, 1)
        if features.ndim == 1:
            features = features.reshape(-1, 1)

        inferred_param_dim = params.shape[1]
        inferred_feature_dim = features.shape[1]

        # Check if user-provided dimensions match inferred ones
        if self.param_dim is not None and self.param_dim != inferred_param_dim:
            print(
                f"Warning: Provided param_dim ({self.param_dim}) doesn't match data ({inferred_param_dim}). Using data dimensions."
            )

        if self.feature_dim is not None and self.feature_dim != inferred_feature_dim:
            print(
                f"Warning: Provided feature_dim ({self.feature_dim}) doesn't match data ({inferred_feature_dim}). Using data dimensions."
            )

        # Set the inferred dimensions
        self.param_dim = inferred_param_dim
        self.feature_dim = inferred_feature_dim
        self._dims_inferred = True

        print(
            f"Inferred dimensions: param_dim={self.param_dim}, feature_dim={self.feature_dim}"
        )

        # Validate inferred dimensions
        if self.param_dim <= 0:
            raise ValueError(
                f"Inferred param_dim must be positive, got {self.param_dim}"
            )
        if self.feature_dim < 0:
            raise ValueError(
                f"Inferred feature_dim must be non-negative, got {self.feature_dim}"
            )

    @abc.abstractmethod
    def _initialize_weights(self, rng: anp.random.RandomState) -> dict:
        """
        Initialize the trainable weights of the model.

        Parameters
        ----------
        rng : autograd.numpy.random.RandomState
            A random number generator for reproducible initialization.

        Returns
        -------
        dict
            A dictionary of initialized weight arrays.
        """
        pass

    @abc.abstractmethod
    def _loss_function(
        self, weights: dict, features: anp.ndarray, params: anp.ndarray
    ) -> float:
        """
        Compute the negative log-likelihood loss for a batch of data.

        Parameters
        ----------
        weights : dict
            A dictionary of the model's trainable weights.
        features : anp.ndarray
            A (N, feature_dim) array of conditional features.
        params : anp.ndarray
            A (N, param_dim) array of target parameters.

        Returns
        -------
        float
            The mean negative log-likelihood of the batch.
        """
        pass

    @abc.abstractmethod
    def sample(
        self, features: anp.ndarray, n_samples: int, rng: anp.random.RandomState
    ) -> anp.ndarray:
        """
        Generate samples from the learned conditional distribution p(params|features).

        Parameters
        ----------
        features : anp.ndarray
            A (n_conditions, feature_dim) array of features to condition on.
        n_samples : int
            The number of samples to generate for each condition.
        rng : autograd.numpy.random.RandomState
            A random number generator for sampling.

        Returns
        -------
        anp.ndarray
            An array of generated samples of shape (n_conditions, n_samples, param_dim).
        """
        if self.weights is None:
            raise RuntimeError("Model has not been trained yet. Call train() first.")

        if not self._dims_inferred:
            raise RuntimeError("Model dimensions not inferred yet. Call train() first.")

    @abc.abstractmethod
    def log_prob(self, features: anp.ndarray, params: anp.ndarray) -> anp.ndarray:
        """
        Compute the log-probability log p(params|features).

        Parameters
        ----------
        features : anp.ndarray
            A (N, feature_dim) array of conditional features.
        params : anp.ndarray
            A (N, param_dim) array of target parameters.

        Returns
        -------
        anp.ndarray
            A (N,) array of log-probabilities.
        """
        if self.weights is None:
            raise RuntimeError("Model has not been trained yet. Call train() first.")

        if not self._dims_inferred:
            raise RuntimeError("Model dimensions not inferred yet. Call train() first.")

    def train(
        self,
        params: anp.ndarray,
        features: anp.ndarray,
        n_iter: int = 2000,
        learning_rate: float = 1e-3,
        seed: int = 0,
        use_tqdm: bool = True,
    ):
        """
        Trains the model using the Adam optimizer.

        Parameters
        ----------
        params : anp.ndarray
            An (N, param_dim) matrix of simulated parameters.
        features : anp.ndarray
            An (N, feature_dim) matrix of corresponding data features.
        n_iter : int, optional
            The number of gradient descent iterations.
        learning_rate : float, optional
            The learning rate for the Adam optimizer.
        seed : int, optional
            Seed for reproducible weight initialization and training.
        use_tqdm : bool, optional
            If True, displays a progress bar during training.
        """
        # --- 1. Data Validation and Dimension Inference ---

        # Convert to arrays first
        params = anp.asarray(params)
        features = anp.asarray(features)

        # Infer dimensions if not already done
        if not self._dims_inferred:
            self._infer_dimensions(params, features)

        # Now validate with known dimensions
        if params.shape[0] != features.shape[0]:
            raise ValueError(
                "Params and features must have the same number of samples."
            )
        if params.shape[1] != self.param_dim or features.shape[1] != self.feature_dim:
            raise ValueError(
                "Data dimensions do not match inferred/expected model dimensions."
            )

        # Filter out non-finite values
        finite_idx = anp.all(anp.isfinite(params), axis=1) & anp.all(
            anp.isfinite(features), axis=1
        )
        params = params[finite_idx].astype("f")
        features = features[finite_idx].astype("f")

        if params.shape[0] == 0:
            raise ValueError("All data points contained non-finite values.")

        # --- 2. Initialization ---
        rng = anp.random.RandomState(seed)
        self.weights = self._initialize_weights(rng)
        self.loss_history = []

        # Adam optimizer state
        m = {key: anp.zeros_like(val) for key, val in self.weights.items()}
        v = {key: anp.zeros_like(val) for key, val in self.weights.items()}
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8

        # --- 3. Optimization Loop ---
        gradient_func = grad(self._loss_function)

        iterator = trange(n_iter, desc="Training", disable=not use_tqdm)
        for i in iterator:
            g = gradient_func(self.weights, features, params)
            loss = self._loss_function(self.weights, features, params)
            self.loss_history.append(loss)

            if not anp.isfinite(loss):
                print(
                    f"Warning: Loss is non-finite at iteration {i}. Stopping training."
                )
                break

            if use_tqdm:
                iterator.set_postfix(loss=f"{loss:.4f}")

            # Adam update step
            for key in self.weights:
                if not anp.all(anp.isfinite(g[key])):
                    print(
                        f"Warning: Non-finite gradient for '{key}' at iteration {i}. Stopping."
                    )
                    return
                m[key] = beta1 * m[key] + (1 - beta1) * g[key]
                v[key] = beta2 * v[key] + (1 - beta2) * (g[key] ** 2)
                m_hat = m[key] / (1 - beta1 ** (i + 1))
                v_hat = v[key] / (1 - beta2 ** (i + 1))
                self.weights[key] -= learning_rate * m_hat / (anp.sqrt(v_hat) + epsilon)


# =============================================================================
# == MDN Implementation
# =============================================================================


@dataclass
class MDNEstimator(ConditionalDensityEstimator):
    """
    Mixture Density Network for conditional density estimation.

    Parameters
    ----------
    param_dim : int, optional
        Dimensionality of the target variable. If None, inferred from training data.
    feature_dim : int, optional
        Dimensionality of the conditional variable. If None, inferred from training data.
    n_components : int, optional
        The number of Gaussian mixture components.
    hidden_sizes : tuple[int, ...], optional
        A tuple specifying the number of units in each hidden layer.
    """

    param_dim: int = None
    feature_dim: int = None
    n_components: int = 5
    hidden_sizes: tuple[int, ...] = (32, 32)

    def __post_init__(self):
        super().__init__(self.param_dim, self.feature_dim)
        # Note: _offdiag_basis will be created after dimensions are inferred

    def _infer_dimensions(self, params: anp.ndarray, features: anp.ndarray):
        """Override to also create basis after dimension inference."""
        super()._infer_dimensions(params, features)
        # Now that dimensions are known, create the off-diagonal basis
        self._offdiag_basis = self._create_offdiag_basis()

    def _create_offdiag_basis(self):
        n_off_diag = self.param_dim * (self.param_dim - 1) // 2
        if n_off_diag == 0:
            return None
        basis = anp.zeros((n_off_diag, self.param_dim, self.param_dim), dtype="f")
        rows, cols = anp.triu_indices(self.param_dim, k=1)
        basis[anp.arange(n_off_diag), rows, cols] = 1
        return basis

    def _initialize_weights(self, rng: anp.random.RandomState) -> dict:
        """Initializes weights for the MLP and GMM output layers."""
        weights = {}
        in_size = self.feature_dim
        for i, out_size in enumerate(self.hidden_sizes):
            weights[f"W{i}"] = (
                rng.randn(in_size, out_size) * anp.sqrt(2.0 / in_size)
            ).astype("f")
            weights[f"b{i}"] = anp.zeros(out_size, dtype="f")
            in_size = out_size

        last_hidden_size = (
            self.hidden_sizes[-1] if self.hidden_sizes else self.feature_dim
        )

        # GMM output layers
        K, D_out = self.n_components, self.param_dim
        weights["W_alpha"] = (rng.randn(last_hidden_size, K) * 0.01).astype("f")
        weights["b_alpha"] = anp.zeros(K, dtype="f")
        weights["W_mu"] = (rng.randn(last_hidden_size, K * D_out) * 0.01).astype("f")
        weights["b_mu"] = anp.zeros(K * D_out, dtype="f")
        weights["W_L_prec_log_diag"] = (
            rng.randn(last_hidden_size, K * D_out) * 0.01
        ).astype("f")
        weights["b_L_prec_log_diag"] = anp.zeros(K * D_out, dtype="f")

        n_off_diag = D_out * (D_out - 1) // 2
        if n_off_diag > 0:
            weights["W_L_prec_offdiag"] = (
                rng.randn(last_hidden_size, K * n_off_diag) * 0.01
            ).astype("f")
            weights["b_L_prec_offdiag"] = anp.zeros(K * n_off_diag, dtype="f")

        return weights

    def _forward_pass(self, weights: dict, features: anp.ndarray):
        """Maps input features to GMM parameters."""
        h = features
        for i in range(len(self.hidden_sizes)):
            h = anp.tanh(h @ weights[f"W{i}"] + weights[f"b{i}"])

        K, D_out = self.n_components, self.param_dim
        log_alpha = h @ weights["W_alpha"] + weights["b_alpha"]
        alpha = anp.exp(log_alpha - logsumexp(log_alpha, axis=1, keepdims=True))
        mu = (h @ weights["W_mu"] + weights["b_mu"]).reshape(-1, K, D_out)

        L_prec_log_diag = (
            h @ weights["W_L_prec_log_diag"] + weights["b_L_prec_log_diag"]
        ).reshape(-1, K, D_out)
        L_prec_diag_mat = anp.einsum(
            "nki,ij->nkij", anp.exp(L_prec_log_diag), anp.eye(D_out, dtype="f")
        )

        n_off_diag = D_out * (D_out - 1) // 2
        if n_off_diag > 0:
            L_prec_offdiag_vals = (
                h @ weights["W_L_prec_offdiag"] + weights["b_L_prec_offdiag"]
            ).reshape(-1, K, n_off_diag)
            L_prec_offdiag_mat = anp.einsum(
                "nkl,lij->nkij", L_prec_offdiag_vals, self._offdiag_basis
            )
            L_prec = L_prec_diag_mat + L_prec_offdiag_mat
        else:
            L_prec = L_prec_diag_mat

        return alpha, mu, L_prec, L_prec_log_diag

    def _loss_function(
        self, weights: dict, features: anp.ndarray, params: anp.ndarray
    ) -> float:
        """Computes the negative log-likelihood of the true parameters under the GMM."""
        alpha, mu, L_prec, L_prec_log_diag = self._forward_pass(weights, features)

        y_true_reshaped = params[:, anp.newaxis, :]
        delta = y_true_reshaped - mu

        z = anp.einsum("nkij,nkj->nki", L_prec, delta)
        quad_term = -0.5 * anp.sum(z**2, axis=2)
        log_det_term = anp.sum(L_prec_log_diag, axis=2)

        log_probs_k = (
            quad_term + log_det_term - 0.5 * self.param_dim * anp.log(2 * math.pi)
        )
        total_log_prob = logsumexp(anp.log(alpha + 1e-9) + log_probs_k, axis=1)

        return -anp.mean(total_log_prob)

    def log_prob(self, features: anp.ndarray, params: anp.ndarray) -> anp.ndarray:
        """
        Computes the log-probability log p(params|features) for each sample.
        """
        super().log_prob(features, params)

        # Perform a forward pass to get GMM parameters
        alpha, mu, L_prec, L_prec_log_diag = self._forward_pass(self.weights, features)

        # Reshape parameters for broadcasting against mixture components
        y_true_reshaped = params[:, anp.newaxis, :]
        delta = y_true_reshaped - mu

        # Compute the log-probability for each component (k) for each sample (n)
        z = anp.einsum("nkij,nkj->nki", L_prec, delta)
        quad_term = -0.5 * anp.sum(z**2, axis=2)
        log_det_term = anp.sum(L_prec_log_diag, axis=2)

        log_probs_k = (
            quad_term + log_det_term - 0.5 * self.param_dim * anp.log(2 * math.pi)
        )

        # Combine component log-probabilities using the mixture weights (alpha)
        # This returns a vector of shape (N,)
        total_log_prob = logsumexp(anp.log(alpha + 1e-9) + log_probs_k, axis=1)

        return total_log_prob

    def sample(
        self, features: anp.ndarray, n_samples: int, rng: anp.random.RandomState
    ) -> anp.ndarray:
        super().sample(features, n_samples, rng)
        features = features.astype("f")
        if features.ndim == 1:
            features = features.reshape(1, -1)

        alpha, mu, L_prec, _ = self._forward_pass(self.weights, features)
        n_cond, K, D_out = mu.shape

        log_alpha = anp.log(alpha + 1e-9)
        gumbel_noise = -anp.log(-anp.log(rng.uniform(size=(n_cond, n_samples, K))))
        component_indices = anp.argmax(
            log_alpha[:, anp.newaxis, :] + gumbel_noise, axis=2
        )

        cond_idx = anp.arange(n_cond)[:, anp.newaxis]
        chosen_mu = mu[cond_idx, component_indices]
        chosen_L_prec = L_prec[cond_idx, component_indices]

        try:
            L_cov_factor = anp.linalg.inv(chosen_L_prec)
        except anp.linalg.LinAlgError:
            print(
                "Warning: Singular precision matrix encountered during sampling. Returning NaNs."
            )
            return anp.full((n_cond, n_samples, D_out), anp.nan)

        z = rng.randn(n_cond, n_samples, D_out)
        samples = chosen_mu + anp.einsum("nsij,nsj->nsi", L_cov_factor, z)

        return samples


# =============================================================================
# == MAF Implementation
# =============================================================================


@dataclass
class MAFEstimator(ConditionalDensityEstimator):
    """
    Masked Autoregressive Flow for conditional density estimation.

    Parameters
    ----------
    param_dim : int, optional
        Dimensionality of the target variable. If None, inferred from training data.
    feature_dim : int, optional
        Dimensionality of the conditional variable. If None, inferred from training data.
    n_flows : int, optional
        The number of flow layers (MADE blocks).
    hidden_units : int, optional
        The number of hidden units in each MADE block.
    """

    param_dim: int = None
    feature_dim: int = None
    n_flows: int = 4
    hidden_units: int = 64

    def __post_init__(self):
        super().__init__(self.param_dim, self.feature_dim)
        self.model_constants = None  # For non-trainable parts like masks

    def _initialize_weights(self, rng: anp.random.RandomState) -> dict:
        """Initializes weights and model constants (masks, permutations)."""
        weights = {}
        layers = []
        D, C, H = self.param_dim, self.feature_dim, self.hidden_units

        for k in range(self.n_flows):
            # MADE masks and permutation
            m_in = anp.arange(1, D + 1)
            m_hidden = rng.randint(1, D, size=H)
            M1 = (m_in[None, :] <= m_hidden[:, None]).astype("f")
            m_out = m_in.copy()
            M2 = (m_hidden[None, :] < m_out[:, None]).astype("f")
            perm = rng.permutation(D)
            inv_perm = anp.empty(D, dtype=int)
            inv_perm[perm] = anp.arange(D)

            layers.append({"M1": M1, "M2": M2, "perm": perm, "inv_perm": inv_perm})

            # Trainable parameters
            w_std = 0.01
            weights[f"W1y_{k}"] = (rng.randn(H, D) * w_std).astype("f")
            weights[f"W1c_{k}"] = (
                (rng.randn(H, C) * w_std).astype("f")
                if C > 0
                else anp.zeros((H, C), dtype="f")
            )
            weights[f"b1_{k}"] = anp.zeros(H, dtype="f")
            weights[f"W2_{k}"] = anp.zeros((2 * D, H), dtype="f")
            weights[f"W2c_{k}"] = (
                anp.zeros((2 * D, C), dtype="f")
                if C > 0
                else anp.zeros((2 * D, C), dtype="f")
            )
            weights[f"b2_{k}"] = anp.zeros(2 * D, dtype="f")

        self.model_constants = {"layers": layers}
        return weights

    def _made_forward(self, y, ctx, layer_const, k, weights):
        """Single forward pass through a MADE block."""
        M1, M2 = layer_const["M1"], layer_const["M2"]
        W1y, W1c, b1 = weights[f"W1y_{k}"], weights[f"W1c_{k}"], weights[f"b1_{k}"]
        W2, W2c, b2 = weights[f"W2_{k}"], weights[f"W2c_{k}"], weights[f"b2_{k}"]

        y_h = anp.dot(y, (W1y * M1).T)
        c_h = anp.dot(ctx, W1c.T) if self.feature_dim > 0 else 0.0
        h = anp.tanh(y_h + c_h + b1)

        M2_tiled = anp.concatenate([M2, M2], axis=0)
        out = anp.dot(h, (W2 * M2_tiled).T)
        if self.feature_dim > 0:
            out = out + anp.dot(ctx, W2c.T)
        out = out + b2

        mu, alpha = out[:, : self.param_dim], anp.clip(
            out[:, self.param_dim :], -7.0, 7.0
        )
        return mu, alpha

    def _get_log_prob(self, weights: dict, features: anp.ndarray, params: anp.ndarray):
        """Computes log probability for the MAF."""
        u = params
        log_det = anp.zeros(params.shape[0])

        for k, layer_const in enumerate(self.model_constants["layers"]):
            u = u[:, layer_const["perm"]]
            mu, alpha = self._made_forward(u, features, layer_const, k, weights)
            u = (u - mu) * anp.exp(-alpha)
            log_det -= anp.sum(alpha, axis=1)

        base_logp = -0.5 * anp.sum(u**2, axis=1) - 0.5 * self.param_dim * anp.log(
            2.0 * anp.pi
        )
        return base_logp + log_det

    def _loss_function(
        self, weights: dict, features: anp.ndarray, params: anp.ndarray
    ) -> float:
        return -anp.mean(self._get_log_prob(weights, features, params))

    def log_prob(self, features: anp.ndarray, params: anp.ndarray) -> anp.ndarray:
        super().log_prob(features, params)
        return self._get_log_prob(self.weights, features, params)

    def sample(
        self, features: anp.ndarray, n_samples: int, rng: anp.random.RandomState
    ) -> anp.ndarray:
        super().sample(features, n_samples, rng)
        features = features.astype("f")
        if features.ndim == 1:
            features = features.reshape(1, -1)

        n_cond = features.shape[0]
        # Broadcast features to match number of samples
        if n_cond != n_samples:
            features = anp.repeat(features, n_samples, axis=0)

        z = rng.randn(n_samples, self.param_dim).astype("f")
        x = z

        # Invert the flow stack
        for k, layer_const in reversed(list(enumerate(self.model_constants["layers"]))):
            y_perm = x
            u = anp.zeros_like(y_perm)
            for i in range(self.param_dim):
                mu, alpha = self._made_forward(
                    u, features, layer_const, k, self.weights
                )
                u[:, i] = y_perm[:, i] * anp.exp(alpha[:, i]) + mu[:, i]
            x = u[:, layer_const["inv_perm"]]

        # Reshape to (n_conditions, n_samples, param_dim)
        return x.reshape(features.shape[0] // n_samples, n_samples, self.param_dim)
