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
from typing import Optional


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

    ############################################################################
    def train(
        self,
        params: anp.ndarray,
        features: anp.ndarray,
        n_iter: int = 2000,
        learning_rate: float = 1e-3,
        seed: int = 0,
        use_tqdm: bool = True,
        patience: int = None,
        min_delta: float = 0.0,
    ):
        """
        Trains the model using the Adam optimizer with optional early stopping for plateaus.

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
        patience : int, optional
            Number of iterations with no improvement after which training will be stopped.
            If None, early stopping is disabled.
        min_delta : float, optional
            Minimum change in the loss to qualify as an improvement.
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

        # Early stopping initialization
        best_loss = anp.inf
        counter = 0
        if patience is not None and patience <= 0:
            raise ValueError("patience must be a positive integer.")

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

            # Early stopping check
            if loss < best_loss - min_delta:
                best_loss = loss
                counter = 0
            else:
                counter += 1
                if patience is not None and counter >= patience:
                    print(f"Early stopping at iteration {i} due to plateau.")
                    break

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
        self,
        features: anp.ndarray,
        n_samples: int,
        rng: anp.random.RandomState,
        log_prob_threshold: float = None,
        oversample_factor: int = 5,
    ) -> anp.ndarray:
        """
        Generate samples from the learned conditional distribution p(params|features),
        with optional rejection sampling to filter low-probability outliers.

        Parameters
        ----------
        features : anp.ndarray
            A (n_conditions, feature_dim) array of features to condition on.
        n_samples : int
            The number of samples to generate for each condition.
        rng : anp.random.RandomState
            A random number generator for sampling.
        log_prob_threshold : float, optional
            If provided, reject samples with log p(sample | features) < threshold.
            Tune based on training data (e.g., min log-prob - 2).
        oversample_factor : int, optional
            Multiplier for initial sample generation to account for rejections.

        Returns
        -------
        anp.ndarray
            An array of generated samples of shape (n_conditions, n_samples, param_dim).
            If filtering is active and fewer than n_samples are retained, it will be padded
            with the last valid sample and a warning printed.
        """
        super().sample(features, n_samples, rng)  # Existing check
        features = features.astype("f")
        if features.ndim == 1:
            features = features.reshape(1, -1)
        n_cond, _, D_out = features.shape[0], features.shape[1], self.param_dim

        # Generate oversampled candidates
        n_candidates = n_samples * oversample_factor
        alpha, mu, L_prec, _ = self._forward_pass(self.weights, features)
        K = self.n_components

        log_alpha = anp.log(alpha + 1e-9)
        gumbel_noise = -anp.log(-anp.log(rng.uniform(size=(n_cond, n_candidates, K))))
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

        z = rng.randn(n_cond, n_candidates, D_out)
        candidate_samples = chosen_mu + anp.einsum("ncsi,ncs->nci", L_cov_factor, z)

        if log_prob_threshold is not None:
            # Compute log_probs for all candidates
            flat_features = anp.tile(features[:, anp.newaxis, :], (1, n_candidates, 1)).reshape(-1, self.feature_dim)
            flat_candidates = candidate_samples.reshape(-1, D_out)
            log_probs = self.log_prob(flat_features, flat_candidates)

            # Filter per condition
            valid_samples = []
            for i in range(n_cond):
                cond_mask = anp.arange(n_candidates) + i * n_candidates
                cond_log_probs = log_probs[cond_mask]
                cond_candidates = flat_candidates[cond_mask]
                valid_mask = cond_log_probs > log_prob_threshold
                valid_cond = cond_candidates[valid_mask]
                if len(valid_cond) == 0:
                    print(f"Warning: No valid samples for condition {i}. Using mean.")
                    valid_cond = anp.tile(mu[i, 0], (n_samples, 1))  # Fallback to first component mean
                elif len(valid_cond) < n_samples:
                    # Pad with last valid or repeat
                    pad_len = n_samples - len(valid_cond)
                    valid_cond = anp.concatenate([valid_cond, anp.tile(valid_cond[-1:], (pad_len, 1))])
                    print(f"Warning: Only {len(valid_cond) - pad_len} valid samples for condition {i}; padded.")
                else:
                    valid_cond = valid_cond[:n_samples]
                valid_samples.append(valid_cond)
            filtered_samples = anp.stack(valid_samples)
        else:
            filtered_samples = candidate_samples[:, :n_samples, :]

        return filtered_samples
    

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
    n_flows : int
        Number of autoregressive transforms (a.k.a. num_transforms).
    hidden_units : int
        Hidden features per MADE block (a.k.a. hidden_features).
    activation : str
        'tanh' (default), 'relu', or 'elu'.
    z_score_theta : bool
        Standardize parameters (θ) internally.
    z_score_x : bool
        Standardize features (x) internally.
    use_actnorm : bool
        Insert ActNorm between flows with data-dependent init.
    embedding_dim : Optional[int]
        If set (E), PCA-reduce features x -> R^E before conditioning.
    """

    param_dim: int = None
    feature_dim: int = None
    n_flows: int = 4
    hidden_units: int = 64
    activation: str = "tanh"
    z_score_theta: bool = True
    z_score_x: bool = True
    use_actnorm: bool = True
    embedding_dim: Optional[int] = None  # PCA embedding for features
    actnorm_eps: float = 1e-6

    # internal state (filled after prepare_* calls / training init)
    def __post_init__(self):
        super().__init__(self.param_dim, self.feature_dim)
        self._dims_inferred = False  # Ensure attribute always exists
        self.model_constants = None  # masks, perms, inv_perms
        self._actnorm_initialized = [False] * self.n_flows

        # Standardization stats
        self.theta_mean = None
        self.theta_std = None
        self.x_mean = None
        self.x_std = None

        # PCA embedding (optional)
        self._use_pca = False
        self._pca_components = None  # (C, E)

    def _warmup_actnorm(self, features: anp.ndarray, params: anp.ndarray):
        """
        Run one forward pass to initialize ActNorm with data-dependent stats,
        outside the autograd graph (avoids mutating weights during grad).
        """
        if not self.use_actnorm:
            return
        # Use a small subset to estimate mean/std (like Glow’s data-dependent init)
        n = features.shape[0]
        k = min(512, n)  # warmup batch size
        _ = self._get_log_prob(self.weights, features[:k], params[:k])
        # After this call, self._actnorm_initialized[*] are True and actnorm params set.

    # ---------- public helpers: call these once before training ----------
    def prepare_normalizers(self, features: anp.ndarray, params: anp.ndarray, rng=None):
        """Compute z-score stats (like sbi) and optional PCA projection for features."""
        assert params.ndim == 2 and params.shape[1] == self.param_dim
        if self.feature_dim > 0:
            assert features.ndim == 2 and features.shape[1] == self.feature_dim

        if self.z_score_theta:
            self.theta_mean = anp.mean(params, axis=0)
            self.theta_std = anp.std(params, axis=0) + 1e-8
        else:
            self.theta_mean = anp.zeros(self.param_dim)
            self.theta_std = anp.ones(self.param_dim)

        if self.feature_dim > 0:
            if self.z_score_x:
                self.x_mean = anp.mean(features, axis=0)
                self.x_std = anp.std(features, axis=0) + 1e-8
            else:
                self.x_mean = anp.zeros(self.feature_dim)
                self.x_std = anp.ones(self.feature_dim)

            if self.embedding_dim is not None and self.embedding_dim < self.feature_dim:
                # PCA via SVD on standardized features
                X = (features - self.x_mean) / self.x_std
                U, S, Vt = anp.linalg.svd(X, full_matrices=False)
                E = self.embedding_dim
                self._pca_components = Vt[:E, :].T  # (C, E)
                self._use_pca = True
            else:
                self._use_pca = False
                self._pca_components = None
        else:
            self.x_mean = anp.zeros(0)
            self.x_std = anp.ones(0)
            self._use_pca = False
            self._pca_components = None

    # ---------- internal transforms ----------
    def _act(self, x):
        if self.activation == "relu":
            return anp.maximum(0.0, x)
        elif self.activation == "elu":
            return anp.where(x > 0.0, x, anp.exp(x) - 1.0)
        else:
            return anp.tanh(x)

    def _z_theta(self, params):
        return (params - self.theta_mean) / self.theta_std

    def _inv_z_theta(self, z):
        return z * self.theta_std + self.theta_mean

    def _z_x(self, features):
        if self.feature_dim == 0:
            return features
        X = (features - self.x_mean) / self.x_std
        if self._use_pca:
            X = anp.dot(X, self._pca_components)  # (N, E)
        return X

    def _ctx_dim(self):
        if self.feature_dim == 0:
            return 0
        return (
            self.embedding_dim
            if (self._use_pca and self.embedding_dim is not None)
            else self.feature_dim
        )

    # ---------- parameters & constants ----------
    def _initialize_weights(self, rng: anp.random.RandomState) -> dict:
        """Initializes weights, masks, permutations, and ActNorm (if enabled)."""
        weights = {}
        layers = []
        D, C_in, H = self.param_dim, self._ctx_dim(), self.hidden_units

        for k in range(self.n_flows):
            # Degrees / masks (classic MADE)
            m_in = anp.arange(1, D + 1)
            # draw hidden degrees in [1, D] inclusive; use D+1 as high because
            # numpy randint is exclusive at the upper bound. This avoids
            # ValueError when D == 1 (low >= high).
            m_hidden = rng.randint(1, D + 1, size=H)
            M1 = (m_in[None, :] <= m_hidden[:, None]).astype("f")
            m_out = m_in.copy()
            M2 = (m_hidden[None, :] < m_out[:, None]).astype("f")

            # Permutation
            perm = rng.permutation(D)
            inv_perm = anp.empty(D, dtype=int)
            inv_perm[perm] = anp.arange(D)

            layers.append({"M1": M1, "M2": M2, "perm": perm, "inv_perm": inv_perm})

            # Trainable parameters
            w_std = 0.01
            weights[f"W1y_{k}"] = (rng.randn(H, D) * w_std).astype("f")
            weights[f"W1c_{k}"] = (
                (rng.randn(H, C_in) * w_std).astype("f")
                if C_in > 0
                else anp.zeros((H, C_in), dtype="f")
            )
            weights[f"b1_{k}"] = anp.zeros(H, dtype="f")

            # Output heads (mu, log_scale)
            # Keep W2/W2c small; set log-scale bias negative (stable)
            weights[f"W2_{k}"] = anp.zeros((2 * D, H), dtype="f")
            weights[f"W2c_{k}"] = anp.zeros((2 * D, C_in), dtype="f")
            b2 = anp.zeros(2 * D, dtype="f")
            b2[D:] = -2.0  # log_sigma bias ~ exp(-2) start
            weights[f"b2_{k}"] = b2.astype("f")

            # ActNorm (per-dim scale & bias)
            if self.use_actnorm:
                weights[f"act_s_{k}"] = anp.ones(D, dtype="f")  # scale
                weights[f"act_b_{k}"] = anp.zeros(D, dtype="f")  # bias (pre-scale)
            else:
                weights[f"act_s_{k}"] = None
                weights[f"act_b_{k}"] = None

        self.model_constants = {"layers": layers}
        return weights

    # ---------- building blocks ----------
    def _made_forward(self, y, ctx, layer_const, k, weights):
        """Single forward pass through a MADE block; returns mu, log_sigma."""
        M1, M2 = layer_const["M1"], layer_const["M2"]
        W1y, W1c, b1 = weights[f"W1y_{k}"], weights[f"W1c_{k}"], weights[f"b1_{k}"]
        W2, W2c, b2 = weights[f"W2_{k}"], weights[f"W2c_{k}"], weights[f"b2_{k}"]

        y_h = anp.dot(y, (W1y * M1).T)
        c_h = anp.dot(ctx, W1c.T) if self._ctx_dim() > 0 else 0.0
        h = self._act(y_h + c_h + b1)

        M2_tiled = anp.concatenate([M2, M2], axis=0)
        out = anp.dot(h, (W2 * M2_tiled).T)
        if self._ctx_dim() > 0:
            out = out + anp.dot(ctx, W2c.T)
        out = out + b2

        mu = out[:, : self.param_dim]
        log_sigma = anp.clip(out[:, self.param_dim :], -7.0, 7.0)
        return mu, log_sigma

    def _apply_actnorm(self, u, k, weights, maybe_data_init=None):
        """ActNorm: y = (u + b) * s ; log_det += sum(log|s|)."""
        if not self.use_actnorm:
            return u, 0.0

        s = weights[f"act_s_{k}"]
        b = weights[f"act_b_{k}"]

        # Data-dependent init (first batch)
        if not self._actnorm_initialized[k] and maybe_data_init is not None:
            m = anp.mean(maybe_data_init, axis=0)
            v = anp.std(maybe_data_init, axis=0) + self.actnorm_eps
            b = -m
            s = 1.0 / v
            weights[f"act_s_{k}"] = s.astype("f")
            weights[f"act_b_{k}"] = b.astype("f")
            self._actnorm_initialized[k] = True

        y = (u + b) * s
        log_abs_s = anp.log(anp.abs(s) + 1e-12)
        log_det = anp.sum(log_abs_s)  # per-sample constant; broadcast by caller
        return y, log_det

    # ---------- core log_prob ----------
    def _get_log_prob(self, weights: dict, features: anp.ndarray, params: anp.ndarray):
        """Computes log probability under the flow (with preprocessing)."""
        # Preprocess to sbi-like standardized spaces
        x = (
            self._z_x(features).astype("f")
            if self._ctx_dim() > 0
            else features.astype("f")
        )
        u = self._z_theta(params).astype("f")

        batch = u.shape[0]
        log_det = anp.zeros(batch, dtype="f")

        for k, layer_const in enumerate(self.model_constants["layers"]):
            # Permute
            u = u[:, layer_const["perm"]]

            # ActNorm (data-dependent init on first batch seen)
            v, ln_det = self._apply_actnorm(u, k, weights, maybe_data_init=u)
            if self.use_actnorm:
                log_det = log_det + ln_det  # add same constant per sample

            # MADE transform
            mu, log_sigma = self._made_forward(v, x, layer_const, k, weights)
            u = (v - mu) * anp.exp(-log_sigma)
            log_det = log_det - anp.sum(log_sigma, axis=1)

        base_logp = -0.5 * anp.sum(u**2, axis=1) - 0.5 * self.param_dim * anp.log(
            2.0 * anp.pi
        )
        return base_logp + log_det

    # ---------- public API ----------
    def log_prob(self, features: anp.ndarray, params: anp.ndarray) -> anp.ndarray:
        super().log_prob(features, params)
        return self._get_log_prob(self.weights, features, params)

    def sample(
        self, features: anp.ndarray, n_samples: int, rng: anp.random.RandomState
    ) -> anp.ndarray:
        """
        Samples from p(theta | features). Returns shape (n_cond, n_samples, D) in original θ space.
        """
        super().sample(features, n_samples, rng)
        if features.ndim == 1 and self.feature_dim > 0:
            features = features.reshape(1, -1)
        n_cond = 1 if self.feature_dim == 0 else features.shape[0]

        # Preprocess features
        x = (
            self._z_x(features).astype("f")
            if self._ctx_dim() > 0
            else (
                features.astype("f")
                if self.feature_dim > 0
                else anp.zeros((n_cond, 0), dtype="f")
            )
        )

        out = anp.zeros((n_cond, n_samples, self.param_dim), dtype="f")
        for c in range(n_cond):
            z = rng.randn(n_samples, self.param_dim).astype("f")
            y = z

            # Invert the flow stack (reverse order)
            for k, layer_const in reversed(
                list(enumerate(self.model_constants["layers"]))
            ):
                u_perm = y  # current state in permuted coordinates we will fill autoregressively
                v = anp.zeros_like(u_perm)
                # Invert autoregressive transform sequentially
                for i in range(self.param_dim):
                    mu, log_sigma = self._made_forward(
                        v,
                        x[c : c + 1].repeat(n_samples, axis=0),
                        layer_const,
                        k,
                        self.weights,
                    )
                    v[:, i] = u_perm[:, i] * anp.exp(log_sigma[:, i]) + mu[:, i]

                # Invert ActNorm: u = v / s - b
                if self.use_actnorm:
                    s = self.weights[f"act_s_{k}"]
                    b = self.weights[f"act_b_{k}"]
                    v = v / (s + 1e-12) - b

                # Invert permutation
                y = v[:, layer_const["inv_perm"]]

            # Map back from z-space to original θ space
            out[c] = self._inv_z_theta(y)

        return out

    # ---------- convenience to (re)build ----------
    def reinitialize(self, rng: Optional[anp.random.RandomState] = None):
        """(Re)build masks/weights; call after prepare_normalizers()."""
        if rng is None:
            rng = anp.random.RandomState(0)
        self.weights = self._initialize_weights(rng)
        self._actnorm_initialized = [False] * self.n_flows

    def _loss_function(
        self, weights: dict, features: anp.ndarray, params: anp.ndarray
    ) -> float:
        return -anp.mean(self._get_log_prob(weights, features, params))

    def train(
        self,
        params: anp.ndarray,
        features: anp.ndarray,
        n_iter: int = 2000,
        learning_rate: float = 1e-3,
        seed: int = 0,
        use_tqdm: bool = True,
        # --- new knobs ---
        validation_fraction: float = 0.1,
        stop_after_epochs: int = 20,  # patience (like sbi)
        early_stopping_delta: float = 0.0,  # required improvement
        clip_max_norm: float = 5.0,  # set None to disable
    ):
        import autograd.numpy as anp
        from autograd import grad

        try:
            from tqdm import trange
        except Exception:

            def trange(N, **kw):
                return range(N)

        # --- 1) arrays + infer dims (same checks as before) ---
        params = anp.asarray(params)
        features = anp.asarray(features)
        if not self._dims_inferred:
            self._infer_dimensions(params, features)

        if params.shape[0] != features.shape[0]:
            raise ValueError(
                "Params and features must have the same number of samples."
            )
        if params.shape[1] != self.param_dim or features.shape[1] != self.feature_dim:
            raise ValueError(
                "Data dimensions do not match inferred/expected model dimensions."
            )

        finite_idx = anp.all(anp.isfinite(params), axis=1) & anp.all(
            anp.isfinite(features), axis=1
        )
        params = params[finite_idx].astype("f")
        features = features[finite_idx].astype("f")
        if params.shape[0] == 0:
            raise ValueError("All data points contained non-finite values.")

        N = params.shape[0]
        rng_np = anp.random.RandomState(seed)
        # --- 2) train/val split ---
        if not (0.0 <= validation_fraction < 1.0):
            raise ValueError("validation_fraction must be in [0,1).")
        n_val = int(N * validation_fraction)
        perm = rng_np.permutation(N)
        val_idx = perm[:n_val] if n_val > 0 else anp.array([], dtype=int)
        train_idx = perm[n_val:]

        params_tr, feats_tr = params[train_idx], features[train_idx]
        params_val, feats_val = (
            (params[val_idx], features[val_idx]) if n_val > 0 else (None, None)
        )

        # --- 3) compute normalizers on TRAIN ONLY (sbi-style) ---
        self.prepare_normalizers(feats_tr, params_tr)

        # --- 4) init weights/masks/perms; reset actnorm flags ---
        rng = anp.random.RandomState(seed)
        self.weights = self._initialize_weights(rng)
        self.loss_history = []
        self.val_loss_history = []  # <-- new
        self._actnorm_initialized = [False] * self.n_flows

        # --- 4.5) one-time ActNorm warmup on TRAIN subset ---
        self._warmup_actnorm(feats_tr, params_tr)

        # --- 5) Adam state ---
        m = {k: anp.zeros_like(v) for k, v in self.weights.items()}
        v = {k: anp.zeros_like(v) for k, v in self.weights.items()}
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8

        gradient_func = grad(self._loss_function)
        iterator = trange(n_iter, desc="Training", disable=not use_tqdm)

        # early stopping bookkeeping
        best_weights = {k: w.copy() for k, w in self.weights.items()}
        best_val = anp.inf if n_val > 0 else None
        epochs_no_improve = 0
        self.best_epoch = -1
        self.best_val_loss = None

        for epoch in iterator:
            # ---- forward/backward on TRAIN ----
            g = gradient_func(self.weights, feats_tr, params_tr)
            train_loss = self._loss_function(self.weights, feats_tr, params_tr)
            self.loss_history.append(float(train_loss))

            # (optional) grad clipping by global norm
            if clip_max_norm is not None:
                # compute global L2 norm over all tensors
                total_sq = 0.0
                for key in g:
                    total_sq += anp.sum(g[key] ** 2)
                global_norm = anp.sqrt(total_sq + 1e-12)
                if global_norm > clip_max_norm:
                    scale = clip_max_norm / (global_norm + 1e-12)
                    for key in g:
                        g[key] = g[key] * scale

            # Adam update
            for key in self.weights:
                if not anp.all(anp.isfinite(g[key])):
                    print(
                        f"Warning: Non-finite gradient for '{key}' at epoch {epoch}. Stopping."
                    )
                    self.weights = best_weights  # rollback to best known
                    return
                m[key] = beta1 * m[key] + (1 - beta1) * g[key]
                v[key] = beta2 * v[key] + (1 - beta2) * (g[key] ** 2)
                m_hat = m[key] / (1 - beta1 ** (epoch + 1))
                v_hat = v[key] / (1 - beta2 ** (epoch + 1))
                self.weights[key] -= learning_rate * m_hat / (anp.sqrt(v_hat) + epsilon)

            # ---- validation & early stopping ----
            if n_val > 0:
                val_loss = self._loss_function(self.weights, feats_val, params_val)
                self.val_loss_history.append(float(val_loss))

                improved = (best_val - val_loss) > early_stopping_delta
                if improved:
                    best_val = float(val_loss)
                    best_weights = {k: w.copy() for k, w in self.weights.items()}
                    epochs_no_improve = 0
                    self.best_epoch = int(epoch)
                    self.best_val_loss = float(val_loss)
                else:
                    epochs_no_improve += 1

                if use_tqdm:
                    try:
                        iterator.set_postfix(
                            train=f"{train_loss:.4f}",
                            val=f"{val_loss:.4f}",
                            patience=f"{epochs_no_improve}/{stop_after_epochs}",
                        )
                    except Exception:
                        pass

                if epochs_no_improve >= stop_after_epochs:
                    # restore best weights and stop
                    self.weights = best_weights
                    break
            else:
                if use_tqdm:
                    try:
                        iterator.set_postfix(train=f"{train_loss:.4f}")
                    except Exception:
                        pass

        # If we never saw validation or never improved, best_weights is initial;
        # in val-enabled runs we already restored on break; ensure final restore here too.
        if n_val > 0:
            self.weights = best_weights



@dataclass
class MAFEstimator0(ConditionalDensityEstimator):
    """
    Masked Autoregressive Flow for conditional density estimation.

    Parameters
    ----------
    param_dim : int
        Dimensionality of the target variable.
    feature_dim : int
        Dimensionality of the conditional variable.
    n_flows : int, optional
        The number of flow layers (MADE blocks).
    hidden_units : int, optional
        The number of hidden units in each MADE block.
    """
    param_dim: int
    feature_dim: int
    n_flows: int = 4
    hidden_units: int = 64

    def __post_init__(self):
        super().__init__(self.param_dim, self.feature_dim)
        self.model_constants = None # For non-trainable parts like masks

    def _initialize_weights(self, rng: anp.random.RandomState) -> dict:
        """Initializes weights and model constants (masks, permutations)."""
        weights = {}
        layers = []
        D, C, H = self.param_dim, self.feature_dim, self.hidden_units

        for k in range(self.n_flows):
            # MADE masks and permutation
            m_in = anp.arange(1, D + 1)
            m_hidden = rng.randint(1, D+1, size=H)
            M1 = (m_in[None, :] <= m_hidden[:, None]).astype('f')
            m_out = m_in.copy()
            M2 = (m_hidden[None, :] < m_out[:, None]).astype('f')
            perm = rng.permutation(D)
            inv_perm = anp.empty(D, dtype=int); inv_perm[perm] = anp.arange(D)

            layers.append({'M1': M1, 'M2': M2, 'perm': perm, 'inv_perm': inv_perm})

            # Trainable parameters
            w_std = 0.01
            weights[f'W1y_{k}'] = (rng.randn(H, D) * w_std).astype('f')
            weights[f'W1c_{k}'] = (rng.randn(H, C) * w_std).astype('f') if C > 0 else anp.zeros((H, C), dtype='f')
            weights[f'b1_{k}'] = anp.zeros(H, dtype='f')
            weights[f'W2_{k}'] = anp.zeros((2 * D, H), dtype='f')
            weights[f'W2c_{k}'] = anp.zeros((2 * D, C), dtype='f') if C > 0 else anp.zeros((2*D, C), dtype='f')
            weights[f'b2_{k}'] = anp.zeros(2 * D, dtype='f')

        self.model_constants = {'layers': layers}
        return weights

    def _made_forward(self, y, ctx, layer_const, k, weights):
        """Single forward pass through a MADE block."""
        M1, M2 = layer_const['M1'], layer_const['M2']
        W1y, W1c, b1 = weights[f'W1y_{k}'], weights[f'W1c_{k}'], weights[f'b1_{k}']
        W2, W2c, b2 = weights[f'W2_{k}'], weights[f'W2c_{k}'], weights[f'b2_{k}']

        y_h = anp.dot(y, (W1y * M1).T)
        c_h = anp.dot(ctx, W1c.T) if self.feature_dim > 0 else 0.0
        h = anp.tanh(y_h + c_h + b1)

        M2_tiled = anp.concatenate([M2, M2], axis=0)
        out = anp.dot(h, (W2 * M2_tiled).T)
        if self.feature_dim > 0:
            out = out + anp.dot(ctx, W2c.T)
        out = out + b2

        mu, alpha = out[:, :self.param_dim], anp.clip(out[:, self.param_dim:], -7.0, 7.0)
        return mu, alpha

    def _get_log_prob(self, weights: dict, features: anp.ndarray, params: anp.ndarray):
        """Computes log probability for the MAF."""
        u = params
        log_det = anp.zeros(params.shape[0])

        for k, layer_const in enumerate(self.model_constants['layers']):
            u = u[:, layer_const['perm']]
            mu, alpha = self._made_forward(u, features, layer_const, k, weights)
            u = (u - mu) * anp.exp(-alpha)
            log_det -= anp.sum(alpha, axis=1)

        base_logp = -0.5 * anp.sum(u**2, axis=1) - 0.5 * self.param_dim * anp.log(2.0 * anp.pi)
        return base_logp + log_det

    def _loss_function(self, weights: dict, features: anp.ndarray, params: anp.ndarray) -> float:
        return -anp.mean(self._get_log_prob(weights, features, params))

    def log_prob(self, features: anp.ndarray, params: anp.ndarray) -> anp.ndarray:
        super().log_prob(features, params)
        return self._get_log_prob(self.weights, features, params)

    def sample(self, features: anp.ndarray, n_samples: int, rng: anp.random.RandomState) -> anp.ndarray:
        super().sample(features, n_samples, rng)
        features = features.astype('f')
        if features.ndim == 1:
            features = features.reshape(1, -1)

        n_cond = features.shape[0]
        # Broadcast features to match number of samples
        if n_cond != n_samples:
            features = anp.repeat(features, n_samples, axis=0)

        z = rng.randn(n_samples, self.param_dim).astype('f')
        x = z

        # Invert the flow stack
        for k, layer_const in reversed(list(enumerate(self.model_constants['layers']))):
            y_perm = x
            u = anp.zeros_like(y_perm)
            for i in range(self.param_dim):
                mu, alpha = self._made_forward(u, features, layer_const, k, self.weights)
                u[:, i] = y_perm[:, i] * anp.exp(alpha[:, i]) + mu[:, i]
            x = u[:, layer_const['inv_perm']]

        # Reshape to (n_conditions, n_samples, param_dim)
        return x.reshape(features.shape[0] // n_samples, n_samples, self.param_dim)
