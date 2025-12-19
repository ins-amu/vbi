CDE API Guide: Conditional Density Estimation
=============================================

This is the comprehensive API documentation for the Conditional Density Estimation (CDE) module in VBI, which provides implementations of Mixture Density Networks (MDN) and Masked Autoregressive Flows (MAF) for parameter inference in brain models.

.. note::
   **Looking for a quick start?** See :doc:`inference_cde_mdn_basic` for a minimal working example with MDN.

Introduction to Conditional Density Estimation
-----------------------------------------------

**What is Conditional Density Estimation?**

Conditional Density Estimation (CDE) addresses the problem of learning the full
conditional probability density

.. math::

   p(y \mid x),

where :math:`x \in \mathcal{X}` denotes observed input variables and
:math:`y \in \mathcal{Y}` denotes a target variable. In contrast to standard
regression methods, which estimate a single point prediction (e.g. the
conditional mean :math:`\mathbb{E}[Y \mid X=x]`), CDE aims to recover the entire
conditional distribution of :math:`Y` given :math:`X=x`. This enables explicit
modeling of uncertainty, heteroskedasticity, skewness, and multimodality in the
output space.

**Problem Setting**

Let

.. math::

   \mathcal{D} = \{(x_i, y_i)\}_{i=1}^N

be a dataset of input–output pairs drawn from an unknown joint distribution
:math:`p(x, y)`. The goal of CDE is to learn a parametric model
:math:`\hat{p}_\phi(y \mid x)` that approximates the true conditional density
:math:`p(y \mid x)` and generalizes to unseen inputs :math:`x^*`.

**Model Formulation**

In CDE, the conditional density is parameterized by a function approximator,
typically a neural network, which maps each input :math:`x` to the parameters
of a conditional distribution over :math:`y`:

.. math::

   x \;\mapsto\; \hat{p}_\phi(y \mid x).

Depending on the chosen model class, this mapping may yield:

- Parameters of a parametric distribution (e.g. mean and variance of a Gaussian)
- Parameters of a mixture model (e.g. mixture weights, component means,
  and covariances)
- Parameters of an invertible transformation defining a normalized density
  (e.g. conditional normalizing flows)

**Training Objective**

CDE models are commonly trained using maximum conditional likelihood estimation.
The parameters :math:`\phi` are optimized by minimizing the negative
log-likelihood (NLL):

.. math::

   \mathcal{L}(\phi)
   = -\frac{1}{N} \sum_{i=1}^N \log \hat{p}_\phi(y_i \mid x_i).

Minimizing this objective encourages the model to assign high probability mass
to observed targets :math:`y_i` conditioned on their corresponding inputs
:math:`x_i`.

**Expressive CDE Models**

Several model families are commonly used for conditional density estimation:

- **Mixture Density Networks (MDNs):**
  Represent :math:`p(y \mid x)` as a mixture of parametric distributions whose
  parameters depend on :math:`x`.
- **Normalizing Flows:**
  Construct highly expressive conditional densities via invertible
  transformations of a simple base distribution.
- **Conditional Kernel Density Estimators:**
  Estimate densities using kernel smoothing conditioned on the input variables.
- **Implicit or likelihood-free models:**
  Approximate conditional densities using simulation-based or amortized
  inference techniques.

**Inference and Usage**

After training, a CDE model can be used to:

- Evaluate conditional likelihoods :math:`\hat{p}(y \mid x)`
- Compute summary statistics such as conditional means, variances, or quantiles
- Draw samples from the conditional distribution :math:`p(y \mid x)`
- Quantify predictive uncertainty for downstream decision-making

**Relation to Bayesian Inference**

Conditional Density Estimation naturally encompasses Bayesian posterior
inference. In simulation-based inference, the posterior
:math:`p(\theta \mid x)` is itself a conditional density, where model parameters
:math:`\theta` are conditioned on observed data :math:`x`. Consequently, many
simulation-based inference methods can be interpreted as specialized instances
of CDE.

**Summary**

Conditional Density Estimation provides a principled framework for learning
full predictive distributions conditioned on observed inputs. By modeling
:math:`p(y \mid x)` directly, CDE enables uncertainty-aware predictions and
forms a core component of modern probabilistic machine learning and inference
pipelines.

VBI Implementation Overview
----------------------------

The CDE module provides two main approaches for conditional density estimation:

**Mixture Density Networks (MDN):**
   - Uses Gaussian mixture models for density approximation
   - Fast training and inference
   - Good for simpler parameter relationships
   - Interpretable mixture components
   - Best for: Low-to-moderate dimensional problems (< 10 parameters)

**Masked Autoregressive Flows (MAF):**
   - Uses normalizing flows for flexible density modeling
   - More expressive for complex distributions
   - Naturally captures parameter dependencies
   - Better for multimodal posteriors
   - Best for: Complex, high-dimensional problems (> 10 parameters)

Both approaches inherit from a common base class that provides standardized training, sampling, and evaluation methods.

Architecture
------------

Base Class: ConditionalDensityEstimator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All density estimators inherit from ``ConditionalDensityEstimator``, which provides:

- **Unified Training Interface**: Adam optimizer with early stopping
- **Dimension Inference**: Automatic parameter/feature dimension detection
- **Data Validation**: Comprehensive input checking and preprocessing
- **Standardized API**: Consistent ``train()``, ``sample()``, and ``log_prob()`` methods

.. code-block:: python

   from vbi.cde import ConditionalDensityEstimator

   # Base class provides common functionality
   estimator = ConditionalDensityEstimator(param_dim=2, feature_dim=3)

Key Features:

- **Automatic Dimension Inference**: Set ``param_dim=None`` and ``feature_dim=None`` for auto-detection
- **Robust Training**: Handles non-finite values, provides convergence monitoring
- **Early Stopping**: Optional plateau detection with customizable patience
- **Progress Monitoring**: tqdm integration for training progress visualization

Mixture Density Network (MDN)
------------------------------

The ``MDNEstimator`` class implements conditional density estimation using Gaussian mixture models.

Class Parameters
~~~~~~~~~~~~~~~~

.. code-block:: python

   from vbi.cde import MDNEstimator

   mdn = MDNEstimator(
       param_dim=2,           # Target parameter dimensionality
       feature_dim=3,         # Conditional feature dimensionality
       n_components=5,        # Number of mixture components (default: 5)
       hidden_sizes=(32, 32)  # Hidden layer sizes (default: (32, 32))
   )

**Parameter Details:**

- ``param_dim``: Dimensionality of parameters to estimate (θ)
- ``feature_dim``: Dimensionality of conditional features (x)
- ``n_components``: Number of Gaussian mixture components (K)
- ``hidden_sizes``: Tuple of hidden layer sizes for the MLP

Training
~~~~~~~~

.. code-block:: python

   # Train the MDN
   loss_history = mdn.train(
       params=theta_train,      # Shape: (N, param_dim)
       features=x_train,        # Shape: (N, feature_dim)
       n_iter=2000,             # Training iterations
       learning_rate=1e-3,      # Adam learning rate
       seed=42,                 # Random seed for reproducibility
       use_tqdm=True,           # Progress bar
       patience=100,            # Early stopping patience
       min_delta=1e-4           # Minimum improvement threshold
   )

**Training Features:**

- **Adam Optimization**: Adaptive learning rate with momentum
- **Early Stopping**: Stops when loss improvement < min_delta for patience iterations
- **Loss Monitoring**: Tracks negative log-likelihood throughout training
- **Data Preprocessing**: Automatic handling of non-finite values

Inference Methods
~~~~~~~~~~~~~~~~~

**Log Probability Evaluation:**

.. code-block:: python

   # Compute log p(θ|x) for each sample
   log_probs = mdn.log_prob(
       features=x_test,    # Shape: (N, feature_dim)
       params=theta_test   # Shape: (N, param_dim)
   )
   # Returns: array of shape (N,) with log probabilities

**Sampling:**

.. code-block:: python

   # Generate samples from posterior p(θ|x)
   samples = mdn.sample(
       features=x_obs,           # Shape: (n_conditions, feature_dim)
       n_samples=1000,           # Samples per condition
       rng=np.random.RandomState(42),
       log_prob_threshold=None,  # Optional rejection sampling
       oversample_factor=5       # Oversampling for rejection
   )
   # Returns: shape (n_conditions, n_samples, param_dim)

**Advanced Sampling Features:**

- **Rejection Sampling**: Filter low-probability samples using ``log_prob_threshold``
- **Oversampling**: Generate extra candidates to account for rejections
- **Fallback Handling**: Graceful degradation when sampling fails

Masked Autoregressive Flow (MAF)
---------------------------------

The ``MAFEstimator`` class provides a more flexible approach using normalizing flows.

Class Parameters
~~~~~~~~~~~~~~~~

.. code-block:: python

   from vbi.cde import MAFEstimator

   maf = MAFEstimator(
       param_dim=2,              # Target parameter dimensionality
       feature_dim=3,            # Conditional feature dimensionality
       n_flows=4,                # Number of flow layers
       hidden_units=64,          # Hidden units per MADE block
       activation='tanh',        # Activation function
       z_score_theta=True,       # Standardize parameters
       z_score_x=True,           # Standardize features
       use_actnorm=True,         # Use ActNorm layers
       embedding_dim=None,       # Optional PCA embedding
   )

**Parameter Details:**

- ``n_flows``: Number of autoregressive transformation layers
- ``hidden_units``: Number of hidden units in each MADE block
- ``activation``: Activation function ('tanh', 'relu', 'elu')
- ``z_score_theta``/``z_score_x``: Internal standardization of parameters/features
- ``use_actnorm``: Data-dependent initialization of normalization layers
- ``embedding_dim``: Optional PCA dimensionality reduction for features

Preprocessing
~~~~~~~~~~~~~

MAF requires preprocessing for optimal performance:

.. code-block:: python

   # Compute normalization statistics (call before training)
   maf.prepare_normalizers(
       features=x_train,
       params=theta_train,
       rng=np.random.RandomState(42)
   )

   # Reinitialize weights and masks
   maf.reinitialize(rng=np.random.RandomState(42))

**Preprocessing Steps:**

1. **Standardization**: Z-score normalization of parameters and features
2. **PCA Embedding**: Optional dimensionality reduction for high-dimensional features
3. **Weight Initialization**: Proper initialization of MADE masks and flow parameters

Training
~~~~~~~~

.. code-block:: python

   # Train the MAF
   maf.train(
       params=theta_train,
       features=x_train,
       n_iter=2000,
       learning_rate=1e-3,
       seed=42,
       use_tqdm=True,
       validation_fraction=0.1,    # Validation split
       stop_after_epochs=20,       # Early stopping patience
       early_stopping_delta=0.0,   # Minimum improvement
       clip_max_norm=5.0           # Gradient clipping
   )

**Advanced Training Features:**

- **Train/Validation Split**: Automatic data splitting for monitoring
- **Gradient Clipping**: Prevents exploding gradients
- **ActNorm Warmup**: Data-dependent initialization of normalization layers
- **Convergence Monitoring**: Validation loss tracking with early stopping

Inference
~~~~~~~~~

**Log Probability:**

.. code-block:: python

   # Compute log probability under the flow
   log_probs = maf.log_prob(features=x_test, params=theta_test)

**Sampling:**

.. code-block:: python

   # Sample from the learned distribution
   samples = maf.sample(
       features=x_obs,
       n_samples=1000,
       rng=np.random.RandomState(42)
   )
   # Returns samples in original parameter space

Comparison: MAF vs MDN
-----------------------

.. list-table:: Backend Performance Comparison
   :header-rows: 1
   :class: color-caption

   * - **Aspect**
     - **MDN**
     - **MAF**
   * - Expressiveness
     - Limited to mixture of Gaussians
     - Highly flexible via flows
   * - Speed
     - Fast training/inference
     - Slower but more accurate
   * - Interpretability
     - Clear mixture components
     - Less interpretable
   * - Dependencies
     - Assumes independence
     - Captures dependencies
   * - Convergence
     - Usually stable
     - May require careful tuning
   * - Memory
     - Lower memory usage
     - Higher memory for flows

Best Practices
--------------

**Data Preparation:**

1. **Scale your data**: Both methods benefit from properly scaled inputs
2. **Handle outliers**: Remove or robustly handle extreme values
3. **Check dimensions**: Ensure consistent feature/parameter dimensions
4. **Sufficient samples**: Use adequate training data for reliable estimation

**Training Tips:**

1. **Monitor convergence**: Use validation splits and early stopping
2. **Tune learning rate**: Start with 1e-3, adjust based on convergence
3. **Gradient clipping**: Essential for MAF to prevent instability
4. **Batch considerations**: Larger batches may improve stability

**Hyperparameter Selection:**

- **MDN**: Focus on ``n_components`` (3-10) and ``hidden_sizes``
- **MAF**: Tune ``n_flows`` (3-6), ``hidden_units`` (32-128), and ``activation``

Example Usage
-------------

Complete example for brain model parameter inference:

.. code-block:: python

   import numpy as np
   from vbi.cde import MAFEstimator

   # Load your brain model simulation data
   theta = np.load('simulation_parameters.npy')  # Shape: (N, param_dim)
   features = np.load('simulation_features.npy')  # Shape: (N, feature_dim)

   # Initialize and configure MAF
   maf = MAFEstimator(
       param_dim=theta.shape[1],
       feature_dim=features.shape[1],
       n_flows=4,
       hidden_units=64
   )

   # Preprocessing
   maf.prepare_normalizers(features, theta)
   maf.reinitialize()

   # Training with monitoring
   maf.train(
       params=theta,
       features=features,
       n_iter=1000,
       validation_fraction=0.2,
       stop_after_epochs=10
   )

   # Inference on new observations
   observed_features = np.load('experimental_data.npy')
   posterior_samples = maf.sample(
       features=observed_features,
       n_samples=5000
   )

   # Analyze posterior
   print(f"Posterior shape: {posterior_samples.shape}")
   print(f"Mean parameters: {np.mean(posterior_samples, axis=1)}")

Troubleshooting
---------------

**Common Issues:**

- **Non-finite losses**: Check for NaN/inf in your data
- **Poor convergence**: Try lower learning rate or gradient clipping
- **Memory errors**: Reduce batch size or model complexity
- **Sampling failures**: Check for singular matrices in MDN

**Performance Optimization:**

- Use ``float32`` precision for faster computation
- Enable GPU acceleration if available
- Consider PCA embedding for high-dimensional features
- Monitor validation loss for overfitting

References
----------

The CDE implementations are based on:

1. **MDN**: Bishop, C. M. (1994). Mixture density networks. Technical Report NCRG/94/004
2. **MAF**: Papamakarios, G., et al. (2017). Masked autoregressive flow for density estimation. NeurIPS
3. **ActNorm**: Kingma, D. P., & Dhariwal, P. (2018). Glow: Generative flow with invertible 1x1 convolutions. NeurIPS

For brain model applications, see the examples directory for complete notebooks demonstrating parameter inference in Jansen-Rit, Wilson-Cowan, and other neural mass models.
