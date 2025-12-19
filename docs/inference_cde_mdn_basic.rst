CDE Quick Start: Mixture Density Networks (MDN)
===============================================

This is a quick-start guide for using Mixture Density Networks (MDN) for parameter inference. For comprehensive documentation covering both MDN and MAF methods, see :doc:`inference_cde_guide`.

.. note::
   **Looking for complete API documentation?** See :doc:`inference_cde_guide` for detailed coverage of both MDN and MAF methods, including all parameters, methods, and best practices.

What is MDN?
------------

Mixture Density Networks (MDN) combine neural networks with Gaussian mixture models to learn conditional probability distributions. In the context of brain model inference:

- **Input**: Features extracted from brain model simulations (e.g., power spectrum, connectivity)
- **Output**: A mixture of Gaussians representing p(θ|x), the probability of parameters θ given features x

**How MDN Works:**

1. Neural network takes features x as input
2. Network outputs mixture parameters: weights, means, and covariances
3. These define a Gaussian mixture model that approximates p(θ|x)
4. You can sample from this mixture to get probable parameter values

**Key Idea:** Instead of predicting a single parameter value, MDN predicts a full probability distribution, capturing uncertainty in the inference.

Why Use MDN?
------------

**Advantages:**

✅ **Lightweight**: No PyTorch or SBI dependencies
✅ **Fast Training**: Typically converges in minutes
✅ **Fast Inference**: Generate thousands of samples in milliseconds
✅ **Interpretable**: Mixture components are easy to visualize and understand
✅ **Flexible**: Works with various brain model architectures

**Limitations:**

⚠️ **Limited expressiveness**: Gaussian mixtures can't capture all distribution types
⚠️ **Independence assumption**: Parameters are modeled as independent within each component
⚠️ **Dimensionality**: Works best with < 10 parameters

**When to use MDN vs MAF:**

- **Use MDN when**: You have < 10 parameters, want fast training, need interpretability
- **Use MAF when**: You have > 10 parameters, need to capture parameter correlations, have complex posteriors

**When to use CDE vs SBI:**

- **Use CDE when**: You want lightweight inference, have limited computational resources, or prefer mathematical transparency
- **Use SBI when**: You need state-of-the-art neural architectures or are working with very high-dimensional problems

Quick Start Example
-------------------

Here's a minimal working example:

.. code-block:: python

   import numpy as np
   from vbi.cde import MDNEstimator
   
   # Initialize MDN estimator
   mdn = MDNEstimator(
       param_dim=2,           # Dimension of parameters θ
       feature_dim=2,         # Dimension of observations x
       n_components=5,        # Number of mixture components
       hidden_sizes=(64, 64)  # Hidden layer dimensions
   )
   
   # Train the estimator
   loss_history = mdn.train(
       params=theta_train,    # Shape: (N, 2)
       features=x_train,      # Shape: (N, 2)
       n_iter=2000,
       learning_rate=1e-3
   )
   
   # Sample from posterior
   posterior_samples = mdn.sample(
       features=x_observed,   # Shape: (1, 2)
       n_samples=2000
   )

Tutorial Workflow
-----------------

1. **Prepare Data**: Generate or load simulation parameters and features
2. **Initialize MDN**: Set parameter/feature dimensions and network architecture
3. **Train Model**: Fit the MDN to learn the conditional density p(θ|x)
4. **Perform Inference**: Sample from posterior for observed data
5. **Analyze Results**: Visualize and evaluate posterior distributions

Complete Examples
-----------------

See these Jupyter notebooks for detailed tutorials:

- :doc:`examples/damp_oscillator_cde` - Damped oscillator with CDE
- :doc:`examples/jansen_rit_sde_numba_cde` - Jansen-Rit neural mass model
- :doc:`examples/vep_sde_numba_cde` - Visual evoked potential model

Next Steps
----------

- **Full Documentation**: See :doc:`inference_cde_guide` for comprehensive API reference
- **Try Examples**: Download complete notebooks from the examples directory
- **Compare Methods**: Learn about MAF (Masked Autoregressive Flows) in :doc:`inference_cde_guide`
- **Brain Models**: Apply CDE to real neural mass models
