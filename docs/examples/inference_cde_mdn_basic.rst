CDE Inference with MDN (Mixture Density Networks)
=================================================

This notebook demonstrates parameter inference using the CDE (Conditional Density Estimation) module with Mixture Density Networks. Unlike SBI-based inference, CDE uses only numpy and autograd, making it lightweight and dependency-free.

.. note::
   This is a placeholder for the full tutorial. The complete interactive notebook will be available in the examples directory.

Overview
--------

**Key Features:**

- **Lightweight**: No PyTorch or SBI dependencies
- **Fast**: Pure numpy implementation with autograd  
- **Flexible**: Suitable for various model types
- **Interpretable**: Clear mathematical foundation

**When to use CDE vs SBI:**

- **Use CDE when**: You want lightweight inference, have limited computational resources, or prefer mathematical transparency
- **Use SBI when**: You need state-of-the-art neural architectures or are working with very high-dimensional problems

Tutorial Content
-----------------

The full tutorial covers:

1. **Generate Synthetic Data**: Create nonlinear parameter-observation relationships
2. **Train MDN Estimator**: Configure and train the MDN to learn conditional density p(θ|x)
3. **Perform Inference**: Use trained model to infer parameters from observations
4. **Visualize Results**: Plot posterior distributions and compare with true values
5. **Model Evaluation**: Assess quality of posterior approximation

.. Code Example
.. ------------

.. .. code-block:: python

..    import numpy as np
..    from vbi.cde import MDNEstimator
..    from vbi.utils import BoxUniform
   
..    # Initialize MDN estimator
..    mdn = MDNEstimator(
..        param_dim=2,      # Dimension of parameters θ
..        feature_dim=2,    # Dimension of observations x
..        n_components=5,   # Number of mixture components
..        hidden_dims=[64, 64],  # Hidden layer dimensions
..    )
   
..    # Train the estimator
..    losses = mdn.train(
..        features=x_train,
..        params=theta_train,
..        n_epochs=100,
..        batch_size=256
..    )
   
..    # Sample from posterior
..    posterior_samples = mdn.sample(
..        features=observed_x,
..        n_samples=2000
..    )

Next Steps
----------

- Download the complete notebook from the examples directory
- Try with different model architectures
- Apply to real brain simulation models
- Compare with SBI-based approaches
