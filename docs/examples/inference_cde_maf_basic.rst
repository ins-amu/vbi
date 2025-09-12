CDE Inference with MAF (Masked Autoregressive Flows)
====================================================

This notebook demonstrates parameter inference using Masked Autoregressive Flows (MAF) from the CDE module. MAF provides a more sophisticated approach to density estimation compared to MDN.

.. note::
   This is a placeholder for the full tutorial. The complete interactive notebook will be available in the examples directory.

Overview
--------

**MAF vs MDN:**

- **MAF**: Uses autoregressive flows for flexible density modeling
- **MDN**: Uses mixture of Gaussians for density approximation  
- **Trade-off**: MAF is more expressive but computationally more intensive

**Key Advantages:**

- **Expressive**: Can model complex, multimodal distributions
- **Autoregressive**: Captures parameter dependencies naturally
- **Scalable**: Efficient for moderate to high-dimensional problems

Tutorial Content
-----------------

The full tutorial demonstrates:

1. **Complex Data Generation**: Multi-dimensional nonlinear relationships
2. **MAF Architecture**: Configure autoregressive flow layers
3. **Advanced Training**: Monitor convergence and performance
4. **3D Visualization**: Comprehensive posterior analysis
5. **Robustness Testing**: Multiple test cases and evaluation metrics

.. Code Example
.. ------------

.. .. code-block:: python

..    import numpy as np
..    from vbi.cde import MAFEstimator
   
..    # Initialize MAF estimator
..    maf = MAFEstimator(
..        param_dim=3,           # Dimension of parameters θ
..        feature_dim=3,         # Dimension of observations x
..        n_layers=3,            # Number of autoregressive layers
..        hidden_dim=64,         # Hidden layer dimension
..    )
   
..    # Train the estimator
..    losses = maf.train(
..        features=x_train,
..        params=theta_train,
..        n_epochs=150,
..        batch_size=256
..    )
   
..    # Sample from complex posterior
..    posterior_samples = maf.sample(
..        features=observed_x,
..        n_samples=3000
..    )

When to Choose MAF vs MDN
-------------------------

**Use MAF when:**

- Complex dependencies between parameters
- Multimodal posterior distributions  
- Higher-dimensional parameter spaces
- Need maximum expressiveness

**Use MDN when:**

- Simpler parameter relationships
- Faster inference required
- Better interpretability needed
- Lower computational resources

Performance Comparison
----------------------

The tutorial includes comprehensive comparison:

- **Accuracy**: Posterior approximation quality
- **Speed**: Training and inference time
- **Robustness**: Performance across multiple test cases
- **Scalability**: Behavior with increasing dimensionality

Next Steps
----------

- Download the complete notebook from examples directory
- Apply to real brain models (Jansen-Rit, Wilson-Cowan, etc.)
- Compare computational efficiency with SBI methods
- Explore different MAF architectures and hyperparameters
