#!/usr/bin/env python
"""
Test to verify that identical empirical and simulated BOLD signals give cost ≈ 0
"""

import numpy as np
import torch
import pytest
from vbi.models.pytorch.utils import fc_correlation_cost


def test_perfect_match_basic():
    """Test that identical BOLD signals result in cost ≈ 0."""
    n_nodes = 68
    n_timesamples = 1200
    n_unique_sets = 5
    n_dup = 3
    n_sets = n_unique_sets * n_dup
    
    # Create empirical BOLD
    np.random.seed(42)
    bold_emp = np.random.randn(n_nodes, n_timesamples)
    
    # Create simulated BOLD as exact duplicates of empirical
    bold_sim = np.tile(bold_emp[:, np.newaxis, :], (1, n_sets, 1))
    
    # Verify they are identical
    for i in range(n_sets):
        assert np.allclose(bold_sim[:, i, :], bold_emp), f"Simulation {i} should be identical to empirical"
    
    cost = fc_correlation_cost(bold_emp, bold_sim, n_dup=n_dup, device='cpu', verbose=False)
    
    # Check if costs are close to zero
    tolerance = 1e-5
    assert np.all(np.abs(cost) < tolerance), \
        f"Expected cost ≈ 0 for identical BOLD, got max |cost| = {np.abs(cost).max():.10f}"


def test_perfect_match_different_ndup():
    """Test perfect match with different n_dup values."""
    n_nodes = 20
    n_timesamples = 500
    tolerance = 1e-5
    
    np.random.seed(42)
    bold_emp = np.random.randn(n_nodes, n_timesamples)
    
    for n_dup in [1, 2, 5, 10]:
        n_sets = 20
        if n_sets % n_dup != 0:
            continue
        
        bold_sim = np.tile(bold_emp[:, np.newaxis, :], (1, n_sets, 1))
        cost = fc_correlation_cost(bold_emp, bold_sim, n_dup=n_dup, device='cpu', verbose=False)
        
        assert np.all(np.abs(cost) < tolerance), \
            f"n_dup={n_dup}: Expected cost ≈ 0, got max |cost| = {np.abs(cost).max():.10f}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_perfect_match_cuda():
    """Test perfect match on CUDA."""
    n_nodes = 68
    n_timesamples = 1200
    n_unique_sets = 5
    n_dup = 3
    n_sets = n_unique_sets * n_dup
    
    np.random.seed(42)
    bold_emp = np.random.randn(n_nodes, n_timesamples)
    bold_sim = np.tile(bold_emp[:, np.newaxis, :], (1, n_sets, 1))
    
    cost_cpu = fc_correlation_cost(bold_emp, bold_sim, n_dup=n_dup, device='cpu', verbose=False)
    cost_cuda = fc_correlation_cost(bold_emp, bold_sim, n_dup=n_dup, device='cuda', verbose=False)
    
    tolerance = 1e-5
    assert np.all(np.abs(cost_cpu) < tolerance), "CPU cost should be ≈ 0"
    assert np.all(np.abs(cost_cuda) < tolerance), "CUDA cost should be ≈ 0"
    
    # Compare CPU and CUDA results
    diff = np.abs(cost_cpu - cost_cuda).max()
    assert diff < 1e-4, f"CPU and CUDA results differ by {diff}"


def test_perfect_match_explanation():
    """
    Test and explain why identical BOLD signals give cost ≈ 0.
    
    When simulated BOLD is identical to empirical BOLD:
    1. FC matrices are identical (correlation = 1.0 for all pairs)
    2. Correlation between vectorized FC matrices = 1.0
    3. Cost = 1 - correlation = 1 - 1.0 = 0.0
    
    Small numerical errors may occur due to:
    - Floating point precision
    - Fisher z-transformation (arctanh)
    - GPU vs CPU computations
    """
    n_nodes = 10
    n_timesamples = 200
    n_unique_sets = 2
    n_dup = 2
    n_sets = n_unique_sets * n_dup
    
    np.random.seed(42)
    bold_emp = np.random.randn(n_nodes, n_timesamples)
    bold_sim = np.tile(bold_emp[:, np.newaxis, :], (1, n_sets, 1))
    
    cost = fc_correlation_cost(bold_emp, bold_sim, n_dup=n_dup, device='cpu', verbose=False)
    
    # The cost should be extremely close to zero
    assert cost.shape == (n_unique_sets,)
    assert np.all(np.abs(cost) < 1e-5), \
        f"Perfect match should give cost ≈ 0, got {cost}"


def test_nearly_perfect_match():
    """Test with very small differences to ensure cost increases appropriately."""
    n_nodes = 50
    n_timesamples = 500
    n_unique_sets = 3
    n_dup = 2
    n_sets = n_unique_sets * n_dup
    
    np.random.seed(42)
    bold_emp = np.random.randn(n_nodes, n_timesamples)
    
    # Perfect match
    bold_sim_perfect = np.tile(bold_emp[:, np.newaxis, :], (1, n_sets, 1))
    cost_perfect = fc_correlation_cost(bold_emp, bold_sim_perfect, n_dup=n_dup, 
                                       device='cpu', verbose=False)
    
    # Nearly perfect match (add small noise)
    bold_sim_noisy = bold_sim_perfect + np.random.randn(n_nodes, n_sets, n_timesamples) * 0.01
    cost_noisy = fc_correlation_cost(bold_emp, bold_sim_noisy, n_dup=n_dup, 
                                     device='cpu', verbose=False)
    
    # Perfect match should have lower cost than noisy match
    assert np.all(cost_perfect < cost_noisy + 1e-4), \
        "Perfect match should have lower cost than noisy match"
    
    # Perfect match cost should still be very close to 0
    assert np.all(np.abs(cost_perfect) < 1e-5), \
        f"Perfect match cost should be ≈ 0, got {cost_perfect}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
