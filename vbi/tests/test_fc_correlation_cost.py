#!/usr/bin/env python
"""
Test script for fc_correlation_cost function
"""

import numpy as np
import torch
import pytest
from vbi.models.pytorch.utils import fc_correlation_cost


def test_fc_correlation_cost_basic():
    """Test the fc_correlation_cost function with normal case without NaN."""
    # Test parameters
    n_nodes = 68
    n_timesamples = 1200
    n_unique_sets = 10
    n_dup = 5
    n_sets = n_unique_sets * n_dup
    
    # Test 1: Normal case without NaN
    np.random.seed(42)
    bold_emp = np.random.randn(n_nodes, n_timesamples)
    bold_sim = np.random.randn(n_nodes, n_sets, n_timesamples)
    
    cost = fc_correlation_cost(bold_emp, bold_sim, n_dup=n_dup, device='cpu', verbose=False)
    
    assert cost.shape == (n_unique_sets,), f"Expected shape ({n_unique_sets},), got {cost.shape}"
    assert not np.any(np.isnan(cost)), "Cost should not contain NaN for valid input"
    assert np.all(cost >= 0), "Cost should be non-negative"


def test_fc_correlation_cost_with_nan():
    """Test case with some NaN values in simulated data."""
    n_nodes = 68
    n_timesamples = 1200
    n_unique_sets = 10
    n_dup = 5
    n_sets = n_unique_sets * n_dup
    
    np.random.seed(42)
    bold_emp = np.random.randn(n_nodes, n_timesamples)
    bold_sim = np.random.randn(n_nodes, n_sets, n_timesamples)
    
    # Add NaN to some duplicates of the first parameter set
    bold_sim_with_nan = bold_sim.copy()
    bold_sim_with_nan[:, 0, :100] = np.nan  # First duplicate, first 100 timepoints
    bold_sim_with_nan[:, 1, 500:600] = np.nan  # Second duplicate, timepoints 500-600
    
    cost_nan = fc_correlation_cost(bold_emp, bold_sim_with_nan, n_dup=n_dup, device='cpu', verbose=False)
    
    assert cost_nan.shape == (n_unique_sets,), f"Expected shape ({n_unique_sets},), got {cost_nan.shape}"
    assert np.sum(cost_nan < 10) > 0, "Should have some valid costs"


def test_fc_correlation_cost_all_nan():
    """Test case with all NaN for one parameter set."""
    n_nodes = 68
    n_timesamples = 1200
    n_unique_sets = 10
    n_dup = 5
    n_sets = n_unique_sets * n_dup
    
    np.random.seed(42)
    bold_emp = np.random.randn(n_nodes, n_timesamples)
    bold_sim = np.random.randn(n_nodes, n_sets, n_timesamples)
    
    # Set all duplicates of the second parameter set (set index 1) to NaN
    # Data arrangement: [set0_dup0, set0_dup1, ..., set0_dup4,
    #                    set1_dup0, set1_dup1, ..., set1_dup4, ...]
    # Set 1 starts at index 1*n_dup = 5
    bold_sim_all_nan = bold_sim.copy()
    for k in range(n_dup):
        bold_sim_all_nan[:, 1*n_dup + k, :] = np.nan  # Set 1, all dups
    
    cost_all_nan = fc_correlation_cost(bold_emp, bold_sim_all_nan, n_dup=n_dup, device='cpu', verbose=False)
    
    assert cost_all_nan.shape == (n_unique_sets,)
    assert cost_all_nan[1] == 10, f"Expected cost of 10 for all-NaN set, got {cost_all_nan[1]}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_fc_correlation_cost_cuda():
    """Test with CUDA if available."""
    n_nodes = 68
    n_timesamples = 1200
    n_unique_sets = 10
    n_dup = 5
    n_sets = n_unique_sets * n_dup
    
    np.random.seed(42)
    bold_emp = np.random.randn(n_nodes, n_timesamples)
    bold_sim = np.random.randn(n_nodes, n_sets, n_timesamples)
    
    cost_cpu = fc_correlation_cost(bold_emp, bold_sim, n_dup=n_dup, device='cpu', verbose=False)
    cost_cuda = fc_correlation_cost(bold_emp, bold_sim, n_dup=n_dup, device='cuda', verbose=False)
    
    assert cost_cuda.shape == (n_unique_sets,)
    # Compare with CPU results
    diff = np.abs(cost_cpu - cost_cuda).max()
    assert diff < 1e-4, f"CUDA and CPU results differ by {diff}"


def test_fc_correlation_cost_shape_validation():
    """Test shape validation."""
    n_nodes = 68
    n_timesamples = 1200
    n_unique_sets = 10
    n_dup = 5
    n_sets = n_unique_sets * n_dup
    
    np.random.seed(42)
    bold_emp = np.random.randn(n_nodes, n_timesamples)
    bold_sim = np.random.randn(n_nodes, n_sets, n_timesamples)
    
    # Wrong empirical shape (1D)
    with pytest.raises(ValueError, match="must be 2D"):
        bad_emp = np.random.randn(n_nodes)
        fc_correlation_cost(bad_emp, bold_sim, n_dup=n_dup, device='cpu', verbose=False)
    
    # Wrong simulated shape (2D)
    with pytest.raises(ValueError, match="must be 3D"):
        bad_sim = np.random.randn(n_nodes, n_sets)
        fc_correlation_cost(bold_emp, bad_sim, n_dup=n_dup, device='cpu', verbose=False)
    
    # Mismatched n_dup
    with pytest.raises(ValueError, match="must be divisible"):
        fc_correlation_cost(bold_emp, bold_sim, n_dup=7, device='cpu', verbose=False)


def test_fc_correlation_cost_perfect_match():
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
    
    cost = fc_correlation_cost(bold_emp, bold_sim, n_dup=n_dup, device='cpu', verbose=False)
    
    # Check if costs are close to zero
    tolerance = 1e-5
    assert np.all(np.abs(cost) < tolerance), f"Expected cost ≈ 0, got max |cost| = {np.abs(cost).max()}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
