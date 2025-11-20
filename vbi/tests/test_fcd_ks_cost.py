#!/usr/bin/env python
"""
Test script for fcd_ks_cost function
"""

import numpy as np
import torch
import pytest
from vbi.models.pytorch.utils import fcd_ks_cost


def test_fcd_ks_cost_basic():
    """Test the fcd_ks_cost function with normal case without NaN."""
    n_nodes = 10
    n_timesamples = 200
    n_unique_sets = 5
    n_dup = 3
    n_sets = n_unique_sets * n_dup
    
    np.random.seed(42)
    bold_emp = np.random.randn(n_nodes, n_timesamples)
    bold_sim = np.random.randn(n_nodes, n_sets, n_timesamples)
    
    cost = fcd_ks_cost(bold_emp, bold_sim, n_dup=n_dup, window_size=50, 
                       device='cpu', verbose=False)
    
    assert cost.shape == (n_unique_sets,), f"Expected shape ({n_unique_sets},), got {cost.shape}"
    assert not np.any(np.isnan(cost)), "Cost should not contain NaN for valid input"
    assert np.all(cost >= 0), "Cost should be non-negative"


def test_fcd_ks_cost_with_nan():
    """Test case with some NaN values in simulated data."""
    n_nodes = 10
    n_timesamples = 200
    n_unique_sets = 5
    n_dup = 3
    n_sets = n_unique_sets * n_dup
    
    np.random.seed(42)
    bold_emp = np.random.randn(n_nodes, n_timesamples)
    bold_sim = np.random.randn(n_nodes, n_sets, n_timesamples)
    
    # Add NaN to some duplicates of the first parameter set
    bold_sim_with_nan = bold_sim.copy()
    bold_sim_with_nan[:, 0, :50] = np.nan  # Set 0, dup 0
    
    cost_nan = fcd_ks_cost(bold_emp, bold_sim_with_nan, n_dup=n_dup, 
                           window_size=50, device='cpu', verbose=False)
    
    assert cost_nan.shape == (n_unique_sets,)
    # Set 0 should still have a valid cost from remaining duplicates
    assert np.sum(cost_nan < 10) > 0, "Should have some valid costs"


def test_fcd_ks_cost_all_nan():
    """Test case with all NaN for one parameter set."""
    n_nodes = 10
    n_timesamples = 200
    n_unique_sets = 5
    n_dup = 3
    n_sets = n_unique_sets * n_dup
    
    np.random.seed(42)
    bold_emp = np.random.randn(n_nodes, n_timesamples)
    bold_sim = np.random.randn(n_nodes, n_sets, n_timesamples)
    
    # Set all duplicates of the second parameter set (set index 1) to NaN
    # Data arrangement: [set0_dup0, set0_dup1, set0_dup2, set1_dup0, ...]
    # Set 1 starts at index 1*n_dup = 3
    bold_sim_all_nan = bold_sim.copy()
    for k in range(n_dup):
        bold_sim_all_nan[:, 1*n_dup + k, :] = np.nan  # Set 1, all dups
    
    cost_all_nan = fcd_ks_cost(bold_emp, bold_sim_all_nan, n_dup=n_dup, 
                               window_size=50, device='cpu', verbose=False)
    
    assert cost_all_nan.shape == (n_unique_sets,)
    assert cost_all_nan[1] == 10, f"Expected cost of 10 for all-NaN set, got {cost_all_nan[1]}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_fcd_ks_cost_cuda():
    """Test with CUDA if available."""
    n_nodes = 10
    n_timesamples = 200
    n_unique_sets = 5
    n_dup = 3
    n_sets = n_unique_sets * n_dup
    
    np.random.seed(42)
    bold_emp = np.random.randn(n_nodes, n_timesamples)
    bold_sim = np.random.randn(n_nodes, n_sets, n_timesamples)
    
    cost_cpu = fcd_ks_cost(bold_emp, bold_sim, n_dup=n_dup, window_size=50,
                           device='cpu', verbose=False)
    cost_cuda = fcd_ks_cost(bold_emp, bold_sim, n_dup=n_dup, window_size=50,
                            device='cuda', verbose=False)
    
    assert cost_cuda.shape == (n_unique_sets,)
    # Results should be very similar between CPU and CUDA
    diff = np.abs(cost_cpu - cost_cuda).max()
    assert diff < 0.01, f"CPU and CUDA results differ significantly: {diff}"


def test_fcd_ks_cost_shape_validation():
    """Test shape validation."""
    n_nodes = 10
    n_timesamples = 200
    n_unique_sets = 5
    n_dup = 3
    n_sets = n_unique_sets * n_dup
    
    np.random.seed(42)
    bold_emp = np.random.randn(n_nodes, n_timesamples)
    bold_sim = np.random.randn(n_nodes, n_sets, n_timesamples)
    
    # Wrong empirical shape (1D)
    with pytest.raises(ValueError, match="must be 2D"):
        bad_emp = np.random.randn(n_nodes)
        fcd_ks_cost(bad_emp, bold_sim, n_dup=n_dup, device='cpu', verbose=False)
    
    # Wrong simulated shape (2D)
    with pytest.raises(ValueError, match="must be 3D"):
        bad_sim = np.random.randn(n_nodes, n_sets)
        fcd_ks_cost(bold_emp, bad_sim, n_dup=n_dup, device='cpu', verbose=False)
    
    # Mismatched n_dup
    with pytest.raises(ValueError, match="must be divisible"):
        fcd_ks_cost(bold_emp, bold_sim, n_dup=7, device='cpu', verbose=False)
    
    # Window size too large
    with pytest.raises(ValueError, match="window_size"):
        fcd_ks_cost(bold_emp, bold_sim, n_dup=n_dup, window_size=300, 
                   device='cpu', verbose=False)


def test_fcd_ks_cost_identical_bold():
    """Test that identical BOLD signals result in very low cost."""
    n_nodes = 10
    n_timesamples = 200
    n_unique_sets = 3
    n_dup = 2
    n_sets = n_unique_sets * n_dup
    
    # Create empirical BOLD
    np.random.seed(42)
    bold_emp = np.random.randn(n_nodes, n_timesamples)
    
    # Create simulated BOLD as exact duplicates of empirical
    bold_sim = np.tile(bold_emp[:, np.newaxis, :], (1, n_sets, 1))
    
    cost = fcd_ks_cost(bold_emp, bold_sim, n_dup=n_dup, window_size=50,
                       device='cpu', verbose=False)
    
    # Cost should be very close to zero for identical signals
    assert np.all(cost < 0.1), \
        f"Expected very low cost for identical BOLD, got max cost = {cost.max():.4f}"


def test_fcd_ks_cost_different_window_sizes():
    """Test with different window sizes."""
    n_nodes = 10
    n_timesamples = 200
    n_unique_sets = 3
    n_dup = 2
    n_sets = n_unique_sets * n_dup
    
    np.random.seed(42)
    bold_emp = np.random.randn(n_nodes, n_timesamples)
    bold_sim = np.random.randn(n_nodes, n_sets, n_timesamples)
    
    for window_size in [30, 50, 83, 100]:
        cost = fcd_ks_cost(bold_emp, bold_sim, n_dup=n_dup, 
                          window_size=window_size, device='cpu', verbose=False)
        assert cost.shape == (n_unique_sets,), \
            f"Window size {window_size}: expected shape ({n_unique_sets},), got {cost.shape}"
        assert np.all(np.isfinite(cost) | (cost == 10)), \
            f"Window size {window_size}: costs should be finite or 10"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
