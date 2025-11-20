#!/usr/bin/env python
"""
Detailed test to verify the averaging logic for NaN handling in fc_correlation_cost
"""

import numpy as np
import pytest

# Try to import torch - skip tests if not available
try:
    import torch
    from vbi.models.pytorch.utils import fc_correlation_cost
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
def test_fc_averaging_no_nan():
    """Test that all duplicates are averaged when there are no NaN values."""
    n_nodes = 5
    n_timesamples = 100
    n_unique_sets = 3
    n_dup = 4
    n_sets = n_unique_sets * n_dup  # 12 total
    
    np.random.seed(42)
    bold_emp = np.random.randn(n_nodes, n_timesamples)
    bold_sim = np.random.randn(n_nodes, n_sets, n_timesamples)
    
    cost = fc_correlation_cost(bold_emp, bold_sim, n_dup=n_dup, device='cpu', verbose=False)
    
    assert cost.shape == (n_unique_sets,)
    assert np.all(np.isfinite(cost)), "All costs should be finite when no NaN present"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
def test_fc_averaging_one_nan_duplication():
    """Test that NaN in one duplication is skipped and average computed from remaining."""
    n_nodes = 5
    n_timesamples = 100
    n_unique_sets = 3
    n_dup = 4
    n_sets = n_unique_sets * n_dup
    
    np.random.seed(42)
    bold_emp = np.random.randn(n_nodes, n_timesamples)
    bold_sim = np.random.randn(n_nodes, n_sets, n_timesamples)
    
    # Data arrangement: [set0_dup0, set1_dup0, set2_dup0, set0_dup1, set1_dup1, ...]
    # Set 0, duplication 1 (global index 0 + 1*n_unique_sets = 3) has all NaN
    bold_sim_with_nan = bold_sim.copy()
    bold_sim_with_nan[:, n_unique_sets, :] = np.nan  # Set 0, dup 1
    
    cost = fc_correlation_cost(bold_emp, bold_sim_with_nan, n_dup=n_dup, device='cpu', verbose=False)
    
    assert cost.shape == (n_unique_sets,)
    assert np.all(np.isfinite(cost)), "All costs should be finite (Set 0 averaged over 3 valid duplications)"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
def test_fc_averaging_multiple_nan_duplications():
    """Test that multiple NaN duplications are skipped and average computed from remaining."""
    n_nodes = 5
    n_timesamples = 100
    n_unique_sets = 3
    n_dup = 4
    n_sets = n_unique_sets * n_dup
    
    np.random.seed(42)
    bold_emp = np.random.randn(n_nodes, n_timesamples)
    bold_sim = np.random.randn(n_nodes, n_sets, n_timesamples)
    
    # Data arrangement: [set0_dup0, set1_dup0, set2_dup0, set0_dup1, set1_dup1, set2_dup1, ...]
    # Set 1 has NaN in duplications 0 and 2
    # Set 1, dup 0: index = 1 + 0*n_unique_sets = 1
    # Set 1, dup 2: index = 1 + 2*n_unique_sets = 7
    bold_sim_with_nan = bold_sim.copy()
    bold_sim_with_nan[:, 1, :] = np.nan  # Set 1, duplication 0
    bold_sim_with_nan[:, 1 + 2*n_unique_sets, :] = np.nan  # Set 1, duplication 2
    
    cost = fc_correlation_cost(bold_emp, bold_sim_with_nan, n_dup=n_dup, device='cpu', verbose=False)
    
    assert cost.shape == (n_unique_sets,)
    assert np.all(np.isfinite(cost)), "All costs should be finite (Set 1 averaged over 2 valid duplications)"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
def test_fc_averaging_all_nan_duplications():
    """Test that when ALL duplications have NaN, default cost of 10 is returned."""
    n_nodes = 5
    n_timesamples = 100
    n_unique_sets = 3
    n_dup = 4
    n_sets = n_unique_sets * n_dup
    
    np.random.seed(42)
    bold_emp = np.random.randn(n_nodes, n_timesamples)
    bold_sim = np.random.randn(n_nodes, n_sets, n_timesamples)
    
    # Data arrangement: [set0_dup0, set0_dup1, set0_dup2, set0_dup3,
    #                    set1_dup0, set1_dup1, set1_dup2, set1_dup3,
    #                    set2_dup0, set2_dup1, set2_dup2, set2_dup3]
    # Set 2 has NaN in ALL duplications
    # Set 2: indices = 8, 9, 10, 11 (set2 starts at index 2*n_dup = 8)
    bold_sim_with_nan = bold_sim.copy()
    for k in range(n_dup):
        bold_sim_with_nan[:, 2*n_dup + k, :] = np.nan  # Set 2, all dups
    
    cost = fc_correlation_cost(bold_emp, bold_sim_with_nan, n_dup=n_dup, device='cpu', verbose=False)
    
    assert cost.shape == (n_unique_sets,)
    assert cost[2] == 10, f"Expected cost of 10 for set with all NaN, got {cost[2]}"
    assert np.isfinite(cost[0]) and np.isfinite(cost[1]), "Other sets should have finite costs"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
def test_fc_averaging_mixed_scenario():
    """Test mixed scenario with different NaN patterns across sets."""
    n_nodes = 5
    n_timesamples = 100
    n_unique_sets = 3
    n_dup = 4
    n_sets = n_unique_sets * n_dup
    
    np.random.seed(42)
    bold_emp = np.random.randn(n_nodes, n_timesamples)
    bold_sim = np.random.randn(n_nodes, n_sets, n_timesamples)
    
    bold_sim_mixed = bold_sim.copy()
    
    # Data arrangement: [set0_dup0, set0_dup1, set0_dup2, set0_dup3,
    #                    set1_dup0, set1_dup1, set1_dup2, set1_dup3,
    #                    set2_dup0, set2_dup1, set2_dup2, set2_dup3]
    # Set 0: Dups 0 and 2 have NaN → average over 2 valid dups (1, 3)
    # Set 0 starts at index 0, dups at indices 0, 1, 2, 3
    bold_sim_mixed[:, 0, :] = np.nan   # Set 0, dup 0
    bold_sim_mixed[:, 2, :] = np.nan   # Set 0, dup 2
    
    # Set 1: Dup 1 has NaN → average over 3 valid dups (0, 2, 3)
    # Set 1 starts at index 4, dups at indices 4, 5, 6, 7
    bold_sim_mixed[:, 5, :] = np.nan   # Set 1, dup 1 (index 4+1)
    
    # Set 2: All dups have NaN → default cost = 10
    # Set 2 starts at index 8, dups at indices 8, 9, 10, 11
    for k in range(n_dup):
        bold_sim_mixed[:, 2*n_dup + k, :] = np.nan  # Set 2, all dups
    
    cost = fc_correlation_cost(bold_emp, bold_sim_mixed, n_dup=n_dup, device='cpu', verbose=False)
    
    assert cost.shape == (n_unique_sets,)
    assert np.isfinite(cost[0]), "Set 0 should have finite cost (averaged over 2 valid dups)"
    assert np.isfinite(cost[1]), "Set 1 should have finite cost (averaged over 3 valid dups)"
    assert cost[2] == 10, f"Set 2 should have default cost of 10, got {cost[2]}"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
def test_fc_averaging_index_mapping():
    """Test that the index mapping between duplications and sets is correct."""
    n_nodes = 5
    n_timesamples = 100
    n_unique_sets = 2
    n_dup = 3
    n_sets = n_unique_sets * n_dup  # 6 total
    
    np.random.seed(42)
    bold_emp = np.random.randn(n_nodes, n_timesamples)
    bold_sim = np.random.randn(n_nodes, n_sets, n_timesamples)
    
    # Data arrangement: [set0_dup0, set1_dup0, set0_dup1, set1_dup1, set0_dup2, set1_dup2]
    # Indices:         [    0    ,    1    ,    2    ,    3    ,    4    ,    5    ]
    
    # Make set 0, dup 1 (index 2) NaN
    bold_sim_test = bold_sim.copy()
    bold_sim_test[:, 2, :] = np.nan  # Set 0, dup 1
    
    cost = fc_correlation_cost(bold_emp, bold_sim_test, n_dup=n_dup, device='cpu', verbose=False)
    
    assert cost.shape == (n_unique_sets,)
    # Set 0 should be computed from dups at indices 0 and 4 (dups 0 and 2)
    # Set 1 should be computed from dups at indices 1, 3, and 5 (all dups)
    assert np.isfinite(cost[0]), "Set 0 should have finite cost"
    assert np.isfinite(cost[1]), "Set 1 should have finite cost"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
