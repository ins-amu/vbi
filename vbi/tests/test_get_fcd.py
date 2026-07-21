#!/usr/bin/env python
"""
Test script for get_fcd function from features_utils
Tests backward compatibility and new overlap parameter
"""

import numpy as np
import pytest
from vbi.feature_extraction.features_utils import get_fcd


@pytest.mark.short
@pytest.mark.fast
class TestGetFcdBackwardCompatibility:
    """Test that get_fcd without overlap parameter produces same results as before."""
    
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.n_regions = 5
        self.n_timepoints = 200
        self.ts = np.random.randn(self.n_regions, self.n_timepoints)
    
    def test_default_parameters(self):
        """Test get_fcd with default parameters (backward compatibility)."""
        fcd = get_fcd(self.ts)
        
        assert 'full' in fcd, "FCD should contain 'full' key"
        assert isinstance(fcd['full'], np.ndarray), "FCD should be numpy array"
        assert fcd['full'].ndim == 2, "FCD should be 2D"
        assert fcd['full'].shape[0] == fcd['full'].shape[1], "FCD should be square"
    
    def test_without_overlap_argument(self):
        """Test that calling without overlap argument works (backward compatibility)."""
        fcd1 = get_fcd(self.ts, tr=1, win_len=30)
        fcd2 = get_fcd(self.ts, tr=1, win_len=30, overlap=None)
        
        # Both should produce identical results
        np.testing.assert_array_equal(
            fcd1['full'], fcd2['full'],
            err_msg="Results should be identical with and without overlap=None"
        )
    
    def test_reproducibility(self):
        """Test that multiple calls with same parameters produce same results."""
        fcd1 = get_fcd(self.ts, tr=1, win_len=30)
        fcd2 = get_fcd(self.ts, tr=1, win_len=30)
        
        np.testing.assert_array_equal(
            fcd1['full'], fcd2['full'],
            err_msg="Multiple calls should produce identical results"
        )
    
    def test_with_positive_flag(self):
        """Test backward compatibility with positive flag."""
        fcd_pos = get_fcd(self.ts, tr=1, win_len=30, positive=True)
        fcd_neg = get_fcd(self.ts, tr=1, win_len=30, positive=False)
        
        # Results should be different
        assert not np.array_equal(fcd_pos['full'], fcd_neg['full']), \
            "positive flag should change results"
        
        # Both should be valid
        assert np.isfinite(fcd_pos['full']).all(), "FCD with positive=True should have finite values"
        assert np.isfinite(fcd_neg['full']).all(), "FCD with positive=False should have finite values"
    
    def test_with_masks(self):
        """Test backward compatibility with masks."""
        mask1 = np.ones((self.n_regions, self.n_regions))
        mask1[0, :] = 0
        mask1[:, 0] = 0
        
        masks = {
            "full": np.ones((self.n_regions, self.n_regions)),
            "mask1": mask1
        }
        
        fcd = get_fcd(self.ts, tr=1, win_len=30, masks=masks)
        
        assert 'full' in fcd, "FCD should contain 'full' key"
        assert 'mask1' in fcd, "FCD should contain 'mask1' key"
        assert not np.array_equal(fcd['full'], fcd['mask1']), \
            "Different masks should produce different results"


@pytest.mark.short
@pytest.mark.fast
class TestGetFcdOverlap:
    """Test the overlap parameter functionality with fraction values (0-1)."""
    
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.n_regions = 5
        self.n_timepoints = 200
        self.ts = np.random.randn(self.n_regions, self.n_timepoints)
    
    def test_overlap_parameter_basic(self):
        """Test that overlap parameter is accepted and produces valid results."""
        fcd = get_fcd(self.ts, tr=1, win_len=30, overlap=0.9)
        
        assert 'full' in fcd, "FCD should contain 'full' key"
        assert isinstance(fcd['full'], np.ndarray), "FCD should be numpy array"
        assert fcd['full'].ndim == 2, "FCD should be 2D"
        assert fcd['full'].shape[0] == fcd['full'].shape[1], "FCD should be square"
        assert np.isfinite(fcd['full']).all(), "FCD should have finite values"
    
    def test_different_overlaps_produce_different_results(self):
        """Test that different overlap values produce different FCD matrices."""
        fcd_overlap90 = get_fcd(self.ts, tr=1, win_len=30, overlap=0.9)  # 90% overlap
        fcd_overlap50 = get_fcd(self.ts, tr=1, win_len=30, overlap=0.5)  # 50% overlap
        fcd_overlap0 = get_fcd(self.ts, tr=1, win_len=30, overlap=0.0)   # 0% overlap
        
        # Different overlaps produce different numbers of windows, hence different FCD shapes
        # Check that all are square matrices
        assert fcd_overlap90['full'].shape[0] == fcd_overlap90['full'].shape[1], \
            "FCD with overlap=0.9 should be square matrix"
        assert fcd_overlap50['full'].shape[0] == fcd_overlap50['full'].shape[1], \
            "FCD with overlap=0.5 should be square matrix"
        assert fcd_overlap0['full'].shape[0] == fcd_overlap0['full'].shape[1], \
            "FCD with overlap=0.0 should be square matrix"
        
        # Higher overlap means more windows, hence larger FCD matrix
        assert fcd_overlap90['full'].shape[0] > fcd_overlap50['full'].shape[0], \
            "Higher overlap (0.9) should produce more windows than lower overlap (0.5)"
        assert fcd_overlap50['full'].shape[0] > fcd_overlap0['full'].shape[0], \
            "Medium overlap (0.5) should produce more windows than no overlap (0.0)"
    
    def test_overlap_zero_vs_no_overlap(self):
        """Test that overlap=0.0 means non-overlapping windows."""
        fcd_no_overlap = get_fcd(self.ts, tr=1, win_len=30, overlap=0.0)
        
        assert 'full' in fcd_no_overlap, "FCD should contain 'full' key"
        assert np.isfinite(fcd_no_overlap['full']).all(), \
            "FCD with overlap=0.0 should have finite values"
    
    def test_overlap_validation(self):
        """Test that invalid overlap values raise appropriate errors."""
        # Overlap > 1.0 should raise error
        with pytest.raises(ValueError, match="overlap must be between 0.0 and 1.0"):
            get_fcd(self.ts, tr=1, win_len=30, overlap=1.5)
        
        # Negative overlap should raise error
        with pytest.raises(ValueError, match="overlap must be between 0.0 and 1.0"):
            get_fcd(self.ts, tr=1, win_len=30, overlap=-0.1)
    
    def test_overlap_with_different_tr(self):
        """Test overlap with different tr values."""
        # tr=2 means win_len is in units of 2ms, but overlap is still fraction
        fcd = get_fcd(self.ts, tr=2, win_len=60, overlap=0.9)
        
        assert 'full' in fcd, "FCD should contain 'full' key"
        assert np.isfinite(fcd['full']).all(), "FCD should have finite values"
    
    def test_overlap_with_positive_flag(self):
        """Test overlap parameter combined with positive flag."""
        fcd = get_fcd(self.ts, tr=1, win_len=30, overlap=0.8, positive=True)
        
        assert 'full' in fcd, "FCD should contain 'full' key"
        assert np.isfinite(fcd['full']).all(), "FCD should have finite values"
    
    def test_overlap_with_masks(self):
        """Test overlap parameter combined with masks."""
        mask1 = np.ones((self.n_regions, self.n_regions))
        mask1[0, :] = 0
        mask1[:, 0] = 0
        
        masks = {
            "full": np.ones((self.n_regions, self.n_regions)),
            "mask1": mask1
        }
        
        fcd = get_fcd(self.ts, tr=1, win_len=30, overlap=0.7, masks=masks)
        
        assert 'full' in fcd, "FCD should contain 'full' key"
        assert 'mask1' in fcd, "FCD should contain 'mask1' key"
    
    def test_overlap_edge_cases(self):
        """Test edge cases: overlap=0.0 and overlap=1.0."""
        # overlap=0.0 should mean no overlap (stride = win_len)
        fcd_0 = get_fcd(self.ts, tr=1, win_len=30, overlap=0.0)
        assert 'full' in fcd_0, "FCD with overlap=0.0 should work"
        
        # overlap=1.0 should mean maximum overlap (stride=1)
        fcd_1 = get_fcd(self.ts, tr=1, win_len=30, overlap=1.0)
        fcd_none = get_fcd(self.ts, tr=1, win_len=30)  # None defaults to stride=1
        
        # Both should produce same number of windows
        assert fcd_1['full'].shape == fcd_none['full'].shape, \
            "overlap=1.0 and overlap=None should produce same result (maximum overlap)"
    
    def test_overlap_intuitive_behavior(self):
        """Test that overlap behaves intuitively: higher value = more overlap."""
        fcd_low = get_fcd(self.ts, tr=1, win_len=30, overlap=0.2)   # 20% overlap
        fcd_high = get_fcd(self.ts, tr=1, win_len=30, overlap=0.95)  # 95% overlap
        
        # Higher overlap should produce more windows
        assert fcd_high['full'].shape[0] > fcd_low['full'].shape[0], \
            "Higher overlap value (0.95) should produce more windows than lower value (0.2)"


@pytest.mark.short
@pytest.mark.fast
class TestGetFcdEdgeCases:
    """Test edge cases and error handling."""
    
    def test_insufficient_timepoints(self):
        """Test that insufficient timepoints raises appropriate error."""
        ts = np.random.randn(5, 50)  # Only 50 timepoints
        
        with pytest.raises(ValueError, match="Length of the time series should be at least 2 times"):
            get_fcd(ts, tr=1, win_len=30)  # Needs at least 60 timepoints
    
    def test_list_input_conversion(self):
        """Test that list input is properly converted to array."""
        # Use random data to avoid correlation issues with constant values
        np.random.seed(456)
        ts_list = np.random.randn(5, 100)  # 5 regions x 100 timepoints
        
        fcd = get_fcd(ts_list, tr=1, win_len=20)
        
        assert 'full' in fcd, "FCD should work with list input"
        assert np.isfinite(fcd['full']).all(), "FCD should have finite values"
    
    def test_deterministic_output(self):
        """Test that output is deterministic for same input."""
        np.random.seed(123)
        ts = np.random.randn(5, 200)
        
        fcd1 = get_fcd(ts, tr=1, win_len=30, overlap=0.8)
        fcd2 = get_fcd(ts, tr=1, win_len=30, overlap=0.8)
        
        np.testing.assert_array_equal(
            fcd1['full'], fcd2['full'],
            err_msg="Same input should produce identical output"
        )
    
    def test_symmetry(self):
        """Test that FCD matrix is symmetric."""
        np.random.seed(456)
        ts = np.random.randn(5, 200)
        
        fcd = get_fcd(ts, tr=1, win_len=30, overlap=0.7)
        fcd_matrix = fcd['full']
        
        # Remove NaN values for comparison
        fcd_matrix_clean = np.nan_to_num(fcd_matrix)
        
        np.testing.assert_allclose(
            fcd_matrix_clean, fcd_matrix_clean.T,
            rtol=1e-10, atol=1e-10,
            err_msg="FCD matrix should be symmetric"
        )


@pytest.mark.short
@pytest.mark.fast
class TestGetFcdParameterNaming:
    """Test backward compatibility for TR -> tr parameter renaming."""
    
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.ts = np.random.randn(5, 200)
    
    def test_tr_parameter(self):
        """Test that new 'tr' parameter works."""
        fcd = get_fcd(self.ts, tr=1, win_len=30)
        
        assert 'full' in fcd, "FCD should work with 'tr' parameter"
        assert np.isfinite(fcd['full']).all(), "FCD should have finite values"
    
    def test_TR_parameter_deprecated(self):
        """Test that old 'TR' parameter still works but raises deprecation warning."""
        with pytest.warns(DeprecationWarning, match="Parameter 'TR' is deprecated"):
            fcd = get_fcd(self.ts, TR=1, win_len=30)
        
        assert 'full' in fcd, "FCD should work with deprecated 'TR' parameter"
        assert np.isfinite(fcd['full']).all(), "FCD should have finite values"
    
    def test_TR_and_tr_same_result(self):
        """Test that TR and tr produce identical results."""
        with pytest.warns(DeprecationWarning):
            fcd_TR = get_fcd(self.ts, TR=1, win_len=30)
        
        fcd_tr = get_fcd(self.ts, tr=1, win_len=30)
        
        np.testing.assert_array_equal(
            fcd_TR['full'], fcd_tr['full'],
            err_msg="TR and tr should produce identical results"
        )
    
    def test_both_TR_and_tr_raises_error(self):
        """Test that specifying both TR and tr raises an error."""
        with pytest.raises(ValueError, match="Cannot specify both 'TR' and 'tr'"):
            get_fcd(self.ts, TR=1, tr=1, win_len=30)
    
    def test_default_tr_value(self):
        """Test that default tr value is 1."""
        fcd_default = get_fcd(self.ts, win_len=30)
        fcd_explicit = get_fcd(self.ts, tr=1, win_len=30)
        
        np.testing.assert_array_equal(
            fcd_default['full'], fcd_explicit['full'],
            err_msg="Default tr should be 1"
        )
    
    def test_tr_with_overlap(self):
        """Test that tr parameter works with overlap."""
        fcd = get_fcd(self.ts, tr=1, win_len=30, overlap=0.8)
        
        assert 'full' in fcd, "FCD should work with tr and overlap"
        assert np.isfinite(fcd['full']).all(), "FCD should have finite values"
    
    def test_TR_with_overlap_deprecated(self):
        """Test that TR parameter works with overlap (with deprecation warning)."""
        with pytest.warns(DeprecationWarning):
            fcd = get_fcd(self.ts, TR=1, win_len=30, overlap=0.8)
        
        assert 'full' in fcd, "FCD should work with TR and overlap"
        assert np.isfinite(fcd['full']).all(), "FCD should have finite values"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
