"""
Tests for vbi.feature_extraction.features_utils

Covers input preparation, mask generation, matrix utilities,
and validation helpers.

Run with:
    pytest vbi/tests/test_features_utils.py -v
"""

import unittest
import numpy as np
import pytest
from parameterized import parameterized

from vbi.feature_extraction.features_utils import (
    prepare_input_ts,
    make_mask,
    get_intrah_mask,
    get_interh_mask,
    get_masks,
    is_sequence,
    set_k_diagonals,
    if_symmetric,
)


@pytest.mark.short
@pytest.mark.fast
class TestIsSequence(unittest.TestCase):
    """Check that is_sequence correctly identifies sequences vs scalars."""

    def test_list_is_sequence(self):
        self.assertTrue(is_sequence([1, 2, 3]))

    def test_tuple_is_sequence(self):
        self.assertTrue(is_sequence((1, 2, 3)))

    def test_ndarray_is_sequence(self):
        self.assertTrue(is_sequence(np.array([1, 2, 3])))

    def test_empty_list_is_sequence(self):
        self.assertTrue(is_sequence([]))

    def test_string_is_not_sequence(self):
        self.assertFalse(is_sequence("hello"))

    def test_int_is_not_sequence(self):
        self.assertFalse(is_sequence(42))

    def test_float_is_not_sequence(self):
        self.assertFalse(is_sequence(3.14))

    def test_none_is_not_sequence(self):
        self.assertFalse(is_sequence(None))

    def test_dict_is_not_sequence(self):
        self.assertFalse(is_sequence({"a": 1}))


@pytest.mark.short
@pytest.mark.fast
class TestMakeMask(unittest.TestCase):
    """Verify mask creation with given indices, diagonal zeroing, and error handling."""

    def test_two_indices(self):
        mask = make_mask(4, [0, 1])
        # only (0,1) and (1,0) should be 1
        self.assertEqual(mask[0, 1], 1)
        self.assertEqual(mask[1, 0], 1)
        # diagonal must be zero
        self.assertEqual(mask[0, 0], 0)
        self.assertEqual(mask[1, 1], 0)
        # everything outside indices block should be zero
        self.assertEqual(mask[2, 3], 0)

    def test_all_indices(self):
        mask = make_mask(3, [0, 1, 2])
        expected = np.ones((3, 3), dtype=np.int64) - np.eye(3, dtype=np.int64)
        np.testing.assert_array_equal(mask, expected)

    def test_single_index_all_zeros(self):
        # single node cant have off-diagonal connections
        mask = make_mask(3, [0])
        np.testing.assert_array_equal(mask, np.zeros((3, 3), dtype=np.int64))

    def test_diagonal_always_zero(self):
        mask = make_mask(5, [0, 1, 2, 3, 4])
        for i in range(5):
            self.assertEqual(mask[i, i], 0)

    def test_mask_is_symmetric(self):
        mask = make_mask(6, [1, 3, 5])
        np.testing.assert_array_equal(mask, mask.T)

    def test_invalid_indices_type_raises(self):
        with self.assertRaises(ValueError):
            make_mask(4, "not a list")

    def test_float_indices_raises(self):
        with self.assertRaises(ValueError):
            make_mask(4, [0.5, 1.5])

    def test_out_of_range_indices_raises(self):
        with self.assertRaises(ValueError):
            make_mask(3, [0, 5])


@pytest.mark.short
@pytest.mark.fast
class TestGetIntrahMask(unittest.TestCase):
    """Intrahemispheric mask: block-diagonal ones, off-diagonal zeros."""

    def test_4_nodes(self):
        mask = get_intrah_mask(4)
        self.assertEqual(mask.shape, (4, 4))
        # top-left 2x2 block should be all ones
        np.testing.assert_array_equal(mask[:2, :2], np.ones((2, 2)))
        # bottom-right 2x2 block should be all ones
        np.testing.assert_array_equal(mask[2:, 2:], np.ones((2, 2)))
        # cross-hemisphere blocks should be zero
        np.testing.assert_array_equal(mask[:2, 2:], np.zeros((2, 2)))
        np.testing.assert_array_equal(mask[2:, :2], np.zeros((2, 2)))

    def test_shape_10_nodes(self):
        mask = get_intrah_mask(10)
        self.assertEqual(mask.shape, (10, 10))
        # top-left block is 5x5
        np.testing.assert_array_equal(mask[:5, :5], np.ones((5, 5)))

    def test_returns_float_array(self):
        mask = get_intrah_mask(6)
        self.assertTrue(mask.dtype in [np.float64, np.float32])


@pytest.mark.short
@pytest.mark.fast
class TestGetInterhMask(unittest.TestCase):
    """Interhemispheric mask: only cross-hemisphere entries are nonzero."""

    def test_4_nodes(self):
        mask = get_interh_mask(4)
        self.assertEqual(mask.shape, (4, 4))
        # intrahemispheric blocks should be zero
        np.testing.assert_array_equal(mask[:2, :2], np.zeros((2, 2)))
        np.testing.assert_array_equal(mask[2:, 2:], np.zeros((2, 2)))

    def test_nonzero_cross_hemisphere(self):
        mask = get_interh_mask(4)
        # at least some cross-hemisphere entries should be nonzero
        cross_block = mask[:2, 2:].sum() + mask[2:, :2].sum()
        self.assertGreater(cross_block, 0)

    def test_shape(self):
        mask = get_interh_mask(8)
        self.assertEqual(mask.shape, (8, 8))


@pytest.mark.short
@pytest.mark.fast
class TestGetMasks(unittest.TestCase):
    """Mask factory function for different network types."""

    def test_full_mask_is_all_ones(self):
        masks = get_masks(4, ["full"])
        self.assertIn("full", masks)
        np.testing.assert_array_equal(masks["full"], np.ones((4, 4)))

    def test_all_three_networks(self):
        masks = get_masks(6, ["full", "intrah", "interh"])
        self.assertEqual(len(masks), 3)
        for key in ["full", "intrah", "interh"]:
            self.assertIn(key, masks)
            self.assertEqual(masks[key].shape, (6, 6))

    def test_string_input_works(self):
        # passing a single string instead of list
        masks = get_masks(4, "full")
        self.assertIn("full", masks)

    def test_invalid_network_raises(self):
        with self.assertRaises(ValueError):
            get_masks(4, ["banana"])

    def test_intrah_interh_dont_overlap(self):
        masks = get_masks(6, ["intrah", "interh"])
        overlap = masks["intrah"] * masks["interh"]
        np.testing.assert_array_equal(overlap, np.zeros((6, 6)))


@pytest.mark.short
@pytest.mark.fast
class TestSetKDiagonals(unittest.TestCase):
    """Setting k diagonals of a matrix to a given value."""

    def test_main_diagonal_to_zero(self):
        A = np.ones((4, 4))
        result = set_k_diagonals(A, k=0, value=0)
        np.testing.assert_array_equal(np.diag(result), np.zeros(4))

    def test_preserves_corners(self):
        # k=0 should only touch the main diagonal
        A = np.ones((3, 3))
        result = set_k_diagonals(A.copy(), k=0, value=0)
        self.assertEqual(result[0, 2], 1)
        self.assertEqual(result[2, 0], 1)

    def test_k1_zeros_three_diagonals(self):
        A = np.ones((5, 5)) * 7.0
        result = set_k_diagonals(A, k=1, value=0)
        # main diagonal should be 0
        for i in range(5):
            self.assertEqual(result[i, i], 0)

    def test_1d_array_raises(self):
        with self.assertRaises(ValueError):
            set_k_diagonals(np.ones(5), k=0, value=0)

    def test_k_too_large_raises(self):
        with self.assertRaises(ValueError):
            set_k_diagonals(np.ones((3, 3)), k=3, value=0)

    def test_float_k_raises(self):
        with self.assertRaises(ValueError):
            set_k_diagonals(np.ones((3, 3)), k=1.5, value=0)

    def test_list_input_converted(self):
        A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        result = set_k_diagonals(A, k=0, value=0)
        self.assertEqual(result[0, 0], 0)
        self.assertEqual(result[1, 1], 0)


@pytest.mark.short
@pytest.mark.fast
class TestIfSymmetric(unittest.TestCase):
    """Symmetry check for matrices."""

    def test_symmetric_matrix(self):
        A = np.array([[1, 2, 3], [2, 5, 6], [3, 6, 9]])
        self.assertTrue(if_symmetric(A))

    def test_nonsymmetric_matrix(self):
        A = np.array([[1, 2], [3, 4]])
        self.assertFalse(if_symmetric(A))

    def test_identity_is_symmetric(self):
        self.assertTrue(if_symmetric(np.eye(5)))

    def test_zeros_is_symmetric(self):
        self.assertTrue(if_symmetric(np.zeros((4, 4))))

    def test_list_input(self):
        self.assertTrue(if_symmetric([[1, 0], [0, 1]]))

    def test_almost_symmetric_within_tol(self):
        A = np.array([[1.0, 2.0], [2.0 + 1e-10, 1.0]])
        self.assertTrue(if_symmetric(A))

    def test_1d_raises(self):
        with self.assertRaises(ValueError):
            if_symmetric(np.array([1, 2, 3]))


@pytest.mark.short
@pytest.mark.fast
class TestPrepareInputTs(unittest.TestCase):
    """Input preparation and validation for time series."""

    def test_valid_2d_input(self):
        ts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        valid, result = prepare_input_ts(ts)
        self.assertTrue(valid)
        self.assertEqual(result.shape, (2, 3))

    def test_selects_correct_indices(self):
        ts = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]])
        valid, result = prepare_input_ts(ts, indices=[0, 2])
        self.assertTrue(valid)
        np.testing.assert_array_equal(result[0], [10, 20, 30])
        np.testing.assert_array_equal(result[1], [70, 80, 90])

    def test_1d_reshaped_to_2d(self):
        ts = np.array([1, 2, 3, 4])
        valid, result = prepare_input_ts(ts, indices=[0])
        self.assertTrue(valid)
        self.assertEqual(result.ndim, 2)

    def test_nan_returns_false(self):
        ts = np.array([[1, np.nan, 3]])
        valid, _ = prepare_input_ts(ts)
        self.assertFalse(valid)

    def test_inf_returns_false(self):
        ts = np.array([[1, np.inf, 3]])
        valid, _ = prepare_input_ts(ts)
        self.assertFalse(valid)

    def test_list_input_converted(self):
        ts = [[1, 2, 3], [4, 5, 6]]
        valid, result = prepare_input_ts(ts)
        self.assertTrue(valid)
        self.assertIsInstance(result, np.ndarray)

    def test_invalid_indices_type_raises(self):
        ts = np.array([[1, 2, 3]])
        with self.assertRaises(ValueError):
            prepare_input_ts(ts, indices="bad")

    def test_float_indices_raises(self):
        ts = np.array([[1, 2, 3]])
        with self.assertRaises(ValueError):
            prepare_input_ts(ts, indices=[0.5])

    def test_out_of_range_indices_raises(self):
        ts = np.array([[1, 2, 3]])
        with self.assertRaises(ValueError):
            prepare_input_ts(ts, indices=[5])

    def test_none_indices_uses_all(self):
        ts = np.array([[1, 2], [3, 4], [5, 6]])
        valid, result = prepare_input_ts(ts)
        self.assertTrue(valid)
        self.assertEqual(result.shape[0], 3)


if __name__ == "__main__":
    unittest.main()
