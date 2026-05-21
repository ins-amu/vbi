import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from vbi.feature_extraction.features_utils_jax import get_fcd as get_fcd_jax


@pytest.fixture
def sample_ts():
    rng = np.random.default_rng(123)
    return rng.standard_normal((5, 200))


@pytest.mark.short
@pytest.mark.fast
def test_get_fcd_jax_default_overlap_matches_max_overlap(sample_ts):
    fcd_default = np.asarray(get_fcd_jax(jnp.asarray(sample_ts), tr=1, win_len=30))
    fcd_max_overlap = np.asarray(
        get_fcd_jax(jnp.asarray(sample_ts), tr=1, win_len=30, overlap=1.0)
    )

    assert fcd_default.shape == fcd_max_overlap.shape
    np.testing.assert_allclose(fcd_default, fcd_max_overlap, atol=1e-6, rtol=1e-6)


@pytest.mark.short
@pytest.mark.fast
def test_get_fcd_jax_returns_finite_symmetric_matrix(sample_ts):
    fcd = np.asarray(get_fcd_jax(jnp.asarray(sample_ts), tr=1, win_len=30, overlap=0.8))

    assert fcd.ndim == 2
    assert fcd.shape[0] == fcd.shape[1]
    assert np.isfinite(fcd).all()
    np.testing.assert_allclose(fcd, fcd.T, atol=1e-6, rtol=1e-6)


@pytest.mark.short
@pytest.mark.fast
def test_get_fcd_jax_higher_overlap_produces_more_windows(sample_ts):
    fcd_high_overlap = np.asarray(
        get_fcd_jax(jnp.asarray(sample_ts), tr=1, win_len=30, overlap=0.9)
    )
    fcd_mid_overlap = np.asarray(
        get_fcd_jax(jnp.asarray(sample_ts), tr=1, win_len=30, overlap=0.5)
    )
    fcd_no_overlap = np.asarray(
        get_fcd_jax(jnp.asarray(sample_ts), tr=1, win_len=30, overlap=0.0)
    )

    assert fcd_high_overlap.shape[0] > fcd_mid_overlap.shape[0]
    assert fcd_mid_overlap.shape[0] > fcd_no_overlap.shape[0]


@pytest.mark.short
@pytest.mark.fast
def test_get_fcd_jax_positive_flag_changes_output(sample_ts):
    fcd_default = np.asarray(
        get_fcd_jax(jnp.asarray(sample_ts), tr=1, win_len=30, overlap=0.8, positive=False)
    )
    fcd_positive = np.asarray(
        get_fcd_jax(jnp.asarray(sample_ts), tr=1, win_len=30, overlap=0.8, positive=True)
    )

    assert fcd_default.shape == fcd_positive.shape
    assert np.isfinite(fcd_positive).all()
    assert not np.allclose(fcd_default, fcd_positive)
