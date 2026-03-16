import jax
import jax.numpy as jnp
from functools import partial


def get_fcd(
    ts: jnp.ndarray,
    tr: float | int,
    win_len: float | int = 30,
    overlap: float | None = None,
    positive: bool = False
):
    """
    JAX-compatible and jittable version of get_fcd for use within vmap/jit contexts.

    Args:
        ts: Time series array (n_regions, n_samples)
        tr: Repetition time in any time unit (milliseconds, seconds, etc.).
            This parameter sets the unit for `win_len`.
        win_len: Sliding window length in the same units as `tr`.
            Default is 30.
        overlap: Overlap between consecutive windows as a fraction (0.0 to 1.0).
            - overlap=0.0: no overlap (stride = win_len, non-overlapping windows)
            - overlap=0.5: 50% overlap (stride = win_len/2)
            - overlap=0.9: 90% overlap (stride = win_len/10, high overlap)
            - overlap=1.0: maximum overlap (stride = 1 sample)
            - If None (default): uses stride of 1 sample (equivalent to overlap=1.0)
            
            **Intuitive rule**: Higher overlap value = MORE overlap between windows
            
        positive: If True, only positive values of FC are considered (negative correlations set to 0).
            Default is False.

    Returns:
        FCD: Functional Connectivity Dynamics matrix (n_windows x n_windows)
        
    Examples:
        >>> import jax.numpy as jnp
        >>> ts = jnp.array(np.random.randn(5, 200))  # 5 regions, 200 timepoints
        >>> 
        >>> # Maximum overlap (stride=1)
        >>> fcd = get_fcd(ts, tr=1, win_len=30)
        >>> 
        >>> # 90% overlap (high overlap)
        >>> fcd = get_fcd(ts, tr=1, win_len=30, overlap=0.9)
        >>> 
        >>> # 50% overlap (moderate)
        >>> fcd = get_fcd(ts, tr=1, win_len=30, overlap=0.5)
        >>> 
        >>> # No overlap
        >>> fcd = get_fcd(ts, tr=1, win_len=30, overlap=0.0)
    """
    # Convert overlap to default value if None (must be done before jit)
    if overlap is None:
        overlap = 1.0
    
    # Calculate window length in samples
    win_len_samples = jnp.floor(win_len / tr).astype(jnp.int32)
    n_regions, n_samples = ts.shape

    # Determine stride based on overlap (fraction-based)
    # overlap=0 → stride=win_len (no overlap)
    # overlap=1 → stride=1 (maximum overlap)
    # overlap=0.5 → stride=win_len/2 (50% overlap)
    stride = jnp.maximum(1, jnp.floor(win_len_samples * (1.0 - overlap)).astype(jnp.int32))
    
    # Calculate number of windows
    n_windows = jnp.floor((n_samples - win_len_samples) / stride + 1).astype(jnp.int32)

    # upper triangle indices
    triu_indices = jnp.triu_indices(n_regions, k=1)
    n_connections = len(triu_indices[0])

    # Compute FC for each window using vmap for efficiency
    def compute_window_fc(i):
        start_idx = stride * i
        # Use dynamic_slice for JAX compatibility with vmap
        window_data = jax.lax.dynamic_slice(
            ts,
            (0, start_idx),  # start indices (n_regions, n_samples)
            (n_regions, win_len_samples),  # slice sizes
        )

        # Compute correlation for this window (data is already n_regions x n_samples)
        fc_window = jnp.corrcoef(window_data)
        fc_window = jnp.nan_to_num(fc_window, nan=0.0)
        
        # Apply positive filter if requested (JAX-compatible way)
        fc_window = jnp.where(positive, jnp.maximum(fc_window, 0.0), fc_window)

        # Extract upper triangle
        return fc_window[triu_indices]

    # Vectorize over all windows
    FC_t = jax.vmap(compute_window_fc)(jnp.arange(n_windows))
    FC_t = FC_t.T  # Shape: (n_connections, n_windows)

    # Compute FCD by correlating the FCs with each other
    FCD = jnp.corrcoef(FC_t.T)
    FCD = jnp.nan_to_num(FCD, nan=0.0)

    return FCD


def get_fc(ts: jnp.ndarray):
    """
    Compute functional connectivity (FC) matrix from time series using JAX.

    Args:
        ts: Time series array (n_regions, n_samples)
    Returns:
        FC: Functional connectivity matrix (n_regions x n_regions)
    """
    fc = jnp.corrcoef(ts)
    fc = jnp.nan_to_num(fc, nan=0.0)
    return fc


# Jittable version of get_fcd
# get_fcd_jit = jax.jit(get_fcd, static_argnums=(1, 2, 3, 4))
