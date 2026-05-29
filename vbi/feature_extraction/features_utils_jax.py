import jax
import jax.numpy as jnp
import math


def get_fcd(
    ts: jnp.ndarray,
    tr: float | int,
    win_len: float | int = 30,
    overlap: float | None = None,
    positive: bool = False
):
    """
    JAX-compatible and jittable version of :func:`get_fcd` for use within
    ``vmap`` and ``jit`` contexts.

    :param ts: Time series array with shape ``(n_regions, n_samples)``.
    :type ts: jnp.ndarray
    :param tr: Repetition time in any time unit. This sets the unit used by
        ``win_len``.
    :type tr: float | int
    :param win_len: Sliding window length in the same units as ``tr``.
        Defaults to ``30``.
    :type win_len: float | int
    :param overlap: Overlap between consecutive windows as a fraction in
        ``[0.0, 1.0]``. If ``None``, a stride of one sample is used, which is
        equivalent to ``overlap=1.0``.

        - ``overlap=0.0``: no overlap, ``stride = win_len``
        - ``overlap=0.5``: 50% overlap, ``stride = win_len / 2``
        - ``overlap=0.9``: 90% overlap, ``stride = win_len / 10``
        - ``overlap=1.0``: maximum overlap, ``stride = 1``

        Higher ``overlap`` means more overlap between windows.
    :type overlap: float | None
    :param positive: If ``True``, only positive FC values are kept and
        negative correlations are set to zero. Defaults to ``False``.
    :type positive: bool
    :returns: Functional Connectivity Dynamics matrix with shape
        ``(n_windows, n_windows)``.
    :rtype: jnp.ndarray

    **Example**

    .. code-block:: python

        import numpy as np
        import jax.numpy as jnp

        ts = jnp.array(np.random.randn(5, 200))

        fcd_max_overlap = get_fcd(ts, tr=1, win_len=30)
        fcd_high_overlap = get_fcd(ts, tr=1, win_len=30, overlap=0.9)
        fcd_medium_overlap = get_fcd(ts, tr=1, win_len=30, overlap=0.5)
        fcd_no_overlap = get_fcd(ts, tr=1, win_len=30, overlap=0.0)
    """
    # Convert overlap to default value if None (must be done before jit)
    if overlap is None:
        overlap = 1.0
    
    n_regions, n_samples = ts.shape
    win_len_samples = int(math.floor(win_len / tr))
    if win_len_samples <= 0:
        raise ValueError("win_len / tr must be at least one sample.")
    if n_samples < win_len_samples:
        raise ValueError(
            f"Time series is shorter than win_len: n_samples={n_samples}, "
            f"win_len_samples={win_len_samples}."
        )

    # Determine stride based on overlap (fraction-based)
    # overlap=0 → stride=win_len (no overlap)
    # overlap=1 → stride=1 (maximum overlap)
    # overlap=0.5 → stride=win_len/2 (50% overlap)
    if not 0.0 <= overlap <= 1.0:
        raise ValueError(f"overlap must be between 0.0 and 1.0, got {overlap}.")
    stride = max(1, int(math.floor(win_len_samples * (1.0 - overlap))))
    
    # Calculate number of windows
    n_windows = int(math.floor((n_samples - win_len_samples) / stride + 1))

    # upper triangle indices
    triu_indices = jnp.triu_indices(n_regions, k=1)

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
        if positive:
            fc_window = jnp.maximum(fc_window, 0.0)

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
