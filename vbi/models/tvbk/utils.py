import tvbk as m
import numpy as np
import scipy.sparse
import collections
from vbi.models.cupy.utils import repmat_vec, is_seq  # TODO move it to vbi.utils


def setup_connectivity(weights, delays=None):
    """
    Sets up connectivity using provided weights and delays.

    Args:
        weights (np.ndarray): The weight matrix.
        delays (np.ndarray): The delay matrix.

    Returns:
        conn: The connection object.
    """

    # Convert weights to sparse matrix
    s_w = scipy.sparse.csr_matrix(weights)
    num_node = weights.shape[0]

    # TODO! check how to handle delays if not provided
    if delays is None:
        delays = np.zeros_like(weights)

    # Ensure delays are valid
    idelays = (delays[weights != 0]).astype(np.uint32) + 2
    assert idelays.max() < delays.shape[1]
    assert idelays.min() >= 2

    # Create the connection object
    conn = m.Conn(num_node, s_w.data.size)
    conn.weights[:] = s_w.data.astype(np.float32)
    conn.indptr[:] = s_w.indptr.astype(np.uint32)
    conn.indices[:] = s_w.indices.astype(np.uint32)
    conn.idelays[:] = idelays

    return conn


def prepare_vec(x, num_batch, dtype=np.float32):
    """
    Check and prepare vector dimension and type.

    Parameters
    ----------
    x: array 1d
        vector to be prepared, if x is a scalar, only the type is modified.
    num_batch: int
        number of batched simulations.

    Returns
    -------
    x: array [len(x), num_batch]
        prepared vector.

    """

    if not is_seq(x):
        return dtype(x)
    else:
        x = np.array(x)
        if x.ndim == 1:
            x = repmat_vec(x, num_batch, "cpu")
        elif x.ndim == 2:
            assert x.shape[1] == num_batch, "second dimension of x must be equal to ns"
            x = x.astype(dtype)
        else:
            raise ValueError("x.ndim must be 1 or 2")
    return x
