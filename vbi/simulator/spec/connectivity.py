"""
Connectivity utilities - load, normalise, and save SC matrices.

Users bring connectivity data in many formats; this module provides a
single entry point that accepts numpy arrays, plain text files (.txt),
CSV files, NPZ archives, and NumPy `.npy` files.

Quick start
-----------
>>> W, D = prepare_connectivity('weights.txt', 'tract_lengths.txt')
>>> spec = SimulationSpec(..., weights=W, tract_lengths=D)

Or save to .npz for use with ``VBIInference.from_config``::

>>> save_connectivity(W, D, 'connectivity.npz')

Config YAML then just needs::

    sim:
      connectivity: connectivity.npz
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _load_array(src) -> np.ndarray:
    """Load a matrix from a path (txt / csv / npz / npy) or return as-is."""
    if isinstance(src, np.ndarray):
        return src.astype(np.float64)

    path = Path(src)
    suffix = path.suffix.lower()

    if suffix in (".txt", ".csv"):
        return np.loadtxt(path, dtype=np.float64)
    if suffix == ".npy":
        return np.load(path).astype(np.float64)
    if suffix == ".npz":
        d = np.load(path)
        # Accept first array if only one, else require 'weights' key
        keys = list(d.keys())
        if "weights" in keys:
            return d["weights"].astype(np.float64)
        if len(keys) == 1:
            return d[keys[0]].astype(np.float64)
        raise ValueError(
            f"{path}: .npz has multiple arrays {keys}; pass the key explicitly "
            f"or use prepare_connectivity_from_npz()."
        )
    raise ValueError(
        f"Unsupported file format {suffix!r} for {path}. "
        f"Supported: .txt, .csv, .npy, .npz"
    )


def prepare_connectivity(
    weights,
    tract_lengths=None,
    normalize: bool = True,
    normalise: bool | None = None,   # British spelling alias
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load and validate SC connectivity matrices from various sources.

    Parameters
    ----------
    weights : array-like | str | Path
        (n_nodes, n_nodes) weight matrix, **or** path to a file:

        * ``.txt`` / ``.csv`` - space- or comma-separated text
        * ``.npy`` - NumPy binary
        * ``.npz`` - NumPy archive with a ``weights`` key (or a single key)

    tract_lengths : array-like | str | Path | None
        (n_nodes, n_nodes) tract-length matrix in **mm**.  Same formats as
        ``weights``.  ``None`` → zero delays (pure ODE / SDE).

    normalize : bool
        Row-sum normalise ``weights`` so each row sums to 1.  Recommended
        for most VBI models (MPR, VEP, GHB) where coupling strength is
        controlled separately by a ``G`` parameter.

    Returns
    -------
    weights       : (n, n) float64
    tract_lengths : (n, n) float64   (zeros if None was supplied)

    Examples
    --------
    >>> W, D = prepare_connectivity('weights.txt', 'tract_lengths.txt')
    >>> W, D = prepare_connectivity(my_array)               # already in memory
    >>> W, D = prepare_connectivity('connectivity.npz')     # npz has 'weights' key
    """
    if normalise is not None:
        normalize = normalise

    W = _load_array(weights)
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError(
            f"weights must be a square 2-D matrix; got shape {W.shape}."
        )
    if np.any(W < 0):
        raise ValueError("weights must be non-negative.")

    if tract_lengths is None:
        D = np.zeros_like(W)
    else:
        D = _load_array(tract_lengths)
        if D.shape != W.shape:
            raise ValueError(
                f"tract_lengths shape {D.shape} must match weights shape {W.shape}."
            )
        if np.any(D < 0):
            raise ValueError("tract_lengths must be non-negative.")

    if normalize:
        row_sums = W.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        W = W / row_sums

    return W, D


def prepare_connectivity_from_npz(
    path,
    weights_key: str = "weights",
    tract_lengths_key: str = "tract_lengths",
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load connectivity from a `.npz` file with explicit key names.

    Parameters
    ----------
    path              : str | Path
    weights_key       : str  key for the weight matrix (default ``'weights'``)
    tract_lengths_key : str  key for tract lengths (default ``'tract_lengths'``)
    normalize         : bool row-sum normalise weights

    Returns
    -------
    weights, tract_lengths  (n, n) float64
    """
    d  = np.load(path)
    W  = d[weights_key].astype(np.float64)
    tl = d[tract_lengths_key].astype(np.float64) if tract_lengths_key in d else None
    return prepare_connectivity(W, tl, normalize=normalize)


def save_connectivity(
    weights,
    tract_lengths=None,
    path: str | Path = "connectivity.npz",
    normalize: bool = False,
) -> Path:
    """
    Save connectivity matrices to a ``.npz`` file compatible with
    ``SimulationSpec.from_dict`` and ``VBIInference.from_config``.

    Parameters
    ----------
    weights       : (n, n) array-like or path (loaded via prepare_connectivity)
    tract_lengths : (n, n) array-like or path | None
    path          : output file path (default ``'connectivity.npz'``)
    normalize     : bool  row-sum normalise before saving (default False -
                    assumes data is already prepared)

    Returns
    -------
    Path  the written file path
    """
    W, D = prepare_connectivity(weights, tract_lengths, normalize=normalize)
    path = Path(path)
    if not path.suffix:
        path = path.with_suffix(".npz")
    np.savez(path, weights=W, tract_lengths=D)
    return path
