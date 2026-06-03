"""
Structural connectivity for brain network models.

Quick start
-----------
>>> conn = Connectivity.from_file('weights.txt', 'tract_lengths.txt')
>>> spec = SimulationSpec(..., connectivity=conn)

Load from a TVB Connectivity object (TVB optional)::

>>> conn = Connectivity.from_tvb(tvb_conn)

Save / load round-trip::

>>> conn.save('connectivity.npz')
>>> conn = Connectivity.load('connectivity.npz')

Config YAML just needs::

    sim:
      connectivity: connectivity.npz
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _load_array(src) -> np.ndarray:
    """Load a matrix from a path (txt/csv/npy/npz) or return as-is."""
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
        keys = list(d.keys())
        if "weights" in keys:
            return d["weights"].astype(np.float64)
        if len(keys) == 1:
            return d[keys[0]].astype(np.float64)
        raise ValueError(
            f"{path}: .npz has multiple arrays {keys}; "
            f"use Connectivity.load() for full archives."
        )
    raise ValueError(
        f"Unsupported file format {suffix!r} for {path}. "
        f"Supported: .txt, .csv, .npy, .npz"
    )


@dataclass
class Connectivity:
    """
    Structural connectivity for a brain network model.

    Parameters
    ----------
    weights : (n, n) array-like
        SC weight matrix (non-negative).
    tract_lengths : (n, n) array-like | None
        Tract lengths in mm. ``None`` → zero delays (pure ODE/SDE).
    speed : float
        Conduction velocity in mm/ms (default 4.0).
    region_labels : (n,) str array, optional
    centres : (n, 3) float array, optional
    areas : (n,) float array, optional
    hemispheres : (n,) bool array, optional
    """

    weights:       np.ndarray
    tract_lengths: np.ndarray | None = None
    speed:         float = 4.0
    normalize:     bool = False
    region_labels: np.ndarray | None = None
    centres:       np.ndarray | None = None
    areas:         np.ndarray | None = None
    hemispheres:   np.ndarray | None = None

    def __post_init__(self):
        object.__setattr__(self, "weights",
                           np.asarray(self.weights, dtype=np.float64))
        w = self.weights
        if w.ndim != 2 or w.shape[0] != w.shape[1]:
            raise ValueError(f"weights must be square 2-D; got shape {w.shape}.")
        if np.any(w < 0):
            raise ValueError("weights must be non-negative.")
        if self.normalize:
            row_sums = w.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            object.__setattr__(self, "weights", w / row_sums)
            object.__setattr__(self, "normalize", False)

        if self.tract_lengths is None:
            object.__setattr__(self, "tract_lengths", np.zeros_like(w))
        else:
            object.__setattr__(self, "tract_lengths",
                               np.asarray(self.tract_lengths, dtype=np.float64))
        tl = self.tract_lengths
        if tl.shape != w.shape:
            raise ValueError(
                f"tract_lengths shape {tl.shape} must match weights shape {w.shape}."
            )
        if np.any(tl < 0):
            raise ValueError("tract_lengths must be non-negative.")
        if self.speed <= 0:
            raise ValueError(f"speed must be > 0 mm/ms; got {self.speed!r}.")

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def n_nodes(self) -> int:
        return self.weights.shape[0]

    @property
    def has_delays(self) -> bool:
        """True when at least one tract length is non-zero."""
        return bool(self.tract_lengths.any())

    def delay_steps(self, dt: float) -> np.ndarray:
        """Return (n, n) int32 array of delay in integration steps."""
        raw = self.tract_lengths / (self.speed * dt)
        return np.round(raw).astype(np.int32)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_file(
        cls,
        weights,
        tract_lengths=None,
        *,
        normalize: bool = True,
        normalise: bool | None = None,
        speed: float = 4.0,
        region_labels=None,
        centres=None,
        areas=None,
        hemispheres=None,
    ) -> "Connectivity":
        """
        Load connectivity from files or arrays.

        Parameters
        ----------
        weights : array-like | str | Path
            Weight matrix or path to a ``.txt`` / ``.csv`` / ``.npy`` /
            ``.npz`` file.
        tract_lengths : array-like | str | Path | None
            Tract lengths in mm. Same formats as ``weights``.
            ``None`` → zero delays.
        normalize : bool
            Row-sum normalise ``weights`` (default ``True``).
        speed : float
            Conduction velocity in mm/ms (default 4.0).
        region_labels, centres, areas, hemispheres : optional
            Rich metadata arrays (see class docstring).
        """
        if normalise is not None:
            normalize = normalise

        W = _load_array(weights)
        if normalize:
            row_sums = W.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            W = W / row_sums

        D = None if tract_lengths is None else _load_array(tract_lengths)

        return cls(
            weights=W,
            tract_lengths=D,
            speed=speed,
            region_labels=np.asarray(region_labels) if region_labels is not None else None,
            centres=np.asarray(centres, dtype=np.float64) if centres is not None else None,
            areas=np.asarray(areas, dtype=np.float64) if areas is not None else None,
            hemispheres=np.asarray(hemispheres, dtype=bool) if hemispheres is not None else None,
        )

    @classmethod
    def from_tvb(cls, conn) -> "Connectivity":
        """
        Build from a ``tvb.datatypes.connectivity.Connectivity`` instance.

        Raises ``ImportError`` if TVB is not installed.
        """
        try:
            from tvb.datatypes.connectivity import Connectivity as _TVBConn  # noqa: F401
        except ImportError:
            raise ImportError(
                "tvb-library is not installed. "
                "Install it with:  pip install tvb-library"
            )

        speed_arr = np.asarray(conn.speed) if conn.speed is not None else None
        speed = float(speed_arr.flat[0]) if speed_arr is not None else 4.0

        return cls(
            weights=conn.weights.copy(),
            tract_lengths=conn.tract_lengths.copy(),
            speed=speed,
            region_labels=getattr(conn, "region_labels", None),
            centres=getattr(conn, "centres", None),
            areas=getattr(conn, "areas", None),
            hemispheres=getattr(conn, "hemispheres", None),
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path = "connectivity.npz") -> Path:
        """
        Save to a ``.npz`` file loadable by :meth:`load` and
        ``SimulationSpec.from_dict``.
        """
        path = Path(path)
        if not path.suffix:
            path = path.with_suffix(".npz")
        arrays: dict = dict(
            weights=self.weights,
            tract_lengths=self.tract_lengths,
            speed=np.array([self.speed]),
        )
        if self.region_labels is not None:
            arrays["region_labels"] = self.region_labels
        if self.centres is not None:
            arrays["centres"] = self.centres
        if self.areas is not None:
            arrays["areas"] = self.areas
        if self.hemispheres is not None:
            arrays["hemispheres"] = self.hemispheres
        np.savez(path, **arrays)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Connectivity":
        """Load a ``.npz`` file saved by :meth:`save`."""
        d = np.load(path, allow_pickle=False)
        speed = float(d["speed"][0]) if "speed" in d else 4.0
        return cls(
            weights=d["weights"].astype(np.float64),
            tract_lengths=d["tract_lengths"].astype(np.float64),
            speed=speed,
            region_labels=d["region_labels"] if "region_labels" in d else None,
            centres=d["centres"].astype(np.float64) if "centres" in d else None,
            areas=d["areas"].astype(np.float64) if "areas" in d else None,
            hemispheres=d["hemispheres"].astype(bool) if "hemispheres" in d else None,
        )
