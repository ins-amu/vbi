"""
Feature pruning for VBI sweep data.

Operates on the post-sweep feature matrix X of shape (n_sims, n_features).
Completely backend-agnostic: the matrix is plain float64 numpy regardless of
which simulation backend (numba, jax, cuda, numpy) produced it.

Typical usage
-------------
    pruner = FeaturePruner(min_std=1e-4, max_corr=0.98)

    theta, x = inf.simulate(N_SIM, DURATION)
    x, labels = pruner.fit_transform(x, feature_labels)
    print(pruner.summary())

    # Prune the observation vector the same way before sampling
    x_obs_pruned = pruner.transform(x_obs[None])[0]
"""
from __future__ import annotations

import numpy as np


class FeaturePruner:
    """
    Prune uninformative and redundant features from a sweep feature matrix.

    Three sequential filters (applied in order):

    1. **NaN/Inf** — columns that contain any non-finite value are always
       removed; they cannot contribute to density estimation.

    2. **Low std** — columns whose standard deviation across simulations is
       below ``min_std``.  Such features do not vary with the parameters and
       carry no discriminative information.

    3. **High correlation** — of each pair of remaining features whose
       absolute Pearson correlation exceeds ``max_corr``, the one with *lower*
       std is dropped (greedy pass, highest-std-first order).  This removes
       redundant features while retaining the most variable representative.

    Parameters
    ----------
    min_std : float
        Minimum acceptable standard deviation across simulations.
        Features with std < min_std are removed. Default: 1e-6.
    max_corr : float
        Maximum acceptable absolute Pearson correlation between any pair of
        retained features.  Must be in (0, 1]. Default: 0.98.
    remove_nan : bool
        Whether to remove columns with NaN/Inf values. Default: True.

    Attributes (set after fit)
    --------------------------
    kept_mask_   : np.ndarray[bool], shape (n_features,)
    kept_labels_ : list[str]
    n_kept       : int
    n_removed    : int

    Notes
    -----
    ``transform`` accepts both 2-D ``(n_sims, n_features)`` and 1-D
    ``(n_features,)`` arrays, making it straightforward to prune a single
    observation vector alongside the training matrix.
    """

    def __init__(
        self,
        min_std: float = 1e-6,
        max_corr: float = 0.98,
        remove_nan: bool = True,
    ) -> None:
        if not (0.0 < max_corr <= 1.0):
            raise ValueError("max_corr must be in (0, 1].")
        if min_std < 0:
            raise ValueError("min_std must be non-negative.")

        self.min_std = float(min_std)
        self.max_corr = float(max_corr)
        self.remove_nan = bool(remove_nan)

        self.kept_mask_: np.ndarray | None = None
        self.kept_labels_: list[str] | None = None
        self._n_original: int | None = None
        self._removed_reason: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, labels: list[str]) -> "FeaturePruner":
        """
        Compute the pruning mask from a sweep feature matrix.

        Parameters
        ----------
        X      : (n_sims, n_features) float64
        labels : list of n_features feature-name strings

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}.")
        n_sims, n_feat = X.shape
        if len(labels) != n_feat:
            raise ValueError(
                f"len(labels)={len(labels)} != X.shape[1]={n_feat}."
            )

        self._n_original = n_feat
        self._removed_reason = {}
        mask = np.ones(n_feat, dtype=bool)

        # 1. NaN / Inf
        if self.remove_nan:
            nan_cols = ~np.all(np.isfinite(X), axis=0)
            for i in np.where(nan_cols)[0]:
                mask[i] = False
                self._removed_reason[labels[i]] = "NaN/Inf"

        # 2. Low std
        stds = X.std(axis=0)
        for i in np.where((stds < self.min_std) & mask)[0]:
            mask[i] = False
            self._removed_reason[labels[i]] = (
                f"low-std ({stds[i]:.2e} < {self.min_std:.2e})"
            )

        # 3. High correlation — greedy, highest-std-first
        if self.max_corr < 1.0:
            live_idx = np.where(mask)[0]
            if len(live_idx) > 1:
                X_live = X[:, live_idx]
                live_stds = stds[live_idx]

                # Sort descending by std — we keep the most variable feature
                # of each correlated group.
                order = np.argsort(-live_stds)
                X_live = X_live[:, order]
                live_idx = live_idx[order]
                live_stds = live_stds[order]

                # Pearson correlation via z-score normalisation.
                # Zero-std columns are already removed so safe to divide.
                mu = X_live.mean(axis=0)
                sig = np.where(live_stds > 0, live_stds, 1.0)
                Z = (X_live - mu) / sig                 # (n_sims, n_live)
                corr = (Z.T @ Z) / max(n_sims, 1)       # (n_live, n_live)

                removed_local: set[int] = set()
                for i in range(len(live_idx)):
                    if i in removed_local:
                        continue
                    for j in range(i + 1, len(live_idx)):
                        if j in removed_local:
                            continue
                        if abs(corr[i, j]) > self.max_corr:
                            removed_local.add(j)
                            orig_j = int(live_idx[j])
                            mask[orig_j] = False
                            self._removed_reason[labels[orig_j]] = (
                                f"corr={corr[i, j]:.3f} "
                                f"with '{labels[int(live_idx[i])]}'"
                            )

        self.kept_mask_ = mask
        self.kept_labels_ = [lb for lb, keep in zip(labels, mask) if keep]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply the fitted pruning mask to a feature array.

        Parameters
        ----------
        X : (n_sims, n_features) or (n_features,) float64

        Returns
        -------
        X_pruned : (n_sims, n_kept) or (n_kept,) float64
        """
        if self.kept_mask_ is None:
            raise RuntimeError("Call fit() before transform().")
        X = np.asarray(X, dtype=np.float64)
        squeeze = X.ndim == 1
        if squeeze:
            X = X[np.newaxis]
        out = X[:, self.kept_mask_]
        return out[0] if squeeze else out

    def fit_transform(
        self, X: np.ndarray, labels: list[str]
    ) -> tuple[np.ndarray, list[str]]:
        """
        Fit and immediately apply the pruning mask.

        Parameters
        ----------
        X      : (n_sims, n_features) float64
        labels : list of n_features feature-name strings

        Returns
        -------
        X_pruned    : (n_sims, n_kept) float64
        kept_labels : list[str]
        """
        self.fit(X, labels)
        return self.transform(X), list(self.kept_labels_)

    # ------------------------------------------------------------------
    # Properties / reporting
    # ------------------------------------------------------------------

    @property
    def n_kept(self) -> int:
        if self.kept_mask_ is None:
            raise RuntimeError("Call fit() first.")
        return int(self.kept_mask_.sum())

    @property
    def n_removed(self) -> int:
        if self._n_original is None:
            raise RuntimeError("Call fit() first.")
        return self._n_original - self.n_kept

    def summary(self) -> str:
        """Return a human-readable pruning report."""
        if self.kept_mask_ is None:
            return "FeaturePruner: not yet fitted."
        lines = [
            f"FeaturePruner  original={self._n_original}  "
            f"kept={self.n_kept}  removed={self.n_removed}",
            f"  settings: min_std={self.min_std:.1e}  "
            f"max_corr={self.max_corr}  remove_nan={self.remove_nan}",
        ]
        if self._removed_reason:
            lines.append("  Removed features:")
            for name, reason in self._removed_reason.items():
                lines.append(f"    - {name:<40s}  [{reason}]")
        else:
            lines.append("  No features removed.")
        return "\n".join(lines)

    def __repr__(self) -> str:
        fitted = self.kept_mask_ is not None
        status = (
            f"kept={self.n_kept}/{self._n_original}" if fitted else "not fitted"
        )
        return (
            f"FeaturePruner(min_std={self.min_std:.1e}, "
            f"max_corr={self.max_corr}, {status})"
        )
