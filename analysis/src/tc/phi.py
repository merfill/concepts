"""Representation maps Phi.

All functions take plain numpy arrays and return numpy arrays, so they are
model-agnostic and testable without downloading anything.

Conventions:
    V : (n, d) matrix, one row per position.
    A : (n, n) row-stochastic attention matrix (after softmax).
"""

from __future__ import annotations

import numpy as np


def _normalize(p: np.ndarray) -> np.ndarray:
    total = p.sum()
    if total <= 0:
        return np.full_like(p, 1.0 / max(p.size, 1))
    return p / total


def phi_spec(V: np.ndarray, d: int | None = None) -> np.ndarray:
    """Spectral invariant: normalized squared singular values of V.

    Eigenvalues of V^T V are the squared singular values of V, so this is
    spec(V^T V) normalized to a probability vector. Invariant to V -> V R for
    orthogonal R and to permutation of rows of V.

    Returns a probability vector; zero-padded to length d if given.
    """
    V = np.asarray(V, dtype=np.float64)
    s = np.linalg.svd(V, compute_uv=False)
    p = _normalize(s**2)
    if d is not None and p.size < d:
        p = np.pad(p, (0, d - p.size))
    return p


def phi_pool(V: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """Baseline: attention/mask-weighted pooled vector, normalized to unit norm."""
    V = np.asarray(V, dtype=np.float64)
    if w is None:
        w = np.ones(V.shape[0], dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    u = w @ V
    nrm = np.linalg.norm(u)
    return u / nrm if nrm > 0 else u


def phi_a(A: np.ndarray, d: int | None = None) -> np.ndarray:
    """Normalized singular-value spectrum of the attention matrix A.

    Note: singular values (and eigenvalues) are identical for A and A^T, so this
    descriptor does NOT encode edge orientation.
    """
    A = np.asarray(A, dtype=np.float64)
    s = np.linalg.svd(A, compute_uv=False)
    p = _normalize(s)
    if d is not None and p.size < d:
        p = np.pad(p, (0, d - p.size))
    return p


def phi_full(V: np.ndarray, A: np.ndarray, d: int | None = None):
    """Tuple (spectrum of A, spectrum of Gram of V)."""
    return phi_a(A, d), phi_spec(V, d)
