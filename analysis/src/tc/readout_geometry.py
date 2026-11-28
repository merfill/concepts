"""Readout geometry (experiment 022): state vs output distances by layer.

Pure helpers: pairwise-distance agreement, eigendecomposition of ``W_UᵀW_U`` and
projection onto its top directions.

Protocol: ``../experiment-022-2026-11-28-readout-geometry.md``.
"""

from __future__ import annotations

import numpy as np

from .controls import spearman, upper_tri
from .slot_relations import pairwise_rdm


def rdm_corr(D1, D2) -> float:
    """Spearman correlation of two RDM upper triangles."""
    return spearman(upper_tri(np.asarray(D1, dtype=np.float64)),
                    upper_tri(np.asarray(D2, dtype=np.float64)))


def layer_alignment(H, D_out, metric: str = "cosine") -> list[float]:
    """Agreement between hidden-state RDM and output RDM, one value per layer."""
    H = np.asarray(H)
    out = []
    for layer in range(H.shape[1]):
        D = pairwise_rdm(H[:, layer, :], metric)
        out.append(rdm_corr(D, D_out))
    return out


def top_directions_from_cov(WtW, k: int = 20):
    """Top right singular directions of ``W_U`` from ``W_UᵀW_U`` (d x d)."""
    WtW = np.asarray(WtW, dtype=np.float64)
    eigvals, eigvecs = np.linalg.eigh(WtW)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    k = min(k, eigvecs.shape[1])
    return eigvecs[:, :k], eigvals


def captured_fraction(eigvals, k: int = 20) -> float:
    eigvals = np.asarray(eigvals, dtype=np.float64)
    eigvals = np.clip(eigvals, 0.0, None)
    total = float(eigvals.sum())
    if total <= 0:
        return float("nan")
    return float(eigvals[:k].sum() / total)


def project(X, V) -> np.ndarray:
    """Coordinates of rows of ``X`` in the orthonormal columns of ``V``."""
    return np.asarray(X, dtype=np.float64) @ np.asarray(V, dtype=np.float64)
