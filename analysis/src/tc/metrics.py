"""Distances and representational dissimilarity matrices (RDM)."""

from __future__ import annotations

import numpy as np


def _pad_pair(p: np.ndarray, q: np.ndarray):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    m = max(p.size, q.size)
    return np.pad(p, (0, m - p.size)), np.pad(q, (0, m - q.size))


def hellinger(p: np.ndarray, q: np.ndarray) -> float:
    """Hellinger distance between two probability vectors."""
    p, q = _pad_pair(p, q)
    return float(np.sqrt(max(0.0, 1.0 - np.sum(np.sqrt(p * q)))))


def cosine_dist(u: np.ndarray, v: np.ndarray) -> float:
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu == 0 or nv == 0:
        return 1.0
    return float(1.0 - np.dot(u, v) / (nu * nv))


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    p, q = _pad_pair(p, q)
    return float(0.5 * np.sum(np.abs(p - q)))


def rdm(vectors, dist=hellinger) -> np.ndarray:
    """Pairwise distance matrix over a list of invariant vectors."""
    vectors = [np.asarray(v, dtype=np.float64) for v in vectors]
    n = len(vectors)
    M = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            M[i, j] = M[j, i] = dist(vectors[i], vectors[j])
    return M
