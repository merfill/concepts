"""Distances and representational dissimilarity matrices (RDM)."""

from __future__ import annotations

import re

import numpy as np
from scipy.spatial.distance import jensenshannon


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


def top_indices(p: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k largest entries of p (descending)."""
    p = np.asarray(p, dtype=np.float64)
    return np.argsort(p)[::-1][: min(k, p.size)]


def shared_topk(p: np.ndarray, q: np.ndarray, k: int):
    """Restrict p and q to the union of their top-k supports, renormalized.

    Comparing two distributions on *different* supports is ill-defined; the
    protocol (experiment 001, §1) fixes a shared support as the union of the
    two top-k index sets.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    idx = np.union1d(top_indices(p, k), top_indices(q, k))
    pp, qq = p[idx], q[idx]
    sp, sq = pp.sum(), qq.sum()
    pp = pp / sp if sp > 0 else np.full_like(pp, 1.0 / pp.size)
    qq = qq / sq if sq > 0 else np.full_like(qq, 1.0 / qq.size)
    return pp, qq


def js_topk(p: np.ndarray, q: np.ndarray, k: int) -> float:
    """Jensen-Shannon divergence (nats) on the shared top-k support."""
    pp, qq = shared_topk(p, q, k)
    return float(jensenshannon(pp, qq) ** 2)


def tv_topk(p: np.ndarray, q: np.ndarray, k: int) -> float:
    """Total-variation distance on the shared top-k support."""
    pp, qq = shared_topk(p, q, k)
    return float(0.5 * np.abs(pp - qq).sum())


def js_union(a: dict, b: dict) -> float:
    """JS divergence between two sparse distributions given as {key: mass}.

    Both are restricted to the union of their supports and renormalized, the
    analogous rule to ``shared_topk`` for multi-token (joint) distributions.
    """
    keys = set(a) | set(b)
    va = np.array([a.get(k, 0.0) for k in keys], dtype=np.float64)
    vb = np.array([b.get(k, 0.0) for k in keys], dtype=np.float64)
    sa, sb = va.sum(), vb.sum()
    if sa <= 0 or sb <= 0:
        return float("nan")
    return float(jensenshannon(va / sa, vb / sb) ** 2)


_WORD = re.compile(r"[A-Za-z\u0400-\u04FF0-9]+")


def words(text: str) -> set[str]:
    """Lower-cased alphanumeric word set (Latin + Cyrillic)."""
    return set(_WORD.findall(text.lower()))


def jaccard_words(a: str, b: str) -> float:
    """Lexical baseline: Jaccard overlap of word sets."""
    A, B = words(a), words(b)
    if not A and not B:
        return 1.0
    return len(A & B) / len(A | B)
