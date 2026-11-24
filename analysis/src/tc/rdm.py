"""Readout-level representational dissimilarity matrices (RDM) and RSA.

Experiment 004 asks whether the *structure of relations* between a fixed set of
objects is stable across different readouts, even when the recovered subspace
orientation is not.  An RDM encodes only pairwise distances, so it is invariant
to rotations/reparameterizations by construction -- unlike coordinates or a
single axis.

Everything here is pure numpy/scipy (no model, no tokenizer), so the instrument
can be validated on synthetic data (``tests/test_rdm.py``) before it is pointed
at a transformer.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from tc.controls import residualize, spearman, upper_tri
from tc.metrics import jaccard_words, js_topk


# --------------------------------------------------------------------------- #
# RDM builders
# --------------------------------------------------------------------------- #

def rdm_js(distributions, k: int = 50) -> np.ndarray:
    """Pairwise Jensen-Shannon RDM on the shared top-k support (experiment 001).

    ``distributions`` is a list of full next-token probability vectors.
    """
    n = len(distributions)
    M = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            M[i, j] = M[j, i] = js_topk(distributions[i], distributions[j], k)
    return M


def rdm_vectors(vectors, metric: str = "euclidean") -> np.ndarray:
    """Pairwise distance RDM over a list of feature vectors.

    ``metric`` in {"euclidean", "cosine", "correlation"} ("correlation" is
    ``1 - Pearson`` over the centered vectors).
    """
    V = [np.asarray(v, dtype=np.float64) for v in vectors]
    n = len(V)
    M = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            M[i, j] = M[j, i] = _distance(V[i], V[j], metric)
    return M


def _distance(u: np.ndarray, v: np.ndarray, metric: str) -> float:
    if metric == "euclidean":
        return float(np.linalg.norm(u - v))
    if metric == "cosine":
        nu, nv = np.linalg.norm(u), np.linalg.norm(v)
        if nu == 0.0 or nv == 0.0:
            return 1.0
        return float(1.0 - np.dot(u, v) / (nu * nv))
    if metric == "correlation":
        uc, vc = u - u.mean(), v - v.mean()
        nu, nv = np.linalg.norm(uc), np.linalg.norm(vc)
        if nu == 0.0 or nv == 0.0:
            return 1.0
        return float(1.0 - np.dot(uc, vc) / (nu * nv))
    raise ValueError(f"unknown metric {metric!r}")


def zscore_columns(M: np.ndarray) -> np.ndarray:
    """Standardize each feature (column) to mean 0 / std 1 across objects."""
    M = np.asarray(M, dtype=np.float64)
    mu = M.mean(axis=0)
    sd = M.std(axis=0)
    sd = np.where(sd == 0.0, 1.0, sd)
    return (M - mu) / sd


def rdm_lexical(texts) -> np.ndarray:
    """Lexical-control RDM: ``1 - Jaccard`` overlap of word sets."""
    n = len(texts)
    M = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            M[i, j] = M[j, i] = 1.0 - jaccard_words(texts[i], texts[j])
    return M


def rdm_length(lengths) -> np.ndarray:
    """Length-control RDM: absolute difference in token count."""
    lengths = np.asarray(lengths, dtype=np.float64)
    return np.abs(lengths[:, None] - lengths[None, :])


# --------------------------------------------------------------------------- #
# RSA between RDMs
# --------------------------------------------------------------------------- #

def rsa(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman correlation of the two RDMs' upper triangles."""
    return spearman(upper_tri(a), upper_tri(b))


def rsa_partial(a: np.ndarray, b: np.ndarray, controls) -> float:
    """Partial Spearman of two RDMs controlling for covariate RDMs."""
    xa, xb = upper_tri(a), upper_tri(b)
    cs = [upper_tri(c) for c in controls]
    return spearman(residualize(xa, cs), residualize(xb, cs))


def rsa_permutation(a: np.ndarray, b: np.ndarray, n_perm: int = 1000, seed: int = 0):
    """Permutation p-value for RSA under relabeling of the objects of ``b``.

    Returns ``(observed, p_value, null)`` with a one-sided test of positive
    structure (``H1: rsa > 0``).
    """
    obs = rsa(a, b)
    n = a.shape[0]
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm, dtype=np.float64)
    for t in range(n_perm):
        p = rng.permutation(n)
        null[t] = rsa(a, b[np.ix_(p, p)])
    pval = (float(np.sum(null >= obs)) + 1.0) / (n_perm + 1.0)
    return obs, pval, null


def baseline_rsa(a: np.ndarray, lexical: np.ndarray, length: np.ndarray) -> float:
    """Largest RSA of ``a`` with the lexical/length control RDMs.

    If an RDM "is stable" only because two readouts share lexical or length
    structure, this is the number the cross-readout RSA must beat.
    """
    return max(rsa(a, lexical), rsa(a, length))


# --------------------------------------------------------------------------- #
# Subspace orientation (contrast with RDM stability)
# --------------------------------------------------------------------------- #

def subspace_basis(X: np.ndarray, k: int = 2, center: bool = True) -> np.ndarray:
    """Top-``k`` right singular vectors of ``X`` (d x k), orthonormal columns."""
    X = np.asarray(X, dtype=np.float64)
    if center:
        X = X - X.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    k = min(k, vt.shape[0])
    return vt[:k].T


def principal_angles(U: np.ndarray, V: np.ndarray):
    """Cosines of the principal angles between two orthonormal bases.

    Returns ``(cosines, mean_cos)``; ``cosines`` is sorted descending, so the
    mean is a summary of subspace agreement (1 = identical, 0 = orthogonal).
    """
    s = np.linalg.svd(U.T @ V, compute_uv=False)
    cos = np.clip(np.sort(s)[::-1], 0.0, 1.0)
    return cos, float(cos.mean()) if cos.size else float("nan")


def orientation_stability(hidden, groups, shards, k: int = 2, center: bool = True):
    """Principal-angle agreement of a within-group subspace across two shards.

    ``hidden`` is an ``(n, d)`` matrix (one vector per object), ``groups`` the
    group label per object, ``shards`` a 0/1 shard label.  In each shard the
    data are centered within groups, a top-``k`` basis is estimated, and the two
    bases are compared by principal angles.  Returns ``(cosines, mean_cos)``.
    """
    X = np.asarray(hidden, dtype=np.float64)
    groups = np.asarray(groups)
    shards = np.asarray(shards)
    bases = []
    for s in (0, 1):
        mask = shards == s
        if mask.sum() <= k:
            return np.array([]), float("nan")
        Xs, gs = X[mask], groups[mask]
        R = Xs.copy()
        for g in np.unique(gs):
            m = gs == g
            if m.sum() > 1:
                R[m] -= R[m].mean(axis=0, keepdims=True)
        bases.append(subspace_basis(R, k=k, center=center))
    return principal_angles(bases[0], bases[1])


def effective_rank(spectrum) -> float:
    """Participation ratio ``(Σσ)² / Σσ²`` of a singular-value spectrum."""
    s = np.asarray(spectrum, dtype=np.float64)
    s = s[s > 0]
    denom = float(np.sum(s ** 2))
    if denom == 0.0:
        return 0.0
    return float(np.sum(s) ** 2 / denom)


# --------------------------------------------------------------------------- #
# Aggregation helpers
# --------------------------------------------------------------------------- #

def verdict(mean_cross: float, cross_minus_baseline: float, mean_orientation: float,
            within: float, n_above_threshold: int, n_cross: int) -> str:
    """Apply the pre-registered §11.4 decision rule to summary statistics."""
    if (mean_cross >= 0.6
            and cross_minus_baseline > 0.15
            and n_above_threshold == n_cross
            and mean_orientation < 0.5):
        return "H4a: structural invariant found"
    if within >= 0.6 and mean_cross < within - 0.3:
        return "H4c: trivial within-channel stability only"
    if mean_cross < 0.3 or cross_minus_baseline <= 0.15:
        return "H4b: no cross-readout invariant"
    return "partial/ambiguous invariant"
