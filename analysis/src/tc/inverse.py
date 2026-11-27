"""Inverse readout: identifiability of the final post-norm state (experiment 014).

``logits = W_U · z`` with ``z = Norm(h)`` and ``p = softmax(logits)``.  Observation
of ``m`` token probabilities constrains ``z`` through the Jacobian
``J_i = p_i (w_i - wbar)``; its null space is what ``top-m`` cannot see.

Pure numpy helpers; model-facing probes (norms, forwards, null perturbations) live
in ``scripts/run_inverse_readout.py``.
"""

from __future__ import annotations

import numpy as np


def softmax(logits) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    z = logits - logits.max()
    e = np.exp(z)
    return e / e.sum()


def topk_indices(p, k: int) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    return np.argsort(-p, kind="stable")[: min(k, p.size)]


def expected_unembedding(W_U, p) -> np.ndarray:
    """``wbar = Σ_j p_j W_U[j]`` (the mean unembedding under ``p``)."""
    return np.asarray(p, dtype=np.float64) @ np.asarray(W_U, dtype=np.float64)


def prob_jacobian(W_U, p, indices) -> np.ndarray:
    """Rows ``p_i (w_i - wbar)``: derivative of observed probabilities wrt ``z``."""
    W_U = np.asarray(W_U, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    wbar = expected_unembedding(W_U, p)
    idx = np.asarray(indices, dtype=int)
    return p[idx, None] * (W_U[idx] - wbar[None, :])


def _effective_tol(s, shape) -> float:
    if s.size == 0:
        return 0.0
    return float(s.max() * max(shape) * np.finfo(np.float64).eps) if s.max() > 0 else 0.0


def rank_nullity(J, tol: float | None = None) -> dict:
    """Numerical rank and null-space dimension of ``J`` (``d - rank``)."""
    J = np.asarray(J, dtype=np.float64)
    if J.size == 0:
        return {"rank": 0, "null_dim": J.shape[1] if J.ndim == 2 else 0}
    s = np.linalg.svd(J, compute_uv=False)
    t = _effective_tol(s, J.shape) if tol is None else tol
    rank = int(np.sum(s > t))
    return {"rank": rank, "null_dim": int(J.shape[1] - rank)}


def null_basis(J, tol: float | None = None) -> np.ndarray:
    """Orthonormal basis of the null space of ``J`` as rows ``(null_dim, d)``."""
    J = np.asarray(J, dtype=np.float64)
    if J.size == 0:
        return np.eye(J.shape[1])
    _, s, Vt = np.linalg.svd(J, full_matrices=True)
    t = _effective_tol(s, J.shape) if tol is None else tol
    rank = int(np.sum(s > t))
    return Vt[rank:]


def kl(p, q, eps: float = 1e-12) -> float:
    """KL(p || q) in nats, ignoring zero-probability entries of ``p``."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    mask = p > 0
    return float(np.sum(p[mask] * (np.log(p[mask]) - np.log(np.clip(q[mask], eps, None)))))


def gauge_dim(kind: str) -> int:
    """Dimension of the ``h``-preimage family under the final norm."""
    return 1 if kind == "rmsnorm" else 2 if kind == "layernorm" else 0


def simplex_interp(p_a, p_b, t: float) -> np.ndarray:
    """Convex combination on the probability simplex (sum stays 1)."""
    p_a = np.asarray(p_a, dtype=np.float64)
    p_b = np.asarray(p_b, dtype=np.float64)
    return (1.0 - t) * p_a + t * p_b


def js(p, q, eps: float = 1e-12) -> float:
    """Jensen-Shannon divergence (nats) over the full support."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    m = 0.5 * (p + q)
    return 0.5 * kl(p, m, eps) + 0.5 * kl(q, m, eps)


def cosine(u, v) -> float:
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu == 0 or nv == 0:
        return float("nan")
    return float(u @ v / (nu * nv))
