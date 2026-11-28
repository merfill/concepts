"""Separation of form and meaning via contrastive subspaces (experiment 017).

Items are vectors ``x`` with two labelings: ``labels_meaning`` (the class, e.g.
the scene) and ``labels_form`` (the within-class variant, e.g. template+synonym).
We build the within-class scatter ``S_w`` (form) and the between-class scatter
``S_b`` (meaning) and solve the generalized eigenproblem ``S_b w = lambda S_w w``.
Directions with large eigenvalues span the meaning subspace; the swapped problem
spans the form subspace.

Pure numpy/scipy; no model code.

Protocol: ``../experiment-017-2026-11-28-form-vs-meaning.md``.
"""

from __future__ import annotations

import numpy as np
from scipy import linalg


def scatter(X, labels) -> dict:
    """Within-class and between-class scatter matrices of ``X`` by ``labels``."""
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels)
    mu = X.mean(axis=0)
    d = X.shape[1]
    S_w = np.zeros((d, d), dtype=np.float64)
    S_b = np.zeros((d, d), dtype=np.float64)
    for c in np.unique(labels):
        Xc = X[labels == c]
        muc = Xc.mean(axis=0)
        D = Xc - muc
        S_w += D.T @ D
        diff = muc - mu
        S_b += Xc.shape[0] * np.outer(diff, diff)
    tw = float(np.trace(S_w))
    tb = float(np.trace(S_b))
    return {
        "within": S_w,
        "between": S_b,
        "within_trace": tw,
        "between_trace": tb,
        "ratio": float(tb / tw) if tw > 0 else float("inf"),
    }


def fisher_ratio(X, labels) -> float:
    return scatter(X, labels)["ratio"]


def _centered_span(X):
    """Center ``X`` and return mean, coordinates in the span, and the basis."""
    mu = X.mean(axis=0)
    Xc = X - mu
    if Xc.size == 0:
        return mu, Xc, np.zeros((X.shape[1], 0))
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    if S.size == 0 or S.max() == 0:
        return mu, np.zeros((X.shape[0], 0)), np.zeros((X.shape[1], 0))
    tol = float(S.max()) * max(Xc.shape) * np.finfo(np.float64).eps
    r = int(np.sum(S > tol))
    r = max(r, 1)
    V = Vt[:r].T
    return mu, Xc @ V, V


def _orthonormalize(W) -> np.ndarray:
    W = np.asarray(W, dtype=np.float64)
    if W.size == 0:
        return W
    Q, _ = np.linalg.qr(W)
    return Q


def fisher_directions(X, labels, n_meaning: int = 5, n_form: int = 5,
                      reg: float = 0.1) -> dict:
    """Meaning and form directions via the two generalized eigenproblems.

    ``reg`` is the LDA shrinkage applied to the denominator of each problem
    (``S_w`` for meaning, ``S_b`` for form).  Without it the small-sample null
    spaces dominate and the recovered directions are spurious.

    Returns directions in the original space; apply ``project`` with the returned
    ``mean`` to get coordinates.
    """
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels)
    mu, Z, V = _centered_span(X)
    d = Z.shape[1]
    sc = scatter(Z, labels) if d > 0 else {
        "within": np.zeros((0, 0)), "between": np.zeros((0, 0)),
        "within_trace": 0.0, "between_trace": 0.0, "ratio": float("nan"),
    }
    S_w, S_b = sc["within"], sc["between"]
    tw = float(np.trace(S_w))
    tb = float(np.trace(S_b))
    alpha = float(np.clip(reg, 1e-6, 0.999))
    eye = np.eye(d) if d > 0 else np.zeros((0, 0))
    S_w_den = (1.0 - alpha) * S_w + alpha * (tw / d) * eye if d > 0 else S_w
    S_b_den = (1.0 - alpha) * S_b + alpha * (tb / d) * eye if d > 0 else S_b

    eig_m, vec_m = linalg.eigh(S_b, S_w_den)
    order = np.argsort(eig_m)[::-1]
    eig_m, vec_m = eig_m[order], vec_m[:, order]
    k_m = min(n_meaning, vec_m.shape[1])
    W_m = _orthonormalize(V @ vec_m[:, :k_m]) if d > 0 else np.zeros((X.shape[1], 0))

    eig_f, vec_f = linalg.eigh(S_w, S_b_den)
    order = np.argsort(eig_f)[::-1]
    eig_f, vec_f = eig_f[order], vec_f[:, order]
    k_f = min(n_form, vec_f.shape[1])
    W_f = _orthonormalize(V @ vec_f[:, :k_f]) if d > 0 else np.zeros((X.shape[1], 0))

    return {
        "mean": mu,
        "W_meaning": W_m,
        "W_form": W_f,
        "eig_meaning": eig_m[:k_m],
        "eig_form": eig_f[:k_f],
        "ratio": sc["ratio"],
    }


def principal_angles(A, B) -> np.ndarray:
    """Principal angles (degrees) between the column spans of ``A`` and ``B``."""
    A = _orthonormalize(A)
    B = _orthonormalize(B)
    if A.size == 0 or B.size == 0:
        return np.zeros(0, dtype=np.float64)
    s = np.linalg.svd(A.T @ B, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    return np.degrees(np.arccos(s))


def mean_principal_angle(A, B) -> float:
    a = principal_angles(A, B)
    return float(a.mean()) if a.size else float("nan")


def project(X, W, mean=None) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    if mean is None:
        mean = np.zeros(X.shape[1])
    return (X - np.asarray(mean, dtype=np.float64)) @ W


def residualize(X, covariates) -> np.ndarray:
    """Residual of each column of ``X`` after OLS on intercept + covariates."""
    X = np.asarray(X, dtype=np.float64)
    cols = [np.ones(X.shape[0])] + [np.asarray(c, dtype=np.float64) for c in covariates]
    D = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(D, X, rcond=None)
    return X - D @ beta


def cosine_rdm(X) -> np.ndarray:
    """Pairwise cosine-distance matrix (1 - cos), diagonal zero."""
    X = np.asarray(X, dtype=np.float64)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    Xn = X / np.where(norms > 0, norms, 1.0)
    S = Xn @ Xn.T
    np.clip(S, -1.0, 1.0, out=S)
    D = 1.0 - S
    np.fill_diagonal(D, 0.0)
    return D


def label_rdm(y) -> np.ndarray:
    """0 when the labels match, 1 otherwise."""
    y = np.asarray(y)
    return (y[:, None] != y[None, :]).astype(np.float64)


def abs_diff_rdm(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return np.abs(v[:, None] - v[None, :])
