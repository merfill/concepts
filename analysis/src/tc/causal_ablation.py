"""Causal ablation of a supervised relation subspace (experiment 021).

The Fisher subspace is fitted on hidden states (leave-one-anchor-out), projected
out of the residual stream at a chosen layer, and the change in the relation
gradient is compared with random and form subspaces of the same rank.

Protocol: ``../experiment-021-2026-11-28-causal-ablation.md``.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from .separation import fisher_directions
from .slot_relations import relation_gradient


def center_by_groups(X, groups) -> tuple[np.ndarray, list[np.ndarray]]:
    """Subtract each group's mean from its rows; return residuals and means."""
    X = np.asarray(X, dtype=np.float64)
    Xc = X.copy()
    means = []
    for idx in groups:
        if len(idx) == 0:
            continue
        mu = X[idx].mean(axis=0)
        Xc[idx] = X[idx] - mu
        means.append(mu)
    return Xc, means


def orthonormal_columns(W) -> np.ndarray:
    W = np.asarray(W, dtype=np.float64)
    if W.size == 0:
        return W
    Q, _ = np.linalg.qr(W)
    return Q


def project_out(X, W) -> np.ndarray:
    """Remove the column space of ``W`` (d x k) from rows of ``X``."""
    X = np.asarray(X, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    if W.size == 0:
        return X.copy()
    return X - (X @ W) @ W.T


def fisher_subspace(X, y, k: int = 3, reg: float = 0.1) -> np.ndarray:
    """Fisher directions (d x k) for the labels ``y``."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return np.zeros((X.shape[1], 0))
    k = min(k, max(1, len(y) - len(np.unique(y))))
    dirs = fisher_directions(X, y, n_meaning=k, n_form=0, reg=reg)
    return orthonormal_columns(dirs["W_meaning"])


def random_subspace(d: int, k: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return orthonormal_columns(rng.normal(size=(d, k)))


def form_subspace(X, covariates, k: int = 3) -> np.ndarray:
    """Directions in ``X`` predicted linearly by ``covariates`` (d x k)."""
    X = np.asarray(X, dtype=np.float64)
    C = np.asarray(covariates, dtype=np.float64)
    if C.ndim == 1:
        C = C[:, None]
    if C.size == 0:
        return np.zeros((X.shape[1], 0))
    B, *_ = np.linalg.lstsq(C, X, rcond=None)  # (c, d)
    _, _, Vt = np.linalg.svd(B, full_matrices=False)
    k = min(k, Vt.shape[0])
    return orthonormal_columns(Vt[:k].T)


def removed_fraction(X, W) -> np.ndarray:
    """Per-row fraction of the norm removed by projecting out ``W``."""
    X = np.asarray(X, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    if W.size == 0:
        return np.zeros(X.shape[0])
    proj = (X @ W) @ W.T
    return np.linalg.norm(proj, axis=1) / (np.linalg.norm(X, axis=1) + 1e-12)


def direction_auc(X, y, W) -> float:
    """AUC of the first subspace direction separating binary labels."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    W = np.asarray(W, dtype=np.float64)
    if W.ndim != 2 or W.shape[1] == 0 or len(np.unique(y)) < 2:
        return float("nan")
    scores = X @ W[:, 0]
    auc = roc_auc_score(y, scores)
    return float(max(auc, 1.0 - auc))


def bootstrap_mean_ci(values, n_boot: int = 2000, seed: int = 0):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        means[i] = rng.choice(values, size=values.size, replace=True).mean()
    return (float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)),
            float(values.mean()))


def bootstrap_diff_ci(a, b, n_boot: int = 2000, seed: int = 0):
    """Paired bootstrap CI of mean(a - b), dropping non-finite pairs."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if a.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, a.size, size=a.size)
        diffs[i] = (a[idx] - b[idx]).mean()
    return (float(np.percentile(diffs, 2.5)),
            float(np.percentile(diffs, 97.5)),
            float((a - b).mean()))


def gradient_per_anchor(D, labels, groups, target: str,
                        control: str) -> list[float]:
    """Relation gradient computed separately inside every anchor."""
    out = []
    for idx in groups:
        value = relation_gradient(D, labels, [idx], [target], control=control)[target]
        out.append(value)
    return out


def gradient_summary(D, labels, groups, target: str, control: str,
                     n_boot: int = 2000, seed: int = 0) -> dict:
    per_anchor = gradient_per_anchor(D, labels, groups, target, control)
    lo, hi, mean = bootstrap_mean_ci(per_anchor, n_boot=n_boot, seed=seed)
    return {
        "mean": mean,
        "ci95": [lo, hi],
        "per_anchor": per_anchor,
    }
