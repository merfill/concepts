"""Statistical controls for the H1 validation step."""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score


def upper_tri(M: np.ndarray) -> np.ndarray:
    i, j = np.triu_indices(M.shape[0], k=1)
    return M[i, j]


def spearman(a, b) -> float:
    return float(stats.spearmanr(a, b).correlation)


def residualize(y: np.ndarray, controls: list[np.ndarray]) -> np.ndarray:
    """Residual of y after OLS regression on intercept + controls."""
    y = np.asarray(y, dtype=np.float64)
    cols = [np.ones_like(y)] + [np.asarray(c, dtype=np.float64) for c in controls]
    X = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def partial_spearman(d1: np.ndarray, d2: np.ndarray, dcontrol: np.ndarray) -> float:
    """Partial Spearman of two RDMs controlling for a third RDM."""
    x, y, z = upper_tri(d1), upper_tri(d2), upper_tri(dcontrol)
    return spearman(residualize(x, [z]), residualize(y, [z]))


def r2_explained(target: np.ndarray, dcontrol: np.ndarray) -> float:
    x, z = upper_tri(target), upper_tri(dcontrol)
    resid = residualize(x, [z])
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((x - x.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def separation_auc(dmat: np.ndarray, groups) -> float:
    """AUC of 'same group' predicted by small distance.

    Higher values mean within-group distances are smaller than between-group.
    """
    groups = np.asarray(groups)
    i, j = np.triu_indices(dmat.shape[0], k=1)
    d = dmat[i, j]
    same = (groups[i] == groups[j]).astype(int)
    if same.min() == same.max():
        return float("nan")
    return float(roc_auc_score(same, -d))


def permutation_test(dmat: np.ndarray, groups, n_perm: int = 500, seed: int = 0):
    """Permutation p-value for the separation AUC under a random relabeling of groups."""
    groups = np.asarray(groups)
    obs = separation_auc(dmat, groups)
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm, dtype=np.float64)
    for k in range(n_perm):
        null[k] = separation_auc(dmat, groups[rng.permutation(groups.size)])
    p = (float(np.sum(null >= obs)) + 1.0) / (n_perm + 1.0)
    return obs, p, null


def split_half_reliability(groups_of_invariants, dist, n_repeats: int = 200, seed: int = 0):
    """Test-retest reliability of concept-level invariants.

    Each concept contributes a list of replicate invariants (paraphrases).
    Replicates are split in two halves; the per-concept invariant is averaged
    within each half; RDMs from the two halves are correlated (Spearman).
    Averaged over random splits.
    """
    rng = np.random.default_rng(seed)
    rhos = []
    for _ in range(n_repeats):
        left, right = [], []
        ok = True
        for reps in groups_of_invariants:
            idx = rng.permutation(len(reps))
            half = len(reps) // 2
            if half == 0:
                ok = False
                break
            l = [reps[i] for i in idx[:half]]
            r = [reps[i] for i in idx[half : 2 * half]]
            left.append(np.mean(l, axis=0))
            right.append(np.mean(r, axis=0))
        if not ok:
            return float("nan"), float("nan")
        R1 = _rdm_flat(left, dist)
        R2 = _rdm_flat(right, dist)
        rho = spearman(R1, R2)
        if np.isfinite(rho):
            rhos.append(rho)
    rhos = np.asarray(rhos)
    sb = (
        2 * rhos.mean() / (1 + rhos.mean())
        if rhos.size and (1 + rhos.mean()) != 0
        else float("nan")
    )
    return float(rhos.mean()), float(sb)


def _rdm_flat(vectors, dist):
    n = len(vectors)
    vals = []
    for i in range(n):
        for j in range(i + 1, n):
            vals.append(dist(vectors[i], vectors[j]))
    return np.asarray(vals)
