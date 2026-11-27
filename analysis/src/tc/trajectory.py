"""Geometry of next-token trajectories in distribution space (experiment 016).

Pure numpy helpers.  A distribution ``p`` is a point of the simplex
``Δ^(V−1)``; the model's reachable set is a low-dimensional submanifold of it
(experiment 014).  The natural metric is Fisher--Rao; in logit coordinates
``z`` (with ``p = softmax(W_U z)``) it is

    g(p) = diag(p) − p pᵀ,

so for a perturbation ``u`` of the logits

    uᵀ g u = Var_p(u) = Σ_i p_i u_i² − (Σ_i p_i u_i)².

``g`` is never materialised (V is large); the quadratic form is O(V).

Model-facing collection (forwards, branching) lives in
``scripts/run_trajectory_geometry.py``.
"""

from __future__ import annotations

import numpy as np

from .inverse import js, softmax

__all__ = [
    "fisher_norm",
    "fisher_path_length",
    "euclid_path_length",
    "simplex_embed",
    "angular_distance",
    "fisher_rao",
    "simplex_geodesic",
    "geodesic_deviation",
    "participation_ratio",
    "token_survival",
    "branch_top_index",
    "nucleus_index",
    "null_index",
    "token_dominance",
    "tail_period",
]


def _as1(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(-1)


def fisher_norm(p, u) -> float:
    """Fisher length of a **logit** perturbation ``u`` at ``p``: ``sqrt(Var_p(u))``.

    ``u`` lives in ``R^V`` (logit space); ``p = softmax(logits)``.  Uses
    ``g = diag(p) − p pᵀ`` without forming the (V, V) matrix.  Returns 0 for a
    constant ``u`` (the softmax gauge direction).  For a hidden-space
    perturbation use ``W_U @ δz`` (the induced logit perturbation).
    """
    p = _as1(p)
    u = _as1(u)
    mean_u = float(np.dot(p, u))
    var = float(np.dot(p, u * u) - mean_u * mean_u)
    return float(np.sqrt(max(0.0, var)))


def fisher_path_length(ps, logits) -> float:
    """Sum of Fisher segment lengths along ``(logits_t)``, metric at ``p̄``.

    ``p_t = softmax(logits_t)``; the logit perturbation between consecutive
    steps is ``δℓ = ℓ_{t+1} − ℓ_t`` (the constant part is gauge and cancels).
    ``p̄ = (p_t + p_{t+1}) / 2`` is the chord midpoint (still a distribution).
    """
    ps = [_as1(p) for p in ps]
    logits = [_as1(l) for l in logits]
    n = min(len(ps), len(logits))
    total = 0.0
    for t in range(n - 1):
        p_bar = 0.5 * (ps[t] + ps[t + 1])
        dl = logits[t + 1] - logits[t]
        total += fisher_norm(p_bar, dl)
    return float(total)


def euclid_path_length(zs) -> float:
    """Sum of Euclidean segment lengths along ``(z_t)``."""
    zs = [_as1(z) for z in zs]
    return float(sum(np.linalg.norm(zs[t + 1] - zs[t]) for t in range(len(zs) - 1)))


def simplex_embed(p) -> np.ndarray:
    """Isometric embedding of the Fisher simplex into a sphere of radius 2."""
    return 2.0 * np.sqrt(np.clip(_as1(p), 0.0, None))


def angular_distance(x, y, radius: float = 2.0) -> float:
    """Great-circle (angular) distance between two points of radius ``radius``."""
    x = _as1(x)
    y = _as1(y)
    nx, ny = np.linalg.norm(x), np.linalg.norm(y)
    if nx == 0 or ny == 0:
        return 0.0
    c = float(np.dot(x, y) / (nx * ny))
    return float(radius * np.arccos(np.clip(c, -1.0, 1.0)))


def fisher_rao(p, q) -> float:
    """Fisher--Rao distance on the simplex = ``2·arccos(Σ√(p_i q_i))``."""
    p = _as1(p)
    q = _as1(q)
    bc = float(np.sum(np.sqrt(np.clip(p, 0.0, None) * np.clip(q, 0.0, None))))
    return float(2.0 * np.arccos(np.clip(bc, 0.0, 1.0)))


def simplex_geodesic(x_a, x_b, s: float) -> np.ndarray:
    """Point at fraction ``s`` of the great-circle geodesic between ``x_a, x_b``."""
    x_a = _as1(x_a)
    x_b = _as1(x_b)
    na, nb = np.linalg.norm(x_a), np.linalg.norm(x_b)
    if na == 0 or nb == 0:
        return (1.0 - s) * x_a + s * x_b
    a, b = x_a / na, x_b / nb
    omega = float(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0)))
    if omega < 1e-12:
        out = (1.0 - s) * a + s * b
        return out / max(np.linalg.norm(out), 1e-12) * na
    out = (np.sin((1.0 - s) * omega) * a + np.sin(s * omega) * b) / np.sin(omega)
    return out * na


def distance_to_arc(x, x_a, x_b, radius: float = 2.0) -> float:
    """Distance from ``x`` to the great-circle arc ``[x_a, x_b]``."""
    x, x_a, x_b = _as1(x), _as1(x_a), _as1(x_b)
    na, nb, nx = np.linalg.norm(x_a), np.linalg.norm(x_b), np.linalg.norm(x)
    if na == 0 or nb == 0 or nx == 0:
        return 0.0
    a, b, u = x_a / na, x_b / nb, x / nx
    omega = float(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0)))
    if omega < 1e-12:
        return float(radius * np.arccos(np.clip(np.dot(u, a), -1.0, 1.0)))
    e2 = b - np.dot(a, b) * a
    ne = np.linalg.norm(e2)
    if ne < 1e-12:
        return float(radius * np.arccos(np.clip(np.dot(u, a), -1.0, 1.0)))
    e2 /= ne
    phi = float(np.arctan2(np.dot(u, e2), np.dot(u, a)))
    t = phi / omega
    if t <= 0.0:
        closest = a
    elif t >= 1.0:
        closest = b
    else:
        inplane = np.cos(phi) * a + np.sin(phi) * e2
        closest = inplane / max(np.linalg.norm(inplane), 1e-12)
    return float(radius * np.arccos(np.clip(np.dot(u, closest), -1.0, 1.0)))


def geodesic_deviation(ps) -> dict:
    """How far a path ``(p_t)`` strays from the geodesic between its endpoints.

    Works through the isometric sphere embedding ``x = 2√p`` (radius 2), so the
    returned numbers are Fisher distances.  Intermediates only; endpoints give 0.
    """
    xs = [simplex_embed(p) for p in ps]
    if len(xs) < 3:
        return {"n_interior": 0, "mean": 0.0, "max": 0.0}
    x_a, x_b = xs[0], xs[-1]
    dev = [distance_to_arc(xs[t], x_a, x_b) for t in range(1, len(xs) - 1)]
    return {
        "n_interior": len(dev),
        "mean": float(np.mean(dev)),
        "max": float(np.max(dev)),
    }


def participation_ratio(cloud, max_points: int = 256, seed: int = 0) -> dict:
    """Participation ratio of a point cloud, capped at ``max_points`` rows.

    ``PR = (Σλ_i)² / Σλ_i²`` from eigenvalues of the centred sample covariance
    (computed through the ``N×N`` Gram matrix, never ``V×V``).  ``PR`` is a
    rough count of directions carrying variance; a single line gives ≈1.
    """
    X = np.asarray(cloud, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < 2:
        return {"pr": 0.0, "n": int(X.shape[0]) if X.ndim == 2 else 0}
    if X.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(X.shape[0], size=max_points, replace=False)
        X = X[idx]
    X = X - X.mean(axis=0, keepdims=True)
    gram = X @ X.T
    lam = np.linalg.eigvalsh(gram)
    lam = np.clip(lam, 0.0, None)
    s1 = float(lam.sum())
    s2 = float((lam * lam).sum())
    pr = float(s1 * s1 / s2) if s2 > 0 else 0.0
    return {"pr": pr, "n": int(X.shape[0])}


def token_survival(tokens_a, tokens_b, start: int = 0) -> float:
    """Fraction of positions ``>= start`` where the two token sequences agree."""
    a = list(tokens_a)
    b = list(tokens_b)
    n = min(len(a), len(b))
    idx = range(max(0, start), n)
    idx = list(idx)
    if not idx:
        return 1.0
    return float(np.mean([a[i] == b[i] for i in idx]))


def branch_top_index(p, rank: int, exclude=()) -> int:
    """Index of the ``rank``-th most probable token (1-indexed), skipping ``exclude``."""
    p = _as1(p)
    order = np.argsort(-p, kind="stable")
    ex = set(int(i) for i in exclude)
    picked = [int(i) for i in order if int(i) not in ex]
    if not picked:
        return int(order[0])
    r = max(1, int(rank))
    return picked[min(r - 1, len(picked) - 1)]


def nucleus_index(p, mass: float = 0.5) -> int:
    """Boundary token of the minimal nucleus whose cumulative mass reaches ``mass``."""
    p = _as1(p)
    order = np.argsort(-p, kind="stable")
    cum = 0.0
    last = int(order[0])
    for i in order:
        cum += float(p[i])
        last = int(i)
        if cum >= mass:
            break
    return last


def null_index(p, exclude=()) -> int:
    """Least probable token (deterministic low-probability control)."""
    p = _as1(p)
    order = np.argsort(p, kind="stable")
    ex = set(int(i) for i in exclude)
    for i in order:
        if int(i) not in ex:
            return int(i)
    return int(order[0])


def token_dominance(tokens) -> dict:
    """Share of the most frequent token in a sequence (``1.0`` = one token only)."""
    toks = [int(t) for t in tokens]
    if not toks:
        return {"dom_share": 0.0, "dom_token": None, "n": 0}
    vals, counts = np.unique(toks, return_counts=True)
    j = int(np.argmax(counts))
    return {"dom_share": float(counts[j] / len(toks)),
            "dom_token": int(vals[j]), "n": len(toks)}


def tail_period(tokens, max_period: int = 8, min_repeats: int = 2) -> int:
    """Smallest period ``p`` repeated at least ``min_repeats`` times at the tail.

    Returns 0 when the tail is not periodic.  ``[...,a,b,a,b,a,b]`` -> 2.
    """
    toks = [int(t) for t in tokens]
    n = len(toks)
    for p in range(1, min(max_period, n // max(1, min_repeats)) + 1):
        w = p * min_repeats
        if w > n:
            continue
        tail = toks[-w:]
        if all(tail[i] == tail[i - p] for i in range(p, w)):
            return int(p)
    return 0
