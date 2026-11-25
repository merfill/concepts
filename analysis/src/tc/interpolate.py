"""Interpolation validity (experiment 006, program stage 4.0).

Tests whether a linear chord ``h_0 + a*(h_K - h_0)`` is a faithful model of the
*natural* input path (a graded ladder) before it is used for steering.  Pure
geometry functions are separated from model hooks and unit-tested offline.

Facts encoded here (see ``../experiment-006-2026-11-25-interpolation-validity.md``):

* a *graded* axis (temperature, size) gives a metric ladder with intermediate
  words; a *binary* factorial ladder is only for path-independence;
* the chord fit is measured by projection monotonicity, a normalised orthogonal
  residual and a length ratio;
* a natural path that is smooth but curved is I2 (parametrisation wrong), not a
  method failure; a jump on a *graded* axis is I3 (discreteness).
"""

from __future__ import annotations

import itertools

import numpy as np
from scipy import stats

from .factorial import ATTRS, render
from .metrics import js_topk

# --------------------------------------------------------------------------- #
# Ladders
# --------------------------------------------------------------------------- #

GRADED = {
    "temperature": {
        "frame": "The water is {}.",
        "words": ["freezing", "cold", "cool", "warm", "hot"],
    },
    "size": {
        "frame": "The animal is {}.",
        "words": ["tiny", "small", "medium-sized", "large", "huge"],
    },
}


def graded_texts(kind: str, n_steps: int | None = None) -> dict:
    spec = GRADED[kind]
    words = spec["words"] if n_steps is None else spec["words"][:n_steps]
    texts = [spec["frame"].format(w) for w in words]
    progress = np.linspace(0.0, 1.0, len(words))
    return {"kind": kind, "words": words, "texts": texts, "progress": progress}


def flip_orders() -> list[tuple]:
    """All 24 orders of flipping the four factorial attributes."""
    return list(itertools.permutations(ATTRS))


def factorial_ladder(order, template: int = 0, syn: int = 0) -> dict:
    """Rung texts from ``all-0`` to ``all-1``, flipping one attribute per step."""
    values = {a: 0 for a in ATTRS}
    scenes = [dict(values)]
    for a in order:
        values[a] = 1
        scenes.append(dict(values))
    texts = [render(v, template, syn) for v in scenes]
    return {"texts": texts, "scenes": scenes, "order": list(order)}


# --------------------------------------------------------------------------- #
# Pure path geometry
# --------------------------------------------------------------------------- #

def projection_progress(H):
    """Return ``t_i`` (chord-progress, ``t_0=0``, ``t_K=1``), ``delta``, ``|delta|``."""
    H = np.asarray(H, dtype=np.float64)
    delta = H[-1] - H[0]
    nd = float(np.linalg.norm(delta))
    if nd == 0:
        return np.zeros(len(H)), delta, nd
    dh = delta / nd
    t = (H - H[0]) @ dh / nd
    return t, delta, nd


def path_metrics(H) -> dict:
    """Geometry of a natural path ``H`` (K+1, d)."""
    H = np.asarray(H, dtype=np.float64)
    t, delta, nd = projection_progress(H)
    K = len(H) - 1
    rho = float(stats.spearmanr(np.arange(len(t)), t).correlation) if len(t) > 2 else float("nan")

    if nd == 0 or K < 1:
        return {
            "t": t.tolist(), "monotonicity": rho,
            "residual_mean": 0.0, "residual_max": 0.0, "length_ratio": 1.0,
            "step_cos_mean": float("nan"), "single_step_share": float("nan"),
        }

    dh = delta / nd
    steps = np.diff(H, axis=0)
    lens = np.linalg.norm(steps, axis=1)
    length_ratio = float(lens.sum() / nd)

    proj = (H - H[0]) @ dh
    resid = (H - H[0]) - np.outer(proj, dh)
    r = np.linalg.norm(resid, axis=1) / nd

    cs = []
    for i in range(1, len(steps)):
        a, b = steps[i - 1], steps[i]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na > 0 and nb > 0:
            cs.append(float(a @ b / (na * nb)))
    step_cos = float(np.mean(cs)) if cs else float("nan")

    proj_steps = steps @ dh
    total = float(np.sum(proj_steps))
    share = float(np.max(np.abs(proj_steps)) / abs(total)) if abs(total) > 0 else float("nan")

    return {
        "t": t.tolist(), "monotonicity": rho,
        "residual_mean": float(np.mean(r)), "residual_max": float(np.max(r)),
        "length_ratio": length_ratio, "step_cos_mean": step_cos,
        "single_step_share": share,
    }


def chord_delta(H) -> np.ndarray:
    H = np.asarray(H, dtype=np.float64)
    return H[-1] - H[0]


# --------------------------------------------------------------------------- #
# Representation variants (the residual stream is NOT on the unit sphere)
# --------------------------------------------------------------------------- #

def unit_rows(H) -> np.ndarray:
    """Direction-only representation: ``h / ||h||`` (scale-invariant)."""
    H = np.asarray(H, dtype=np.float64)
    n = np.linalg.norm(H, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return H / n


def center_rows(H, mu) -> np.ndarray:
    return np.asarray(H, dtype=np.float64) - np.asarray(mu, dtype=np.float64)


def background_stats(bg, n_pcs: int = 3):
    """Mean/std of a background set and its top principal directions (whitened space)."""
    bg = np.asarray(bg, dtype=np.float64)
    mu = bg.mean(axis=0)
    sigma = bg.std(axis=0)
    sigma[sigma < 1e-8] = 1.0
    Xw = (bg - mu) / sigma
    _, _, vt = np.linalg.svd(Xw, full_matrices=False)
    return mu, sigma, vt[:n_pcs]


def standardize_rows(H, mu, sigma) -> np.ndarray:
    return (np.asarray(H, dtype=np.float64) - np.asarray(mu, dtype=np.float64)) / np.asarray(sigma, dtype=np.float64)


def remove_pcs_rows(H, mu, sigma, components) -> np.ndarray:
    """Standardize, then project out the top background principal components."""
    Xw = standardize_rows(H, mu, sigma)
    C = np.asarray(components, dtype=np.float64)
    if C.size == 0:
        return Xw
    return Xw - (Xw @ C.T) @ C


def angular_metrics(H) -> dict:
    """Geodesic (great-circle) diagnostics on the unit sphere."""
    U = unit_rows(H)
    cs = np.clip(np.sum(U[:-1] * U[1:], axis=1), -1.0, 1.0)
    angles = np.arccos(cs)
    total = float(angles.sum())
    chord = float(np.arccos(np.clip(float(U[0] @ U[-1]), -1.0, 1.0)))
    ratio = total / chord if chord > 0 else float("nan")
    return {
        "angle_path": total, "angle_chord": chord,
        "geodesic_length_ratio": ratio,
        "step_angle_mean": float(angles.mean()) if len(angles) else float("nan"),
    }


# --------------------------------------------------------------------------- #
# Behavioral profiling (next-token readout)
# --------------------------------------------------------------------------- #

def behavioral_profile(probs, k: int = 50) -> dict:
    """Per-rung behavioral distinctness: is the ladder graded at all?"""
    probs = [np.asarray(p, dtype=np.float64) for p in probs]
    p0, pK = probs[0], probs[-1]
    from_start = [float(js_topk(p0, p, k)) for p in probs]
    adjacent = [float(js_topk(probs[i - 1], probs[i], k)) for i in range(1, len(probs))]
    mono = (float(stats.spearmanr(np.arange(len(from_start)), from_start).correlation)
            if len(from_start) > 2 else float("nan"))
    return {
        "js_from_start": from_start, "js_adjacent": adjacent,
        "js_endpoints": float(js_topk(p0, pK, k)),
        "js_from_start_monotonicity": mono,
    }


# --------------------------------------------------------------------------- #
# Behavioral curve comparison (next-token readout)
# --------------------------------------------------------------------------- #

def _union_support(probs_list, k: int) -> np.ndarray:
    idx: set[int] = set()
    for p in probs_list:
        idx |= set(np.argsort(np.asarray(p))[::-1][:k].tolist())
    return np.array(sorted(idx), dtype=int)


def behavioral_curve(natural_probs, steered_probs, k: int = 50) -> dict:
    """Project natural/steered next-token distributions on the ``p_K - p_0`` axis.

    Both lists must be progress-aligned (same number of entries).  Returns the
    scalar curves and their rank correlation (the behavioral match).
    """
    natural_probs = [np.asarray(p, dtype=np.float64) for p in natural_probs]
    steered_probs = [np.asarray(p, dtype=np.float64) for p in steered_probs]
    idx = _union_support(natural_probs + steered_probs, k)

    def prep(p):
        q = p[idx]
        s = q.sum()
        return q / s if s > 0 else np.full_like(q, 1.0 / q.size)

    N = np.stack([prep(p) for p in natural_probs])
    S = np.stack([prep(p) for p in steered_probs])
    p0, pK = N[0], N[-1]
    v = pK - p0
    nv = float(v @ v)
    if nv <= 0:
        return {"natural_scalar": [0.0] * len(N), "steered_scalar": [0.0] * len(S),
                "match_rho": float("nan"), "natural_monotonicity": float("nan")}
    sn = (N - p0) @ v / nv
    ss = (S - p0) @ v / nv
    match = float(stats.spearmanr(sn, ss).correlation) if len(sn) > 2 else float("nan")
    mono = float(stats.spearmanr(np.arange(len(sn)), sn).correlation) if len(sn) > 2 else float("nan")
    return {
        "natural_scalar": sn.tolist(), "steered_scalar": ss.tolist(),
        "match_rho": match, "natural_monotonicity": mono,
    }


# --------------------------------------------------------------------------- #
# Path independence across flip orders
# --------------------------------------------------------------------------- #

def order_step_signature(H, order) -> np.ndarray:
    """Steps of one order, reindexed to canonical attribute order, concatenated."""
    H = np.asarray(H, dtype=np.float64)
    steps = np.diff(H, axis=0)
    by_attr = {a: v for a, v in zip(order, steps)}
    return np.concatenate([by_attr[a] for a in ATTRS])


def pairwise_consistency(vectors) -> float:
    """Mean pairwise cosine between vectors (path-independence score)."""
    vs = [np.asarray(v, dtype=np.float64) for v in vectors]
    cs = []
    for i in range(len(vs)):
        for j in range(i + 1, len(vs)):
            ni, nj = np.linalg.norm(vs[i]), np.linalg.norm(vs[j])
            if ni > 0 and nj > 0:
                cs.append(float(vs[i] @ vs[j] / (ni * nj)))
    return float(np.mean(cs)) if cs else float("nan")


# --------------------------------------------------------------------------- #
# Verdict
# --------------------------------------------------------------------------- #

def interpolation_verdict(geom: dict, behavioral_match: float | None,
                          residual_tau: float = 0.15, length_tau: float = 1.15,
                          mono_tau: float = 0.8, cos_tau: float = 0.5,
                          match_tau: float = 0.8) -> str:
    monotone = geom["monotonicity"] >= mono_tau
    jump = (not monotone) or (
        not np.isnan(geom["single_step_share"]) and geom["single_step_share"] > 0.6)
    if jump:
        return "I3: jump/non-monotone (graded axis)"

    # Zigzag steps (low cos) are still *curved*, not discrete: I2, not I3.
    curved = (geom["residual_mean"] > residual_tau or
              geom["length_ratio"] > length_tau or
              (not np.isnan(geom["step_cos_mean"]) and geom["step_cos_mean"] < cos_tau))
    if curved:
        return "I2: smooth but curved (use polyline/geodesic)"
    if behavioral_match is not None and behavioral_match >= match_tau:
        return "I1: chord valid (straight + behavioral match)"
    return "I1-geometry, behavioral mismatch (partial)"
