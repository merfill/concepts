"""Concept-space continuity as a *response function* on an on-manifold graded
axis (experiment 013, program stage 4).

The verdict is read off the *output* distribution along a real word ladder, not
off a straight line between hidden states (the lesson of 006: the chord is a bad
path model).  Primary coordinate is the projection of each rung's next-token
distribution onto the ``q_last - q_first`` axis in output space; an external
expected-valence coordinate is optional (norms are not bundled).

Stateless next-token readout means an "ascending vs descending" sweep yields
identical distributions, so direction is *not* evidence of hysteresis.  Instead
we test independence from the parameterisation: agreement across frames and
across a second synonymous ladder.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy import stats

from .metrics import js_topk

# --------------------------------------------------------------------------- #
# Pre-registered constants (experiment 013, §5/§7)
# --------------------------------------------------------------------------- #

DEFAULT_K = 50
MONO_TAU = 0.8          # Spearman(rank, phi)
CLIFF_TAU = 2.0         # max(d) / median(d) for "smooth"
CLIFF_DISCRETE = 4.0    # max(d) / median(d) for "discrete"
SHARE_TAU = 0.5         # one step's share of total JS
NONCE_TAU = 0.3         # |rho| for a flat nonce null
CONTRAST_TAU = 0.05     # endpoint JS below which the axis is not used
FRAME_TAU = 0.8         # cross-frame Spearman agreement

FRAMES = {
    "F1": "The movie was {w}.{f} Overall, it was ___",
    "F2": "I found the meal {w}.{f} All in all, it was ___",
    "F3": "Her performance seemed {w}.{f} On the whole, it was ___",
}

NEUTRAL_FRAME = "The movie was {w}.{f} The most fitting word is ___"

# Carrier-connotation counterbalance (§6): the cliff *location* must not move.
CARRIER_POS = "The masterpiece was {w}.{f} Overall, it was ___"
CARRIER_NEG = "The flop was {w}.{f} Overall, it was ___"

DEFAULT_NONCE = ["blorp", "flim", "zorb", "quax", "vimp", "trog", "sneld"]


# --------------------------------------------------------------------------- #
# Ladder data
# --------------------------------------------------------------------------- #

def load_ladder(path: str | Path, column: str = "word") -> list[dict]:
    """Read a ``rung / word / word_alt`` TSV into an ordered rung list."""
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            word = (row.get(column) or "").strip()
            if not word:
                continue
            rows.append({"rung": int(row["rung"]), "word": word})
    rows.sort(key=lambda d: d["rung"])
    return rows


def ladder_words(ladder) -> list[str]:
    return [r["word"] for r in ladder]


def build_prompt(template: str, word: str, filler: str = "") -> str:
    """Fill a frame template; ``{f}`` is an optional filler between rung and probe."""
    f = f" {filler.strip()}" if filler and filler.strip() else ""
    return template.format(w=word, f=f)


# --------------------------------------------------------------------------- #
# Behavioral signature
# --------------------------------------------------------------------------- #

def js_steps(probs, k: int = DEFAULT_K) -> list[float]:
    """Adjacent Jensen-Shannon distances on the union top-k support."""
    probs = [np.asarray(p, dtype=np.float64) for p in probs]
    return [float(js_topk(probs[i - 1], probs[i], k)) for i in range(1, len(probs))]


def cliff_ratio(steps) -> float:
    """``max / median`` of adjacent steps; large -> a dominant jump."""
    steps = [float(s) for s in steps]
    if not steps:
        return float("nan")
    med = float(np.median(steps))
    mx = float(np.max(steps))
    if med > 0:
        return mx / med
    return float("inf") if mx > 0 else 1.0


def single_step_share(steps) -> float:
    """Largest single step as a share of the total JS travelled."""
    steps = [abs(float(s)) for s in steps]
    total = float(np.sum(steps))
    return float(np.max(steps) / total) if total > 0 else float("nan")


def monotonicity(values) -> float:
    """Spearman correlation between the rung index and a scalar curve."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 3:
        return float("nan")
    return float(stats.spearmanr(np.arange(len(values)), values).correlation)


def _union_support(probs_list, k: int) -> np.ndarray:
    idx: set[int] = set()
    for p in probs_list:
        idx |= set(np.argsort(np.asarray(p))[::-1][:k].tolist())
    return np.array(sorted(idx), dtype=int)


def axis_projection(probs, k: int = DEFAULT_K) -> dict:
    """Project each distribution onto the ``q_last - q_first`` axis (output side).

    ``phi`` is 0 at the first rung and 1 at the last by construction; the test is
    whether the interior moves monotonically *along* that axis (cf. 006's
    ``behavioral_curve``).
    """
    probs = [np.asarray(p, dtype=np.float64) for p in probs]
    if len(probs) < 2:
        raise ValueError("axis_projection needs at least two distributions")
    idx = _union_support(probs, k)

    def prep(p):
        q = p[idx]
        s = q.sum()
        return q / s if s > 0 else np.full_like(q, 1.0 / q.size)

    P = np.stack([prep(p) for p in probs])
    p0, pK = P[0], P[-1]
    v = pK - p0
    vv = float(v @ v)
    if vv <= 0:
        phi = np.zeros(len(P), dtype=np.float64)
    else:
        phi = (P - p0) @ v / vv
    return {
        "phi": phi.tolist(),
        "support_size": int(idx.size),
        "js_endpoints": float(js_topk(probs[0], probs[-1], k)),
    }


def expected_valence(probs, id_valence: dict[int, float]) -> float:
    """Expected valence from a fixed, normed candidate lexicon (optional readout).

    ``id_valence`` maps token id -> human valence.  Returns ``nan`` if the
    candidates carry no mass.  Purely associative: no norms are bundled.
    """
    probs = np.asarray(probs, dtype=np.float64)
    ids = [i for i in id_valence if 0 <= i < probs.size]
    if not ids:
        return float("nan")
    m = np.array([probs[i] for i in ids], dtype=np.float64)
    s = float(m.sum())
    if s <= 0:
        return float("nan")
    v = np.array([id_valence[i] for i in ids], dtype=np.float64)
    return float((m / s) @ v)


# --------------------------------------------------------------------------- #
# Control / robustness helpers
# --------------------------------------------------------------------------- #

def path_independence(curve_a, curve_b) -> dict:
    """Agreement of two parameterisations' curves (rank corr + max abs gap)."""
    a = np.asarray(curve_a, dtype=np.float64)
    b = np.asarray(curve_b, dtype=np.float64)
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    rho = float(stats.spearmanr(a, b).correlation) if n > 2 else float("nan")
    return {"rho": rho, "max_abs_diff": float(np.max(np.abs(a - b))) if n else float("nan")}


def frame_robustness(phi_by_frame: dict, steps_by_frame: dict | None = None) -> dict:
    """Cross-frame agreement of the concept-coordinate curves and cliff location."""
    names = list(phi_by_frame)
    pairwise = {}
    rhos = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = phi_by_frame[names[i]], phi_by_frame[names[j]]
            n = min(len(a), len(b))
            r = float(stats.spearmanr(a[:n], b[:n]).correlation) if n > 2 else float("nan")
            pairwise[f"{names[i]}~{names[j]}"] = r
            if not np.isnan(r):
                rhos.append(r)

    cliff_locations = {}
    if steps_by_frame:
        for name, steps in steps_by_frame.items():
            steps = [abs(float(s)) for s in steps]
            cliff_locations[name] = int(np.argmax(steps)) if steps else -1

    agree = True
    if len(cliff_locations) >= 2:
        locs = list(cliff_locations.values())
        agree = (max(locs) - min(locs)) <= 1

    mean_rho = float(np.mean(rhos)) if rhos else float("nan")
    return {
        "pairwise_rho": pairwise,
        "mean_rho": mean_rho,
        "cliff_locations": cliff_locations,
        "cliff_agree": bool(agree),
    }


def frequency_residual(phi, zipf, lengths) -> dict:
    """Residualise ``phi`` on log-frequency and word length; re-test monotonicity."""
    phi = np.asarray(phi, dtype=np.float64)
    zipf = np.asarray(zipf, dtype=np.float64)
    lengths = np.asarray(lengths, dtype=np.float64)
    X = np.column_stack([np.ones_like(phi), zipf, lengths])
    beta, *_ = np.linalg.lstsq(X, phi, rcond=None)
    resid = phi - X @ beta
    return {"rho_resid": monotonicity(resid), "resid": resid.tolist(),
            "beta": beta.tolist()}


# --------------------------------------------------------------------------- #
# Verdict
# --------------------------------------------------------------------------- #

def continuity_verdict(phi, steps, rho_rank: float, contrast: float,
                       frame_robust: dict, nonce_rho: float,
                       resid_rho: float) -> dict:
    """Map metrics to ``continuous`` / ``discrete`` / ``absent`` / ``inconclusive``."""
    cliff = cliff_ratio(steps)
    share = single_step_share(steps)

    conditions = {
        "uses_axis": bool(rho_rank >= MONO_TAU),
        "smooth": bool(cliff <= CLIFF_TAU and share < SHARE_TAU),
        "frame_ok": bool(frame_robust.get("ok", False)),
        "nonce_ok": bool(abs(nonce_rho) <= NONCE_TAU) if not np.isnan(nonce_rho) else None,
        "freq_ok": bool(resid_rho >= MONO_TAU) if not np.isnan(resid_rho) else None,
        "cliff": None if np.isnan(cliff) else float(cliff),
        "single_share": None if np.isnan(share) else float(share),
    }

    if contrast < CONTRAST_TAU:
        label = "absent (contrast below tau)"
    elif rho_rank < NONCE_TAU:
        label = "absent (no monotone signal)"
    elif cliff > CLIFF_DISCRETE or share > SHARE_TAU:
        label = "discrete/bistable"
    elif (conditions["uses_axis"] and conditions["smooth"]
          and conditions["frame_ok"] and conditions["nonce_ok"] is not False
          and conditions["freq_ok"] is not False):
        label = "continuous (frame-robust)"
    elif conditions["uses_axis"] and conditions["smooth"]:
        label = "continuous (partial controls)"
    else:
        label = "inconclusive"
    return {"label": label, "cliff": conditions["cliff"],
            "single_share": conditions["single_share"], "conditions": conditions}
