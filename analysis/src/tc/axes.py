"""Graded-axis validation (experiment 008, program step 4.0a).

Before measuring a path (experiment 006) we must confirm that the model treats
the rungs as an *ordered* progression.  Pure helpers here are model-free; the
model-facing probes live in ``scripts/run_axis_validation.py``.

Single-token is a *preference*, not a hard filter: multi-token rungs are allowed
and scored with length-normalised log-probs.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from .interpolate import behavioral_profile

AXES = {
    "temperature": {
        "words": ["cold", "cool", "warm", "hot"],
        "frame": "The water is {w}.", "dim": "hotter", "opposite": "colder",
        "low_anchor": "coldest", "high_anchor": "hottest",
    },
    "size": {
        "words": ["tiny", "small", "medium", "large", "huge"],
        "frame": "The animal is {w}.", "dim": "larger", "opposite": "smaller",
        "low_anchor": "smallest", "high_anchor": "largest",
    },
    "quality": {
        "words": ["bad", "weak", "fair", "good", "great"],
        "frame": "The food is {w}.", "dim": "better", "opposite": "worse",
        "low_anchor": "worst", "high_anchor": "best",
    },
    "loudness": {
        "words": ["quiet", "soft", "loud", "booming"],
        "frame": "The sound is {w}.", "dim": "louder", "opposite": "quieter",
        "low_anchor": "quietest", "high_anchor": "loudest",
    },
    "speed": {
        "words": ["slow", "steady", "fast", "rapid"],
        "frame": "The car is {w}.", "dim": "faster", "opposite": "slower",
        "low_anchor": "slowest", "high_anchor": "fastest",
    },
    "brightness": {
        "words": ["dim", "bright", "glowing", "glaring"],
        "frame": "The light is {w}.", "dim": "brighter", "opposite": "dimmer",
        "low_anchor": "darkest", "high_anchor": "brightest",
    },
}


SHUFFLE_ORDER = (2, 0, 4, 1, 3)


def order_prompt(name: str, words, reverse: bool = False) -> str:
    """Generative ordering prompt with a shuffled input order (no copying)."""
    spec = AXES[name]
    lo, hi = (spec["high_anchor"], spec["low_anchor"]) if reverse else \
             (spec["low_anchor"], spec["high_anchor"])
    shuffled = [words[i] for i in SHUFFLE_ORDER if i < len(words)]
    return f"Order these words from {lo} to {hi}: {', '.join(shuffled)}.\nAnswer:"


def parse_order(text: str, words) -> list[str]:
    """Recover the generated ordering of ``words`` by first occurrence."""
    import re
    low = text.lower()
    found = []
    for w in words:
        m = re.search(r"\b" + re.escape(w.lower()) + r"\b", low)
        if m:
            found.append((m.start(), w))
    found.sort()
    return [w for _, w in found]


def likert_prompt(name: str, word: str) -> str:
    """Anchored 1..5 rating prompt (explicitly sets the scale attractor)."""
    spec = AXES[name]
    phrase = spec["frame"].format(w=word)
    return (f"Rate the property on a scale from 1 ({spec['low_anchor']}) "
            f"to 5 ({spec['high_anchor']}).\n"
            f"Phrase: \"{phrase}\"\nRating:")


def likert_prompt_rev(name: str, word: str) -> str:
    """Reversed-anchor rating: 1 = high extreme, 5 = low extreme."""
    spec = AXES[name]
    phrase = spec["frame"].format(w=word)
    return (f"Rate the property on a scale from 1 ({spec['high_anchor']}) "
            f"to 5 ({spec['low_anchor']}).\n"
            f"Phrase: \"{phrase}\"\nRating:")


def likert_expected(probs, digit_ids, values=(1, 2, 3, 4, 5)) -> float:
    """Expected rating from the next-token mass on the digit tokens."""
    probs = np.asarray(probs, dtype=np.float64)
    m = np.array([float(probs[i]) for i in digit_ids], dtype=np.float64)
    s = m.sum()
    if s <= 0:
        return float("nan")
    return float((m / s) @ np.asarray(values, dtype=np.float64))

FC_TAU = 0.8
JUDGE_TAU = 0.8
JS_END_TAU = 0.05
JUMP_RATIO = 3.0


def names() -> list[str]:
    return list(AXES)


def ladder(name: str) -> dict:
    spec = AXES[name]
    words = list(spec["words"])
    return {
        "name": name, "words": words, "dim": spec["dim"],
        "opposite": spec.get("opposite"),
        "frame": spec["frame"],
        "texts": [spec["frame"].format(w=w) for w in words],
    }


def adjacent(words) -> list[tuple[str, str]]:
    return [(words[i], words[i + 1]) for i in range(len(words) - 1)]


def two_way(logp_a: float, logp_b: float) -> tuple[float, float]:
    """Softmax over two log-probs -> (P(a), P(b))."""
    m = max(logp_a, logp_b)
    ea, eb = np.exp(logp_a - m), np.exp(logp_b - m)
    z = ea + eb
    return float(ea / z), float(eb / z)


def ordering_accuracy(scores) -> float:
    """Scores are (logp_low, logp_high): correct iff ``high > low``."""
    scores = list(scores)
    if not scores:
        return float("nan")
    return float(np.mean([1.0 if hi > lo else 0.0 for lo, hi in scores]))


def spearman_index(values) -> float:
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 3:
        return float("nan")
    return float(stats.spearmanr(np.arange(len(values)), values).correlation)


def axis_verdict(fc_acc: float, fc_acc_alt: float, judge_rho: float,
                 cont_profile: dict | None) -> dict:
    channels = []
    if fc_acc >= FC_TAU and fc_acc_alt >= FC_TAU:
        channels.append("forced_choice")
    if judge_rho >= JUDGE_TAU:
        channels.append("judgment")

    endpoint_ok = bool(fc_acc >= 1.0 or (
        cont_profile is not None and cont_profile.get("js_endpoints", 0.0) >= JS_END_TAU))

    no_jump = True
    if cont_profile is not None:
        adj = [float(x) for x in cont_profile.get("js_adjacent", [])]
        med = float(np.median(adj)) if adj else 0.0
        if med > 0:
            no_jump = bool(max(adj) / med <= JUMP_RATIO)

    if len(channels) >= 2 and endpoint_ok and no_jump:
        verdict = "valid"
    elif len(channels) >= 2:
        verdict = "valid-order (contrast/jump weak)"
    elif len(channels) == 1:
        verdict = "partial (one channel)"
    else:
        verdict = "weak/unsupported"

    return {"verdict": verdict, "channels": channels,
            "endpoint_ok": endpoint_ok, "no_jump": no_jump}


def axis_verdict_anchored(fc_bidir: float, likert_rho: float,
                          cont_profile: dict | None) -> dict:
    """Verdict for the anchored probes: bidirectional FC + Likert monotonicity."""
    channels = []
    if not np.isnan(fc_bidir) and fc_bidir >= FC_TAU:
        channels.append("forced_choice")
    if not np.isnan(likert_rho) and likert_rho >= JUDGE_TAU:
        channels.append("likert")

    endpoint_ok = bool(
        (not np.isnan(fc_bidir) and fc_bidir >= 1.0) or (
            cont_profile is not None and cont_profile.get("js_endpoints", 0.0) >= JS_END_TAU))

    no_jump = True
    if cont_profile is not None:
        adj = [float(x) for x in cont_profile.get("js_adjacent", [])]
        med = float(np.median(adj)) if adj else 0.0
        if med > 0:
            no_jump = bool(max(adj) / med <= JUMP_RATIO)

    if len(channels) >= 2 and no_jump:
        verdict = "valid" if endpoint_ok else "valid-order (contrast weak)"
    elif len(channels) >= 2:
        verdict = "valid-order (contrast/jump weak)"
    elif len(channels) == 1:
        verdict = "partial (one channel)"
    else:
        verdict = "weak/unsupported"

    return {"verdict": verdict, "channels": channels,
            "endpoint_ok": endpoint_ok, "no_jump": no_jump}
