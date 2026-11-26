"""Hierarchy instruments for experiment 012 (stage 3.0 template calibration).

Hard-coded lessons from experiments 009/011:

* the free next-token top-k measures **syntax** (``a/an/the/...`` dominate and
  the top-10 is almost invariant across the inserted word), so every read here
  is over a **constrained candidate set**, never the free continuum;
* a template is only usable if it is **level-invariant in how it is read**
  (``Q``: crispness of the candidate distribution is ~equal on super/basic/sub)
  while still **tracking content** (``MI``: the read distribution depends on
  which concept was inserted).  Either alone is uninformative: Q-invariance
  without MI is the 009 syntax artefact; high MI without Q-invariance means the
  template is read differently at different levels.

The model-free core (rendering, normalisation, ``Q``, ``MI``, entropy,
frame-agreement) is importable and unit-tested on synthetic data before it is
pointed at a transformer.  Model-facing helpers import torch lazily.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from tc.controls import spearman
from tc.lexicon import normalize


# --------------------------------------------------------------------------- #
# Triad data
# --------------------------------------------------------------------------- #

def load_triads(path: str | Path, split: str | None = None) -> list[dict]:
    """Read ``super/basic/sub/split`` triads, optionally filtering by ``split``."""
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            row = {k: (v.strip() if isinstance(v, str) else v) for k, v in row.items()}
            if split is not None and row["split"] != split:
                continue
            rows.append(row)
    return rows


def triad_concepts(triads: Sequence[Mapping[str, str]]) -> list[tuple[str, str]]:
    """Flatten triads to ``(word, level)`` pairs (super/basic/sub), with triad id."""
    out = []
    for i, t in enumerate(triads):
        out.append((t["super"], "super", i))
        out.append((t["basic"], "basic", i))
        out.append((t["sub"], "sub", i))
    return out

# --------------------------------------------------------------------------- #
# Templates (descent = G, entailment = E).  ``{X}`` is the concept slot,
# ``{Y}`` the second slot of an entailment question; ``{aX}`` / ``{aY}`` insert
# the word with an indefinite article.  These are *candidates*: stage 3.0 keeps
# only those that pass the calibration gate.
# --------------------------------------------------------------------------- #

TEMPLATES_G = {
    "types_include": "The types of {X} include",
    "examples_include": "Examples of things in the category {X} include",
    "such_as": "{X}, such as",
    "can_be": "{X} can be a",
    "kinds_are": "The kinds of {X} are",
    "category_includes": "The category {X} includes",
}

TEMPLATES_E = {
    "is_a": "Is {aX} a {Y}? Answer Yes or No.\nAnswer:",
    "every_is_a": "Is every {X} a {Y}? Answer Yes or No.\nAnswer:",
    "kind_of": "Is {X} a kind of {Y}? Answer Yes or No.\nAnswer:",
}

_VOWELS = set("aeiou")


def article(noun: str) -> str:
    """Indefinite article by first letter (good enough for these nouns)."""
    return "an" if noun[:1].lower() in _VOWELS else "a"


def render(template: str, x: str | None = None, y: str | None = None) -> str:
    """Fill ``{X}``/``{Y}`` (and their article forms) in a template."""
    out = template
    if x is not None:
        out = out.replace("{aX}", f"{article(x)} {x}").replace("{X}", x)
    if y is not None:
        out = out.replace("{aY}", f"{article(y)} {y}").replace("{Y}", y)
    return out[:1].upper() + out[1:] if out else out


# --------------------------------------------------------------------------- #
# Normalisation and per-concept read
# --------------------------------------------------------------------------- #

def normalize_probs(raw: Mapping[str, float], vocab: Sequence[str] | None = None) -> dict[str, float]:
    """Normalise ``raw`` word probabilities to sum to 1 over ``vocab``.

    Word keys are compared in :func:`tc.lexicon.normalize` form.  Missing
    entries are treated as zero.  ``vocab`` defaults to ``raw``'s keys.
    """
    norm_raw = {normalize(k): float(v) for k, v in raw.items()}
    words = list(vocab) if vocab is not None else list(raw)
    keys = [normalize(w) for w in words]
    values = np.array([norm_raw.get(k, 0.0) for k in keys], dtype=np.float64)
    total = float(values.sum())
    if total <= 0.0:
        return {w: 0.0 for w in words}
    values /= total
    return {w: float(v) for w, v in zip(words, values)}


def entropy(dist: Mapping[str, float]) -> float:
    """Shannon entropy (nats) of a distribution given as ``{word: prob}``."""
    p = np.array([v for v in dist.values() if v > 0.0], dtype=np.float64)
    if p.size == 0:
        return 0.0
    p = p / p.sum()
    return float(-np.sum(p * np.log(p)))


def mutual_information(dists: Sequence[Mapping[str, float]],
                       vocab: Sequence[str] | None = None) -> float:
    """``I(concept; word)`` for a list of per-concept candidate distributions.

    Each item of ``dists`` is the normalised read for one concept over a shared
    vocabulary.  Returns ``H(W) - H(W|C)`` in nats; 0 if all reads are identical
    (the 009 failure mode), ``log n_concepts`` if each concept has its own
    unique support.
    """
    if not dists:
        return 0.0
    words = list(vocab) if vocab is not None else list(dists[0])
    P = np.array([[float(d.get(w, 0.0)) for w in words] for d in dists], dtype=np.float64)
    P = np.clip(P, 0.0, None)
    rows = P.sum(axis=1, keepdims=True)
    rows[rows == 0.0] = 1.0
    P = P / rows
    q = P.mean(axis=0)
    h_word = _entropy_vec(q)
    h_cond = float(np.mean([_entropy_vec(p) for p in P]))
    return float(h_word - h_cond)


def _entropy_vec(p: np.ndarray) -> float:
    p = p[p > 0.0]
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * np.log(p)))


def frame_agreement(scores_a: Sequence[float], scores_b: Sequence[float]) -> float:
    """Spearman agreement of a per-concept score across two templates."""
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    if a.size < 2 or b.size != a.size:
        return float("nan")
    return spearman(a, b)


def permutation_pvalue(observed: float, null: Sequence[float]) -> float:
    """One-sided permutation p-value with the +1 correction."""
    null = np.asarray(null, dtype=np.float64)
    return float((np.sum(null >= observed) + 1.0) / (null.size + 1.0))


# --------------------------------------------------------------------------- #
# Candidate-token reads from a single next-token vector (no model here)
# --------------------------------------------------------------------------- #

def _single_token_id(tokenizer, word: str, leading_space: bool = True) -> int | None:
    """Token id of ``word`` as one token (with a leading space by default)."""
    text = (" " + word) if leading_space else word
    ids = tokenizer.encode(text, add_special_tokens=False)
    return int(ids[0]) if len(ids) == 1 else None


def readable_vocab(tokenizer, vocab: Sequence[str]) -> list[str]:
    """Keep only candidates that are a single token after a leading space."""
    return [w for w in vocab if _single_token_id(tokenizer, w) is not None]


def candidate_probs(probs: np.ndarray, tokenizer,
                    vocab: Sequence[str]) -> tuple[dict[str, float], dict[str, float]]:
    """Read ``P(candidate token)`` for each word from one next-token vector.

    Only single-token candidates are read (leading-space form, as it appears
    after a prompt); multi-token words are reported as ``nan`` in ``raw`` and
    dropped from ``normalised``.  This is the described *candidate-token
    distribution*: the read is over a constrained content vocabulary, never the
    free top-k, so the syntactic mass (``a/an/the/...``) is ignored by
    construction.
    """
    raw: dict[str, float] = {}
    for word in vocab:
        tid = _single_token_id(tokenizer, word)
        raw[word] = float(probs[tid]) if tid is not None else float("nan")
    found = [w for w in vocab if not math.isnan(raw[w])]
    return raw, normalize_probs({w: raw[w] for w in found}, found)


def entailment_mass(probs: np.ndarray, tokenizer) -> tuple[float, float]:
    """Return ``(P(Yes)/(P(Yes)+P(No)), P(Yes)+P(No))`` from one vector."""
    yes_id = _single_token_id(tokenizer, "Yes")
    no_id = _single_token_id(tokenizer, "No")
    yes = float(probs[yes_id]) if yes_id is not None else 0.0
    no = float(probs[no_id]) if no_id is not None else 0.0
    total = yes + no
    return (yes / total if total > 0.0 else 0.0), total


# --------------------------------------------------------------------------- #
# Main hierarchy test: targeted, labelled candidate sets per triad
# --------------------------------------------------------------------------- #

def build_candidates(triads: Sequence[Mapping[str, str]], tid: int,
                     n_sib: int = 4, n_unrel: int = 4, seed: int = 0) -> dict:
    """Labelled candidates for the super and basic of triad ``tid``.

    Returns ``{(word, level): {candidate: label}}`` where the label is one of
    ``desc`` (true child, exactly one level down), ``anc`` (parent),
    ``desc_wrong_level`` (in the subtree but two levels down), ``sib_branch``
    (same level, other branch), ``unrelated`` (other branch, other level).
    These labels drive the factor control that separates "right branch" from
    "right level" from "just a frequent word".
    """
    t = triads[tid]
    s, b, u = t["super"], t["basic"], t["sub"]
    others = [t2 for j, t2 in enumerate(triads) if j != tid]
    basics = sorted({t2["basic"] for t2 in others})
    subs = sorted({t2["sub"] for t2 in others})
    rng = np.random.default_rng(seed)

    def sample(pool, n):
        if len(pool) <= n:
            return list(pool)
        return sorted(rng.choice(pool, size=n, replace=False).tolist())

    sib, unrel = sample(basics, n_sib), sample(subs, n_unrel)

    cand_s = {b: "desc", u: "desc_wrong_level"}
    for w in sib:
        cand_s.setdefault(w, "sib_branch")
    for w in unrel:
        cand_s.setdefault(w, "unrelated")

    cand_b = {u: "desc", s: "anc"}
    for w in sib:
        cand_b.setdefault(w, "sib_branch")
    for w in unrel:
        cand_b.setdefault(w, "unrelated")

    return {(s, "super"): cand_s, (b, "basic"): cand_b}


def auc(scores, positive) -> float:
    """ROC-AUC of ``scores`` separating ``positive`` items from the rest."""
    from sklearn.metrics import roc_auc_score

    scores = np.asarray(scores, dtype=np.float64)
    positive = np.asarray(positive, dtype=bool)
    if positive.sum() == 0 or (~positive).sum() == 0:
        return float("nan")
    return float(roc_auc_score(positive, scores))


def descendant_metrics(dist: Mapping[str, float], labels: Mapping[str, str]) -> dict:
    """Per-concept descent read: AUC and mass on ``desc`` vs everything else."""
    words = list(dist)
    scores = [dist[w] for w in words]
    positive = [labels.get(w) == "desc" for w in words]
    top = max(words, key=lambda w: dist[w]) if words else None
    return {
        "auc": auc(scores, positive),
        "desc_mass": float(sum(dist[w] for w, p in zip(words, positive) if p)),
        "top": top,
        "top_is_desc": bool(top is not None and labels.get(top) == "desc"),
    }


def read_candidates(model, tokenizer, prompt: str,
                    words: Sequence[str]) -> tuple[dict[str, float], dict[str, float]]:
    """Sequence-probability read of each candidate word after ``prompt``.

    Unlike :func:`candidate_probs` this handles multi-token words (``poodle``,
    ``armchair``) via :func:`tc.centrality.word_probability`, so the sub level is
    not systematically dropped.
    """
    from tc.centrality import word_probability

    raw = {w: word_probability(model, tokenizer, prompt, w)[0] for w in words}
    return raw, normalize_probs(raw, list(words))
