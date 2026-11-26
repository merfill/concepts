"""Category centrality via instance naming (experiment 010).

Model-free helpers (token classification, purity, matching, partial correlation)
are importable without torch/transformers/WordNet data so they can be tested
directly.  Model-facing helpers (top-k distribution, sequence log-prob,
membership judgment) take a HuggingFace model/tokenizer and import lazily.
"""

from __future__ import annotations

import math
import re

import numpy as np

from tc.lexicon import normalize

DESCRIPTORS = {
    "mammal": "mammal",
    "bird": "bird",
    "fruit": "fruit",
    "vegetable": "vegetable",
    "furniture": "piece of furniture",
    "vehicle": "vehicle for transport",
    "tool": "tool used for work",
    "weapon": "weapon",
    "clothing": "item of clothing",
    "musical instrument": "instrument used to make music",
}

# Frames fixed after structural inspection (protocol §3, §10).  Only categories
# whose phrasing was inspected are listed; others fall back to a generic pair.
FRAMES = {
    "mammal": ["A mammal, such as a", "Name one mammal: a"],
}

def article(noun: str, capitalize: bool = False) -> str:
    """Indefinite article by first letter (good enough for these nouns)."""
    art = "an" if noun[:1].lower() in set("aeiou") else "a"
    return art.capitalize() if capitalize else art


def candidate_frames(category: str) -> list[str]:
    """Grammatically safe candidate frames to scan structurally (purity)."""
    desc = DESCRIPTORS.get(category, category)
    art = article(desc)
    return [
        f"Name one {desc}: a",
        f"One example of {art} {desc} is a",
        f"A common {desc} is a",
        f"{article(desc, capitalize=True)} {desc}, such as a",
    ]


FUNCTION_WORDS = set(
    """a an the of to in on at for and or but nor so yet as is are was were be been being
    am do does did have has had it its this that these those there here with without from by
    about into over under again further then once how what when where why which who whom whose
    i you he she we they me him her us them my your his their our one something someone
    anything nothing some any no not very more most such own same other another two three many
    all first either neither both several few""".split()
)

_ALPHA = re.compile(r"[A-Za-z]")


def is_function_token(token: str) -> bool:
    """True for punctuation/format tokens and closed-class function words."""
    text = token.strip().lower()
    if not text:
        return True
    if not _ALPHA.search(text):
        return True
    return text in FUNCTION_WORDS


def member_match(token: str, member_set: set[str], lemmatizer=None) -> str | None:
    """Return the member lemma matching a surface token, or None."""
    candidate = normalize(token)
    if candidate in member_set:
        return candidate
    if lemmatizer is not None:
        lemma = normalize(lemmatizer.lemmatize(candidate.replace("_", " "), "n"))
        if lemma in member_set:
            return lemma
    for suffix in ("s", "es"):
        if candidate.endswith(suffix) and len(candidate) > len(suffix) + 1:
            stem = candidate[: -len(suffix)]
            if stem in member_set:
                return stem
    return None


def compute_purity(top_items, member_set: set[str], lemmatizer=None) -> dict:
    """Split a top-k probability window into member / function / other mass.

    ``top_items`` is a sequence of ``(token, probability)`` pairs.
    """
    member_mass = function_mass = other_mass = 0.0
    members: list[dict] = []
    others: list[dict] = []
    for token, prob in top_items:
        prob = float(prob)
        if is_function_token(token):
            function_mass += prob
            continue
        matched = member_match(token, member_set, lemmatizer)
        if matched is not None:
            member_mass += prob
            members.append({"token": token, "lemma": matched, "p": prob})
        else:
            other_mass += prob
            others.append({"token": token, "p": prob})
    window = member_mass + function_mass + other_mass
    return {
        "window_mass": window,
        "member_mass": member_mass,
        "function_mass": function_mass,
        "other_mass": other_mass,
        "purity": (member_mass / window) if window > 0 else float("nan"),
        "members": members,
        "other": others,
    }


def partial_pearson(x, y, controls) -> float:
    """Pearson correlation of x and y after linear residualisation on controls."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    controls = np.asarray(controls, dtype=float)
    if controls.ndim == 1:
        controls = controls[:, None]

    def residual(a):
        design = np.column_stack([np.ones(len(a)), controls])
        beta, *_ = np.linalg.lstsq(design, a, rcond=None)
        return a - design @ beta

    rx, ry = residual(x), residual(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def topk_items(tokenizer, probs, k: int = 50) -> list[tuple[str, float]]:
    """Top-k ``(decoded_token, probability)`` pairs from a next-token vector."""
    idx = np.argsort(probs)[-k:][::-1]
    return [(tokenizer.decode([int(i)]).strip(), float(probs[i])) for i in idx]


def sequence_logprob(model, tokenizer, prompt: str, continuation: str) -> tuple[float, int]:
    """Teacher-forced log-prob of ``continuation`` after ``prompt`` and its length."""
    import torch

    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids
    cont_ids = tokenizer(continuation, add_special_tokens=False, return_tensors="pt").input_ids
    n_tokens = int(cont_ids.shape[1])
    if n_tokens == 0:
        return float("nan"), 0
    ids = torch.cat([prompt_ids, cont_ids], dim=1)
    with torch.no_grad():
        out = model(input_ids=ids)
    logprobs = torch.log_softmax(out.logits[0], dim=-1).double().numpy()
    start = prompt_ids.shape[1]
    values = [logprobs[start - 1 + i, int(cont_ids[0, i])] for i in range(n_tokens)]
    return float(np.sum(values)), n_tokens


def word_probability(model, tokenizer, prompt: str, word: str) -> tuple[float, int]:
    """Probability that the model's next word is ``word`` (handles multi-token)."""
    logprob, n_tokens = sequence_logprob(model, tokenizer, prompt, " " + word)
    if n_tokens == 0:
        return 0.0, 0
    return float(math.exp(logprob)), n_tokens


def item_probability(model, tokenizer, prompt: str, word: str,
                     prefixes=("pair of",)) -> tuple[float, dict]:
    """P(model names ``word``), summing auxiliary English prefixes (e.g. 'pair of').

    Returns ``(total, breakdown)`` where ``breakdown['direct']`` is ``word`` as the
    immediate next word and each other key is a prefix (mutually exclusive events,
    so the probabilities add).
    """
    base, n_direct = word_probability(model, tokenizer, prompt, word)
    total = base
    breakdown = {"direct": base}
    for prefix in prefixes:
        logprob, n_tokens = sequence_logprob(model, tokenizer, prompt, " " + prefix + " " + word)
        prob = float(math.exp(logprob)) if n_tokens else 0.0
        total += prob
        breakdown[prefix] = prob
    return float(total), breakdown


def find_token_ids(tokenizer, word: str) -> list[int]:
    """Token ids whose decoded form equals ``word`` (case-insensitive)."""
    target = word.strip().lower()
    ids = [i for i in range(len(tokenizer)) if tokenizer.decode([i]).strip().lower() == target]
    if not ids:
        for text in (f" {word}", word):
            encoded = tokenizer(text, add_special_tokens=False).input_ids
            if len(encoded) == 1:
                ids.append(int(encoded[0]))
    return sorted(set(ids))


def membership_prob(model, tokenizer, item: str, label: str, yes_ids, no_ids) -> float:
    """P(Yes) / (P(Yes) + P(No)) for ``Is "item" a label?``."""
    from tc.extract import next_token_probs

    prompt = f'Is "{item}" a {label}? Answer Yes or No.\nAnswer:'
    probs = next_token_probs(model, tokenizer, prompt)
    yes = float(probs[yes_ids].sum())
    no = float(probs[no_ids].sum())
    return yes / (yes + no) if (yes + no) > 0 else 0.0
