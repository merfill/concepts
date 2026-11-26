"""Text-only channels for centrality (surface vs meaning, black-box friendly).

Channels that need no token probabilities:
- free listing   -> ordered names (prototypes first);
- rating 1-7     -> a score per word.

Parsers are pure (model-free) and unit-tested.  Generation lives in scripts.
"""

from __future__ import annotations

import re

from tc.centrality import member_match

LABELS = {
    "tool": "hand tools used for work",
    "clothing": "items of clothing",
    "mammal": "mammals",
    "fruit": "fruits",
    "vegetable": "vegetables",
    "furniture": "pieces of furniture",
    "vehicle": "vehicles",
    "weapon": "weapons",
    "bird": "birds",
    "musical instrument": "musical instruments",
}

_LEAD_NUM = re.compile(r"^\s*\d+\s*[\.\)\:]\s*")
_LEAD_BULLET = re.compile(r"^\s*[\-\*\u2022]\s*")
_LEAD_ARTICLE = ("a pair of ", "an pair of ", "pair of ", "a ", "an ", "the ")
_EDGE = "\"'`.,;:!?()[]{}*<>-_/"
_RATING_NUM = re.compile(r"\d+(?:\.\d+)?")


def _strip_lead(text: str) -> str:
    text = _LEAD_NUM.sub("", text)
    text = _LEAD_BULLET.sub("", text)
    return text


def normalize_phrase(text: str) -> str:
    """Lower-case, strip list marking/articles/'pair of', collapse spaces."""
    text = _strip_lead(text).strip().lower()
    changed = True
    while changed:
        changed = False
        for prefix in _LEAD_ARTICLE:
            if text.startswith(prefix):
                text = text[len(prefix):]
                changed = True
    text = text.strip().strip(_EDGE).strip()
    return re.sub(r"\s+", " ", text)


def _label(category: str) -> str:
    return LABELS.get(category, category)


def list_prompt(category: str, n: int = 10) -> str:
    return (f"List {n} {_label(category)}, most typical first. "
            f"Reply with names only, separated by commas, no commentary.")


def rank_prompt(category: str, words, seed: int = 0) -> str:
    import random

    shuffled = list(words)
    random.Random(seed).shuffle(shuffled)
    return (f"Here are 20 {_label(category)}: {', '.join(shuffled)}.\n"
            f"Rank them from most to least typical of a {category}. "
            f"Reply with the 20 words in order, one per line, no commentary.")


def rating_prompt(category: str, words, seed: int = 0) -> str:
    import random

    shuffled = list(words)
    random.Random(seed).shuffle(shuffled)
    return (f"Rate how typical each of these {_label(category)} is on a scale "
            f"from 1 (least typical) to 7 (most typical):\n{', '.join(shuffled)}.\n"
            f"Reply with one line per word: the word, a colon, then the number. No commentary.")


def parse_list(text: str, member_set: set[str], lemmatizer=None) -> dict:
    """Split a free-list answer into matched member lemmas (ordered) and leftovers."""
    chunks = re.split(r"[,\n;]+", text)
    matched: list[str] = []
    unmatched: list[str] = []
    for chunk in chunks:
        candidate = normalize_phrase(chunk)
        if not candidate or not re.search(r"[a-zA-Z]", candidate):
            continue
        hit = member_match(candidate, member_set, lemmatizer)
        if hit is not None:
            matched.append(hit)
        else:
            unmatched.append(chunk.strip())
    seen: set[str] = set()
    ordered: list[str] = []
    for lemma in matched:
        if lemma not in seen:
            seen.add(lemma)
            ordered.append(lemma)
    return {"matched": ordered, "unmatched": unmatched}


def parse_ranking(text: str, words) -> dict:
    """Locate each known word in the ranked answer; return positions and order."""
    working = text.lower()
    positions: dict[str, int] = {}
    for word in sorted(words, key=len, reverse=True):
        phrase = word.lower().replace("_", " ")
        pattern = r"(?<![a-z])" + re.escape(phrase) + r"(?![a-z])"
        match = re.search(pattern, working)
        if match:
            positions[word] = match.start()
            working = working[:match.start()] + "\x00" * len(match.group(0)) + working[match.end():]
    ordered = sorted(positions, key=lambda w: positions[w])
    return {"positions": positions, "ordered": ordered}


def parse_rating(text: str, words) -> dict:
    """Parse 'word: number' lines into ``{word: rating}`` for known words."""
    keys = {normalize_phrase(w): w for w in words}
    keys_sorted = sorted(keys, key=len, reverse=True)
    ratings: dict[str, float] = {}
    for line in re.split(r"[\n;]+", text):
        stripped = _strip_lead(line.strip())
        if not stripped:
            continue
        match = _RATING_NUM.search(stripped)
        if not match:
            continue
        value = float(match.group(0))
        remainder = normalize_phrase(stripped[:match.start()] + " " + stripped[match.end():])
        if not remainder:
            continue
        for key in keys_sorted:
            if key and re.search(r"(?<![a-z])" + re.escape(key) + r"(?![a-z])", remainder):
                original = keys[key]
                ratings.setdefault(original, value)
                break
    return ratings


def ranking_spearman(parsed_order, words, human_by_word) -> float:
    """Spearman between model order (first = most typical) and human typicality."""
    import numpy as np
    from scipy.stats import spearmanr

    model_score, human = [], []
    for rank, word in enumerate(parsed_order):
        if word in human_by_word:
            model_score.append(-rank)
            human.append(human_by_word[word])
    if len(model_score) < 3:
        return float("nan")
    return float(spearmanr(np.array(model_score), np.array(human))[0])
