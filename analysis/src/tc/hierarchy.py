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
from collections import deque
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


# --------------------------------------------------------------------------- #
# WordNet fragment (arm b): model-free graph core + lazy WordNet adapter.
#
# A fragment is ``{"roots": [...], "nodes": {name: node}}`` where each node has
# ``word`` (surface form), ``depth`` (shortest hypernym distance from a root),
# ``root`` (which root it descends from), ``parents`` and ``children`` (names).
# Depth and the descendant/ancestor sets come from the graph, not from a human
# choosing examples.  The WordNet adapter only builds the graph; all class
# building and testing below is pure and unit-tested on synthetic fragments.
# --------------------------------------------------------------------------- #

def build_fragment(edges: Sequence[tuple[str, str]],
                   words: Mapping[str, str],
                   roots: Sequence[str]) -> dict:
    """Build a fragment from ``(parent, child)`` edges and surface words.

    Depth is the shortest distance from any root; ``root`` records the first
    root that reaches a node (deterministic BFS in the order of ``roots``).
    """
    children: dict[str, set[str]] = {}
    parents: dict[str, set[str]] = {}
    for r in roots:
        children.setdefault(r, set())
        parents.setdefault(r, set())
    for p, c in edges:
        children.setdefault(p, set()).add(c)
        parents.setdefault(c, set()).add(p)
        children.setdefault(c, set())
        parents.setdefault(p, set())

    depth: dict[str, int] = {}
    root_of: dict[str, str] = {}
    queue = deque((r, 0, r) for r in roots)
    while queue:
        name, d, root = queue.popleft()
        if name in depth:
            continue
        depth[name] = d
        root_of[name] = root
        for child in sorted(children.get(name, ())):
            if child not in depth:
                queue.append((child, d + 1, root))

    nodes = {
        name: {
            "word": words.get(name, name),
            "depth": d,
            "root": root_of[name],
            "parents": sorted(parents.get(name, ())),
            "children": sorted(children.get(name, ())),
        }
        for name, d in depth.items()
    }
    return {"roots": list(roots), "nodes": nodes}


def fragment_descendants(fragment: Mapping, name: str) -> set[str]:
    """Transitive hyponyms of ``name`` inside the fragment."""
    nodes = fragment["nodes"]
    out: set[str] = set()
    stack = list(nodes[name]["children"])
    while stack:
        child = stack.pop()
        if child in out:
            continue
        out.add(child)
        stack.extend(nodes[child]["children"])
    return out


def fragment_ancestors(fragment: Mapping, name: str) -> set[str]:
    """Transitive hypernyms of ``name`` inside the fragment."""
    nodes = fragment["nodes"]
    out: set[str] = set()
    stack = list(nodes[name]["parents"])
    while stack:
        parent = stack.pop()
        if parent in out or parent not in nodes:
            continue
        out.add(parent)
        stack.extend(nodes[parent]["parents"])
    return out


def fragment_concepts(fragment: Mapping, min_children: int = 1) -> list[str]:
    """Concept nodes usable as test targets: depth >= 1 and with children."""
    nodes = fragment["nodes"]
    return [
        name
        for name, node in sorted(nodes.items(), key=lambda kv: (kv[1]["root"], kv[0]))
        if node["depth"] >= 1 and len(node["children"]) >= min_children
    ]


def labelled_candidates(fragment: Mapping, name: str, n_desc: int = 6,
                        n_anc: int = 2, n_wrong: int = 2, n_sib: int = 4,
                        n_unrel: int = 4, seed: int = 0) -> dict[str, str]:
    """Label candidate words for concept ``name`` from the fragment graph.

    Labels mirror the triads arm so the same metrics apply:
    ``desc`` (direct child, +1), ``anc`` (parent, -1), ``desc_wrong_level``
    (deeper descendant, >+1), ``sib_branch`` (same depth, other branch, same
    root), ``unrelated`` (different root).
    """
    nodes = fragment["nodes"]
    node = nodes[name]
    depth, root = node["depth"], node["root"]
    descendants = fragment_descendants(fragment, name)

    desc = list(node["children"])
    wrong = [n for n in descendants if nodes[n]["depth"] >= depth + 2]
    anc = [n for n in node["parents"] if n in nodes and nodes[n]["depth"] == depth - 1]
    sib = [n for n in nodes
           if n != name and nodes[n]["root"] == root and nodes[n]["depth"] == depth]
    unrel = [n for n in nodes if nodes[n]["root"] != root]

    rng = np.random.default_rng(seed)

    def take(pool: Sequence[str], k: int) -> list[str]:
        pool = sorted(set(pool))
        if len(pool) <= k:
            return pool
        return sorted(rng.choice(pool, size=k, replace=False).tolist())

    labels: dict[str, str] = {}
    # Descendants are the target: take them first so distractors never mask them.
    for n in take(desc, n_desc):
        labels[nodes[n]["word"]] = "desc"
    for n in take(anc, n_anc):
        labels.setdefault(nodes[n]["word"], "anc")
    for n in take(wrong, n_wrong):
        labels.setdefault(nodes[n]["word"], "desc_wrong_level")
    for n in take(sib, n_sib):
        labels.setdefault(nodes[n]["word"], "sib_branch")
    for n in take(unrel, n_unrel):
        labels.setdefault(nodes[n]["word"], "unrelated")
    return labels


def synset_is_abstract(name: str) -> bool:
    """True if the synset is not under ``physical_entity.n.01`` (WordNet test)."""
    from nltk.corpus import wordnet as wn

    synset = wn.synset(name)
    closure = synset.closure(lambda item: item.hypernyms())
    return not any(item.name() == "physical_entity.n.01" for item in closure)


def _non_root_nodes(fragment: Mapping) -> list[str]:
    """All non-root nodes (usable as the subject of an entailment pair)."""
    nodes = fragment["nodes"]
    return [
        name
        for name, node in sorted(nodes.items(), key=lambda kv: (kv[1]["root"], kv[0]))
        if node["depth"] >= 1
    ]


def fragment_entailment_pairs(fragment: Mapping, concepts: Sequence[str] | None = None,
                              max_distance: int = 2,
                              seed: int = 0) -> list[tuple[str, str, int, int]]:
    """``(x, y, is_true, distance)`` pairs built around ``concepts``.

    For each concept C: true pairs ``(C, ancestor)`` 1..``max_distance`` steps
    up; false pairs are the reverse ``(ancestor, C)``, a direct child, a
    same-level other-branch word, and a cross-root word.  ``distance`` is
    defined for true pairs only (0 for false ones).  If ``concepts`` is None,
    all non-root nodes are used.
    """
    nodes = fragment["nodes"]
    if concepts is None:
        concepts = _non_root_nodes(fragment)
    rng = np.random.default_rng(seed)
    pairs: list[tuple[str, str, int, int]] = []
    seen: set[tuple[str, str]] = set()

    def add(x: str, y: str, is_true: int, distance: int) -> None:
        wx, wy = nodes[x]["word"], nodes[y]["word"]
        if wx == wy or (x, y) in seen:
            return
        seen.add((x, y))
        pairs.append((wx, wy, is_true, distance))

    for name in concepts:
        node = nodes[name]
        root, depth = node["root"], node["depth"]

        cur = node["parents"][0] if node["parents"] else None
        distance = 1
        while cur is not None and cur in nodes and distance <= max_distance:
            add(name, cur, 1, distance)
            add(cur, name, 0, 0)
            parents = nodes[cur]["parents"]
            cur = parents[0] if parents else None
            distance += 1

        if node["children"]:
            add(name, node["children"][0], 0, 0)
        same = sorted(n for n in nodes if n != name
                      and nodes[n]["root"] == root and nodes[n]["depth"] == depth)
        if same:
            add(name, same[int(rng.integers(len(same)))], 0, 0)
        other = sorted(n for n in nodes if nodes[n]["root"] != root)
        if other:
            add(name, other[int(rng.integers(len(other)))], 0, 0)
    return pairs


def load_fragment(path: str | Path) -> dict:
    """Read a fragment TSV with columns ``parent``/``synset``/``word``.

    A row with an empty ``parent`` declares a root.
    """
    edges: list[tuple[str, str]] = []
    words: dict[str, str] = {}
    roots: list[str] = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            parent = (row.get("parent") or "").strip()
            name = row["synset"].strip()
            words[name] = row["word"].strip()
            if not parent:
                roots.append(name)
            else:
                edges.append((parent, name))
    return build_fragment(edges, words, roots)


def write_fragment(fragment: Mapping, path: str | Path) -> None:
    """Write a fragment as TSV (``parent``/``synset``/``word``), sorted."""
    nodes = fragment["nodes"]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, node in sorted(nodes.items(), key=lambda kv: (kv[1]["root"], kv[1]["depth"], kv[0])):
        parents = node["parents"] if node["depth"] > 0 else [""]
        for parent in parents:
            rows.append((parent, name, node["word"]))
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["parent", "synset", "word"])
        writer.writerows(rows)


def _dominant_surface(synset, wn) -> str | None:
    """First lemma of ``synset`` if it is that word's most frequent noun sense.

    This is the fixed polysemy rule: a word is used only in its first noun
    sense, so depth and descendants are unambiguous.  Words whose first sense
    is a different synset are dropped from the fragment.
    """
    lemmas = synset.lemma_names()
    if not lemmas:
        return None
    word = lemmas[0].replace("_", " ")
    senses = wn.synsets(word, pos="n")
    if not senses or senses[0].name() != synset.name():
        return None
    return word


def wordnet_seed_fragment(min_zipf: float = 4.0, min_children: int = 3,
                          top_n: int = 50000) -> dict:
    """Fragment built from common concepts and their common relatives.

    Instead of walking down from broad roots (where common words sit behind rare
    technical intermediates), this takes everyday concepts directly: a common
    noun (Zipf >= ``min_zipf``) is a seed if it has at least ``min_children``
    common hyponyms and a common hypernym.  For each seed it adds the seed, its
    children (depth +1) and up to two common ancestors (depth -1, -2).  Every
    node is therefore a common word; levels still come from the WordNet graph.
    """
    from nltk.corpus import wordnet as wn
    from wordfreq import top_n_list, zipf_frequency

    def surface(synset) -> str | None:
        word = _dominant_surface(synset, wn)
        if word is None or zipf_frequency(word, "en") < min_zipf:
            return None
        return word

    edges: list[tuple[str, str]] = []
    words: dict[str, str] = {}

    for raw in top_n_list("en", top_n):
        if not raw.isalpha():
            continue
        synsets = wn.synsets(raw, pos="n")
        if not synsets:
            continue
        seed = synsets[0]
        seed_word = _dominant_surface(seed, wn)
        if seed_word != raw or zipf_frequency(seed_word, "en") < min_zipf:
            continue
        children = []
        for child in seed.hyponyms():
            child_word = surface(child)
            if child_word is not None:
                children.append((child, child_word))
        if len(children) < min_children:
            continue

        chain = []
        current = seed
        for _ in range(2):
            parents = [p for p in current.hypernyms() if surface(p) is not None]
            if not parents:
                break
            current = parents[0]
            chain.append(current)
        if not chain:
            continue

        words.setdefault(seed.name(), seed_word)
        edges.append((chain[0].name(), seed.name()))
        if len(chain) >= 2:
            edges.append((chain[1].name(), chain[0].name()))
        for ancestor in chain:
            words.setdefault(ancestor.name(), surface(ancestor))
        for child, child_word in children:
            words.setdefault(child.name(), child_word)
            edges.append((seed.name(), child.name()))

    child_names = {child for _, child in edges}
    roots = sorted({parent for parent, _ in edges} - child_names)
    return build_fragment(edges, words, roots)


def wordnet_fragment(roots: Sequence[str], max_depth: int = 3,
                     max_children: int = 6) -> dict:
    """Build a fragment from WordNet under ``roots`` (lazy nltk import).

    Deterministic: children are sorted by synset name and capped at
    ``max_children``; only the dominant noun sense of each word is kept.
    """
    from nltk.corpus import wordnet as wn

    edges: list[tuple[str, str]] = []
    words: dict[str, str] = {}
    kept_roots: list[str] = []

    for root_name in roots:
        root = wn.synset(root_name)
        word = _dominant_surface(root, wn)
        if word is None:
            continue
        kept_roots.append(root.name())
        words.setdefault(root.name(), word)
        seen = {root.name()}
        queue = deque([(root, 0)])
        while queue:
            synset, depth = queue.popleft()
            if depth >= max_depth:
                continue
            scored = []
            for child in synset.hyponyms():
                child_word = _dominant_surface(child, wn)
                if child_word is None:
                    continue
                branching = sum(
                    1 for grandchild in child.hyponyms()
                    if _dominant_surface(grandchild, wn) is not None
                )
                scored.append((branching, child.name(), child, child_word))
            # Prefer branching subtrees (protocol: branches with branching),
            # then order by synset name so the fragment is deterministic.
            scored.sort(key=lambda item: (-item[0], item[1]))
            for _, _, child, child_word in scored[:max_children]:
                edges.append((synset.name(), child.name()))
                words.setdefault(child.name(), child_word)
                if child.name() not in seen:
                    seen.add(child.name())
                    queue.append((child, depth + 1))

    return build_fragment(edges, words, kept_roots)
