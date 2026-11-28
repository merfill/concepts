"""Slot relations (experiment 020): fixed frame, WordNet relations in the slot.

One frame, one slot, only the filler word changes.  Fillers are words related to
an anchor by different WordNet relations (siblings, hypernym, hyponym, synonym,
meronym, holonym, derivation; antonyms on a small adjective arm) plus unrelated
controls.  We read three channels: the full next-token distribution, the layerwise
logit-lens distribution, and the residual-stream state at the slot position.

Protocol: ``../experiment-020-2026-11-28-wordnet-slot-relations.md``.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from wordfreq import zipf_frequency

from .controls import residualize, spearman, upper_tri
from .separation import abs_diff_rdm, label_rdm

FRAMES = [
    "When we say {x}, we mean",
    "The expression {x} refers to",
    "In everyday language, {x} is",
]

NOUN_ANCHORS = [
    "dog", "car", "tree", "bird", "fish", "house", "chair", "table", "knife",
    "hand", "eye", "river", "mountain", "flower", "book", "computer",
    "telephone", "shirt", "shoe", "bicycle",
]
ADJ_ANCHORS = [
    "hot", "big", "fast", "happy", "bright", "young", "strong", "clean",
    "empty", "near",
]

NOUN_RELATIONS = [
    "self", "sibling", "hypernym", "hyponym", "synonym", "meronym", "holonym",
    "derivation",
]
ADJ_RELATIONS = ["self", "antonym", "synonym", "similar", "derivation"]
RELATIONS_BY_POS = {"noun": NOUN_RELATIONS, "adjective": ADJ_RELATIONS}
CONTROL_RELATION = "unrelated"
HARD_CONTROL_RELATION = "unrelated_same"
CONTROL_RELATIONS = (CONTROL_RELATION, HARD_CONTROL_RELATION)

_WORD_RE = re.compile(r"[a-z]+(?: [a-z]+)*")

TSV_COLUMNS = ["pos", "anchor_id", "anchor", "relation", "word", "zipf", "n_chars"]


# ---------------------------------------------------------------------------
# WordNet stimuli
# ---------------------------------------------------------------------------

def clean_name(name) -> str:
    """Lower-case lemma name with underscores turned into spaces."""
    return str(name).replace("_", " ").strip().lower()


def _lemmas(synsets) -> list[str]:
    return [lemma.name() for synset in synsets for lemma in synset.lemmas()]


def anchor_synset(word: str, pos: str):
    """First synset of the requested part of speech (fixed sense, as in 012)."""
    from nltk.corpus import wordnet as wn

    tag = wn.NOUN if pos == "noun" else wn.ADJ
    synsets = wn.synsets(word, pos=tag)
    return synsets[0] if synsets else None


def raw_relatives(word: str, pos: str):
    """Map relation -> raw lemma names for one anchor, or ``None`` if no synset."""
    synset = anchor_synset(word, pos)
    if synset is None:
        return None
    rel: dict[str, list[str]] = {"self": [word]}
    if pos == "noun":
        hypernyms = synset.hypernyms()
        rel["hypernym"] = _lemmas(hypernyms)
        rel["hyponym"] = _lemmas(synset.hyponyms())
        rel["synonym"] = _lemmas([synset])
        rel["meronym"] = _lemmas(synset.part_meronyms())
        rel["holonym"] = _lemmas(synset.part_holonyms())
        rel["derivation"] = [
            form.name()
            for lemma in synset.lemmas()
            for form in lemma.derivationally_related_forms()
        ]
        best = max(hypernyms, key=lambda h: len(h.hyponyms()), default=None)
        if best is not None:
            rel["sibling"] = [
                name
                for sister in best.hyponyms()
                if sister.name() != synset.name()
                for name in _lemmas([sister])
            ]
    else:
        rel["antonym"] = [
            antonym.name()
            for lemma in synset.lemmas()
            for antonym in lemma.antonyms()
        ]
        rel["synonym"] = _lemmas([synset])
        rel["similar"] = _lemmas(synset.similar_tos())
        rel["derivation"] = [
            form.name()
            for lemma in synset.lemmas()
            for form in lemma.derivationally_related_forms()
        ]
    return rel


def select_relation_words(cands, anchor: str, max_n: int = 4,
                          min_zipf: float = 4.0) -> list[str]:
    """Clean, filter and deterministically cap relation words (alphabetical)."""
    out, seen = [], {clean_name(anchor)}
    for cand in cands:
        name = clean_name(cand)
        if not name or name in seen:
            continue
        if not _WORD_RE.fullmatch(name):
            continue
        if zipf_frequency(name, "en") < min_zipf:
            continue
        seen.add(name)
        out.append(name)
    out.sort()
    return out[:max_n]


def wordnet_pool(pos: str, min_zipf: float = 4.0) -> dict[str, tuple[float, str]]:
    """Common same-POS words with Zipf score and lexicographer class."""
    from nltk.corpus import wordnet as wn

    tag = wn.NOUN if pos == "noun" else wn.ADJ
    pool: dict[str, tuple[float, str]] = {}
    for synset in wn.all_synsets(tag):
        lexicon = synset.lexname()
        for lemma in synset.lemmas():
            name = clean_name(lemma.name())
            if not name or name in pool or not _WORD_RE.fullmatch(name):
                continue
            score = float(zipf_frequency(name, "en"))
            if score >= min_zipf:
                pool[name] = (score, lexicon)
    return pool


def select_controls(pool: dict[str, tuple[float, str]], used: set[str],
                    target_zipf: float, n: int = 4,
                    anchor_lexname: str | None = None,
                    same_lexname: bool = False) -> list[str]:
    """Deterministic controls, frequency-matched and single-word preferred.

    ``same_lexname=False`` gives easy controls from a different lexicographer
    class (broad-category baseline).  ``same_lexname=True`` gives hard controls
    from the anchor's own class that are not among its direct relation words
    (fine-grained baseline).
    """
    cands = []
    for name, (score, lexicon) in pool.items():
        if name in used:
            continue
        if anchor_lexname is not None:
            if same_lexname and lexicon != anchor_lexname:
                continue
            if not same_lexname and lexicon == anchor_lexname:
                continue
        cands.append(((" " in name), abs(score - target_zipf), name))
    cands.sort()
    return [name for _, _, name in cands[:n]]


def _anchor_lexname(word: str, pos: str):
    synset = anchor_synset(word, pos)
    return synset.lexname() if synset is not None else None


def build_stimuli(max_per_relation: int = 4, min_zipf: float = 4.0,
                  noun_anchors=None, adj_anchors=None, n_controls: int = 4,
                  relatives_fn=raw_relatives, pool_fn=wordnet_pool,
                  lexname_fn=_anchor_lexname) -> list[dict]:
    """Full stimulus table (frozen into a TSV by the build script)."""
    items: list[dict] = []
    arms = [
        ("noun", NOUN_ANCHORS if noun_anchors is None else noun_anchors),
        ("adjective", ADJ_ANCHORS if adj_anchors is None else adj_anchors),
    ]
    for pos, anchors in arms:
        pool = pool_fn(pos, min_zipf) if n_controls else {}
        for anchor_id, anchor in enumerate(anchors):
            rel = relatives_fn(anchor, pos)
            if rel is None:
                continue
            selected: dict[str, list[str]] = {}
            used: set[str] = {clean_name(anchor)}
            for relation in RELATIONS_BY_POS[pos]:
                raw = rel.get(relation, [])
                if relation == "self":
                    chosen = [anchor]
                else:
                    chosen = select_relation_words(raw, anchor, max_per_relation, min_zipf)
                used.update(clean_name(w) for w in raw)
                if chosen:
                    selected[relation] = chosen
            if n_controls:
                relatives = [w for words in selected.values() for w in words if w != anchor]
                if relatives:
                    target = float(np.mean([zipf_frequency(w, "en") for w in relatives]))
                else:
                    target = float(zipf_frequency(anchor, "en"))
                lexicon = lexname_fn(anchor, pos)
                easy = select_controls(pool, used, target, n_controls, lexicon,
                                       same_lexname=False)
                if easy:
                    selected[CONTROL_RELATION] = easy
                hard = select_controls(pool, used, target, n_controls, lexicon,
                                       same_lexname=True)
                if hard:
                    selected[HARD_CONTROL_RELATION] = hard
            for relation, words in selected.items():
                for word in words:
                    items.append({
                        "pos": pos,
                        "anchor_id": anchor_id,
                        "anchor": anchor,
                        "relation": relation,
                        "word": word,
                        "zipf": float(zipf_frequency(word, "en")),
                        "n_chars": len(word),
                    })
    return items


def stimuli_text(items, frame: str) -> list[str]:
    return [frame.format(x=it["word"]) for it in items]


def write_tsv(items, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=TSV_COLUMNS, delimiter="\t")
        writer.writeheader()
        for it in items:
            writer.writerow({k: it[k] for k in TSV_COLUMNS})


def read_tsv(path) -> list[dict]:
    items = []
    with Path(path).open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            items.append({
                "pos": row["pos"],
                "anchor_id": int(row["anchor_id"]),
                "anchor": row["anchor"],
                "relation": row["relation"],
                "word": row["word"],
                "zipf": float(row["zipf"]),
                "n_chars": int(row["n_chars"]),
            })
    return items


# ---------------------------------------------------------------------------
# Grouping and relation geometry
# ---------------------------------------------------------------------------

def anchor_groups(items) -> list[list[int]]:
    groups: dict[tuple, list[int]] = {}
    for i, it in enumerate(items):
        groups.setdefault((it["pos"], it["anchor_id"]), []).append(i)
    return list(groups.values())


def relation_labels(items) -> np.ndarray:
    return np.asarray([it["relation"] for it in items], dtype=object)


def relation_names(pos: str, items) -> list[str]:
    names = list(RELATIONS_BY_POS[pos])
    for control in CONTROL_RELATIONS:
        if any(it["relation"] == control for it in items):
            names.append(control)
    return names


def relation_profile(D, labels, groups, names) -> dict:
    """Mean distance for every unordered pair of relations, within anchors."""
    D = np.asarray(D, dtype=np.float64)
    index = {name: k for k, name in enumerate(names)}
    R = len(names)
    sums = np.zeros((R, R), dtype=np.float64)
    counts = np.zeros((R, R), dtype=np.int64)
    for idx in groups:
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                i, j = idx[a], idx[b]
                x, y = index[labels[i]], index[labels[j]]
                if x > y:
                    x, y = y, x
                sums[x, y] += D[i, j]
                counts[x, y] += 1
    mean = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    mean = mean + mean.T - np.diag(np.diag(mean))
    return {
        "relations": list(names),
        "mean": mean.tolist(),
        "count": counts.tolist(),
    }


def relation_pair_distance(D, labels, groups, rel_a: str, rel_b: str) -> dict:
    """Mean distance between two relations inside the same anchor."""
    D = np.asarray(D, dtype=np.float64)
    total, count = 0.0, 0
    for idx in groups:
        A = [i for i in idx if labels[i] == rel_a]
        B = [i for i in idx if labels[i] == rel_b]
        for i in A:
            for j in B:
                if i == j:
                    continue
                if rel_a == rel_b and i > j:
                    continue
                total += D[i, j]
                count += 1
    return {"mean": float(total / count) if count else float("nan"), "count": count}


def _mean_auc(D, labels, groups, target: str) -> float:
    """AUC that target items sit closer to each other than to the rest.

    Only pairs with at least one target item are scored (the rest-rest pairs
    carry no information about this relation).
    """
    D = np.asarray(D, dtype=np.float64)
    total, weight = 0.0, 0.0
    for idx in groups:
        if len(idx) < 2:
            continue
        mask = np.asarray([labels[i] == target for i in idx])
        if mask.all() or not mask.any():
            continue
        sub = D[np.ix_(idx, idx)]
        iu = np.triu_indices(len(idx), k=1)
        mi, mj = mask[iu[0]], mask[iu[1]]
        pos = mi & mj
        neg = mi ^ mj
        if not pos.any() or not neg.any():
            continue
        y = np.concatenate([np.ones(int(pos.sum())), np.zeros(int(neg.sum()))])
        d = np.concatenate([sub[iu][pos], sub[iu][neg]])
        auc = float(roc_auc_score(y, -d))
        w = float(mask.sum() * (~mask).sum())
        total += auc * w
        weight += w
    return float(total / weight) if weight else float("nan")


def relation_separation(D, labels, groups, names) -> dict:
    return {name: _mean_auc(D, labels, groups, name) for name in names}


def separation_permutation(D, labels, groups, target: str,
                           n_perm: int = 200, seed: int = 0) -> dict:
    """Permutation null for one relation's within-anchor separation AUC."""
    labels = np.asarray(labels, dtype=object)
    obs = _mean_auc(D, labels, groups, target)
    if not np.isfinite(obs):
        return {"auc": float("nan"), "null_mean": float("nan"),
                "p": float("nan"), "n_perm": n_perm}
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm, dtype=np.float64)
    perm = labels.copy()
    for k in range(n_perm):
        for idx in groups:
            perm[idx] = rng.permutation(perm[idx])
        null[k] = _mean_auc(D, perm, groups, target)
    p = (float(np.sum(null >= obs)) + 1.0) / (n_perm + 1.0)
    return {"auc": obs, "null_mean": float(np.mean(null)), "p": p, "n_perm": n_perm}


def separation_table(D, labels, groups, names, n_perm: int = 200,
                     seed: int = 0) -> dict:
    return {
        name: separation_permutation(D, labels, groups, name, n_perm, seed)
        for name in names
        if name != "self"
    }


def _mean_gradient(D, labels, groups, target: str, reference: str = "self",
                   control: str = CONTROL_RELATION) -> float:
    """AUC that target words are closer to the anchor than unrelated words.

    Works for a single word per anchor (hypernym, antonym) where the
    within-relation separation AUC is undefined.
    """
    D = np.asarray(D, dtype=np.float64)
    total, weight = 0.0, 0.0
    for idx in groups:
        ref = [i for i in idx if labels[i] == reference]
        tgt = [i for i in idx if labels[i] == target]
        ctl = [i for i in idx if labels[i] == control]
        if not ref or not tgt or not ctl:
            continue
        pos = [D[i, j] for i in ref for j in tgt if i != j]
        neg = [D[i, j] for i in ref for j in ctl if i != j]
        if not pos or not neg:
            continue
        y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        d = np.asarray(pos + neg, dtype=np.float64)
        auc = float(roc_auc_score(y, -d))
        w = float(len(pos) * len(neg))
        total += auc * w
        weight += w
    return float(total / weight) if weight else float("nan")


def relation_gradient(D, labels, groups, names,
                      control: str = CONTROL_RELATION) -> dict:
    skip = ("self",) + CONTROL_RELATIONS
    return {
        name: _mean_gradient(D, labels, groups, name, control=control)
        for name in names
        if name not in skip
    }


def gradient_permutation(D, labels, groups, target: str,
                         n_perm: int = 200, seed: int = 0,
                         control: str = CONTROL_RELATION) -> dict:
    labels = np.asarray(labels, dtype=object)
    obs = _mean_gradient(D, labels, groups, target, control=control)
    if not np.isfinite(obs):
        return {"auc": float("nan"), "null_mean": float("nan"),
                "p": float("nan"), "n_perm": n_perm}
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm, dtype=np.float64)
    perm = labels.copy()
    for k in range(n_perm):
        for idx in groups:
            perm[idx] = rng.permutation(perm[idx])
        null[k] = _mean_gradient(D, perm, groups, target, control=control)
    p = (float(np.sum(null >= obs)) + 1.0) / (n_perm + 1.0)
    return {"auc": obs, "null_mean": float(np.mean(null)), "p": p, "n_perm": n_perm}


def gradient_table(D, labels, groups, names, n_perm: int = 200,
                   seed: int = 0, control: str = CONTROL_RELATION) -> dict:
    skip = ("self",) + CONTROL_RELATIONS
    return {
        name: gradient_permutation(D, labels, groups, name, n_perm, seed, control)
        for name in names
        if name not in skip
    }


# ---------------------------------------------------------------------------
# Distances and per-layer metrics
# ---------------------------------------------------------------------------

def _float_dtype(arr):
    return arr.dtype if np.issubdtype(arr.dtype, np.floating) else np.float64


def hellinger_rdm(P) -> np.ndarray:
    P = np.asarray(P)
    X = np.sqrt(np.clip(P.astype(_float_dtype(P), copy=False), 0.0, None))
    D = np.sqrt(np.clip(1.0 - X @ X.T, 0.0, None))
    np.fill_diagonal(D, 0.0)
    return D


def pairwise_rdm(X, metric: str = "cosine") -> np.ndarray:
    X = np.asarray(X)
    X = X.astype(_float_dtype(X), copy=False)
    if metric == "cosine":
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        Xn = X / np.where(norms > 0, norms, 1.0)
        D = 1.0 - Xn @ Xn.T
    elif metric == "euclid":
        sq = (X * X).sum(axis=1)
        D = sq[:, None] + sq[None, :] - 2.0 * X @ X.T
        D = np.sqrt(np.clip(D, 0.0, None))
    else:
        raise ValueError(f"unknown metric: {metric}")
    np.fill_diagonal(D, 0.0)
    return np.clip(D, 0.0, None)


def js_distance(p, q, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    m = 0.5 * (p + q)
    kl = 0.5 * np.sum(p * (np.log(np.clip(p, eps, None)) - np.log(np.clip(m, eps, None))))
    kl += 0.5 * np.sum(q * (np.log(np.clip(q, eps, None)) - np.log(np.clip(m, eps, None))))
    return float(np.sqrt(max(kl, 0.0)))


def sym_kl(p, q, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    kl_pq = np.sum(p * (np.log(np.clip(p, eps, None)) - np.log(np.clip(q, eps, None))))
    kl_qp = np.sum(q * (np.log(np.clip(q, eps, None)) - np.log(np.clip(p, eps, None))))
    return float(max(0.5 * (kl_pq + kl_qp), 0.0))


def pair_divergence(P, labels, groups, rel_a: str, rel_b: str,
                    kind: str = "js") -> dict:
    fn = js_distance if kind == "js" else sym_kl
    total, count = 0.0, 0
    for idx in groups:
        A = [i for i in idx if labels[i] == rel_a]
        B = [i for i in idx if labels[i] == rel_b]
        for i in A:
            for j in B:
                if i == j:
                    continue
                if rel_a == rel_b and i > j:
                    continue
                total += fn(P[i], P[j])
                count += 1
    return {"mean": float(total / count) if count else float("nan"), "count": count}


def residualize_coords(X, covariates) -> np.ndarray:
    """Residual of every column of ``X`` after OLS on intercept + covariates.

    Uses the small ``n x n`` projection matrix, so a wide ``X`` is never cast to
    float64 as a whole.
    """
    X = np.asarray(X)
    dtype = _float_dtype(X)
    C = np.asarray(covariates, dtype=dtype)
    A = np.column_stack([np.ones(X.shape[0], dtype=dtype), C])
    proj = (A @ np.linalg.pinv((A.T @ A).astype(np.float64)) @ A.T).astype(dtype)
    return X - proj @ X


def increment_stats(H) -> dict:
    """Per-transition norm and direction of ``h_{l+1} - h_l`` (layers on axis 0)."""
    H = np.asarray(H, dtype=np.float64)
    delta = H[1:] - H[:-1]
    dn = np.linalg.norm(delta, axis=2)
    hn = np.linalg.norm(H[:-1], axis=2)
    cos = (delta * H[:-1]).sum(axis=2) / (dn * hn + 1e-12)
    return {
        "norm_mean": dn.mean(axis=1).tolist(),
        "relative_mean": (dn / (hn + 1e-12)).mean(axis=1).tolist(),
        "cos_mean": cos.mean(axis=1).tolist(),
    }


def state_layer_metrics(H, labels, groups, names, metric: str = "cosine") -> list[dict]:
    """Separation AUC and self-relation distances for every layer.

    ``H`` has shape ``(n_items, n_states, d)``.
    """
    H = np.asarray(H)
    out = []
    for layer in range(H.shape[1]):
        D = pairwise_rdm(H[:, layer, :], metric)
        sep = relation_separation(D, labels, groups, names)
        grad = relation_gradient(D, labels, groups, names)
        grad_hard = (
            relation_gradient(D, labels, groups, names,
                              control=HARD_CONTROL_RELATION)
            if HARD_CONTROL_RELATION in names else {}
        )
        self_dist = {
            name: relation_pair_distance(D, labels, groups, "self", name)["mean"]
            for name in names
            if name != "self"
        }
        out.append({
            "layer": layer,
            "separation": sep,
            "gradient": grad,
            "gradient_hard": grad_hard,
            "self_distance": self_dist,
        })
    return out


def profile_correlation(profile_a, profile_b) -> float:
    A = np.asarray(profile_a["mean"], dtype=np.float64)
    B = np.asarray(profile_b["mean"], dtype=np.float64)
    i, j = np.triu_indices(A.shape[0], k=1)
    return spearman(A[i, j], B[i, j])


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------

def _partial_two(d1, d2, c1, c2) -> float:
    x, y = upper_tri(np.asarray(d1, float)), upper_tri(np.asarray(d2, float))
    z1, z2 = upper_tri(np.asarray(c1, float)), upper_tri(np.asarray(c2, float))
    return spearman(residualize(x, [z1, z2]), residualize(y, [z1, z2]))


def rdm_controls(D, items, labels) -> dict:
    length = np.asarray([
        it.get("n_tokens_slot", it.get("n_chars", len(it["word"]))) for it in items
    ], dtype=np.float64)
    freq = np.asarray([it["zipf"] for it in items], dtype=np.float64)
    anchor = np.asarray([f"{it['pos']}:{it['anchor_id']}" for it in items], dtype=object)
    relation = np.asarray(labels, dtype=object)
    length_rdm = abs_diff_rdm(length)
    freq_rdm = abs_diff_rdm(freq)
    return {
        "rho_model_anchor": spearman(upper_tri(D), upper_tri(label_rdm(anchor))),
        "rho_model_relation": spearman(upper_tri(D), upper_tri(label_rdm(relation))),
        "rho_model_length": spearman(upper_tri(D), upper_tri(length_rdm)),
        "rho_model_freq": spearman(upper_tri(D), upper_tri(freq_rdm)),
        "partial_relation_given_length_freq":
            _partial_two(D, label_rdm(relation), length_rdm, freq_rdm),
    }


def covariates(items) -> np.ndarray:
    return np.column_stack([
        np.asarray([it["zipf"] for it in items], dtype=np.float64),
        np.asarray([it["n_chars"] for it in items], dtype=np.float64),
        np.asarray([
            it.get("n_tokens_slot", it.get("n_chars", len(it["word"]))) for it in items
        ], dtype=np.float64),
    ])


def analyze_distribution(P, items, n_perm: int = 200) -> dict:
    """Distribution channel: raw and covariate-residualized geometry."""
    groups = anchor_groups(items)
    labels = relation_labels(items)
    pos = items[0]["pos"] if items else "noun"
    names = relation_names(pos, items)
    D = hellinger_rdm(P)
    raw = {
        "profile": relation_profile(D, labels, groups, names),
        "separation": separation_table(D, labels, groups, names, n_perm),
        "gradient": gradient_table(D, labels, groups, names, n_perm),
        "gradient_hard": (
            gradient_table(D, labels, groups, names, n_perm,
                           control=HARD_CONTROL_RELATION)
            if HARD_CONTROL_RELATION in names else {}
        ),
        "controls": rdm_controls(D, items, labels),
        "divergences": {
            f"{a}__{b}": {
                kind: pair_divergence(P, labels, groups, a, b, kind)
                for kind in ("js", "kl")
            }
            for a, b in (("self", "sibling"), ("self", CONTROL_RELATION),
                         ("sibling", CONTROL_RELATION))
            if any(labels == a) and any(labels == b)
        },
    }
    X = residualize_coords(np.sqrt(np.clip(np.asarray(P, float), 0.0, None)),
                           covariates(items))
    D_res = pairwise_rdm(X, "cosine")
    resid = {
        "profile": relation_profile(D_res, labels, groups, names),
        "separation": relation_separation(D_res, labels, groups, names),
        "gradient": relation_gradient(D_res, labels, groups, names),
        "gradient_hard": (
            relation_gradient(D_res, labels, groups, names,
                              control=HARD_CONTROL_RELATION)
            if HARD_CONTROL_RELATION in names else {}
        ),
        "controls": rdm_controls(D_res, items, labels),
    }
    return {"n_items": len(items), "names": names, "raw": raw, "residualized": resid}
