"""Stage 1: typicality gradient and category prototype (experiment 009).

Pure (model-free) helpers: norm loading/selection, probe text builders, rank
correlations with controls, permutation nulls, the behavioural prototype and
its held-out test, and the pre-registered verdicts.  Model-facing probes live in
``scripts/run_typicality.py``.

The question is whether the model reproduces the human typicality gradient and
whether a central prototype generalises to held-out members, on at least two
readouts (judgment, ranking) and two behavioural-signature frames.
"""

from __future__ import annotations

import csv
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

# --------------------------------------------------------------------------- #
# Pre-registered data selection and thresholds
# --------------------------------------------------------------------------- #

SELECTED_CATEGORIES = (
    "mammal",
    "bird",
    "fruit",
    "vegetable",
    "furniture",
    "vehicle",
    "tool",
    "weapon",
    "clothing",
    "musical instrument",
)
PER_CATEGORY = 20
SOURCE = "THINGSplus:typicality53_mean-ratings.tsv"

TAU_RHO = 0.50
TAU_PARTIAL = 0.30
TAU_MARGIN = 0.10
TAU_HELDOUT = 0.40
TAU_HELDOUT_PARTIAL = 0.25
TAU_P = 0.01
MIN_POSITIVE_FRAC = 0.70

JUDGMENT_TEMPLATES = (
    'Is "{item}" a typical example of {category}? Answer Yes or No.\nAnswer:',
    'Would you call "{item}" a typical {category}? Answer Yes or No.\nAnswer:',
)
RANKING_FRAMES = (
    ("A typical example of {category} is", " {item}."),
    ("The most typical kind of {category} is", " {item}."),
)
SIGNATURE_TEMPLATES = (
    "People usually describe {item} as",
    "When you see {item}, you notice",
)


def judgment_prompts(item: str, category: str) -> list[str]:
    return [t.format(item=item, category=category) for t in JUDGMENT_TEMPLATES]


def ranking_probes(item: str, category: str) -> list[tuple[str, str]]:
    return [(p.format(category=category), c.format(item=item)) for p, c in RANKING_FRAMES]


def signature_prompts(item: str) -> list[str]:
    return [t.format(item=item) for t in SIGNATURE_TEMPLATES]


def judgment_score(p_yes: float, p_no: float) -> float:
    total = p_yes + p_no
    return float(p_yes / total) if total > 0 else 0.0


# --------------------------------------------------------------------------- #
# Norm loading, writing, deterministic subset selection
# --------------------------------------------------------------------------- #

def _clean_item(value) -> str:
    return str(value).replace("_", " ").strip()


def load_norms(path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            rows.append({
                "category": row["category"].strip(),
                "item": _clean_item(row["item"]),
                "typicality": float(row["typicality"]),
                "source": row.get("source", SOURCE),
            })
    return rows


def write_norms(rows, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["category", "item", "typicality", "source"],
                                delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "category": row["category"],
                "item": row["item"],
                "typicality": f"{float(row['typicality']):.6f}",
                "source": row.get("source", SOURCE),
            })


def _even_indices(n: int, k: int) -> list[int]:
    if n == 0 or k <= 0:
        return []
    k = min(k, n)
    if k == 1:
        return [n // 2]
    idx = sorted({int(round(i * (n - 1) / (k - 1))) for i in range(k)})
    if len(idx) < k:
        for j in range(n):
            if j not in idx:
                idx.append(j)
            if len(idx) == k:
                break
        idx = sorted(idx)
    return idx


def _stratified(values, k: int) -> list:
    values = sorted(values, key=lambda x: (x[0], x[1]))
    return [values[i] for i in _even_indices(len(values), k)]


def select_subset(rows, categories=SELECTED_CATEGORIES, per_category: int = PER_CATEGORY):
    """Deterministic stratified subset: no cross-category duplicates, no self-named items."""
    cats = list(categories)
    by_category: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        category = row["category"]
        if category in cats:
            by_category[category][_clean_item(row["item"])] = float(row["typicality"])

    word_categories: dict[str, set[str]] = defaultdict(set)
    for category in cats:
        for item in by_category.get(category, {}):
            word_categories[item.lower()].add(category)

    selected = []
    for category in cats:
        values = [
            (typicality, item)
            for item, typicality in by_category.get(category, {}).items()
            if len(word_categories[item.lower()]) == 1
            and item.lower() != category.lower()
        ]
        for typicality, item in _stratified(values, per_category):
            selected.append({
                "category": category,
                "item": item,
                "typicality": typicality,
                "source": SOURCE,
            })
    return selected


def limit_per_category(rows, limit: int) -> list[dict]:
    if limit <= 0:
        return list(rows)
    by = defaultdict(list)
    for row in rows:
        by[row["category"]].append(row)
    out = []
    for category in selected_categories(rows):
        group = sorted(by[category], key=lambda r: (float(r["typicality"]), r["item"]))
        out.extend(group[i] for i in _even_indices(len(group), limit))
    return out


def selected_categories(rows) -> list[str]:
    seen = []
    for row in rows:
        if row["category"] not in seen:
            seen.append(row["category"])
    return seen


# --------------------------------------------------------------------------- #
# Rank statistics with category fixed effects
# --------------------------------------------------------------------------- #

def _groups(cats):
    cats = np.asarray(cats)
    return {c: np.where(cats == c)[0] for c in sorted(set(cats.tolist()))}


def _finite_rho(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    rho = float(stats.spearmanr(a, b).correlation)
    return rho if np.isfinite(rho) else float("nan")


def per_category_rho(score, target, cats) -> dict[str, float]:
    score = np.asarray(score, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    return {c: _finite_rho(score[idx], target[idx]) for c, idx in _groups(cats).items()}


def mean_within_rho(score, target, cats) -> float:
    values = [v for v in per_category_rho(score, target, cats).values() if np.isfinite(v)]
    return float(np.mean(values)) if values else float("nan")


def permutation_pvalue(score, target, cats, n_perm: int = 1000, seed: int = 0):
    target = np.asarray(target, dtype=np.float64)
    groups = _groups(cats)
    observed = mean_within_rho(score, target, cats)
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm, dtype=np.float64)
    for k in range(n_perm):
        shuffled = target.copy()
        for idx in groups.values():
            shuffled[idx] = target[idx][rng.permutation(idx.size)]
        null[k] = mean_within_rho(score, shuffled, cats)
    p = (float(np.sum(null >= observed)) + 1.0) / (n_perm + 1.0)
    return observed, p, null


def _residualize(y, columns):
    y = np.asarray(y, dtype=np.float64)
    cols = [np.ones_like(y)] + [np.asarray(c, dtype=np.float64) for c in columns]
    design = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return y - design @ beta


def mean_partial_rho(score, target, controls, cats) -> float:
    score = np.asarray(score, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    controls = [np.asarray(c, dtype=np.float64) for c in controls]
    values = []
    for _, idx in _groups(cats).items():
        rs = _residualize(score[idx], [c[idx] for c in controls])
        rt = _residualize(target[idx], [c[idx] for c in controls])
        rho = _finite_rho(rs, rt)
        if np.isfinite(rho):
            values.append(rho)
    return float(np.mean(values)) if values else float("nan")


def multiple_baseline_rho(target, controls, cats) -> float:
    """Leave-one-out multiple rank correlation of the controls with the target.

    Fitting and evaluating on the same items inflates the baseline when categories
    are small, so each prediction is made without the item it is scored on.
    """
    target = np.asarray(target, dtype=np.float64)
    controls = [np.asarray(c, dtype=np.float64) for c in controls]
    values = []
    for _, idx in _groups(cats).items():
        y = target[idx]
        design = np.column_stack([np.ones_like(y)] + [c[idx] for c in controls])
        n = y.size
        if n < 4:
            continue
        predicted = np.empty_like(y)
        for i in range(n):
            train = np.arange(n) != i
            beta, *_ = np.linalg.lstsq(design[train], y[train], rcond=None)
            predicted[i] = design[i] @ beta
        rho = _finite_rho(predicted, y)
        if np.isfinite(rho):
            values.append(rho)
    return float(np.mean(values)) if values else float("nan")


def frequency_zipf(item: str) -> float:
    from wordfreq import zipf_frequency

    tokens = [t for t in re.split(r"[\s_-]+", item.lower()) if t]
    if not tokens:
        return 0.0
    return float(np.mean([zipf_frequency(t, "en") for t in tokens]))


def count_positive(per_category: dict) -> int:
    return sum(1 for v in per_category.values() if np.isfinite(v) and v > 0)


def _nanmax(values):
    finite = [v for v in values if v is not None and np.isfinite(v)]
    return max(finite) if finite else float("nan")


# --------------------------------------------------------------------------- #
# Behavioural prototype in signature space
# --------------------------------------------------------------------------- #

def signature_matrix(signatures, k: int = 50):
    """Restrict full next-token distributions to a shared top-k union support."""
    signatures = [np.asarray(s, dtype=np.float64) for s in signatures]
    support: set[int] = set()
    for prob in signatures:
        support |= set(np.argsort(prob)[-k:].tolist())
    idx = np.array(sorted(support), dtype=int)
    matrix = []
    for prob in signatures:
        q = prob[idx]
        total = q.sum()
        matrix.append(q / total if total > 0 else np.full(idx.size, 1.0 / idx.size))
    return np.stack(matrix), idx


def _folds(n: int, k: int, seed: int) -> list[list[int]]:
    order = np.random.default_rng(seed).permutation(n)
    return [order[i::k].tolist() for i in range(k)]


def _centroid_log(matrix) -> np.ndarray:
    centroid = np.asarray(matrix, dtype=np.float64).mean(axis=0)
    return np.log(np.clip(centroid, 1e-12, None))


def prototype_cv(signatures, cats, items, k: int = 50, folds: int = 4, seed: int = 0):
    """Held-out likelihood under a category centroid of behavioural signatures.

    The prototype is the arithmetic centroid (the maximum-likelihood point for
    the pooled signatures).  An item's score is its expected log-likelihood under
    the centroid: higher means closer to the prototype.  Returns one score per
    input item (aligned to the input order) and, per category, the three real
    items with the highest likelihood under the full centroid.
    """
    signatures = [np.asarray(s, dtype=np.float64) for s in signatures]
    cats = np.asarray(cats)
    items = np.asarray(items)
    scores = np.full(len(signatures), np.nan, dtype=np.float64)
    nearest: dict[str, list[str]] = {}
    for category, idx in _groups(cats).items():
        idx = np.asarray(idx, dtype=int)
        if idx.size < 3:
            continue
        matrix, _ = signature_matrix([signatures[i] for i in idx], k=k)
        full_log = _centroid_log(matrix)
        order = np.argsort(matrix @ full_log)[::-1]
        nearest[category] = [str(items[i]) for i in idx[order[:3]]]
        fold_size = min(folds, idx.size - 2)
        for test in _folds(idx.size, fold_size, seed):
            test = np.asarray(test, dtype=int)
            train = np.setdiff1d(np.arange(idx.size), test)
            if train.size == 0:
                continue
            train_log = _centroid_log(matrix[train])
            scores[idx[test]] = matrix[test] @ train_log
    return scores, nearest


# --------------------------------------------------------------------------- #
# Channel and prototype summaries / verdicts
# --------------------------------------------------------------------------- #

def _channel_passed(stats_dict):
    per = stats_dict["per_category"]
    n_cat = sum(1 for v in per.values() if np.isfinite(v))
    n_pos = count_positive(per)
    passed_a = bool(
        np.isfinite(stats_dict["rho"]) and stats_dict["rho"] >= TAU_RHO
        and n_cat > 0 and n_pos >= math.ceil(MIN_POSITIVE_FRAC * n_cat)
        and np.isfinite(stats_dict["p"]) and stats_dict["p"] < TAU_P
    )
    passed_b = bool(
        np.isfinite(stats_dict["partial"]) and stats_dict["partial"] >= TAU_PARTIAL
        and np.isfinite(stats_dict["margin"]) and stats_dict["margin"] >= TAU_MARGIN
    )
    if passed_a and passed_b:
        verdict = "P1a+P1b"
    elif passed_a:
        verdict = "P1a only"
    elif passed_b:
        verdict = "P1b only"
    else:
        verdict = "absent"
    return passed_a, passed_b, verdict


def channel_summary(name: str, score, target, cats, frequency, length,
                    static_cos=None, n_perm: int = 1000, seed: int = 0) -> dict:
    per = per_category_rho(score, target, cats)
    rho, p, _ = permutation_pvalue(score, target, cats, n_perm=n_perm, seed=seed)
    partial = mean_partial_rho(score, target, [frequency, length], cats)
    rho_l0 = multiple_baseline_rho(target, [frequency, length], cats)
    rho_l1 = mean_within_rho(static_cos, target, cats) if static_cos is not None else float("nan")
    baseline = _nanmax([rho_l0, rho_l1])
    margin = float(rho - baseline) if np.isfinite(rho) and np.isfinite(baseline) else float("nan")
    out = {
        "name": name, "rho": rho, "p": p, "partial": partial,
        "rho_l0": rho_l0, "rho_l1": rho_l1, "baseline": baseline, "margin": margin,
        "per_category": per, "n_positive": count_positive(per),
    }
    passed_a, passed_b, verdict = _channel_passed(out)
    out.update({"P1a": passed_a, "P1b": passed_b, "verdict": verdict})
    return out


def prototype_summary(name: str, scores, target, cats, frequency, length,
                      n_perm: int = 1000, seed: int = 0, nearest=None) -> dict:
    score = np.asarray(scores, dtype=np.float64)
    per = per_category_rho(score, target, cats)
    rho, p, _ = permutation_pvalue(score, target, cats, n_perm=n_perm, seed=seed)
    partial = mean_partial_rho(score, target, [frequency, length], cats)
    baseline = multiple_baseline_rho(target, [frequency, length], cats)
    margin = float(rho - baseline) if np.isfinite(rho) and np.isfinite(baseline) else float("nan")
    passed = bool(
        np.isfinite(rho) and rho >= TAU_HELDOUT
        and np.isfinite(partial) and partial >= TAU_HELDOUT_PARTIAL
        and np.isfinite(margin) and margin >= TAU_MARGIN
    )
    return {
        "name": name, "rho": rho, "p": p, "partial": partial,
        "baseline": baseline, "margin": margin, "per_category": per,
        "nearest": nearest or {}, "P1c": passed,
        "verdict": "P1c" if passed else "absent",
    }


def overall_verdict(channels: dict, prototypes: dict) -> dict:
    robust = [name for name, c in channels.items() if c["P1a"] and c["P1b"]]
    proto_ok = [name for name, p in prototypes.items() if p["P1c"]]
    if len(robust) >= 2:
        typicality = "readout-robust"
    elif len(robust) == 1:
        typicality = "readout-local"
    else:
        typicality = "absent"
    if len(proto_ok) >= 2:
        prototype = "readout-robust"
    elif len(proto_ok) == 1:
        prototype = "readout-local"
    else:
        prototype = "absent"
    return {
        "typicality": typicality,
        "prototype": prototype,
        "robust_channels": robust,
        "prototype_channels": proto_ok,
    }
