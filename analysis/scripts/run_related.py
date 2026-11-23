#!/usr/bin/env python
"""Related concepts test: fine-grained discrimination (concept vs its neighbour).

Three pair levels: same concept < related concepts (bee/wasp) < unrelated.
Question: do pooled directions still separate, and do relations (Gram spectrum)
gain an advantage on the fine same-vs-related contrast?

Run:  uv run python scripts/run_related.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tc.corpus import RELATED_CONCEPTS, RELATED_PAIRS, flat_related
from tc.extract import load, token_count
from tc.joint import cosine_sim, encode_alone, encode_pair
from tc.metrics import hellinger
from tc.phi import phi_spec

STOP = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "at", "from", "with",
    "by", "for", "that", "this", "these", "those", "when", "as", "is", "are",
    "was", "were", "it", "its", "into", "across", "via", "much", "many", "their",
    "they", "you", "your", "during", "below", "through", "down", "out", "up",
    "them", "object", "machine", "device",
}


def content_jaccard(a: str, b: str) -> float:
    A = {w for w in a.lower().replace(".", " ").replace(",", " ").split() if w not in STOP}
    B = {w for w in b.lower().replace(".", " ").replace(",", " ").split() if w not in STOP}
    return len(A & B) / len(A | B) if (A | B) else 0.0


def make_pairs(cids, lens, name_to_cid, rng):
    N = len(cids)
    same = [(i, j) for i in range(N) for j in range(i + 1, N) if cids[i] == cids[j]]
    rel_cid = [frozenset((name_to_cid[a], name_to_cid[b])) for a, b in RELATED_PAIRS]
    related = [
        (i, j)
        for i in range(N)
        for j in range(i + 1, N)
        if frozenset((cids[i], cids[j])) in rel_cid
    ]
    # unrelated: different, non-related pairs, length-matched
    unrel = []
    for tol in (2, 3, 4, 5):
        unrel = [
            (i, j)
            for i in range(N)
            for j in range(i + 1, N)
            if cids[i] != cids[j]
            and frozenset((cids[i], cids[j])) not in rel_cid
            and abs(lens[i] - lens[j]) <= tol
        ]
        if len(unrel) >= 36:
            break
    unrel = [unrel[k] for k in rng.permutation(len(unrel))[:36]]
    return same, related, unrel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/related_summary.json")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    model, tok = load(args.model)
    print(f"[model] {args.model}")

    texts, cids, names, name_to_cid = flat_related()
    lens = [token_count(t, tok) for t in texts]
    alone = {}
    for t in texts:
        V, pooled = encode_alone(t, model, tok)
        alone[t] = (phi_spec(V), pooled)

    same, related, unrel = make_pairs(cids, lens, name_to_cid, rng)
    groups = {"same": same, "related": related, "unrelated": unrel}

    sig = {"s_spec": {}, "s_alonecos": {}, "s_pool": {}}
    jac = {}
    for level, pairs in groups.items():
        for k in sig:
            sig[k][level] = []
        jac[level] = []
        for (i, j) in pairs:
            a, b = texts[i], texts[j]
            phi_a, pool_a = alone[a]
            phi_b, pool_b = alone[b]
            sig["s_spec"][level].append(-hellinger(phi_a, phi_b))
            sig["s_alonecos"][level].append(cosine_sim(pool_a, pool_b))
            pa, pb, _, _, _ = encode_pair(a, b, model, tok)
            sig["s_pool"][level].append(cosine_sim(pa, pb))
            jac[level].append(content_jaccard(a, b))
        for k in sig:
            sig[k][level] = np.asarray(sig[k][level])
        jac[level] = np.asarray(jac[level])

    print(f"\n[pairs] same={len(same)} related={len(related)} unrelated={len(unrel)}")
    for level in groups:
        print(f"[overlap] {level:10s} content-jaccard mean={jac[level].mean():.3f}")

    print("\n=== mean similarity by level (higher = more similar) ===")
    for k in sig:
        m = {lvl: float(sig[k][lvl].mean()) for lvl in groups}
        print(f"  {k:12s} same={m['same']:.3f}  related={m['related']:.3f}  unrelated={m['unrelated']:.3f}")

    def auc(a_vals, b_vals):
        y = np.r_[np.ones(len(a_vals)), np.zeros(len(b_vals))]
        s = np.r_[a_vals, b_vals]
        return float(roc_auc_score(y, s))

    print("\n=== AUC contrasts (1.0 = perfectly orders the two groups) ===")
    print("  same vs related   (hard: concept vs its neighbour)")
    for k in sig:
        print(f"    {k:12s} AUC = {auc(sig[k]['same'], sig[k]['related']):.3f}")
    print("  related vs unrelated")
    for k in sig:
        print(f"    {k:12s} AUC = {auc(sig[k]['related'], sig[k]['unrelated']):.3f}")

    # ordinal ordering same < related < unrelated (similarity should decrease)
    print("\n=== ordering (Spearman of similarity with level 0..2; negative = correct) ===")
    for k in sig:
        vals, lv = [], []
        for li, lvl in enumerate(["same", "related", "unrelated"]):
            vals.extend(sig[k][lvl])
            lv.extend([li] * len(sig[k][lvl]))
        r = stats.spearmanr(vals, lv).correlation
        print(f"  {k:12s} rho = {r:+.3f}")

    report = {
        "model": args.model,
        "n_pairs": {k: len(v) for k, v in groups.items()},
        "mean_similarity": {
            k: {lvl: float(sig[k][lvl].mean()) for lvl in groups} for k in sig
        },
        "auc_same_vs_related": {k: auc(sig[k]["same"], sig[k]["related"]) for k in sig},
        "auc_related_vs_unrelated": {k: auc(sig[k]["related"], sig[k]["unrelated"]) for k in sig},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
