#!/usr/bin/env python
"""Layer scan for the pooled-direction signal on related concepts.

For every layer, mean-pool separately-encoded texts and measure how well the
cosine similarity orders: same < related < unrelated, and how well it separates
the hard contrast same-vs-related.

Run:  uv run python scripts/run_layer_scan.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tc.corpus import RELATED_PAIRS, flat_related
from tc.extract import load, token_count
from tc.joint import cosine_sim, encode_alone


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


def auc(a_vals, b_vals):
    y = np.r_[np.ones(len(a_vals)), np.zeros(len(b_vals))]
    return float(roc_auc_score(y, np.r_[a_vals, b_vals]))


def main():
    model_name = "distilbert-base-uncased"
    model, tok = load(model_name)
    texts, cids, names, name_to_cid = flat_related()
    lens = [token_count(t, tok) for t in texts]
    rng = np.random.default_rng(0)
    same, related, unrel = make_pairs(cids, lens, name_to_cid, rng)

    n_layers = model.config.num_hidden_layers
    print(f"model={model_name}  hidden_layers={n_layers}  (layer 0 = embeddings)")
    print(f"{'layer':>5} {'same':>7} {'related':>8} {'unrel':>7} {'AUC(s/r)':>9} {'AUC(r/u)':>9} {'rho':>7}")

    best = (-1, -1.0)
    rows = []
    for layer in range(0, n_layers + 1):
        reps = {}
        for t in texts:
            _, pooled = encode_alone(t, model, tok, layer=layer)
            reps[t] = pooled

        def sim(i, j):
            return cosine_sim(reps[texts[i]], reps[texts[j]])

        s_same = np.array([sim(i, j) for (i, j) in same])
        s_rel = np.array([sim(i, j) for (i, j) in related])
        s_un = np.array([sim(i, j) for (i, j) in unrel])
        a_sr = auc(s_same, s_rel)
        a_ru = auc(s_rel, s_un)
        vals = np.r_[s_same, s_rel, s_un]
        lv = np.r_[np.zeros(len(s_same)), np.ones(len(s_rel)), np.full(len(s_un), 2)]
        rho = stats.spearmanr(vals, lv).correlation
        if a_sr > best[1]:
            best = (layer, a_sr)
        rows.append(
            {"layer": layer, "same": float(s_same.mean()), "related": float(s_rel.mean()),
             "unrelated": float(s_un.mean()), "auc_same_related": a_sr,
             "auc_related_unrelated": a_ru, "rho": float(rho)}
        )
        print(f"{layer:>5} {s_same.mean():>7.3f} {s_rel.mean():>8.3f} {s_un.mean():>7.3f} "
              f"{a_sr:>9.3f} {a_ru:>9.3f} {rho:>7.3f}")

    print(f"\nbest layer for same-vs-related: {best[0]} (AUC={best[1]:.3f})")
    out = Path("results/layer_scan_summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": model_name, "rows": rows, "best_layer": best[0]},
                              indent=2, ensure_ascii=False))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
