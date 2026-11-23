#!/usr/bin/env python
"""Strengthened test: hard paraphrases (disjoint content words), length-matched.

Direct test of (a) artifact-of-shared-words vs (b) real semantic signal.
If pooled-direction similarity still separates same/different concepts when
content-word overlap is ~0 and lengths are matched, the signal is not lexical.

Run:  uv run python scripts/run_hard.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tc import controls
from tc.corpus import flat_hard
from tc.extract import load, token_count
from tc.joint import cosine_sim, encode_alone, encode_pair
from tc.metrics import hellinger
from tc.phi import phi_spec

STOP = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "at", "from", "with",
    "by", "for", "that", "this", "these", "those", "when", "as", "is", "are",
    "was", "were", "it", "its", "into", "across", "via", "much", "many", "their",
    "they", "you", "your", "during", "below", "through", "down", "out", "up",
    "their", "them", "object", "machine", "device",
}


def content_tokens(t: str):
    raw = t.lower().replace(".", " ").replace(",", " ").split()
    return {w for w in raw if w not in STOP}


def content_jaccard(a: str, b: str) -> float:
    A, B = content_tokens(a), content_tokens(b)
    return len(A & B) / len(A | B) if (A | B) else 0.0


def build_pairs(texts, gids, lens, rng):
    N = len(texts)
    same = [(i, j) for i in range(N) for j in range(i + 1, N) if gids[i] == gids[j]]
    n_same = len(same)
    cand = None
    for tol in (2, 3, 4, 5, 6):
        cand = [
            (i, j)
            for i in range(N)
            for j in range(i + 1, N)
            if gids[i] != gids[j] and abs(lens[i] - lens[j]) <= tol
        ]
        if len(cand) >= n_same:
            break
    idx = rng.permutation(len(cand))[:n_same]
    diff = [cand[k] for k in idx]
    return same, diff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/hard_summary.json")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    model, tok = load(args.model)
    print(f"[model] {args.model}")

    texts, gids, _ = flat_hard()
    N = len(texts)
    lens = [token_count(t, tok) for t in texts]

    alone = {}
    for t in texts:
        V, pooled = encode_alone(t, model, tok)
        alone[t] = (phi_spec(V), pooled)

    same, diff = build_pairs(texts, gids, lens, rng)
    pairs = same + diff
    labels = np.array([1] * len(same) + [0] * len(diff))

    sig = {k: [] for k in ["s_spec", "s_alonecos", "s_pool", "s_cross"]}
    jac, ldiff, jac_raw = [], [], []
    pair_texts = []
    for (i, j) in pairs:
        a, b = texts[i], texts[j]
        pair_texts.append((a, b))
        phi_a, pool_a = alone[a]
        phi_b, pool_b = alone[b]
        sig["s_spec"].append(-hellinger(phi_a, phi_b))
        sig["s_alonecos"].append(cosine_sim(pool_a, pool_b))
        pa, pb, cscore, _, _ = encode_pair(a, b, model, tok)
        sig["s_pool"].append(cosine_sim(pa, pb))
        sig["s_cross"].append(cscore)
        jac.append(content_jaccard(a, b))
        jac_raw.append(len(set(a.lower().split()) & set(b.lower().split())) /
                       max(1, len(set(a.lower().split()) | set(b.lower().split()))))
        ldiff.append(abs(token_count(a, tok) - token_count(b, tok)))

    for k in sig:
        sig[k] = np.asarray(sig[k])
    jac = np.asarray(jac)
    ldiff = np.asarray(ldiff)

    ns = len(same)
    print(f"\n[pairs] same={len(same)} diff={len(diff)} total={len(pairs)}")
    print(f"[overlap] content-jaccard same: mean={jac[:ns].mean():.3f} max={jac[:ns].max():.3f}")
    print(f"[overlap] content-jaccard diff: mean={jac[ns:].mean():.3f} max={jac[ns:].max():.3f}")
    print(f"[length] token-length diff same: mean={ldiff[:ns].mean():.1f} ; diff: mean={ldiff[ns:].mean():.1f}")

    print("\n=== AUC for same/different (raw) ===")
    for k in sig:
        print(f"  {k:12s} AUC = {roc_auc_score(labels, sig[k]):.3f}")
    print(f"  {'lexical':12s} AUC = {roc_auc_score(labels, jac):.3f}  (content-jaccard baseline)")

    print("\n=== AUC controlling content-jaccard and length ===")
    med = float(np.median(jac))
    low = jac <= med
    controlled = {}
    for k in sig:
        resid = controls.residualize(sig[k], [jac, ldiff])
        auc_res = roc_auc_score(labels, resid)
        auc_low = (
            roc_auc_score(labels[low], sig[k][low])
            if labels[low].min() != labels[low].max()
            else float("nan")
        )
        controlled[k] = float(auc_res)
        print(f"  {k:12s} AUC|resid={auc_res:.3f}   AUC|low-overlap={auc_low:.3f}")

    # Phase 2: learned metric on separate pooled reps
    reps = {t: alone[t][1] for t in texts}
    X = np.array([-(reps[a] - reps[b]) ** 2 for (a, b) in pair_texts])
    y = sig["s_pool"]
    kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    pred = np.zeros_like(y)
    for tr, te in kf.split(X):
        pipe = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        pipe.fit(X[tr], y[tr])
        pred[te] = pipe.predict(X[te])
    r_learned = stats.spearmanr(pred, y).correlation
    print("\n=== Phase 2: reproduce joint signal from separate encoding ===")
    print(f"  learned metric CV r={r_learned:.3f}  AUC(same)={roc_auc_score(labels, pred):.3f}")
    print(f"  baseline alone-cosine r={stats.spearmanr(sig['s_alonecos'], y).correlation:.3f}")
    print(f"  baseline Phi_spec     r={stats.spearmanr(sig['s_spec'], y).correlation:.3f}")

    report = {
        "model": args.model,
        "n_pairs": {"same": len(same), "diff": len(diff)},
        "content_jaccard_same_mean": float(jac[:ns].mean()),
        "content_jaccard_diff_mean": float(jac[ns:].mean()),
        "phase1_auc": {k: float(roc_auc_score(labels, sig[k])) for k in sig},
        "lexical_auc": float(roc_auc_score(labels, jac)),
        "auc_controlling_confounds": controlled,
        "phase2": {
            "learned_cv_r": float(r_learned),
            "learned_cv_auc": float(roc_auc_score(labels, pred)),
            "baseline_alone_cosine_r": float(stats.spearmanr(sig["s_alonecos"], y).correlation),
            "baseline_phi_spec_r": float(stats.spearmanr(sig["s_spec"], y).correlation),
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
