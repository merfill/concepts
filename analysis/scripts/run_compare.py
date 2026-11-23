#!/usr/bin/env python
"""Phase 1 + Phase 2: exported fingerprint vs native (joint) comparison.

Phase 1: do separate-encoding fingerprints (d_export) agree with the model's
in-context comparison signals (cross-attention, joint pooling)?

Phase 2: can a *learned* metric on separate encodings reproduce the joint signal
(the "inductively derived fingerprint")?

Run:  uv run python scripts/run_compare.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tc.corpus import flat_paraphrases
from tc.extract import load, token_count
from tc.joint import cosine_sim, encode_alone, encode_pair
from tc.metrics import hellinger
from tc.phi import phi_spec


def _toks(t: str):
    return set(t.lower().replace(".", "").replace(",", "").split())


def jaccard(a: str, b: str) -> float:
    A, B = _toks(a), _toks(b)
    return len(A & B) / len(A | B) if (A | B) else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/compare_summary.json")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    model, tok = load(args.model)
    print(f"[model] {args.model}")

    texts, gids, _ = flat_paraphrases()
    N = len(texts)

    # separate encodings
    alone = {}
    for t in texts:
        V, pooled = encode_alone(t, model, tok)
        alone[t] = (phi_spec(V), pooled)
    d = next(iter(alone.values()))[1].size

    # pairs
    same = [(i, j) for i in range(N) for j in range(i + 1, N) if gids[i] == gids[j]]
    cross = [(i, j) for i in range(N) for j in range(i + 1, N) if gids[i] != gids[j]]
    n_diff = min(len(same), len(cross))
    diff = [cross[k] for k in rng.permutation(len(cross))[:n_diff]]
    pairs = same + diff
    labels = np.array([1] * len(same) + [0] * len(diff))

    sig = {k: [] for k in ["s_spec", "s_alonecos", "s_pool", "s_cross"]}
    jac, ldiff = [], []
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
        jac.append(jaccard(a, b))
        ldiff.append(abs(token_count(a, tok) - token_count(b, tok)))

    for k in sig:
        sig[k] = np.asarray(sig[k])
    jac = np.asarray(jac)
    ldiff = np.asarray(ldiff)

    print(f"\n[pairs] same={len(same)} diff={len(diff)} total={len(pairs)}")
    print("\n=== Phase 1: AUC for same/different (1.0 best, 0.5 chance) ===")
    for k in sig:
        print(f"  {k:12s} AUC = {roc_auc_score(labels, sig[k]):.3f}")

    print("\n=== correlations between signals (Spearman) ===")
    keys = list(sig)
    for a_i in range(len(keys)):
        for b_i in range(a_i + 1, len(keys)):
            r = stats.spearmanr(sig[keys[a_i]], sig[keys[b_i]]).correlation
            print(f"  {keys[a_i]:12s} ~ {keys[b_i]:12s} r={r:.3f}")

    print("\n=== confound check: Spearman(signal, shared-words jaccard) ===")
    for k in sig:
        print(f"  {k:12s} ~ jaccard r={stats.spearmanr(sig[k], jac).correlation:.3f}")

    from tc import controls

    print("\n=== AUC controlling for shared words (jaccard) and length ===")
    controlled_auc = {}
    med = float(np.median(jac))
    low = jac <= med
    for k in sig:
        resid = controls.residualize(sig[k], [jac, ldiff])
        auc_res = roc_auc_score(labels, resid)
        auc_low = (
            roc_auc_score(labels[low], sig[k][low])
            if labels[low].min() != labels[low].max()
            else float("nan")
        )
        controlled_auc[k] = float(auc_res)
        print(f"  {k:12s} AUC|resid={auc_res:.3f}   AUC|low-jaccard={auc_low:.3f}")

    # Phase 2: learn a diagonal metric on separate pooled reps to reproduce s_pool
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
    auc_learned = roc_auc_score(labels, pred)
    r_spec = stats.spearmanr(sig["s_spec"], y).correlation
    r_cos = stats.spearmanr(sig["s_alonecos"], y).correlation
    # unweighted L2 baseline (no learning)
    l2 = X.sum(axis=1)
    r_l2 = stats.spearmanr(l2, y).correlation

    print("\n=== Phase 2: reproduce joint signal (s_pool) from separate encoding ===")
    print(f"  target = s_pool (joint contextualized cosine)")
    print(f"  learned diagonal metric (5-fold CV) r = {r_learned:.3f}  AUC(same)={auc_learned:.3f}")
    print(f"  baseline  Phi_spec        r = {r_spec:.3f}")
    print(f"  baseline  alone-cosine    r = {r_cos:.3f}")
    print(f"  baseline  unweighted L2   r = {r_l2:.3f}")

    report = {
        "model": args.model,
        "n_pairs": {"same": len(same), "diff": len(diff)},
        "phase1_auc": {k: float(roc_auc_score(labels, sig[k])) for k in sig},
        "signal_corr": {
            f"{keys[a]}_{keys[b]}": float(stats.spearmanr(sig[keys[a]], sig[keys[b]]).correlation)
            for a in range(len(keys))
            for b in range(a + 1, len(keys))
        },
        "jaccard_corr": {k: float(stats.spearmanr(sig[k], jac).correlation) for k in sig},
        "auc_controlling_confounds": controlled_auc,
        "phase2": {
            "target": "s_pool",
            "learned_cv_r": float(r_learned),
            "learned_cv_auc": float(auc_learned),
            "baseline_phi_spec_r": float(r_spec),
            "baseline_alone_cosine_r": float(r_cos),
            "baseline_unweighted_l2_r": float(r_l2),
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
