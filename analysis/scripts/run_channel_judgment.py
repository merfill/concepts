#!/usr/bin/env python
"""Experiment 001 (judgment arm): does the model *say* "same concept"?

Companion to ``run_channel_dissociation.py``. There, the declared channel was
*generation* (the model rewrites the seed). Here it is *judgment*: a fixed
yes/no question, read deterministically as P(Yes)/(P(Yes)+P(No)) at the answer
position, averaged over A/B order.

It reuses the pairs and functional distances (JS) already saved by the
dissociation run, so only the judgment forwards are new.

Question (H1'b): on near pairs (bee/wasp) does the model *declare* "same" (high
declared score) while the functional channel keeps them apart (high JS)? That
gap is the dissociation "says equivalent, does not do equivalence".

Run:  uv run python scripts/run_channel_dissociation.py   # first (provides pairs)
      uv run python scripts/run_channel_judgment.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tc.corpus import HARD_GROUPS
from tc.extract import next_token_probs

PROMPT = (
    "Decide whether two descriptions refer to the same concept.\n\n"
    "Description A: {a}\n"
    "Description B: {b}\n\n"
    "Do they refer to the same concept? Answer Yes or No.\n"
    "Answer:"
)


def class_token_ids(tok, target: str):
    """Vocabulary ids whose decoded, stripped form equals `target` (lower)."""
    t = target.strip().lower()
    return [i for i in range(len(tok)) if tok.decode([i]).strip().lower() == t]


def declared_same(model, tok, a, b, yes_ids, no_ids) -> float:
    p = next_token_probs(model, tok, PROMPT.format(a=a, b=b))
    py, pn = float(p[yes_ids].sum()), float(p[no_ids].sum())
    return py / (py + pn + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="results/channel_dissociation_summary.json")
    ap.add_argument("--out", default="results/channel_judgment_summary.json")
    args = ap.parse_args()

    src = json.loads(Path(args.pairs).read_text())
    model_name = src["model"]
    records = src["records"]
    seeds = list(src["generated"].keys())

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    yes_ids = class_token_ids(tok, "yes")
    no_ids = class_token_ids(tok, "no")
    if not yes_ids or not no_ids:
        raise SystemExit(f"could not locate Yes/No tokens (yes={yes_ids}, no={no_ids})")
    print(f"[model] {model_name}  yes_ids={yes_ids[:5]} no_ids={no_ids[:5]}\n")

    seed_text = {s: HARD_GROUPS[s][0] for s in seeds}
    for s in seeds:
        a = seed_text[s]
        d = 0.5 * (declared_same(model, tok, a, a, yes_ids, no_ids)
                   + declared_same(model, tok, a, a, yes_ids, no_ids))
        print(f"  [self-control] {s:9s} declared_same(seed,seed) = {d:.3f}")

    print("\npair judgments:")
    for r in records:
        a = seed_text[r["seed"]]
        d1 = declared_same(model, tok, a, r["text"], yes_ids, no_ids)
        d2 = declared_same(model, tok, r["text"], a, yes_ids, no_ids)
        r["declared"] = 0.5 * (d1 + d2)
        print(f"  {r['seed']:9s} {r['category']:9s} D={r['declared']:.3f}  JS={r['js']:.3f}")

    cats = ["human", "generated", "near", "unrelated"]
    summary = {"categories": {}, "auc_per_seed": {}, "auc_mean": {}}
    for c in cats:
        D = [r["declared"] for r in records if r["category"] == c]
        F = [r["js"] for r in records if r["category"] == c]
        summary["categories"][c] = {
            "n": len(D),
            "declared_mean": float(np.mean(D)),
            "frac_declared_same": float(np.mean([d > 0.5 for d in D])),
            "js_mean": float(np.mean(F)),
        }

    def per_seed_auc():
        vals = []
        for s in seeds:
            same = [r["declared"] for r in records if r["seed"] == s and r["category"] in ("human", "generated")]
            diff = [r["declared"] for r in records if r["seed"] == s and r["category"] in ("near", "unrelated")]
            if same and diff:
                y = np.r_[np.ones(len(same)), np.zeros(len(diff))]
                vals.append(float(roc_auc_score(y, np.r_[same, diff])))
        return vals

    vals = per_seed_auc()
    summary["auc_per_seed"]["declared_same_vs_different"] = vals
    summary["auc_mean"]["declared_same_vs_different"] = float(np.nanmean(vals))

    D = np.array([r["declared"] for r in records])
    F = np.array([r["js"] for r in records])
    summary["declared_vs_functional_spearman"] = float(stats.spearmanr(D, F).correlation)

    # dissociation cells: declared high but functional far
    near = [r for r in records if r["category"] == "near"]
    summary["near_declared_mean"] = float(np.mean([r["declared"] for r in near]))
    summary["near_js_mean"] = float(np.mean([r["js"] for r in near]))
    summary["dissociation_near_frac (D>0.5 & JS above same-median)"] = float(
        np.mean([
            r["declared"] > 0.5
            and r["js"] > np.median([x["js"] for x in records if x["category"] in ("human", "generated")])
            for r in near
        ])
    )

    print("\ncategory      n   declared   frac>0.5   JS")
    for c in cats:
        s = summary["categories"][c]
        print(f"  {c:11s} {s['n']:2d}   {s['declared_mean']:.3f}     {s['frac_declared_same']:.2f}    {s['js_mean']:.3f}")
    print(f"\nAUC declared (same vs different): {summary['auc_mean']['declared_same_vs_different']:.3f}")
    print(f"Spearman(declared, functional JS): {summary['declared_vs_functional_spearman']:+.3f}")
    print(f"near: declared={summary['near_declared_mean']:.3f}  JS={summary['near_js_mean']:.3f}  "
          f"dissociation frac={summary['dissociation_near_frac (D>0.5 & JS above same-median)']:.2f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": model_name, "source": args.pairs,
        "prompt": PROMPT, "records": records, "summary": summary,
    }, indent=2, ensure_ascii=False))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
