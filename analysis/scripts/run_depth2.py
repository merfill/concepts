#!/usr/bin/env python
"""Experiment 001 (depth 2): does functional equivalence persist past the first token?

Companion to ``run_channel_dissociation.py``. Reuses its pairs (seed, category,
text) and re-measures each prefix under the *same* probe, now over the joint
two-token distribution:

    q(t1, t2) = p1(t1) * p2(t2 | t1)

truncated to top-K1 x top-K2, with the chosen first token appended by *id* (no
re-tokenization). Two prefixes are compared by JS on the union of their pair
supports (``metrics.js_union``). Depth-1 distance is recomputed with the same
K for a fair comparison.

The point is the attractor diagnostic (protocol §6): compare same-concept vs
different-concept separation at depth 1 and depth 2. If different-concept pairs
collapse as much as same-concept pairs, depth 2 is uninformative (generic
attractor); if only same-concept pairs stay together, equivalence persists.

Pre-registered: K1 = K2 = 10, probe "It is a" (frame-swap already checked at
depth 1).  Run:  uv run python scripts/run_depth2.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tc.corpus import HARD_GROUPS
from tc.metrics import js_topk, js_union

PROBE = "It is a"


def top2_joint(model, tok, prefix, k1, k2):
    """Return (p1_full, joint{(t1,t2): mass}) for a prefix."""
    enc = tok(prefix, return_tensors="pt")
    ids = enc["input_ids"]
    am = enc["attention_mask"]
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=am)
    p1 = torch.softmax(out.logits[0, -1], dim=-1).double().numpy()
    top1 = np.argsort(p1)[::-1][:k1]
    joint = {}
    for t1 in top1:
        ids2 = torch.cat([ids, torch.tensor([[int(t1)]])], dim=1)
        am2 = torch.cat([am, torch.ones((1, 1), dtype=am.dtype)], dim=1)
        with torch.no_grad():
            out2 = model(input_ids=ids2, attention_mask=am2)
        p2 = torch.softmax(out2.logits[0, -1], dim=-1).double().numpy()
        for t2 in np.argsort(p2)[::-1][:k2]:
            joint[(int(t1), int(t2))] = float(p1[t1] * p2[t2])
    return p1, joint


def auc_smaller(a, b) -> float:
    a, b = list(a), list(b)
    if not a or not b:
        return float("nan")
    y = np.r_[np.ones(len(a)), np.zeros(len(b))]
    return float(roc_auc_score(y, -np.r_[a, b]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="results/channel_dissociation_summary.json")
    ap.add_argument("--k1", type=int, default=10)
    ap.add_argument("--k2", type=int, default=10)
    ap.add_argument("--out", default="results/depth2_summary.json")
    args = ap.parse_args()

    src = json.loads(Path(args.pairs).read_text())
    model_name = src["model"]
    records = src["records"]
    seeds = list(src["generated"].keys())

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    print(f"[model] {model_name}  probe={PROBE!r}  K1={args.k1} K2={args.k2}\n")

    seed_text = {s: HARD_GROUPS[s][0] for s in seeds}
    ref = {s: top2_joint(model, tok, f"{seed_text[s]}\n\n{PROBE}", args.k1, args.k2) for s in seeds}
    print("reference prefixes ready\n")

    for r in records:
        a = seed_text[r["seed"]]
        p1_seed, j_seed = ref[r["seed"]]
        p1, joint = top2_joint(model, tok, f"{r['text']}\n\n{PROBE}", args.k1, args.k2)
        r["js_d1"] = js_topk(p1_seed, p1, args.k1)
        r["js_d2"] = js_union(j_seed, joint)

    cats = ["human", "generated", "near", "unrelated"]
    summary = {"categories": {}, "auc_per_seed": {}, "auc_mean": {}}
    for c in cats:
        xs = [r for r in records if r["category"] == c]
        summary["categories"][c] = {
            "n": len(xs),
            "js_d1_mean": float(np.mean([r["js_d1"] for r in xs])),
            "js_d2_mean": float(np.mean([r["js_d2"] for r in xs])),
        }

    def per_seed_auc(key):
        vals = []
        for s in seeds:
            same = [r[key] for r in records if r["seed"] == s and r["category"] in ("human", "generated")]
            diff = [r[key] for r in records if r["seed"] == s and r["category"] in ("near", "unrelated")]
            vals.append(auc_smaller(same, diff))
        return vals

    for depth in ("js_d1", "js_d2"):
        vals = per_seed_auc(depth)
        summary["auc_per_seed"][depth] = vals
        summary["auc_mean"][depth] = float(np.nanmean(vals))

    same_c = summary["categories"]["human"]["js_d1_mean"], summary["categories"]["generated"]["js_d1_mean"]
    same_d1 = float(np.mean(same_c))
    same_d2 = float(np.mean([summary["categories"]["human"]["js_d2_mean"],
                             summary["categories"]["generated"]["js_d2_mean"]]))
    diff_d1 = float(np.mean([summary["categories"]["near"]["js_d1_mean"],
                             summary["categories"]["unrelated"]["js_d1_mean"]]))
    diff_d2 = float(np.mean([summary["categories"]["near"]["js_d2_mean"],
                             summary["categories"]["unrelated"]["js_d2_mean"]]))
    summary["attractor"] = {
        "same_d1": same_d1, "same_d2": same_d2,
        "diff_d1": diff_d1, "diff_d2": diff_d2,
        "gap_d1": diff_d1 - same_d1, "gap_d2": diff_d2 - same_d2,
        "ratio_same": same_d2 / same_d1 if same_d1 else float("nan"),
        "ratio_diff": diff_d2 / diff_d1 if diff_d1 else float("nan"),
    }

    print("category      n   JS depth1   JS depth2")
    for c in cats:
        s = summary["categories"][c]
        print(f"  {c:11s} {s['n']:2d}   {s['js_d1_mean']:.4f}      {s['js_d2_mean']:.4f}")
    a = summary["attractor"]
    print(f"\nAUC same-vs-different: depth1={summary['auc_mean']['js_d1']:.3f}  depth2={summary['auc_mean']['js_d2']:.3f}")
    print(f"gap(diff-same): depth1={a['gap_d1']:+.4f}  depth2={a['gap_d2']:+.4f}")
    print(f"growth same/diff: same x{a['ratio_same']:.2f}  diff x{a['ratio_diff']:.2f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": model_name, "probe": PROBE, "k1": args.k1, "k2": args.k2,
        "source": args.pairs, "records": records, "summary": summary,
    }, indent=2, ensure_ascii=False))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
