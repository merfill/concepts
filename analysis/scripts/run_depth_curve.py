#!/usr/bin/env python
"""Experiment 001 (depth curve): separation vs generation depth (1..D).

Full joint over d tokens explodes (K^d). Instead we approximate the joint by a
*beam* of the B most probable continuations at each depth, and compare two
prefixes by JS on the union of their beam supports (``metrics.js_union``). This
gives one functional distance per depth, so we can see *where* same-vs-different
separation starts to decay (transient -> generic attractor).

Beam: at each depth, expand every beam item by its top-`expand` tokens (one
forward per beam item, full distribution reused), keep the global top-B by mass.
Cost per prefix = 1 + B*(D-1) forwards.

Run:  uv run python scripts/run_depth_curve.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tc.corpus import HARD_GROUPS
from tc.metrics import js_union

PROBE = "It is a"


def beam_profiles(model, tok, prefix, B, D, expand):
    """Return profiles[0..D-1]: {seq_tuple: mass} of the beam at each depth."""
    enc = tok(prefix, return_tensors="pt")
    ids0, am0 = enc["input_ids"], enc["attention_mask"]
    with torch.no_grad():
        out = model(input_ids=ids0, attention_mask=am0)
    p = torch.softmax(out.logits[0, -1], dim=-1).double().numpy()
    beam = [((int(t),), float(p[t])) for t in np.argsort(p)[::-1][:B]]
    profiles = [{seq: m for seq, m in beam}]
    for _ in range(2, D + 1):
        cand = {}
        for seq, mass in beam:
            ids = torch.cat([ids0, torch.tensor([list(seq)])], dim=1)
            am = torch.cat([am0, torch.ones((1, len(seq)), dtype=am0.dtype)], dim=1)
            with torch.no_grad():
                out = model(input_ids=ids, attention_mask=am)
            p = torch.softmax(out.logits[0, -1], dim=-1).double().numpy()
            for t in np.argsort(p)[::-1][:expand]:
                nseq = seq + (int(t),)
                cand[nseq] = cand.get(nseq, 0.0) + mass * float(p[t])
        beam = sorted(cand.items(), key=lambda kv: kv[1], reverse=True)[:B]
        profiles.append(dict(beam))
    return profiles


def auc_smaller(a, b) -> float:
    a, b = list(a), list(b)
    if not a or not b:
        return float("nan")
    y = np.r_[np.ones(len(a)), np.zeros(len(b))]
    return float(roc_auc_score(y, -np.r_[a, b]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="results/channel_dissociation_summary.json")
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--beam", type=int, default=8)
    ap.add_argument("--expand", type=int, default=20)
    ap.add_argument("--out", default="results/depth_curve_summary.json")
    args = ap.parse_args()

    src = json.loads(Path(args.pairs).read_text())
    model_name, records = src["model"], src["records"]
    seeds = list(src["generated"].keys())

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    print(f"[model] {model_name}  probe={PROBE!r}  D={args.depth} B={args.beam}\n")

    seed_text = {s: HARD_GROUPS[s][0] for s in seeds}
    ref = {s: beam_profiles(model, tok, f"{seed_text[s]}\n\n{PROBE}", args.beam, args.depth, args.expand)
           for s in seeds}
    print("reference prefixes ready\n")

    for r in records:
        prof = beam_profiles(model, tok, f"{r['text']}\n\n{PROBE}", args.beam, args.depth, args.expand)
        r["js_by_depth"] = [js_union(ref[r["seed"]][d], prof[d]) for d in range(args.depth)]

    cats = ["human", "generated", "near", "unrelated"]
    summary = {"by_depth": [], "categories": {}}
    same_base = ("human", "generated")
    diff_base = ("near", "unrelated")
    for d in range(args.depth):
        row = {"depth": d + 1, "categories": {}}
        for c in cats:
            xs = [r["js_by_depth"][d] for r in records if r["category"] == c]
            row["categories"][c] = {"mean": float(np.mean(xs)), "n": len(xs)}
        def per_seed(*keys):
            vals = []
            for s in seeds:
                same = [r["js_by_depth"][d] for r in records if r["seed"] == s and r["category"] in same_base]
                diff = [r["js_by_depth"][d] for r in records if r["seed"] == s and r["category"] in diff_base]
                vals.append(auc_smaller(same, diff))
            return float(np.nanmean(vals))
        row["auc_same_vs_diff"] = per_seed()
        same = np.mean([row["categories"][c]["mean"] for c in same_base])
        diff = np.mean([row["categories"][c]["mean"] for c in diff_base])
        row["gap_diff_minus_same"] = float(diff - same)
        summary["by_depth"].append(row)

    summary["categories"] = {
        c: [r["js_by_depth"][d] for r in records if r["category"] == c] for c in cats
    }

    print("depth  human  generated  near  unrelated   AUC    gap")
    for row in summary["by_depth"]:
        cc = row["categories"]
        print(f"  {row['depth']}    {cc['human']['mean']:.3f}   {cc['generated']['mean']:.3f}     "
              f"{cc['near']['mean']:.3f}   {cc['unrelated']['mean']:.3f}      "
              f"{row['auc_same_vs_diff']:.3f}  {row['gap_diff_minus_same']:+.3f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": model_name, "probe": PROBE, "depth": args.depth,
        "beam": args.beam, "expand": args.expand, "source": args.pairs,
        "records": records, "summary": summary,
    }, indent=2, ensure_ascii=False))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
