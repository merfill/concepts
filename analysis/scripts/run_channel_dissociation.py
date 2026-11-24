#!/usr/bin/env python
"""Experiment 001: declared vs functional equivalence.

Protocol: ``../experiment-001-2026-11-24-channel-dissociation.md``.

Two channels of one network, measured on the same expressions:
  * declared channel   -- the model *generates* paraphrases of a seed, thereby
                          asserting they are equivalent (generation arm, §1);
  * functional channel -- each expression induces a next-token distribution
                          under a *fixed* probe; distance = shared-support
                          Jensen-Shannon over top-k.

Question (H1'): do the model's own paraphrases behave functionally like the seed,
i.e. is ``JS(seed, generated)`` as small as ``JS(seed, human)`` and smaller than
``JS(seed, near)`` / ``JS(seed, unrelated)``? A positive dissociation gap means
the model *says* "equivalent" but does not *do* equivalence.

Pre-registered parameters (fixed before looking at results):
    PROBE_PRIMARY = "It is a"
    PROBE_ALT     = "In short, it is a"   (frame-swap stability check)
    K             = 50                    (shared-support top-k)

Run:  uv run python scripts/run_channel_dissociation.py
      uv run python scripts/run_channel_dissociation.py --concepts 2   # smoke
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tc.controls import residualize
from tc.corpus import HARD_GROUPS, RELATED_CONCEPTS, RELATED_PAIRS
from tc.extract import next_token_probs, token_count
from tc.metrics import jaccard_words, js_topk, tv_topk

PROBE_PRIMARY = "It is a"
PROBE_ALT = "In short, it is a"
K = 50
UNRELATED = "clock"
GEN_TEMPLATE = (
    "Rewrite the following description using different words while keeping the "
    "same meaning. Reply with the rewrite only.\n\nDescription: {seed}\n\nRewrite:"
)


_REWRITE = re.compile(r"\s*rewrite\s*:", re.IGNORECASE)


def clean(text: str) -> str:
    text = text.strip()
    text = _REWRITE.split(text)[0]
    text = text.split("\n")[0].strip()
    for pref in ("rewritten:", "output:", "-", "\u2022"):
        if text.lower().startswith(pref):
            text = text[len(pref):].strip()
    m = re.match(r"(.+?[.!?])(?:\s|$)", text)
    if m:
        text = m.group(1)
    return text.strip(" \"'")


def generate_paraphrases(model, tok, seed, n, max_new_tokens):
    enc = tok(GEN_TEMPLATE.format(seed=seed), return_tensors="pt")
    out, seen = [], set()
    for _ in range(n):
        with torch.no_grad():
            gen = model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=True,
                temperature=0.8, top_p=0.9, pad_token_id=tok.eos_token_id,
            )
        t = clean(tok.decode(gen[0][enc["input_ids"].shape[1]:], skip_special_tokens=True))
        if len(t.split()) < 3 or t.lower() in seen or t.lower() == seed.lower():
            continue
        seen.add(t.lower())
        out.append(t)
    return out


def probe_dist(model, tok, text, probe):
    return next_token_probs(model, tok, f"{text}\n\n{probe}")


def auc_smaller(a, b) -> float:
    """AUC that distances in `a` are smaller than those in `b`."""
    a, b = list(a), list(b)
    if not a or not b:
        return float("nan")
    y = np.r_[np.ones(len(a)), np.zeros(len(b))]
    return float(roc_auc_score(y, -np.r_[a, b]))


def partial_spearman_vec(a, b, controls_list) -> float:
    """Partial Spearman between a and b controlling for covariate vectors."""
    ra = residualize(np.asarray(a, dtype=np.float64), controls_list)
    rb = residualize(np.asarray(b, dtype=np.float64), controls_list)
    return float(stats.spearmanr(ra, rb).correlation)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--concepts", type=int, default=0, help="limit to first N seeds (0=all)")
    ap.add_argument("--gen", type=int, default=3, help="paraphrases to sample per seed")
    ap.add_argument("--max-new-tokens", type=int, default=40)
    ap.add_argument("--out", default="results/channel_dissociation_summary.json")
    args = ap.parse_args()

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()
    print(f"[model] {args.model}\n[probe] {PROBE_PRIMARY!r} / {PROBE_ALT!r}  k={K}\n")

    partner = dict(RELATED_PAIRS)
    seeds = list(partner)
    if args.concepts:
        seeds = seeds[: args.concepts]

    unrelated_texts = HARD_GROUPS[UNRELATED]
    records, generated = [], {}

    for seed_concept in seeds:
        seed = HARD_GROUPS[seed_concept][0]
        cats = {
            "human": HARD_GROUPS[seed_concept][1:],
            "near": RELATED_CONCEPTS[partner[seed_concept]],
            "unrelated": unrelated_texts,
        }
        cats["generated"] = generate_paraphrases(
            model, tok, seed, args.gen, args.max_new_tokens
        )
        generated[seed_concept] = cats["generated"]
        print(f"=== {seed_concept} (seed: {seed[:50]!r}) ===")
        print(f"  generated: {cats['generated']}")

        ps = probe_dist(model, tok, seed, PROBE_PRIMARY)
        ps_alt = probe_dist(model, tok, seed, PROBE_ALT)
        seed_len = token_count(seed, tok)
        for cat, texts in cats.items():
            for t in texts:
                pt = probe_dist(model, tok, t, PROBE_PRIMARY)
                pt_alt = probe_dist(model, tok, t, PROBE_ALT)
                records.append({
                    "seed": seed_concept, "category": cat, "text": t,
                    "js": js_topk(ps, pt, K),
                    "js_alt": js_topk(ps_alt, pt_alt, K),
                    "tv": tv_topk(ps, pt, K),
                    "lexical": jaccard_words(seed, t),
                    "len_delta": abs(token_count(t, tok) - seed_len),
                })
        print()

    # ---- aggregation -------------------------------------------------------
    cats_all = ["human", "generated", "near", "unrelated"]
    summary = {"categories": {}, "auc_per_seed": {}, "auc_mean": {}}
    for c in cats_all:
        js = [r["js"] for r in records if r["category"] == c]
        summary["categories"][c] = {
            "n": len(js),
            "js_mean": float(np.mean(js)) if js else float("nan"),
            "js_median": float(np.median(js)) if js else float("nan"),
            "tv_mean": float(np.mean([r["tv"] for r in records if r["category"] == c])) if js else float("nan"),
            "lex_mean": float(np.mean([r["lexical"] for r in records if r["category"] == c])) if js else float("nan"),
            "len_delta_mean": float(np.mean([r["len_delta"] for r in records if r["category"] == c])) if js else float("nan"),
        }

    def per_seed_auc(cat_a, cat_b_list):
        vals = []
        for s in seeds:
            a = [r["js"] for r in records if r["seed"] == s and r["category"] == cat_a]
            b = [r["js"] for r in records if r["seed"] == s and r["category"] in cat_b_list]
            vals.append(auc_smaller(a, b))
        return vals

    comparisons = {
        "generated_vs_near": ("generated", ["near"]),
        "generated_vs_unrelated": ("generated", ["unrelated"]),
        "generated_vs_different": ("generated", ["near", "unrelated"]),
        "human_vs_different": ("human", ["near", "unrelated"]),
    }
    for name, (a, bl) in comparisons.items():
        vals = per_seed_auc(a, bl)
        summary["auc_per_seed"][name] = vals
        summary["auc_mean"][name] = float(np.nanmean(vals)) if vals else float("nan")

    js_gen = summary["categories"]["generated"]["js_mean"]
    js_hum = summary["categories"]["human"]["js_mean"]
    summary["dissociation_gap"] = float(js_gen - js_hum)

    all_js = [r["js"] for r in records]
    all_alt = [r["js_alt"] for r in records]
    summary["frame_stability_spearman"] = float(stats.spearmanr(all_js, all_alt).correlation)
    summary["js_vs_lexical_spearman"] = float(
        stats.spearmanr(all_js, [r["lexical"] for r in records]).correlation
    )
    labels = np.array([1.0 if r["category"] in ("human", "generated") else 0.0 for r in records])
    lex = np.array([r["lexical"] for r in records])
    lens = np.array([r["len_delta"] for r in records])
    summary["label_js_spearman"] = float(stats.spearmanr(labels, all_js).correlation)
    summary["label_js_partial_lex_len"] = partial_spearman_vec(labels, all_js, [lex, lens])

    # ---- print -------------------------------------------------------------
    print("category       n   JS_mean  JS_med   TV_mean  lex_mean  len_delta")
    for c in cats_all:
        s = summary["categories"][c]
        print(f"  {c:11s} {s['n']:2d}  {s['js_mean']:.4f}  {s['js_median']:.4f}  "
              f"{s['tv_mean']:.4f}  {s['lex_mean']:.3f}    {s['len_delta_mean']:.2f}")
    print("\nAUC (per-seed mean), smaller distance = same concept:")
    for name in comparisons:
        print(f"  {name:24s} {summary['auc_mean'][name]:.3f}")
    print(f"\ndissociation gap (gen - human JS): {summary['dissociation_gap']:+.4f}")
    print(f"frame-swap stability (rho JS1 vs JS2): {summary['frame_stability_spearman']:.3f}")
    print(f"JS vs lexical overlap (rho):           {summary['js_vs_lexical_spearman']:.3f}")
    print(f"same-vs-different vs JS (rho):         {summary['label_js_spearman']:.3f}")
    print(f"  partial | lex+len (rho):             {summary['label_js_partial_lex_len']:.3f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": args.model, "probe_primary": PROBE_PRIMARY, "probe_alt": PROBE_ALT,
        "k": K, "gen_per_seed": args.gen, "unrelated": UNRELATED,
        "generated": generated, "records": records, "summary": summary,
    }, indent=2, ensure_ascii=False))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
