#!/usr/bin/env python
"""Experiment 010 / stage 1 extension: category centrality via instance naming.

For each category: scan candidate frames structurally (purity of the top-k naming
distribution), keep the best, then measure centrality ``q(w)`` vs THINGSplus
typicality and probe graded membership ``Is "x" a C?``.

Protocol: ``../experiment-010-2026-11-26-category-centrality.md``.

Run:
    uv run python scripts/run_centrality.py --categories bird,fruit,vegetable
    uv run python scripts/run_centrality.py --category mammal --skip-membership
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from wordfreq import zipf_frequency

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from transformers import AutoModelForCausalLM, AutoTokenizer

from tc.centrality import (
    DESCRIPTORS,
    FRAMES,
    candidate_frames,
    compute_purity,
    find_token_ids,
    membership_prob,
    partial_pearson,
    topk_items,
    word_probability,
)
from tc.extract import next_token_probs
from tc.lexicon import member_lemmas, normalize
from tc.typicality import load_norms

try:
    from nltk.stem import WordNetLemmatizer

    LEMMATIZER = WordNetLemmatizer()
except Exception:  # pragma: no cover - WordNet data optional
    LEMMATIZER = None

DEFAULT_DATA = Path(__file__).resolve().parents[1] / "data" / "typicality.tsv"
DEFAULT_PROBES = ["lion", "cat", "elephant", "rabbit",
                  "bird", "reptile", "fish", "insect", "table", "chair", "large", "type"]


def model_slug(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_")


def members_in_top(top, purity):
    rank = {tok: i for i, (tok, _) in enumerate(top, 1)}
    return [(m["lemma"], m["p"], rank[m["token"]])
            for m in purity["members"] if m["token"] in rank]


def evaluate_frame(model, tokenizer, frame, k, members):
    probs = next_token_probs(model, tokenizer, frame)
    top = topk_items(tokenizer, probs, k)
    purity = compute_purity(top, members, LEMMATIZER)
    return top, purity


def centrality_stats(model, tokenizer, frame, words, human):
    q, lengths, zipfs = [], [], []
    for word in words:
        prob, n_tokens = word_probability(model, tokenizer, frame, word)
        q.append(prob)
        lengths.append(n_tokens)
        zipfs.append(zipf_frequency(word, "en"))
    q = np.array(q)
    pear = pearsonr(q, human)
    spear = spearmanr(q, human)
    partial = partial_pearson(q, human, np.column_stack([lengths, zipfs]))
    return {
        "pearson": {"r": float(pear[0]), "p": float(pear[1])},
        "spearman": {"rho": float(spear[0]), "p": float(spear[1])},
        "partial_pearson": float(partial),
        "per_word": [
            {"word": w, "human": float(h), "q": float(p), "n_tokens": int(n), "zipf": float(z)}
            for w, h, p, n, z in zip(words, human, q, lengths, zipfs)
        ],
    }


def run_category(category, args, model, tokenizer, yes_ids, no_ids, norms):
    rows = [r for r in norms if r["category"] == category]
    if not rows:
        print(f"[skip] no rows for {category!r}")
        return None
    words = [r["item"] for r in rows]
    human = np.array([float(r["typicality"]) for r in rows])
    members = member_lemmas(category, extra=words)

    fixed = FRAMES.get(category)
    candidates = fixed if fixed else candidate_frames(category)
    print(f"\n===== {category!r}  (n={len(words)}, WordNet members={len(members)}) =====")
    print(f"  descriptor={DESCRIPTORS.get(category, category)!r}")
    print(f"  frame candidates: {candidates}")

    scans = {}
    for frame in candidates:
        top, purity = evaluate_frame(model, tokenizer, frame, args.k, members)
        mit = members_in_top(top, purity)
        scans[frame] = {
            "purity": purity,
            "n_members": len(mit),
            "members_in_top": [{"lemma": l, "p": p, "rank": r} for l, p, r in mit],
        }
        print(f"    purity={purity['purity']:.3f} n_mem={len(mit):2d} | {frame!r}")

    best = max(candidates, key=lambda f: (scans[f]["n_members"], scans[f]["purity"]["purity"]))
    print(f"  -> chosen (max member coverage, tie-break purity): {best!r}")

    top, purity = evaluate_frame(model, tokenizer, best, args.k, members)
    stats = centrality_stats(model, tokenizer, best, words, human)
    print(f"  purity={purity['purity']:.3f} (member={purity['member_mass']:.3f} "
          f"function={purity['function_mass']:.3f} other={purity['other_mass']:.3f})")
    print("  members in top-k: %s" % (
        ", ".join(f"r{r}:{l}({p:.3f})" for l, p, r in members_in_top(top, purity)) or "<none>"))
    print("  other content: %s" % (
        ", ".join(f"{o['token']}({o['p']:.3f})" for o in purity["other"][:15]) or "<none>"))
    print(f"  Pearson(q)={stats['pearson']['r']:+.3f} (p={stats['pearson']['p']:.3f})  "
          f"Spearman={stats['spearman']['rho']:+.3f}  "
          f"partial(len+zipf)={stats['partial_pearson']:+.3f}")

    membership = {}
    if not args.skip_membership:
        extra = args.probes or DEFAULT_PROBES
        candidates_m = words + [w for w in extra if w not in words]
        for item in candidates_m:
            membership[item] = {
                "mu": membership_prob(model, tokenizer, item, category, yes_ids, no_ids),
                "in_annotation": item in words,
                "in_lexicon": normalize(item) in members,
            }
        ann = [v["mu"] for v in membership.values() if v["in_annotation"]]
        mem_ctrl = [v["mu"] for v in membership.values()
                    if not v["in_annotation"] and v["in_lexicon"]]
        non_ctrl = [v["mu"] for v in membership.values()
                    if not v["in_annotation"] and not v["in_lexicon"]]
        print("  membership: ann=%.3f (n=%d)" % (np.mean(ann), len(ann)), end="")
        if mem_ctrl:
            print("  member-ctrl=%.3f (n=%d)" % (np.mean(mem_ctrl), len(mem_ctrl)), end="")
        if non_ctrl:
            print("  non-member-ctrl=%.3f (n=%d)" % (np.mean(non_ctrl), len(non_ctrl)), end="")
        print()
        low = [f"{w}({v['mu']:.2f})" for w, v in membership.items()
               if v["in_annotation"] and v["mu"] < 0.5]
        if low:
            print("  low-mu annotated members: %s" % ", ".join(low))

    return {
        "category": category,
        "descriptor": DESCRIPTORS.get(category, category),
        "frames": candidates,
        "scan": {f: {"purity": s["purity"]["purity"], "n_members": s["n_members"]}
                 for f, s in scans.items()},
        "chosen_frame": best,
        "purity": purity,
        "centrality": stats,
        "membership": membership,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--category", default="")
    ap.add_argument("--categories", default="",
                    help="comma-separated list; overrides --category")
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--probes", default="", help="comma-separated membership probes")
    ap.add_argument("--skip-membership", action="store_true")
    args = ap.parse_args()
    args.probes = [w.strip() for w in args.probes.split(",") if w.strip()]

    categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    if not categories:
        categories = [args.category or "mammal"]

    norms = load_norms(args.data)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()
    yes_ids = find_token_ids(tokenizer, "yes")
    no_ids = find_token_ids(tokenizer, "no")
    if not yes_ids or not no_ids:
        raise SystemExit("could not locate Yes/No tokens")

    outdir = Path("results")
    outdir.mkdir(parents=True, exist_ok=True)
    for category in categories:
        result = run_category(category, args, model, tokenizer, yes_ids, no_ids, norms)
        if result is None:
            continue
        out = outdir / f"centrality_{model_slug(args.model)}_{model_slug(category)}.json"
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"  saved -> {out}")


if __name__ == "__main__":
    main()
