#!/usr/bin/env python
"""Recount centrality with 'pair of X' handling (default: clothing).

Compares, per prompt: Pearson(q, typicality) using direct naming only vs using
direct + 'pair of X' (mutually exclusive, so probabilities add).  Reports the
top prototypes' probability breakdown.

Run:
    uv run python scripts/run_pair_recount.py --model distilgpt2
    uv run python scripts/run_pair_recount.py --model Qwen/Qwen2.5-1.5B-Instruct
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr
from wordfreq import zipf_frequency

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from transformers import AutoModelForCausalLM, AutoTokenizer

from tc.centrality import item_probability, partial_pearson, word_probability
from tc.typicality import load_norms

DEFAULT_DATA = Path(__file__).resolve().parents[1] / "data" / "typicality.tsv"

PROMPTS = {
    "clothing": [
        "Name one item of clothing: a",
        "An item of clothing people wear, such as a",
        "Something you wear on your body, such as a",
        "An everyday garment, such as a",
        "A piece of clothing, such as a",
    ],
}
PROTOTYPES = {"clothing": ["jeans", "sweater", "underwear"]}


def model_slug(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilgpt2")
    ap.add_argument("--category", default="clothing")
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--prefixes", default="pair of")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    prefixes = tuple(p.strip() for p in args.prefixes.split(",") if p.strip())
    protos = PROTOTYPES.get(args.category, [])

    rows = [r for r in load_norms(args.data) if r["category"] == args.category]
    words = [r["item"] for r in rows]
    human = np.array([float(r["typicality"]) for r in rows])

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()

    print(f"\n##### {args.model} | {args.category} | prefixes={prefixes} #####")
    payload = {"model": args.model, "category": args.category, "prefixes": list(prefixes),
               "prompts": {}}

    for prompt in PROMPTS[args.category]:
        direct, combined, lengths, zipfs = [], [], [], []
        breakdown = {}
        for word in words:
            d, n = word_probability(model, tokenizer, prompt, word)
            total, bd = item_probability(model, tokenizer, prompt, word, prefixes=prefixes)
            direct.append(d)
            combined.append(total)
            lengths.append(n)
            zipfs.append(zipf_frequency(word, "en"))
            breakdown[word] = bd
        direct = np.array(direct)
        combined = np.array(combined)
        controls = np.column_stack([lengths, zipfs])

        rd = pearsonr(direct, human)
        rc = pearsonr(combined, human)
        sc = spearmanr(combined, human)
        pc = partial_pearson(combined, human, controls)

        proto = {p: breakdown[p] for p in protos if p in breakdown}
        proto_sum = {p: sum(breakdown[p].values()) for p in proto}

        print(f"\n  {prompt!r}")
        print(f"    Pearson direct={rd[0]:+.3f}  |  combined={rc[0]:+.3f}(p={rc[1]:.3f})  "
              f"Spearman={sc[0]:+.3f}  partial={pc:+.3f}")
        for p in proto:
            bd = proto[p]
            parts = ", ".join(f"{k}={v:.4f}" for k, v in bd.items())
            print(f"    {p:10s} combined={proto_sum[p]:.4f}  ({parts})")

        payload["prompts"][prompt] = {
            "pearson_direct": {"r": float(rd[0]), "p": float(rd[1])},
            "pearson_combined": {"r": float(rc[0]), "p": float(rc[1])},
            "spearman_combined": {"rho": float(sc[0]), "p": float(sc[1])},
            "partial_combined": float(pc),
            "prototype_breakdown": proto,
            "prototype_combined": proto_sum,
        }

    out = Path(args.out) if args.out else Path(
        f"results/pair_recount_{model_slug(args.model)}_{model_slug(args.category)}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
