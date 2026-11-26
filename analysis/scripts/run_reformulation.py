#!/usr/bin/env python
"""Reformulation test for tool/clothing (surface vs meaning, fixed phrasings).

For each fixed phrasing: purity of the top-k naming distribution, the raw top
tokens, the probability of the human prototypes, and Pearson(q, typicality)
over the 20 annotation words.

Run:
    uv run python scripts/run_reformulation.py --model distilgpt2
    uv run python scripts/run_reformulation.py --model Qwen/Qwen2.5-1.5B-Instruct
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

from tc.centrality import compute_purity, partial_pearson, topk_items, word_probability
from tc.extract import next_token_probs
from tc.lexicon import member_lemmas
from tc.typicality import load_norms

try:
    from nltk.stem import WordNetLemmatizer

    LEMMATIZER = WordNetLemmatizer()
except Exception:  # pragma: no cover
    LEMMATIZER = None

DEFAULT_DATA = Path(__file__).resolve().parents[1] / "data" / "typicality.tsv"

PROMPTS = {
    "tool": [
        "Name one tool used for work: a",
        "A hand tool, such as a",
        "A manual tool, such as a",
        "A tool you use with your hands, such as a",
        "A tool you hold in your hand, such as a",
    ],
    "clothing": [
        "Name one item of clothing: a",
        "An item of clothing people wear, such as a",
        "Something you wear on your body, such as a",
        "An everyday garment, such as a",
        "A piece of clothing, such as a",
    ],
}
# top-3 by human typicality in our THINGSplus subset
PROTOTYPES = {
    "tool": ["screwdriver", "tape measure", "ratchet"],
    "clothing": ["jeans", "sweater", "underwear"],
}


def model_slug(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilgpt2")
    ap.add_argument("--categories", default="tool,clothing")
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    norms = load_norms(args.data)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()

    categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    payload = {"model": args.model, "k": args.k, "categories": {}}

    for category in categories:
        rows = [r for r in norms if r["category"] == category]
        words = [r["item"] for r in rows]
        human = np.array([float(r["typicality"]) for r in rows])
        members = member_lemmas(category, extra=words)
        protos = PROTOTYPES[category]

        print(f"\n===== {category}  (n={len(words)}) =====")
        cat_out = {"prompts": {}}
        for prompt in PROMPTS[category]:
            probs = next_token_probs(model, tokenizer, prompt)
            top = topk_items(tokenizer, probs, args.k)
            purity = compute_purity(top, members, LEMMATIZER)

            q, lengths, zipfs = [], [], []
            for word in words:
                prob, n_tokens = word_probability(model, tokenizer, prompt, word)
                q.append(prob)
                lengths.append(n_tokens)
                zipfs.append(zipf_frequency(word, "en"))
            q = np.array(q)
            pr = pearsonr(q, human)
            sp = spearmanr(q, human)
            part = partial_pearson(q, human, np.column_stack([lengths, zipfs]))

            proto_q = {}
            for p in protos:
                if p in words:
                    idx = words.index(p)
                    proto_q[p] = float(q[idx])
            proto_sum = float(sum(proto_q.values()))

            top8 = ", ".join(f"{t}({p:.3f})" for t, p in top[:8])
            print(f"\n  {prompt!r}")
            print(f"    purity={purity['purity']:.3f}  Pearson={pr[0]:+.3f}(p={pr[1]:.3f})  "
                  f"Spearman={sp[0]:+.3f}  partial={part:+.3f}")
            print(f"    top: {top8}")
            print(f"    prototypes ({', '.join(protos)}): "
                  + ", ".join(f"{k}={v:.4f}" for k, v in proto_q.items())
                  + f"  sum={proto_sum:.4f}")

            cat_out["prompts"][prompt] = {
                "purity": purity["purity"],
                "pearson": {"r": float(pr[0]), "p": float(pr[1])},
                "spearman": {"rho": float(sp[0]), "p": float(sp[1])},
                "partial_pearson": float(part),
                "top8": top[:8],
                "prototype_q": proto_q,
                "prototype_sum": proto_sum,
                "per_word": [
                    {"word": w, "human": float(h), "q": float(p), "n_tokens": int(n)}
                    for w, h, p, n in zip(words, human, q, lengths)
                ],
            }
        payload["categories"][category] = cat_out

    out = Path(args.out) if args.out else Path(
        f"results/reformulation_{model_slug(args.model)}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
