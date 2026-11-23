#!/usr/bin/env python
"""Next-token distribution probe: the surface the model is actually trained on.

For completion-style prompts, print the top-k next tokens and compare the full
next-token distributions between contexts (identical / related / unrelated) via
Jensen-Shannon divergence.

Run:  uv run python scripts/probe_topk.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.distance import jensenshannon
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

TRIPLES = [
    {
        "name": "мед: оса / пчела / вулкан",
        "identical": "Пчела делает мёд, а пчела",
        "related": "Пчела делает мёд, а оса",
        "unrelated": "Пчела делает мёд, а вулкан",
    },
    {
        "name": "отличается: осы / пчелы / вулкана",
        "identical": "Пчела отличается от пчелы тем, что",
        "related": "Пчела отличается от осы тем, что",
        "unrelated": "Пчела отличается от вулкана тем, что",
    },
]


def next_dist(model, tok, text, k=15):
    enc = tok(text, return_tensors="pt")
    with torch.no_grad():
        out = model(**enc)
    logits = out.logits[0, -1]
    probs = torch.softmax(logits, dim=-1).double().numpy()
    top = np.argsort(probs)[::-1][:k]
    return probs, [(tok.decode([int(i)]), float(probs[i])) for i in top]


def js(p, q):
    return float(jensenshannon(p, q) ** 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--k", type=int, default=15)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()
    print(f"[model] {args.model}\n")

    for tri in TRIPLES:
        print(f"===== {tri['name']} =====")
        dists = {}
        for cond in ("identical", "related", "unrelated"):
            prompt = tri[cond]
            probs, top = next_dist(model, tok, prompt, args.k)
            dists[cond] = probs
            shown = ", ".join(f"{t!r}:{p:.3f}" for t, p in top)
            print(f"\n[{cond:9s}] «{prompt}»")
            print(f"  top-{args.k}: {shown}")

        js_rel = js(dists["related"], dists["identical"])
        js_unr = js(dists["related"], dists["unrelated"])
        js_id = js(dists["identical"], dists["unrelated"])
        print(f"\n  JS(related, identical)   = {js_rel:.4f}")
        print(f"  JS(related, unrelated)   = {js_unr:.4f}")
        print(f"  JS(identical, unrelated) = {js_id:.4f}")
        verdict = "related CLOSER to identical" if js_rel < js_unr else "related closer to UNRELATED (!)"
        print(f"  -> {verdict}")
        print()


if __name__ == "__main__":
    main()
