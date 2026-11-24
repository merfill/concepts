#!/usr/bin/env python
"""Experiment 002: find a stronger readout probe for weak attributes.

For attributes whose completion probe barely separates base/source (body,
setting), scan candidate probe phrasings: measure mean JS(base, source) of the
next-token distribution on train minimal pairs (select) and held-out pairs
(report). Selection is done on train; held-out is the honest number.

Run:  uv run python scripts/run_probe_search.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tc.factorial import ATTRIBUTES_ALL, PROBE, build_corpus, build_pairs, prompt
from tc.interchange import extract
from tc.metrics import js_topk

CANDIDATES = {
    "body": [
        "Its body is", "The body is", "Its skin is", "It has a", "Its body looks",
        "The insect's body is", "It is covered with", "Its surface is", "The body of this insect is",
    ],
    "setting": [
        "It is found", "It lives", "It is located", "You can find it", "It is usually found",
        "Its habitat is", "It occurs", "It can be found", "It is found mostly",
    ],
}
N = 8


def sample_pairs(pairs, attr, template_set, n):
    ps = [p for p in pairs if p["attr"] == attr and p["template"] in template_set]
    return ps[:n]


def sep(model, tok, pairs, probe, k=50):
    vals = []
    for p in pairs:
        _, pb = extract(model, tok, prompt(p["base"], probe))
        _, ps = extract(model, tok, prompt(p["source"], probe))
        vals.append(js_topk(pb, ps, k))
    return float(np.mean(vals)) if vals else float("nan")


def main():
    tok = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    model.eval()

    pairs = build_pairs(ATTRIBUTES_ALL)
    for attr, cands in CANDIDATES.items():
        tr = sample_pairs(pairs, attr, {0, 1}, N)
        he = sample_pairs(pairs, attr, {2}, N)
        rows = []
        for pr in cands:
            rows.append((pr, sep(model, tok, tr, pr), sep(model, tok, he, pr)))
        rows.sort(key=lambda r: r[1], reverse=True)
        print(f"\n=== {attr} (train/held js_sep) ===")
        for pr, a, b in rows:
            print(f"  {pr:28s} train={a:.4f}  held={b:.4f}")


if __name__ == "__main__":
    main()
