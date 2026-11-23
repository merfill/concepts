#!/usr/bin/env python
"""Direct probe: feed a comparison prompt and look at tokens + attention.

Russian prompts, multilingual model. For each prompt we report, per layer, the
cosine between the contextual representations of the two compared words, and at
the last layer the cross-attention between them and how the verb "отличается"
attends to each.

Run:  uv run python scripts/probe_compare.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tc.extract import load
from tc.joint import cosine_sim


PROMPTS = [
    # (group, label, prompt, wordA, wordB)
    ("bee", "identical", "Чем пчела отличается от пчелы?", "пчела", "пчелы"),
    ("bee", "related", "Чем пчела отличается от осы?", "пчела", "осы"),
    ("bee", "unrelated", "Чем пчела отличается от вулкана?", "пчела", "вулкана"),
    ("river", "identical", "Чем река отличается от реки?", "река", "реки"),
    ("river", "related", "Чем река отличается от ручья?", "река", "ручья"),
    ("river", "unrelated", "Чем река отличается от гитары?", "река", "гитары"),
]


def span_positions(offsets, text, word):
    s = text.find(word)
    if s < 0:
        raise ValueError(f"word {word!r} not found in {text!r}")
    e = s + len(word)
    return [i for i, (a, b) in enumerate(offsets) if b > a and not (b <= s or a >= e)]


def mean_rep(V, pos):
    return V[pos].mean(0) if pos else np.zeros(V.shape[1])


def main():
    model, tok = load("distilbert-base-multilingual-cased")

    print("=== per-letter reference: tokenization ===")
    for _, _, prompt, wa, wb in PROMPTS:
        enc = tok(prompt, return_offsets_mapping=True)
        print(f"  {prompt}")
        print(f"    tokens: {enc.tokens()}")

    print("\n=== cosine between compared words, by layer ===")
    print(f"{'group':7s} {'label':10s} " + " ".join(f"L{l:<5d}" for l in range(model.config.num_hidden_layers + 1)))

    cache = {}
    for group, label, prompt, wa, wb in PROMPTS:
        enc = tok(prompt, return_tensors="pt", return_offsets_mapping=True)
        offsets = enc["offset_mapping"][0].tolist()
        with torch.no_grad():
            out = model(**{k: v for k, v in enc.items() if k != "offset_mapping"},
                        output_hidden_states=True, output_attentions=True)
        pa = span_positions(offsets, prompt, wa)
        pb = span_positions(offsets, prompt, wb)
        cos_by_layer = []
        for layer in range(model.config.num_hidden_layers + 1):
            V = out.hidden_states[layer][0].double().numpy()
            cos_by_layer.append(cosine_sim(mean_rep(V, pa), mean_rep(V, pb)))
        cache[(group, label)] = (cos_by_layer, out, enc, offsets, pa, pb, prompt)
        print(f"{group:7s} {label:10s} " + " ".join(f"{c:+.3f}" for c in cos_by_layer))

    print("\n=== last-layer cross-attention (mean over heads) ===")
    for group, label, prompt, wa, wb in PROMPTS:
        cos_by_layer, out, enc, offsets, pa, pb, _ = cache[(group, label)]
        A = out.attentions[-1][0].mean(0).double().numpy()
        ab = A[np.ix_(pb, pa)].mean()
        ba = A[np.ix_(pa, pb)].mean()
        pp = span_positions(offsets, prompt, "отличается")
        verb_to_a = A[np.ix_(pp, pa)].mean() if pp else float("nan")
        verb_to_b = A[np.ix_(pp, pb)].mean() if pp else float("nan")
        print(f"  {group:7s} {label:10s} {wa}->{wb}={ab:.3f}  {wb}->{wa}={ba:.3f}  "
              f"«отличается»->{wa}={verb_to_a:.3f}  ->{wb}={verb_to_b:.3f}")


if __name__ == "__main__":
    main()
