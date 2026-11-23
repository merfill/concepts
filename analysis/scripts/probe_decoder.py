#!/usr/bin/env python
"""Small Russian decoder probe: generate an answer and inspect tokens/attention.

For the comparison prompt, we (1) generate a short continuation, (2) report the
per-layer cosine between the two compared words, and (3) the last-layer
attention between them and from the verb.

Run:  uv run python scripts/probe_decoder.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tc.joint import cosine_sim

MODEL = "ai-forever/rugpt3small_based_on_gpt2"

PROMPTS = [
    ("identical", "Чем пчела отличается от пчелы?", "пчела", "пчелы"),
    ("related", "Чем пчела отличается от осы?", "пчела", "осы"),
    ("unrelated", "Чем пчела отличается от вулкана?", "пчела", "вулкана"),
]


def span_positions(offsets, text, word):
    s = text.find(word)
    if s < 0:
        return []
    e = s + len(word)
    return [i for i, (a, b) in enumerate(offsets) if b > a and not (b <= s or a >= e)]


def mean_rep(V, pos):
    return V[pos].mean(0) if pos else np.zeros(V.shape[1])


def main():
    torch.use_deterministic_algorithms(True, warn_only=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, attn_implementation="eager")
    model.eval()
    n_layers = model.config.num_hidden_layers

    print("=== generation ===")
    for label, prompt, wa, wb in PROMPTS:
        ids = tok(prompt, return_tensors="pt")
        with torch.no_grad():
            gen = model.generate(**ids, max_new_tokens=30, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        text = tok.decode(gen[0], skip_special_tokens=True)
        print(f"\n[{label}] {text!r}")

    print("\n=== per-layer cosine between compared words ===")
    print(f"{'label':10s} " + " ".join(f"L{l:<5d}" for l in range(n_layers + 1)))
    cache = {}
    for label, prompt, wa, wb in PROMPTS:
        enc = tok(prompt, return_tensors="pt", return_offsets_mapping=True)
        offsets = enc["offset_mapping"][0].tolist()
        fwd = {k: v for k, v in enc.items() if k != "offset_mapping"}
        with torch.no_grad():
            out = model(**fwd, output_hidden_states=True, output_attentions=True)
        pa = span_positions(offsets, prompt, wa)
        pb = span_positions(offsets, prompt, wb)
        if not pa or not pb:
            print(f"  [warn] span not found for {label}: {wa}={pa} {wb}={pb}")
        cos = []
        for layer in range(n_layers + 1):
            V = out.hidden_states[layer][0].double().numpy()
            cos.append(cosine_sim(mean_rep(V, pa), mean_rep(V, pb)))
        cache[label] = (out, offsets, pa, pb, prompt)
        print(f"{label:10s} " + " ".join(f"{c:+.3f}" for c in cos))

    print("\n=== last-layer attention ===")
    for label, prompt, wa, wb in PROMPTS:
        out, offsets, pa, pb, _ = cache[label]
        A = out.attentions[-1][0].mean(0).double().numpy()
        ab = A[np.ix_(pb, pa)].mean()  # later word -> earlier word (causal)
        pp = span_positions(offsets, prompt, "отличается")
        verb_a = A[np.ix_(pp, pa)].mean() if pp else float("nan")
        verb_b = A[np.ix_(pp, pb)].mean() if pp else float("nan")
        last = A[-1]
        last_a = last[pa].mean()
        last_b = last[pb].mean()
        print(f"  {label:10s} {wb}->{wa}={ab:.3f}  «отличается»->{wa}={verb_a:.3f} "
              f"->{wb}={verb_b:.3f}  «?»->{wa}={last_a:.3f} ->{wb}={last_b:.3f}")


if __name__ == "__main__":
    main()
