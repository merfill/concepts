#!/usr/bin/env python
"""Small instruction model probe (Russian): does it actually compare, and where?

For each prompt we (1) let the model answer, (2) inspect the per-layer cosine
between the two compared words in the user turn, and (3) the last-layer
attention between them.

Run:  uv run python scripts/probe_instruct.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tc.joint import cosine_sim

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

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
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()
    torch.use_deterministic_algorithms(True, warn_only=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, attn_implementation="eager")
    model.eval()
    n_layers = model.config.num_hidden_layers
    print(f"[model] {args.model}")

    print("=== answers ===")
    for label, prompt, _, _ in PROMPTS:
        messages = [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        enc = tok(text, return_tensors="pt", add_special_tokens=False)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=120, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        new = gen[0][enc["input_ids"].shape[1]:]
        print(f"\n[{label}] {prompt}\n  -> {tok.decode(new, skip_special_tokens=True).strip()!r}")

    print("\n=== per-layer cosine between compared words (user turn) ===")
    print(f"{'label':10s} " + " ".join(f"L{l:<4d}" for l in range(0, n_layers + 1, 2)))
    cache = {}
    for label, prompt, wa, wb in PROMPTS:
        messages = [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        enc = tok(text, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False)
        offsets = enc["offset_mapping"][0].tolist()
        fwd = {k: v for k, v in enc.items() if k != "offset_mapping"}
        with torch.no_grad():
            out = model(**fwd, output_hidden_states=True, output_attentions=True)
        pa = span_positions(offsets, text, wa)
        pb = span_positions(offsets, text, wb)
        cos = []
        for layer in range(n_layers + 1):
            V = out.hidden_states[layer][0].double().numpy()
            cos.append(cosine_sim(mean_rep(V, pa), mean_rep(V, pb)))
        cache[label] = (out, offsets, pa, pb, text)
        sel = cos[::2]
        print(f"{label:10s} " + " ".join(f"{c:+.3f}" for c in sel))

    print("\n=== last-layer attention ===")
    for label, prompt, wa, wb in PROMPTS:
        out, offsets, pa, pb, text = cache[label]
        A = out.attentions[-1][0].mean(0).double().numpy()
        ab = A[np.ix_(pb, pa)].mean()
        verb = span_positions(offsets, text, "отличается")
        va = A[np.ix_(verb, pa)].mean() if verb else float("nan")
        vb = A[np.ix_(verb, pb)].mean() if verb else float("nan")
        print(f"  {label:10s} {wb}->{wa}={ab:.3f}  «отличается»->{wa}={va:.3f} ->{wb}={vb:.3f}")


if __name__ == "__main__":
    main()
