#!/usr/bin/env python
"""Experiment 002c: isolation validation on the instruct model (Qwen).

On an instruct model the natural per-attribute readout is the *judgment*
("Does it ...? Yes/No", P(Yes)/(P(Yes)+P(No))), not a bare completion (which
fails there). For focus attributes (nest, body) and several layers:
* sufficiency + specificity (cross-attribute interchange matrix);
* necessity (ablation from source);
* identifiability (shard principal angle);
* random controls.

Run:  uv run python scripts/run_qwen_validation.py --layers 8,16,20,24
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tc.extract import next_token_probs
from tc.factorial import ATTRIBUTES_ALL, build_corpus, build_pairs, prompt
from tc.interchange import ablated_probs, extract, patched_probs
from tc.metrics import top_indices

QUESTIONS = {
    "diet": "Does this insect hunt other insects?",
    "sociality": "Does this insect live mostly alone?",
    "body": "Does this insect have a smooth body?",
    "nest": "Does this insect build nests of paper?",
    "setting": "Is it found near a mountain?",
}
FOCUS = ["nest", "body"]
N_TR = 18
N_HE = 4


def jp(text, attr):
    return f"{text}\n\n{QUESTIONS[attr]} Answer Yes or No.\nAnswer:"


def yn_ids(tok):
    yes, no = [], []
    for i in range(len(tok)):
        s = tok.decode([i]).strip().lower()
        if s == "yes":
            yes.append(i)
        elif s == "no":
            no.append(i)
    return yes, no


def r_yes(probs, yes, no):
    py = float(np.asarray(probs)[yes].sum())
    pn = float(np.asarray(probs)[no].sum())
    return py / (py + pn + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--layers", default="8,16,20,24")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()
    yes, no = yn_ids(tok)
    D = model.config.hidden_size
    layers = [int(x) for x in args.layers.split(",")]
    print(f"[model] {args.model} layers={layers} yes/no ids={yes[:3]}/{no[:3]}\n")

    corpus = [c for c in build_corpus(ATTRIBUTES_ALL) if c["template"] in {0, 1}]
    pairs = build_pairs(ATTRIBUTES_ALL)
    he = {b: [p for p in pairs if p["attr"] == b and p["template"] == 2][:N_HE] for b in ATTRIBUTES_ALL}

    # held: base/source judgment readout + source hidden (per target b)
    held = {b: [] for b in ATTRIBUTES_ALL}
    for b in ATTRIBUTES_ALL:
        for p in he[b]:
            hb, _ = extract(model, tok, jp(p["base"], b))
            hs, _ = extract(model, tok, jp(p["source"], b))
            rb = r_yes(next_token_probs(model, tok, jp(p["base"], b)), yes, no)
            rs = r_yes(next_token_probs(model, tok, jp(p["source"], b)), yes, no)
            held[b].append({"pb": jp(p["base"], b), "sb": jp(p["source"], b),
                            "rb": rb, "rs": rs, "gap": rs - rb, "hs": hs})
    print("readout gaps by attribute:", {b: round(float(np.mean([h['gap'] for h in held[b]])), 2)
                                         for b in ATTRIBUTES_ALL})

    def closing(target, layer, v):
        vals = []
        for h in held[target]:
            if abs(h["gap"]) < 1e-6:
                continue
            patched = patched_probs(model, tok, h["pb"], layer, v, h["hs"][layer + 1])
            vals.append((r_yes(patched, yes, no) - h["rb"]) / h["gap"])
        return float(np.mean(vals)) if vals else float("nan")

    def necessity(target, layer, v):
        vals = []
        for h in held[target]:
            if abs(h["gap"]) < 1e-6:
                continue
            ab = ablated_probs(model, tok, h["sb"], layer, np.asarray(v).reshape(-1, 1))
            vals.append((h["rs"] - r_yes(ab, yes, no)) / h["gap"])
        return float(np.mean(vals)) if vals else float("nan")

    rng = np.random.default_rng(0)
    for a in FOCUS:
        train = [c for c in corpus if c["values"][a] in (0, 1)]
        pick = [train[i] for i in rng.permutation(len(train))[:N_TR]]
        cache = [(extract(model, tok, jp(c["text"], a))[0], c["values"][a], c["syn"]) for c in pick]
        print(f"=== {a} ===")
        for layer in layers:
            H1 = [h[layer + 1] for h, val, s in cache if val == 1]
            H0 = [h[layer + 1] for h, val, s in cache if val == 0]
            v = np.mean(H1, 0) - np.mean(H0, 0); v /= np.linalg.norm(v)
            v0 = np.mean([h[layer + 1] for h, val, s in cache if s == 0 and val == 1], 0) - \
                 np.mean([h[layer + 1] for h, val, s in cache if s == 0 and val == 0], 0)
            v1 = np.mean([h[layer + 1] for h, val, s in cache if s != 0 and val == 1], 0) - \
                 np.mean([h[layer + 1] for h, val, s in cache if s != 0 and val == 0], 0)
            shard = abs(float(np.dot(v0 / np.linalg.norm(v0), v1 / np.linalg.norm(v1))))
            rv = rng.standard_normal(D); rv /= np.linalg.norm(rv)

            row = {b: closing(b, layer, v) for b in ATTRIBUTES_ALL}
            rand = closing(a, layer, rv)
            nec, necr = necessity(a, layer, v), necessity(a, layer, rv)
            print(f"  L{layer}: diag={row[a]:+.3f} rand={rand:+.3f} " +
                  " ".join(f"{b}={row[b]:+.2f}" for b in ATTRIBUTES_ALL if b != a) +
                  f" | nec={nec:+.3f}/{necr:+.3f} shard={shard:.2f}")


if __name__ == "__main__":
    main()
