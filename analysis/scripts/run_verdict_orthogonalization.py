#!/usr/bin/env python
"""Experiment 002d: decouple attribute content from the Yes/No verdict direction.

002c showed that at the judgment answer position a single shared direction (the
verdict Yes<->No) closes every attribute. Here we *remove* that shared component:
v_common = normalize(sum_a v_a); v_a_perp = v_a - (v_a . v_common) v_common.
Then re-measure specificity. If off-diagonal drops while the diagonal survives,
the content is separable from the decision.

Run:  uv run python scripts/run_verdict_orthogonalization.py --layers 20,24
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tc.extract import next_token_probs
from tc.factorial import ATTRIBUTES_ALL, build_corpus, build_pairs
from tc.interchange import extract, patched_probs

QUESTIONS = {
    "diet": "Does this insect hunt other insects?",
    "sociality": "Does this insect live mostly alone?",
    "body": "Does this insect have a smooth body?",
    "nest": "Does this insect build nests of paper?",
    "setting": "Is it found near a mountain?",
}
N_TR = 16
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
    py = float(np.asarray(probs)[yes].sum()); pn = float(np.asarray(probs)[no].sum())
    return py / (py + pn + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--layers", default="20,24")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()
    yes, no = yn_ids(tok)
    layers = [int(x) for x in args.layers.split(",")]
    rng = np.random.default_rng(0)
    print(f"[model] {args.model} layers={layers}\n")

    corpus = [c for c in build_corpus(ATTRIBUTES_ALL) if c["template"] in {0, 1}]
    pairs = build_pairs(ATTRIBUTES_ALL)
    he = {b: [p for p in pairs if p["attr"] == b and p["template"] == 2][:N_HE] for b in ATTRIBUTES_ALL}

    held = {}
    for b in ATTRIBUTES_ALL:
        held[b] = []
        for p in he[b]:
            hs, _ = extract(model, tok, jp(p["source"], b))
            rb = r_yes(next_token_probs(model, tok, jp(p["base"], b)), yes, no)
            rs = r_yes(next_token_probs(model, tok, jp(p["source"], b)), yes, no)
            held[b].append({"pb": jp(p["base"], b), "rb": rb, "rs": rs,
                            "gap": rs - rb, "hs": hs})

    # per-attribute directions from train scenes
    def attr_dir(a, layer):
        train = [c for c in corpus if c["values"][a] in (0, 1)]
        pick = [train[i] for i in rng.permutation(len(train))[:N_TR]]
        cache = [(extract(model, tok, jp(c["text"], a))[0][layer + 1], c["values"][a]) for c in pick]
        v = np.mean([h for h, val in cache if val == 1], 0) - np.mean([h for h, val in cache if val == 0], 0)
        return v / np.linalg.norm(v)

    def closing(target, layer, v):
        vals = []
        for h in held[target]:
            if abs(h["gap"]) < 1e-6:
                continue
            patched = patched_probs(model, tok, h["pb"], layer, v, h["hs"][layer + 1])
            vals.append((r_yes(patched, yes, no) - h["rb"]) / h["gap"])
        return float(np.mean(vals)) if vals else float("nan")

    for layer in layers:
        V = {a: attr_dir(a, layer) for a in ATTRIBUTES_ALL}
        vc = np.sum([V[a] for a in ATTRIBUTES_ALL], 0); vc /= np.linalg.norm(vc)
        print(f"=== layer {layer} ===  cos(v_a, v_verdict): " +
              " ".join(f"{a}={float(np.dot(V[a], vc)):.2f}" for a in ATTRIBUTES_ALL))
        S, Sp = {}, {}
        for a in ATTRIBUTES_ALL:
            vp = V[a] - np.dot(V[a], vc) * vc; vp /= np.linalg.norm(vp)
            S[a] = {b: closing(b, layer, V[a]) for b in ATTRIBUTES_ALL}
            Sp[a] = {b: closing(b, layer, vp) for b in ATTRIBUTES_ALL}
        for a in ATTRIBUTES_ALL:
            diag = S[a][a]; off = np.mean([S[a][b] for b in ATTRIBUTES_ALL if b != a])
            diagp = Sp[a][a]; offp = np.mean([Sp[a][b] for b in ATTRIBUTES_ALL if b != a])
            print(f"  {a:9s} orig diag={diag:+.2f} off={off:+.2f} | perp diag={diagp:+.2f} off={offp:+.2f}")


if __name__ == "__main__":
    main()
