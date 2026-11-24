#!/usr/bin/env python
"""Experiment 002, phase 0.5: stronger readout + proper controls.

Phase 0 used the weak generic probe "It is a" (attribute barely changed the
continuation) and only a random-direction control. Here:

* readout = attribute-specific *completion* probe (predict the attribute's own
  content word from the stated clause), e.g. "Its body is" -> hairy/smooth.
  Works on a base LM (no instruction-following needed) and gives a real gap.
* 5th attribute ``setting`` (near a lake / a mountain), unrelated to the insect,
  as a transfer control (its direction must stay isolated and must not close the
  insect attributes, and vice versa).
* controls: random direction, label-shuffle null, grand-mean ("always-on")
  direction.

Run:  uv run python scripts/run_localization_phase05.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tc.factorial import ATTRIBUTES_ALL, build_corpus, build_pairs, prompt
from tc.interchange import direction, extract, get_blocks, patched_probs
from tc.metrics import js_topk

TRAIN_TEMPLATES = {0, 1}
HELD_TEMPLATE = 2
PROBES = {
    "diet": "It feeds on",
    "sociality": "It lives",
    "body": "It has a",
    "nest": "It builds",
    "setting": "It is found",
}


def mean_closing(dirs, layer, cache, b, k, model, tok):
    closes = []
    for c in cache[b]:
        p = c["base_prompt"]
        patched = patched_probs(model, tok, p, layer, dirs[layer], c["source_hidden"][layer + 1])
        closes.append((c["js0"] - js_topk(patched, c["source_probs"], k)) / c["js0"])
    return float(np.mean(closes)) if closes else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilgpt2")
    ap.add_argument("--layers", default="")
    ap.add_argument("--max-pairs", type=int, default=10)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--out", default="results/localization_phase05.json")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()
    n_blocks = len(get_blocks(model))
    # exclude the last block: patching right before final LayerNorm is off-manifold
    layers = [int(x) for x in args.layers.split(",")] if args.layers else list(range(n_blocks - 1))
    print(f"[model] {args.model} blocks={n_blocks} layers={layers}\n")

    corpus = build_corpus(ATTRIBUTES_ALL)
    pairs = build_pairs(ATTRIBUTES_ALL)
    train = [c for c in corpus if c["template"] in TRAIN_TEMPLATES]

    # directions per attribute (own completion probe), + shuffle and grand-mean
    rng = np.random.default_rng(0)
    dirs, dirs_shuf, dirs_gm = {}, {}, {}
    for a in ATTRIBUTES_ALL:
        H, labels = [], []
        for c in train:
            h, _ = extract(model, tok, prompt(c["text"], PROBES[a]))
            H.append(h)
            labels.append(c["values"][a])
        dirs[a] = {}
        dirs_shuf[a] = {}
        dirs_gm[a] = {}
        perm = rng.permutation(len(labels))
        shuffled = [labels[i] for i in perm]
        for layer in layers:
            idx = layer + 1
            hs = [h[idx] for h in H]
            dirs[a][layer] = direction(hs, labels)
            dirs_shuf[a][layer] = direction(hs, shuffled)
            gm = np.mean(hs, 0)
            dirs_gm[a][layer] = gm / (np.linalg.norm(gm) + 1e-12)
    print("directions ready\n")

    # held-out pairs cache under each target attribute's own probe
    held = [p for p in pairs if p["template"] == HELD_TEMPLATE]
    cache = {b: [] for b in ATTRIBUTES_ALL}
    for b in ATTRIBUTES_ALL:
        for p in [q for q in held if q["attr"] == b][: args.max_pairs]:
            base_prompt = prompt(p["base"], PROBES[b])
            src_prompt = prompt(p["source"], PROBES[b])
            bh, bp = extract(model, tok, base_prompt)
            sh, sp = extract(model, tok, src_prompt)
            js0 = js_topk(bp, sp, args.k)
            if js0 < 1e-6:
                continue
            cache[b].append({"base_prompt": base_prompt, "source_hidden": sh,
                             "source_probs": sp, "js0": js0})

    js_sep = {b: float(np.mean([c["js0"] for c in cache[b]])) for b in ATTRIBUTES_ALL}
    print("readout separation js_sep:", {b: round(v, 4) for b, v in js_sep.items()}, "\n")

    # random directions per layer (same dim)
    d = len(dirs[ATTRIBUTES_ALL[0]][layers[0]])
    rand = {layer: (lambda u: u / np.linalg.norm(u))(rng.standard_normal(d)) for layer in layers}

    S = {layer: {a: {} for a in ATTRIBUTES_ALL} for layer in layers}
    ctrl = {layer: {"random": {}, "shuffle": {}, "grandmean": {}} for layer in layers}
    for layer in layers:
        for a in ATTRIBUTES_ALL:
            for b in ATTRIBUTES_ALL:
                S[layer][a][b] = mean_closing(dirs[a], layer, cache, b, args.k, model, tok)
            ctrl[layer]["random"][a] = mean_closing(
                {layer: rand[layer]}, layer, cache, a, args.k, model, tok)
            ctrl[layer]["shuffle"][a] = mean_closing(dirs_shuf[a], layer, cache, a, args.k, model, tok)
            ctrl[layer]["grandmean"][a] = mean_closing(dirs_gm[a], layer, cache, a, args.k, model, tok)
        diag = np.mean([S[layer][a][a] for a in ATTRIBUTES_ALL])
        off = np.mean([S[layer][a][b] for a in ATTRIBUTES_ALL for b in ATTRIBUTES_ALL if a != b])
        cm = {k: np.mean(list(v.values())) for k, v in ctrl[layer].items()}
        print(f"layer {layer}: diag={diag:+.3f} offdiag={off:+.3f} "
              f"random={cm['random']:+.3f} shuffle={cm['shuffle']:+.3f} grandmean={cm['grandmean']:+.3f}")

    best = {}
    for a in ATTRIBUTES_ALL:
        layer = max(layers, key=lambda l: S[l][a][a] if not np.isnan(S[l][a][a]) else -9)
        best[a] = {"layer": layer, "diag": S[layer][a][a],
                   "row": {b: S[layer][a][b] for b in ATTRIBUTES_ALL},
                   "random": ctrl[layer]["random"][a],
                   "shuffle": ctrl[layer]["shuffle"][a],
                   "grandmean": ctrl[layer]["grandmean"][a],
                   "js_sep": js_sep[a]}

    print("\nbest layer per attribute:")
    for a in ATTRIBUTES_ALL:
        b = best[a]
        row = " ".join(f"{k[:4]}={v:+.2f}" for k, v in b["row"].items())
        print(f"  {a:9s} L{b['layer']:2d} diag={b['diag']:+.3f} rand={b['random']:+.3f} "
              f"shuf={b['shuffle']:+.3f} gm={b['grandmean']:+.3f} js_sep={b['js_sep']:.3f} | {row}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": args.model, "layers": layers, "k": args.k,
        "js_sep": js_sep,
        "specificity": {str(l): S[l] for l in layers},
        "controls": {str(l): ctrl[l] for l in layers},
        "best": best,
    }, indent=2, ensure_ascii=False))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
