#!/usr/bin/env python
"""Experiment 002, phase 0: attribute localization by diff-in-means interchange.

For each binary attribute: compute the difference-in-means direction of the
last-position residual stream at each block, then interchange that direction
from a source run (attr=1) into a base run (attr=0) on held-out minimal pairs.
Effect = fraction of the base->source next-token JS that closes. Specificity
matrix S[a][b]: direction of attribute a applied to pairs differing in b
(diagonal should be larger if the attribute is isolated). Random-direction
baseline included.

Sandbox model: distilgpt2 (cached GPT-2-family small), CPU.

Run:  uv run python scripts/run_localization.py
      uv run python scripts/run_localization.py --layers 0,2,4,5   # quick
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tc.factorial import ATTRS, PROBE, build_corpus, build_pairs, prompt
from tc.interchange import extract, get_blocks, patched_probs
from tc.metrics import js_topk

TRAIN_TEMPLATES = {0, 1}
HELD_TEMPLATE = 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilgpt2")
    ap.add_argument("--layers", default="", help="comma list of block indices (default all)")
    ap.add_argument("--max-pairs", type=int, default=12, help="held-out pairs per attribute")
    ap.add_argument("--probe", default=PROBE)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--out", default="results/localization_phase0.json")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()
    n_blocks = len(get_blocks(model))
    layers = [int(x) for x in args.layers.split(",")] if args.layers else list(range(n_blocks))
    print(f"[model] {args.model} blocks={n_blocks}  layers={layers}  probe={args.probe!r}\n")

    corpus = build_corpus()
    pairs = build_pairs()

    # cache last-position hidden states + next-token probs for every prompt
    cache = {}
    for c in corpus:
        kw = prompt(c["text"], args.probe)
        if kw not in cache:
            cache[kw] = extract(model, tok, kw)
    print(f"cached {len(cache)} prompts")

    train = [c for c in corpus if c["template"] in TRAIN_TEMPLATES]
    dirs = {a: {} for a in ATTRS}
    for a in ATTRS:
        h0 = [cache[prompt(c["text"], args.probe)][0] for c in train if c["values"][a] == 0]
        h1 = [cache[prompt(c["text"], args.probe)][0] for c in train if c["values"][a] == 1]
        for layer in layers:
            v = np.mean([h[layer + 1] for h in h1], 0) - np.mean([h[layer + 1] for h in h0], 0)
            n = np.linalg.norm(v)
            dirs[a][layer] = v / n if n > 0 else v
    print("directions ready\n")

    held = [p for p in pairs if p["template"] == HELD_TEMPLATE]
    held_by_attr = {a: [p for p in held if p["attr"] == a][: args.max_pairs] for a in ATTRS}

    rng = np.random.default_rng(0)
    d = len(dirs[ATTRS[0]][layers[0]])
    rand_vec = {layer: (lambda u: u / np.linalg.norm(u))(rng.standard_normal(d)) for layer in layers}

    S = {layer: {a: {} for a in ATTRS} for layer in layers}
    rand_diag = {layer: {} for layer in layers}
    js_sep = {a: [] for a in ATTRS}

    for layer in layers:
        for a in ATTRS:
            for b in ATTRS:
                closes = []
                for p in held_by_attr[b]:
                    kb, ks = prompt(p["base"], args.probe), prompt(p["source"], args.probe)
                    p0, p1 = cache[kb][1], cache[ks][1]
                    js0 = js_topk(p0, p1, args.k)
                    if js0 < 1e-6:
                        continue
                    src = cache[ks][0][layer + 1]
                    patched = patched_probs(model, tok, kb, layer, dirs[a][layer], src)
                    js1 = js_topk(patched, p1, args.k)
                    closes.append((js0 - js1) / js0)
                    if a == b:
                        js_sep[a].append(js0)
                S[layer][a][b] = float(np.mean(closes)) if closes else float("nan")
            # random baseline on this attribute's own pairs
            closes = []
            for p in held_by_attr[a]:
                kb, ks = prompt(p["base"], args.probe), prompt(p["source"], args.probe)
                p0, p1 = cache[kb][1], cache[ks][1]
                js0 = js_topk(p0, p1, args.k)
                if js0 < 1e-6:
                    continue
                src = cache[ks][0][layer + 1]
                patched = patched_probs(model, tok, kb, layer, rand_vec[layer], src)
                closes.append((js0 - js_topk(patched, p1, args.k)) / js0)
            rand_diag[layer][a] = float(np.mean(closes)) if closes else float("nan")
        diag = np.mean([S[layer][a][a] for a in ATTRS])
        off = np.mean([S[layer][a][b] for a in ATTRS for b in ATTRS if a != b])
        print(f"layer {layer}: diag={diag:+.3f}  offdiag={off:+.3f}  random={np.mean(list(rand_diag[layer].values())):+.3f}")

    # best layer per attribute (max diagonal)
    best = {}
    for a in ATTRS:
        layer = max(layers, key=lambda l: (S[l][a][a] if not np.isnan(S[l][a][a]) else -9))
        best[a] = {
            "layer": layer,
            "diag": S[layer][a][a],
            "row": {b: S[layer][a][b] for b in ATTRS},
            "random": rand_diag[layer][a],
            "js_separation": float(np.mean(js_sep[a])) if js_sep[a] else float("nan"),
        }

    print("\nbest layer per attribute (diag = own effect, row = specificity, random = control):")
    for a in ATTRS:
        b = best[a]
        row = "  ".join(f"{k}={v:+.3f}" for k, v in b["row"].items())
        print(f"  {a:9s} L{b['layer']:2d}  diag={b['diag']:+.3f}  random={b['random']:+.3f}  "
              f"js_sep={b['js_separation']:.3f}  [{row}]")

    s = S  # JSON-friendly
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": args.model, "probe": args.probe, "k": args.k, "layers": layers,
        "specificity": {str(l): s[l] for l in layers},
        "random_diag": {str(l): rand_diag[l] for l in layers},
        "best": best,
    }, indent=2, ensure_ascii=False))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
