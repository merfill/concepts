#!/usr/bin/env python
"""Experiment 022: where does hidden-state geometry agree with the output?

One forward per stimulus; compare pairwise hidden distances with pairwise output
distances by layer, split the last step (pre/post norm, penultimate/last block),
and project the last layer onto the top right singular directions of ``W_U``.

Protocol: ``../experiment-022-2026-11-28-readout-geometry.md``.

Run:
    uv run python scripts/run_readout_geometry.py --model distilgpt2
    uv run python scripts/run_readout_geometry.py --model Qwen/Qwen2.5-1.5B-Instruct
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tc.causal_ablation import random_subspace
from tc.readout_geometry import (
    captured_fraction,
    layer_alignment,
    project,
    rdm_corr,
    top_directions_from_cov,
)
from tc.slot_relations import (
    FRAMES,
    hellinger_rdm,
    pairwise_rdm,
    read_tsv,
    stimuli_text,
)


def model_slug(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_")


def find_final_norm(model):
    for path in ("model.norm", "transformer.ln_f", "gpt_neox.final_layer_norm"):
        obj, ok = model, True
        for part in path.split("."):
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                ok = False
                break
        if ok:
            return obj
    raise AttributeError("final norm not found")


def extract_all(model, tok, texts, label: str = ""):
    hidden, probs = [], []
    for k, text in enumerate(texts):
        enc = tok(text, return_tensors="pt")
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        hidden.append(np.stack([h[0, -1, :].double().numpy() for h in out.hidden_states]))
        probs.append(torch.softmax(out.logits[0, -1].float(), dim=-1).numpy())
        if label and (k + 1) % 50 == 0:
            print(f"  [{label}] {k + 1}/{len(texts)}")
    return np.stack(hidden), np.stack(probs)


def run(model, tok, items, args):
    pos_items = [it for it in items if it["pos"] == args.pos]
    if args.limit_anchors:
        pos_items = [it for it in pos_items if it["anchor_id"] < args.limit_anchors]
    frame = FRAMES[args.frame]
    texts = stimuli_text(pos_items, frame)
    print(f"[items] {len(pos_items)}  pos={args.pos}  frame={frame!r}")

    H, P = extract_all(model, tok, texts, label=args.pos)
    n, n_states, d = H.shape
    D_out = hellinger_rdm(P)
    layer_rho = layer_alignment(H, D_out)
    h_last = H[:, -1, :]

    norm = find_final_norm(model)
    with torch.no_grad():
        normed = norm(torch.as_tensor(h_last, dtype=torch.float32)).double().cpu().numpy()
    rho_pre = rdm_corr(pairwise_rdm(h_last, "cosine"), D_out)
    rho_post = rdm_corr(pairwise_rdm(normed, "cosine"), D_out)

    W = model.lm_head.weight.detach().float()
    WtW = (W.T @ W).double().cpu().numpy()
    V_top, eigvals = top_directions_from_cov(WtW, k=args.k)
    capture = captured_fraction(eigvals, args.k)
    Z_top = project(h_last, V_top)
    rho_top = rdm_corr(pairwise_rdm(Z_top, "cosine"), D_out)

    rand_rhos = []
    for s in range(args.n_random):
        V_r = random_subspace(d, args.k, args.seed + 100 + s)
        Z_r = project(h_last, V_r)
        rand_rhos.append(rdm_corr(pairwise_rdm(Z_r, "cosine"), D_out))

    result = {
        "model": args.model,
        "pos": args.pos,
        "frame": frame,
        "n_items": len(pos_items),
        "d": int(d),
        "n_states": int(n_states),
        "k": args.k,
        "layer_alignment": layer_rho,
        "last_step": {
            "rho_pre_norm": rho_pre,
            "rho_post_norm": rho_post,
            "delta_norm": rho_post - rho_pre,
        },
        "last_block": {
            "rho_penultimate": layer_rho[-2] if len(layer_rho) >= 2 else float("nan"),
            "rho_last": layer_rho[-1],
            "delta_last_block": (
                layer_rho[-1] - layer_rho[-2] if len(layer_rho) >= 2 else float("nan")
            ),
        },
        "unembedding": {
            "captured_fraction_top_k": capture,
            "rho_top_k": rho_top,
            "rho_full": rho_pre,
            "rho_random_mean": float(np.mean(rand_rhos)),
            "rho_random_sd": float(np.std(rand_rhos)),
            "random_values": rand_rhos,
        },
    }
    print(f"  layers: first={layer_rho[0]:+.3f}  penultimate={layer_rho[-2]:+.3f}  "
          f"last={layer_rho[-1]:+.3f}  max=L{int(np.nanargmax(layer_rho))}:{np.nanmax(layer_rho):+.3f}")
    print(f"  norm: pre={rho_pre:+.3f}  post={rho_post:+.3f}  delta={rho_post - rho_pre:+.3f}")
    print(f"  last block: penult={layer_rho[-2]:+.3f}  last={layer_rho[-1]:+.3f}  "
          f"delta={layer_rho[-1] - layer_rho[-2]:+.3f}")
    print(f"  W_U top-{args.k}: capture={capture:.3f}  rho_top={rho_top:+.3f}  "
          f"random={np.mean(rand_rhos):+.3f}±{np.std(rand_rhos):.3f}  full={rho_pre:+.3f}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilgpt2")
    ap.add_argument("--data", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--pos", default="noun", choices=["noun", "adjective"])
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--n-random", type=int, default=5)
    ap.add_argument("--limit-anchors", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data = Path(args.data) if args.data else \
        Path(__file__).resolve().parents[1] / "data" / "slot_relations.tsv"
    items = read_tsv(data)

    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    model.eval()
    dtype = str(next(model.parameters()).dtype).replace("torch.", "")
    print(f"[model] {args.model}  dtype={dtype}")

    result = run(model, tok, items, args)
    result["dtype"] = dtype
    out = Path(args.out) if args.out else \
        Path(__file__).resolve().parents[1] / "results" / \
        f"readout_geometry_{model_slug(args.model)}_{args.pos}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
