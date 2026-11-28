#!/usr/bin/env python
"""Experiment 020: fixed frame, WordNet relations in the slot.

Reads the frozen stimuli (``data/slot_relations.tsv``) and, for every frame,
measures three channels: the full next-token distribution, the layerwise
logit-lens distribution, and the residual-stream state at the slot position.

Protocol: ``../experiment-020-2026-11-28-wordnet-slot-relations.md``.

Run:
    uv run python scripts/run_slot_relations.py --model distilgpt2
    uv run python scripts/run_slot_relations.py --model Qwen/Qwen2.5-1.5B-Instruct
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
from tc.controls import spearman, upper_tri
from tc.slot_relations import (
    FRAMES,
    HARD_CONTROL_RELATION,
    pairwise_rdm,
    analyze_distribution,
    anchor_groups,
    hellinger_rdm,
    increment_stats,
    profile_correlation,
    read_tsv,
    relation_gradient,
    relation_labels,
    relation_names,
    relation_separation,
    state_layer_metrics,
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


def extract(model, tok, texts, label: str = "") -> tuple[np.ndarray, np.ndarray]:
    probs, hidden = [], []
    for k, text in enumerate(texts):
        enc = tok(text, return_tensors="pt")
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        probs.append(torch.softmax(out.logits[0, -1].float(), dim=-1).numpy())
        hidden.append(np.stack([h[0, -1].float().numpy() for h in out.hidden_states]))
        if label and (k + 1) % 50 == 0:
            print(f"    [{label}] {k + 1}/{len(texts)}")
    return np.stack(probs), np.stack(hidden)


def candidate_ids(P, tok, words, topk: int, max_candidates: int) -> list[int]:
    slot_ids = set()
    for word in words:
        slot_ids.update(int(i) for i in tok(word, add_special_tokens=False)["input_ids"])
    chosen = set(slot_ids)
    order = np.argsort(-P, axis=1)[:, :topk]
    for j in order.ravel():
        if len(chosen) >= max_candidates:
            break
        chosen.add(int(j))
    return sorted(chosen)


def logit_lens(H, P, model, tok, words, labels, groups, names,
               topk: int, max_candidates: int) -> list[dict]:
    norm = find_final_norm(model)
    W = model.lm_head.weight.detach()
    ids = torch.as_tensor(candidate_ids(P, tok, words, topk, max_candidates),
                          dtype=torch.long)
    Wc = W[ids]
    final = P[:, ids.numpy()].astype(np.float64)
    final /= final.sum(axis=1, keepdims=True)
    out = []
    for layer in range(H.shape[1]):
        x = torch.as_tensor(H[:, layer, :], dtype=torch.float32)
        with torch.no_grad():
            z = norm(x)
            p = torch.softmax((z @ Wc.T).float(), dim=-1).numpy().astype(np.float64)
        p /= p.sum(axis=1, keepdims=True)
        D = hellinger_rdm(p)
        cross = np.sqrt(np.clip(
            1.0 - (np.sqrt(p) * np.sqrt(final)).sum(axis=1), 0.0, None))
        out.append({
            "layer": layer,
            "separation": relation_separation(D, labels, groups, names),
            "gradient": relation_gradient(D, labels, groups, names),
            "gradient_hard": (
                relation_gradient(D, labels, groups, names,
                                  control=HARD_CONTROL_RELATION)
                if HARD_CONTROL_RELATION in names else {}
            ),
            "mean_distance_to_final": float(cross.mean()),
        })
    return out


def run(model, tok, items, args):
    for it in items:
        it["n_tokens_slot"] = len(tok(it["word"], add_special_tokens=False)["input_ids"])

    frames = [FRAMES[i] for i in args.frames]
    positions = sorted({it["pos"] for it in items})
    result = {
        "model": args.model,
        "data": str(args.data),
        "frames": frames,
        "n_perm": args.n_perm,
        "logit_lens": not args.no_logit_lens,
        "n_items": len(items),
        "per_frame": {},
    }
    for frame in frames:
        frame_result = {}
        for pos in positions:
            arm = [it for it in items if it["pos"] == pos]
            texts = stimuli_text(arm, frame)
            labels = relation_labels(arm)
            groups = anchor_groups(arm)
            names = relation_names(pos, arm)
            print(f"[frame] {frame!r}  pos={pos}  items={len(arm)}")
            P, H = extract(model, tok, texts, label=f"{pos}")

            dist = analyze_distribution(P, arm, n_perm=args.n_perm)
            states = {
                "cosine": state_layer_metrics(H, labels, groups, names, "cosine"),
                "euclid": state_layer_metrics(H, labels, groups, names, "euclid"),
            }
            increment = increment_stats(np.transpose(H, (1, 0, 2)))
            D_dist = hellinger_rdm(P)
            rdm_corr = [
                spearman(upper_tri(D_dist),
                         upper_tri(pairwise_rdm(H[:, layer, :], "cosine")))
                for layer in range(H.shape[1])
            ]
            lens = None
            if not args.no_logit_lens:
                lens = logit_lens(H, P, model, tok, [it["word"] for it in arm],
                                  labels, groups, names,
                                  args.lens_topk, args.lens_max_candidates)
            frame_result[pos] = {
                "n_items": len(arm),
                "names": names,
                "distribution": dist,
                "states": states,
                "increment": increment,
                "rdm_corr_state_vs_dist": rdm_corr,
                "logit_lens": lens,
            }
            grad = dist["raw"]["gradient"]
            grad_h = dist["raw"]["gradient_hard"]
            keys = [k for k in ("sibling", "antonym", "hypernym", "hyponym") if k in grad]
            print("    dist grad(easy): " + "  ".join(
                f"{k}={grad[k]['auc']:.3f}(p={grad[k]['p']:.3f})" for k in keys))
            if grad_h:
                print("    dist grad(hard): " + "  ".join(
                    f"{k}={grad_h[k]['auc']:.3f}(p={grad_h[k]['p']:.3f})"
                    for k in keys if k in grad_h))
            del P, H, states
        result["per_frame"][frame] = frame_result

    result["cross_frame"] = {}
    for pos in positions:
        prof = {
            frame: result["per_frame"][frame][pos]["distribution"]["raw"]["profile"]
            for frame in frames
        }
        result["cross_frame"][pos] = {
            f"{a}__{b}": profile_correlation(prof[a], prof[b])
            for i, a in enumerate(frames) for b in frames[i + 1:]
        }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilgpt2")
    ap.add_argument("--data", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--frames", default="0,1,2")
    ap.add_argument("--limit-anchors", type=int, default=0)
    ap.add_argument("--no-logit-lens", action="store_true")
    ap.add_argument("--lens-topk", type=int, default=50)
    ap.add_argument("--lens-max-candidates", type=int, default=4000)
    args = ap.parse_args()
    args.frames = [int(x) for x in str(args.frames).split(",") if x != ""]

    data = Path(args.data) if args.data else \
        Path(__file__).resolve().parents[1] / "data" / "slot_relations.tsv"
    args.data = data
    items = read_tsv(data)
    if args.limit_anchors:
        items = [it for it in items if it["anchor_id"] < args.limit_anchors]
    print(f"[data] {data}  items={len(items)}")

    torch.manual_seed(0)
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
        f"slot_relations_{model_slug(args.model)}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
