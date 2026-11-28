#!/usr/bin/env python
"""Experiment 021: is the supervised relation subspace causally used?

Leave-one-anchor-out Fisher subspace on hidden states, projected out of the
residual stream at a layer; the relation gradient is compared with random and
form subspaces of the same rank.

Protocol: ``../experiment-021-2026-11-28-causal-ablation.md``.

Run:
    uv run python scripts/run_causal_ablation.py --model distilgpt2
    uv run python scripts/run_causal_ablation.py --model Qwen/Qwen2.5-1.5B-Instruct --pos adjective
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
from tc.causal_ablation import (
    bootstrap_diff_ci,
    direction_auc,
    fisher_subspace,
    form_subspace,
    gradient_summary,
    orthonormal_columns,
    random_subspace,
    removed_fraction,
)
from tc.interchange import ablated_probs, extract, get_blocks
from tc.slot_relations import (
    FRAMES,
    HARD_CONTROL_RELATION,
    anchor_groups,
    hellinger_rdm,
    read_tsv,
    relation_labels,
    stimuli_text,
)


def model_slug(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_")


def select_arm(items, pos: str, relation: str, control: str):
    keep_rels = {"self", relation, control}
    arm = [it for it in items if it["pos"] == pos and it["relation"] in keep_rels]
    by = {}
    for it in arm:
        by.setdefault(it["anchor_id"], {}).setdefault(it["relation"], 0)
        by[it["anchor_id"]][it["relation"]] += 1
    anchors = {
        a for a, d in by.items()
        if d.get("self") and d.get(relation) and d.get(control)
    }
    return [it for it in arm if it["anchor_id"] in anchors]


def pad_subspace(W, d: int, rank: int, seed: int) -> np.ndarray:
    W = np.asarray(W, dtype=np.float64)
    if W.shape[1] >= rank:
        return W[:, :rank]
    extra = random_subspace(d, rank - W.shape[1], seed)
    extra = extra - W @ (W.T @ extra)
    extra = orthonormal_columns(extra)
    return np.hstack([W, extra[:, : max(0, rank - W.shape[1])]])


def extract_all(model, tok, texts, label: str = ""):
    hidden, probs = [], []
    for k, text in enumerate(texts):
        hs, p = extract(model, tok, text)
        hidden.append(hs)
        probs.append(p.astype(np.float32))
        if label and (k + 1) % 50 == 0:
            print(f"    [{label}] {k + 1}/{len(texts)}")
    return np.stack(hidden), np.stack(probs)


def run(model, tok, items, args):
    arm = [it for it in items if it["pos"] == args.pos]
    relation = args.relation or ("sibling" if args.pos == "noun" else "antonym")
    sel = select_arm(arm, args.pos, relation, args.control)
    # reindex anchors densely for group bookkeeping
    remap = {a: i for i, a in enumerate(sorted({it["anchor_id"] for it in sel}))}
    for it in sel:
        it["anchor_id"] = remap[it["anchor_id"]]
        it["n_tokens_slot"] = len(tok(it["word"], add_special_tokens=False)["input_ids"])
    if args.limit_anchors:
        sel = [it for it in sel if it["anchor_id"] < args.limit_anchors]

    groups = anchor_groups(sel)
    labels = relation_labels(sel)
    frame = FRAMES[args.frame]
    texts = stimuli_text(sel, frame)
    print(f"[items] {len(sel)}  anchors={len(groups)}  relation={relation}  "
          f"control={args.control}  frame={frame!r}")

    H, P_base = extract_all(model, tok, texts, label="base")
    n, n_states, d = H.shape
    blocks = get_blocks(model)
    n_blocks = len(blocks)
    if args.layers:
        layers = [int(x) for x in str(args.layers).split(",") if x != ""]
    else:
        layers = sorted({1, max(1, n_blocks // 2), n_blocks})
    print(f"[model] {args.model}  blocks={n_blocks}  states={n_states}  d={d}  "
          f"layers={layers}")

    cov = np.column_stack([
        np.asarray([it["zipf"] for it in sel], dtype=np.float64),
        np.asarray([it["n_tokens_slot"] for it in sel], dtype=np.float64),
        np.asarray([it["n_chars"] for it in sel], dtype=np.float64),
    ])
    rng = np.random.default_rng(args.seed)
    result = {
        "model": args.model, "pos": args.pos, "relation": relation,
        "control": args.control, "frame": frame, "n_items": len(sel),
        "n_anchors": len(groups), "k": args.k, "n_random": args.n_random,
        "n_boot": args.n_boot, "layers": [],
    }

    for layer in layers:
        block = layer - 1
        if not (0 <= block < n_blocks):
            continue
        Hl = H[:, layer, :]
        rel_ctl = [[i for i in g if labels[i] in {relation, args.control}]
                   for g in groups]
        P_meaning = np.full_like(P_base, np.nan)
        P_form = np.full_like(P_base, np.nan)
        P_random = [np.full_like(P_base, np.nan) for _ in range(args.n_random)]
        dir_aucs = []
        removed = {"meaning": [], "form": [],
                   "random": [[] for _ in range(args.n_random)]}

        for gi, g in enumerate(groups):
            train_rows, y_train, c_train = [], [], []
            for gj, gg in enumerate(groups):
                if gj == gi or not rel_ctl[gj]:
                    continue
                idx = rel_ctl[gj]
                mu = Hl[idx].mean(axis=0)
                train_rows.append(Hl[idx] - mu)
                y_train += [1 if labels[i] == relation else 0 for i in idx]
                c_train.append(cov[idx] - cov[idx].mean(axis=0))
            if not train_rows:
                continue
            Xtr = np.vstack(train_rows)
            ytr = np.asarray(y_train, dtype=int)
            Ctr = np.vstack(c_train)
            W_m = fisher_subspace(Xtr, ytr, k=args.k)
            rank = W_m.shape[1]
            if rank == 0:
                continue
            W_f = pad_subspace(form_subspace(Xtr, Ctr, k=rank), d, rank, args.seed)
            W_rand = [random_subspace(d, rank, args.seed + 1000 + s)
                      for s in range(args.n_random)]

            test_rel = rel_ctl[gi]
            if test_rel:
                mu = Hl[test_rel].mean(axis=0)
                dir_aucs.append(direction_auc(
                    Hl[test_rel] - mu,
                    np.asarray([labels[i] == relation for i in test_rel], dtype=int),
                    W_m,
                ))

            Xtest = Hl[g]
            removed["meaning"].extend(removed_fraction(Xtest, W_m))
            removed["form"].extend(removed_fraction(Xtest, W_f))
            for s, W in enumerate(W_rand):
                removed["random"][s].extend(removed_fraction(Xtest, W))

            test_idx = g
            test_texts = [texts[i] for i in test_idx]
            P_meaning[test_idx] = np.stack([
                ablated_probs(model, tok, t, block, W_m) for t in test_texts
            ]).astype(np.float32)
            P_form[test_idx] = np.stack([
                ablated_probs(model, tok, t, block, W_f) for t in test_texts
            ]).astype(np.float32)
            for s, W in enumerate(W_rand):
                P_random[s][test_idx] = np.stack([
                    ablated_probs(model, tok, t, block, W) for t in test_texts
                ]).astype(np.float32)

        def summarize(P):
            return gradient_summary(hellinger_rdm(P), labels, groups, relation,
                                    args.control, n_boot=args.n_boot, seed=args.seed)

        base = summarize(P_base)
        meaning = summarize(P_meaning)
        form = summarize(P_form)
        random_summ = [summarize(P) for P in P_random]
        random_per = np.nanmean(
            np.vstack([s["per_anchor"] for s in random_summ]), axis=0
        ).tolist()
        random_mean = float(np.nanmean(random_per))
        seed_means = [s["mean"] for s in random_summ]
        delta_meaning = bootstrap_diff_ci(base["per_anchor"], meaning["per_anchor"],
                                          n_boot=args.n_boot, seed=args.seed)
        delta_form = bootstrap_diff_ci(base["per_anchor"], form["per_anchor"],
                                       n_boot=args.n_boot, seed=args.seed)
        delta_random = bootstrap_diff_ci(base["per_anchor"], random_per,
                                         n_boot=args.n_boot, seed=args.seed)
        difference = bootstrap_diff_ci(random_per, meaning["per_anchor"],
                                       n_boot=args.n_boot, seed=args.seed)
        removed_summary = {
            "meaning": float(np.nanmean(removed["meaning"])) if removed["meaning"] else float("nan"),
            "form": float(np.nanmean(removed["form"])) if removed["form"] else float("nan"),
            "random_seed_means": [float(np.nanmean(v)) for v in removed["random"]],
        }
        removed_summary["random_mean"] = (
            float(np.nanmean(removed_summary["random_seed_means"]))
            if removed_summary["random_seed_means"] else float("nan")
        )
        entry = {
            "layer": layer, "block": block,
            "removed_fraction": removed_summary,
            "direction_auc": float(np.nanmean(dir_aucs)) if dir_aucs else float("nan"),
            "baseline": base,
            "meaning": meaning,
            "form": form,
            "random": {"mean": random_mean, "sd": float(np.nanstd(seed_means)),
                       "seed_means": seed_means, "per_anchor": random_per},
            "delta_meaning": {"mean": delta_meaning[2], "ci95": list(delta_meaning[:2])},
            "delta_form": {"mean": delta_form[2], "ci95": list(delta_form[:2])},
            "delta_random": {"mean": delta_random[2], "ci95": list(delta_random[:2])},
            "difference_random_minus_meaning": {
                "mean": difference[2], "ci95": list(difference[:2])
            },
        }
        result["layers"].append(entry)
        print(f"  [L{layer}] base={base['mean']:.3f}  meaning={meaning['mean']:.3f}  "
              f"random={random_mean:.3f}  form={form['mean']:.3f}  "
              f"dirAUC={entry['direction_auc']:.3f}  "
              f"rand-meaning={difference[2]:+.3f} "
              f"CI[{difference[0]:+.3f},{difference[1]:+.3f}]  "
              f"removed: mean={removed_summary['meaning']:.3f} "
              f"rand={removed_summary['random_mean']:.3f} "
              f"form={removed_summary['form']:.3f}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilgpt2")
    ap.add_argument("--data", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--pos", default="noun", choices=["noun", "adjective"])
    ap.add_argument("--relation", default="")
    ap.add_argument("--control", default=HARD_CONTROL_RELATION)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--layers", default="")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--n-random", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--limit-anchors", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data = Path(args.data) if args.data else \
        Path(__file__).resolve().parents[1] / "data" / "slot_relations.tsv"
    args.data = data
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
        f"causal_ablation_{model_slug(args.model)}_{args.pos}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
