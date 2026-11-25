#!/usr/bin/env python
"""Experiment 009 / stage 1: typicality gradient and category prototype.

Protocol: ``../experiment-009-2026-11-25-typicality-prototype.md``.

Readouts: judgment (P(Yes)), ranking (length-normalised cloze log-prob), and a
next-token behavioural signature for the prototype.  The prototype is a
category centroid of signatures; its held-out distance is tested against the
human typicality gradient, controlling for frequency and length.

Run:
    uv run python scripts/run_typicality.py --model distilgpt2 --limit 5
    uv run python scripts/run_typicality.py --model Qwen/Qwen2.5-1.5B-Instruct
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tc.extract import next_token_probs
from tc.typicality import (
    channel_summary,
    frequency_zipf,
    judgment_prompts,
    judgment_score,
    limit_per_category,
    load_norms,
    mean_within_rho,
    overall_verdict,
    prototype_cv,
    prototype_summary,
    ranking_probes,
    selected_categories,
    signature_prompts,
)

DEFAULT_DATA = Path(__file__).resolve().parents[1] / "data" / "typicality.tsv"
SIG_TOPK_STORE = 200


def model_slug(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_")


def token_ids_for_word(tokenizer, word: str) -> list[int]:
    target = word.strip().lower()
    ids = [i for i in range(len(tokenizer))
           if tokenizer.decode([i]).strip().lower() == target]
    if not ids:
        for text in (f" {word}", word):
            encoded = tokenizer(text, add_special_tokens=False).input_ids
            if len(encoded) == 1:
                ids.append(int(encoded[0]))
    return sorted(set(ids))


def seq_logprob(model, tokenizer, prompt: str, continuation: str) -> float:
    encoded_prompt = tokenizer(prompt, return_tensors="pt")
    encoded_cont = tokenizer(continuation, add_special_tokens=False, return_tensors="pt")
    if encoded_cont.input_ids.shape[1] == 0:
        return float("nan")
    ids = torch.cat([encoded_prompt.input_ids, encoded_cont.input_ids], dim=1)
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=torch.ones_like(ids))
    logprobs = torch.log_softmax(out.logits[0], dim=-1).double()
    start = encoded_prompt.input_ids.shape[1]
    values = [float(logprobs[start - 1 + i, int(tid)])
              for i, tid in enumerate(encoded_cont.input_ids[0])]
    return float(np.mean(values))


def mean_input_embedding(model, tokenizer, text: str) -> np.ndarray:
    ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids
    emb = model.get_input_embeddings()
    with torch.no_grad():
        vectors = emb(ids)[0].double().numpy()
    return vectors.mean(axis=0) if vectors.size else np.zeros(emb.weight.shape[1])


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def top_tokens(tokenizer, probs: np.ndarray, k: int = 10) -> list[dict]:
    idx = np.argsort(probs)[-k:][::-1]
    return [{"token": tokenizer.decode([int(i)]).strip(), "p": float(probs[i])}
            for i in idx]


def sparse_signature(probs: np.ndarray, k: int = SIG_TOPK_STORE):
    idx = np.argsort(probs)[-k:]
    return idx.astype(np.int64), probs[idx].astype(np.float64)


def reconstruct_signatures(sparse: list[tuple[np.ndarray, np.ndarray]], vocab_size: int):
    out = []
    for idx, vals in sparse:
        full = np.zeros(vocab_size, dtype=np.float64)
        full[idx] = vals
        out.append(full)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--limit", type=int, default=0, help="items per category (0 = all)")
    ap.add_argument("--k", type=int, default=50, help="top-k support for signature JS")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--perm", type=int, default=1000)
    ap.add_argument("--unrelated-limit", type=int, default=5)
    ap.add_argument("--skip-prototype", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    rows = load_norms(args.data)
    if args.limit > 0:
        rows = limit_per_category(rows, args.limit)
    categories = selected_categories(rows)
    print(f"[data] {args.data}  rows={len(rows)}  categories={categories}")

    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()
    dtype = str(next(model.parameters()).dtype).replace("torch.", "")
    vocab_size = int(model.get_input_embeddings().weight.shape[0])
    yes_ids = token_ids_for_word(tokenizer, "yes")
    no_ids = token_ids_for_word(tokenizer, "no")
    if not yes_ids or not no_ids:
        raise SystemExit(f"could not locate Yes/No tokens (yes={yes_ids}, no={no_ids})")
    print(f"[model] {args.model}  dtype={dtype}  yes={yes_ids[:4]} no={no_ids[:4]}")

    category_embedding = {
        c: mean_input_embedding(model, tokenizer, c) for c in categories
    }

    records, signatures = [], []
    for i, row in enumerate(rows):
        item, category = row["item"], row["category"]
        judgments = []
        for prompt in judgment_prompts(item, category):
            probs = next_token_probs(model, tokenizer, prompt)
            judgments.append(judgment_score(float(probs[yes_ids].sum()),
                                            float(probs[no_ids].sum())))
        rankings = [seq_logprob(model, tokenizer, prompt, continuation)
                    for prompt, continuation in ranking_probes(item, category)]
        sig_probs = [next_token_probs(model, tokenizer, prompt)
                     for prompt in signature_prompts(item)]
        signatures.append([sparse_signature(p) for p in sig_probs])
        static_cos = cosine(mean_input_embedding(model, tokenizer, item),
                            category_embedding[category])
        records.append({
            "category": category, "item": item,
            "typicality": float(row["typicality"]),
            "judgment_parts": judgments, "judgment": float(np.mean(judgments)),
            "ranking_parts": rankings, "ranking": float(np.mean(rankings)),
            "frequency_zipf": frequency_zipf(item),
            "length": int(len(tokenizer(item, add_special_tokens=False).input_ids)),
            "static_cos": static_cos,
            "signature_top": [top_tokens(tokenizer, p, k=10) for p in sig_probs],
        })
        print(f"  [{i + 1:3d}/{len(rows)}] {category:20s} {item:18s} "
              f"J={records[-1]['judgment']:.3f} R={records[-1]['ranking']:+.3f}")

    target = np.array([r["typicality"] for r in records])
    cats = np.array([r["category"] for r in records])
    items = np.array([r["item"] for r in records])
    frequency = np.array([r["frequency_zipf"] for r in records])
    length = np.array([r["length"] for r in records])
    static_cos = np.array([r["static_cos"] for r in records])

    channels = {}
    for name, key in (("judgment", "judgment"), ("ranking", "ranking")):
        score = np.array([r[key] for r in records])
        channels[name] = channel_summary(
            name, score, target, cats, frequency, length,
            static_cos=static_cos, n_perm=args.perm, seed=0,
        )
        parts = np.array([r[f"{key}_parts"] for r in records])
        channels[name]["frame_agreement"] = mean_within_rho(parts[:, 0], parts[:, 1], cats)

    prototypes = {}
    if not args.skip_prototype:
        for frame_index in range(len(signature_prompts("{item}"))):
            scores = np.full(len(records), np.nan)
            nearest = {}
            by_category = defaultdict(list)
            for i, r in enumerate(records):
                by_category[r["category"]].append(i)
            for category, index in by_category.items():
                full = reconstruct_signatures(
                    [signatures[i][frame_index] for i in index], vocab_size)
                local_scores, local_nearest = prototype_cv(
                    full, [category] * len(index), [items[i] for i in index],
                    k=args.k, folds=args.folds, seed=0,
                )
                scores[index] = local_scores
                nearest.update(local_nearest)
            prototypes[f"signature_{frame_index}"] = prototype_summary(
                f"signature_{frame_index}", scores, target, cats, frequency, length,
                n_perm=args.perm, seed=0, nearest=nearest,
            )

    unrelated = {}
    if args.unrelated_limit > 0:
        by_category = defaultdict(list)
        for i, r in enumerate(records):
            by_category[r["category"]].append(i)
        own, other, labels = [], [], []
        for ci, category in enumerate(categories):
            other_category = categories[(ci + 1) % len(categories)]
            if other_category == category:
                continue
            for i in by_category[category][: args.unrelated_limit]:
                record = records[i]
                other_judgments = []
                for prompt in judgment_prompts(record["item"], other_category):
                    probs = next_token_probs(model, tokenizer, prompt)
                    other_judgments.append(
                        judgment_score(float(probs[yes_ids].sum()),
                                       float(probs[no_ids].sum())))
                other_j = float(np.mean(other_judgments))
                other_r = float(np.mean([
                    seq_logprob(model, tokenizer, prompt, cont)
                    for prompt, cont in ranking_probes(record["item"], other_category)
                ]))
                own.append((record["judgment"], record["ranking"]))
                other.append((other_j, other_r))
        own = np.array(own)
        other = np.array(other)
        for j, key in enumerate(("judgment", "ranking")):
            scores = np.r_[own[:, j], other[:, j]]
            y = np.r_[np.ones(len(own)), np.zeros(len(other))]
            unrelated[key] = {
                "mean_own": float(own[:, j].mean()),
                "mean_other": float(other[:, j].mean()),
                "gap": float(own[:, j].mean() - other[:, j].mean()),
                "auc": float(roc_auc_score(y, scores)) if len(set(y)) > 1 else float("nan"),
            }

    verdict = overall_verdict(channels, prototypes)
    payload = {
        "model": args.model, "dtype": dtype, "data": args.data,
        "limit": args.limit, "k": args.k, "folds": args.folds, "perm": args.perm,
        "n_items": len(records), "categories": categories,
        "channels": channels, "prototypes": prototypes,
        "unrelated": unrelated, "verdict": verdict,
        "records": records,
    }
    out = Path(args.out) if args.out else Path(
        f"results/typicality_{model_slug(args.model)}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    print("\nchannel      rho    p     partial  L0     L1     margin  frame   verdict")
    for name, ch in channels.items():
        print(f"{name:10s} {ch['rho']:+.3f} {ch['p']:.4f} {ch['partial']:+.3f} "
              f"{ch['rho_l0']:+.3f} {ch['rho_l1']:+.3f} {ch['margin']:+.3f} "
              f"{ch['frame_agreement']:+.3f}  {ch['verdict']}")
    for name, proto in prototypes.items():
        print(f"{name:10s} heldout={proto['rho']:+.3f} p={proto['p']:.4f} "
              f"partial={proto['partial']:+.3f} margin={proto['margin']:+.3f} "
              f"-> {proto['verdict']}")
    for name, ctrl in unrelated.items():
        print(f"unrelated {name}: gap={ctrl['gap']:+.3f} auc={ctrl['auc']:.3f}")
    print(f"\nverdict: typicality={verdict['typicality']}  "
          f"prototype={verdict['prototype']}  robust={verdict['robust_channels']}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
