#!/usr/bin/env python
"""Attractor test: does the *generated answer* separate what instantaneous
representations do not?

For each concept we give the model its three disjoint-word descriptions and ask
it to name the concept. We then check whether the generated answers cluster by
concept (same < related < unrelated), and compare with the input-side pooled
representation.

Run:  uv run python scripts/run_attractor.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tc.corpus import RELATED_CONCEPTS, RELATED_PAIRS
from tc.joint import cosine_sim

INSTR = "Name in one or two words the single object or concept described by this text:"


def pool_last_hidden(text, model, tok):
    enc = tok(text, return_tensors="pt")
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
    V = out.hidden_states[-1][0].double().numpy()
    return V.mean(0)


def make_pairs(cids, name_to_cid):
    N = len(cids)
    same = [(i, j) for i in range(N) for j in range(i + 1, N) if cids[i] == cids[j]]
    rel_cid = [frozenset((name_to_cid[a], name_to_cid[b])) for a, b in RELATED_PAIRS]
    related = [
        (i, j) for i in range(N) for j in range(i + 1, N)
        if frozenset((cids[i], cids[j])) in rel_cid
    ]
    unrel = [
        (i, j) for i in range(N) for j in range(i + 1, N)
        if cids[i] != cids[j] and frozenset((cids[i], cids[j])) not in rel_cid
    ]
    return same, related, unrel


def auc(a_vals, b_vals):
    y = np.r_[np.ones(len(a_vals)), np.zeros(len(b_vals))]
    return float(roc_auc_score(y, np.r_[a_vals, b_vals]))


def sims(embs, pairs):
    return np.array([cosine_sim(embs[i], embs[j]) for (i, j) in pairs])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", default="results/attractor_summary.json")
    args = ap.parse_args()

    torch.use_deterministic_algorithms(True, warn_only=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()
    print(f"[model] {args.model}\n")

    names = list(RELATED_CONCEPTS.keys())
    name_to_cid = {n: i for i, n in enumerate(names)}
    texts, cids = [], []
    for cid, name in enumerate(names):
        for desc in RELATED_CONCEPTS[name]:
            texts.append(desc)
            cids.append(cid)

    answers = []
    print("=== answers (concept -> generated names) ===")
    for cid, name in enumerate(names):
        got = []
        for desc in RELATED_CONCEPTS[name]:
            messages = [{"role": "user", "content": f"{INSTR}\n\n{desc}"}]
            chat = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            enc = tok(chat, return_tensors="pt", add_special_tokens=False)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=8, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
            new = gen[0][enc["input_ids"].shape[1]:]
            ans = tok.decode(new, skip_special_tokens=True).strip().splitlines()[0].strip()
            answers.append(ans)
            got.append(ans)
        print(f"  {name:10s} -> {got}")

    # embeddings of generated answers and of input descriptions
    ans_emb = [pool_last_hidden(a if a else "?", model, tok) for a in answers]
    desc_emb = [pool_last_hidden(t, model, tok) for t in texts]

    same, related, unrel = make_pairs(cids, name_to_cid)
    print(f"\n[pairs] same={len(same)} related={len(related)} unrelated={len(unrel)}")

    for label, embs in (("answer(attractor)", ans_emb), ("input(description)", desc_emb)):
        s_same, s_rel, s_un = sims(embs, same), sims(embs, related), sims(embs, unrel)
        print(f"\n=== {label} ===")
        print(f"  mean sim: same={s_same.mean():+.3f} related={s_rel.mean():+.3f} unrel={s_un.mean():+.3f}")
        print(f"  AUC same-vs-related = {auc(s_same, s_rel):.3f}")
        print(f"  AUC related-vs-unrel = {auc(s_rel, s_un):.3f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": args.model, "answers": answers,
                               "concepts": [n for n in names]}, indent=2, ensure_ascii=False))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
