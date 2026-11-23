#!/usr/bin/env python
"""Trajectory-attractor test (v2).

Same fixed question to every description; look at the *short generated
trajectory* (3-6 tokens) and compare its distribution across descriptions of the
same concept vs related vs unrelated concepts.

Signals:
  A) exact first-token distribution at the question position (deterministic), JS.
  B) sampled 6-token trajectories, embedded, average pairwise similarity.

Run:  uv run python scripts/run_trajectory.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tc.corpus import RELATED_CONCEPTS, RELATED_PAIRS
from tc.joint import cosine_sim

QUESTION = "It is a"
N_SAMPLES = 6
N_TOKENS = 6


def make_pairs(cids, name_to_cid):
    N = len(cids)
    same = [(i, j) for i in range(N) for j in range(i + 1, N) if cids[i] == cids[j]]
    rel_cid = [frozenset((name_to_cid[a], name_to_cid[b])) for a, b in RELATED_PAIRS]
    related = [(i, j) for i in range(N) for j in range(i + 1, N)
               if frozenset((cids[i], cids[j])) in rel_cid]
    unrel = [(i, j) for i in range(N) for j in range(i + 1, N)
             if cids[i] != cids[j] and frozenset((cids[i], cids[j])) not in rel_cid]
    return same, related, unrel


def auc(a_vals, b_vals):
    y = np.r_[np.ones(len(a_vals)), np.zeros(len(b_vals))]
    return float(roc_auc_score(y, np.r_[a_vals, b_vals]))


def mean_pool(text, model, tok):
    enc = tok(text if text.strip() else "?", return_tensors="pt")
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
    return out.hidden_states[-1][0].double().numpy().mean(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", default="results/trajectory_summary.json")
    args = ap.parse_args()

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()
    print(f"[model] {args.model}   Q={QUESTION!r}\n")

    names = list(RELATED_CONCEPTS.keys())
    name_to_cid = {n: i for i, n in enumerate(names)}
    texts, cids = [], []
    for cid, name in enumerate(names):
        for desc in RELATED_CONCEPTS[name]:
            texts.append(desc)
            cids.append(cid)

    first_dists, traj_embs, samples_by_cid = [], [], {i: [] for i in cids}
    print("=== sampled trajectories (description -> 6-token continuation) ===")
    for idx, (desc, cid) in enumerate(zip(texts, cids)):
        prompt = f"{desc}\n\n{QUESTION}"
        enc = tok(prompt, return_tensors="pt")
        with torch.no_grad():
            out = model(**enc)
            probs = torch.softmax(out.logits[0, -1], dim=-1).double().numpy()
        first_dists.append(probs)

        embs = []
        texts_gen = []
        for _ in range(N_SAMPLES):
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=N_TOKENS, do_sample=True,
                                     temperature=0.8, top_p=0.9, pad_token_id=tok.eos_token_id)
            cont = tok.decode(gen[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            texts_gen.append(cont.strip())
            embs.append(mean_pool(cont, model, tok))
        traj_embs.append(embs)
        samples_by_cid[cid].append(texts_gen)
        print(f"  {names[cid]:9s} |{desc[:40]:40s}| -> {texts_gen[0]!r} ...")

    # condition-level similarity
    def cond_sim(i, j, use_emb):
        if not use_emb:
            return -float(jensenshannon(first_dists[i], first_dists[j]) ** 2)
        return float(np.mean([[cosine_sim(a, b) for b in traj_embs[j]] for a in traj_embs[i]]))

    same, related, unrel = make_pairs(cids, name_to_cid)
    print(f"\n[pairs] same={len(same)} related={len(related)} unrelated={len(unrel)}")
    for lab, use_emb in (("first-token dist (JS)", False), ("sampled trajectories (emb)", True)):
        s_same = [cond_sim(i, j, use_emb) for i, j in same]
        s_rel = [cond_sim(i, j, use_emb) for i, j in related]
        s_un = [cond_sim(i, j, use_emb) for i, j in unrel]
        print(f"\n=== {lab} ===")
        print(f"  mean sim: same={np.mean(s_same):+.3f} related={np.mean(s_rel):+.3f} unrel={np.mean(s_un):+.3f}")
        print(f"  AUC same-vs-related  = {auc(s_same, s_rel):.3f}")
        print(f"  AUC related-vs-unrel = {auc(s_rel, s_un):.3f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": args.model, "question": QUESTION,
                               "samples_by_concept": {names[k]: v for k, v in samples_by_cid.items()}},
                              indent=2, ensure_ascii=False))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
