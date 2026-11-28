#!/usr/bin/env python
"""Experiment 017: separation of form and meaning via contrastive subspaces.

Protocol: ``../experiment-017-2026-11-28-form-vs-meaning.md``.

Items come from ``tc.form_meaning`` (factorial scenes x templates x synonyms).
We read hidden states at the probe position for every layer, build the
within-scene (form) and between-scene (meaning) scatter matrices, and solve the
two generalized eigenproblems (Fisher / linear discriminant).  Per layer we
report the scatter ratio, the angle between the meaning and form subspaces, RSA
with the meaning and form RDMs, partial correlations controlling for the other
factor and length, and the ratio after projecting on each subspace.

Run:
    uv run python scripts/run_separation.py --model distilgpt2
    uv run python scripts/run_separation.py --model Qwen/Qwen2.5-1.5B-Instruct
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
from tc.controls import residualize as residualize_vec
from tc.controls import separation_auc, spearman, upper_tri
from tc.form_meaning import build_items, labels
from tc.separation import (
    abs_diff_rdm,
    cosine_rdm,
    fisher_directions,
    fisher_ratio,
    label_rdm,
    mean_principal_angle,
    project,
    residualize,
)


def model_slug(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_")


def extract(model, tok, texts):
    """Hidden states at the last position for every layer, plus token lengths."""
    per_item = []
    lengths = []
    for text in texts:
        enc = tok(text, return_tensors="pt")
        lengths.append(int(enc["input_ids"].shape[1]))
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        hs = [h[0, -1].float().cpu().numpy().astype(np.float64)
              for h in out.hidden_states]
        per_item.append(hs)
    n_layers = len(per_item[0])
    layers = [np.stack([it[l] for it in per_item], axis=0) for l in range(n_layers)]
    return layers, np.asarray(lengths, dtype=np.float64)


def partial_two(a, b, c1, c2) -> float:
    """Partial Spearman of a and b controlling for c1 and c2 (RDM matrices)."""
    cs = [upper_tri(c1), upper_tri(c2)]
    return spearman(residualize_vec(upper_tri(a), cs),
                    residualize_vec(upper_tri(b), cs))


def cv_group_metrics(X, y_meaning, groups, n_dir) -> dict:
    """Leave-one-group-out: fit the subspaces on train, score on the held-out group.

    ``groups`` is the form factor (template or synonym).  The meaning subspace is
    never fitted on the held-out group, so the test checks generalization to an
    unseen surface form.
    """
    groups = np.asarray(groups)
    folds = []
    for g in np.unique(groups):
        test = groups == g
        train = ~test
        y_test = y_meaning[test]
        if np.unique(y_test).size < 2 or np.unique(y_meaning[train]).size < 2:
            continue
        dirs = fisher_directions(X[train], y_meaning[train], n_meaning=n_dir, n_form=n_dir)
        Xp = project(X[test], dirs["W_meaning"], dirs["mean"])
        Xf = project(X[test], dirs["W_form"], dirs["mean"])
        rdm_p = cosine_rdm(Xp)
        rdm_f = cosine_rdm(Xf)
        rdm_r = cosine_rdm(X[test])
        y_rdm = label_rdm(y_test)
        folds.append({
            "group": int(g),
            "n_test": int(test.sum()),
            "auc_proj": separation_auc(rdm_p, y_test),
            "auc_formproj": separation_auc(rdm_f, y_test),
            "auc_raw": separation_auc(rdm_r, y_test),
            "rho_proj": spearman(upper_tri(rdm_p), upper_tri(y_rdm)),
            "rho_raw": spearman(upper_tri(rdm_r), upper_tri(y_rdm)),
            "ratio_proj": float(fisher_ratio(Xp, y_test)),
            "ratio_raw": float(fisher_ratio(X[test], y_test)),
        })
    keys = ["auc_proj", "auc_formproj", "auc_raw", "rho_proj", "rho_raw",
            "ratio_proj", "ratio_raw"]
    mean = {k: float(np.nanmean([f[k] for f in folds])) if folds else float("nan")
            for k in keys}
    return {"folds": folds, "mean": mean}


def layer_metrics(X, y_meaning, y_form, length, n_dir) -> dict:
    X = np.asarray(X, dtype=np.float64)
    dirs = fisher_directions(X, y_meaning, n_meaning=n_dir, n_form=n_dir)
    rdm_repr = cosine_rdm(X)
    rdm_meaning = label_rdm(y_meaning)
    rdm_form = label_rdm(y_form)
    rdm_len = abs_diff_rdm(length)

    a_repr = upper_tri(rdm_repr)
    rho_meaning = spearman(a_repr, upper_tri(rdm_meaning))
    rho_form = spearman(a_repr, upper_tri(rdm_form))
    partial_meaning = partial_two(rdm_repr, rdm_meaning, rdm_form, rdm_len)
    partial_form = partial_two(rdm_repr, rdm_form, rdm_meaning, rdm_len)

    Xm = project(X, dirs["W_meaning"], dirs["mean"])
    Xf = project(X, dirs["W_form"], dirs["mean"])
    return {
        "ratio": float(fisher_ratio(X, y_meaning)),
        "ratio_meaning_proj": float(fisher_ratio(Xm, y_meaning)) if Xm.shape[1] else float("nan"),
        "ratio_form_proj": float(fisher_ratio(Xf, y_meaning)) if Xf.shape[1] else float("nan"),
        "angle_meaning_form": mean_principal_angle(dirs["W_meaning"], dirs["W_form"]),
        "eig_meaning_top": float(np.max(dirs["eig_meaning"])) if dirs["eig_meaning"].size else float("nan"),
        "rho_meaning": rho_meaning,
        "rho_form": rho_form,
        "partial_meaning": partial_meaning,
        "partial_form": partial_form,
        "auc_meaning": separation_auc(rdm_repr, y_meaning),
        "auc_form": separation_auc(rdm_repr, y_form),
    }


def run(model, tok, args):
    items = build_items()
    if args.max_items:
        items = items[: args.max_items]
    texts = [it["text"] for it in items]
    y_meaning, y_form = labels(items)
    groups_template = np.asarray([it["template"] for it in items])
    groups_syn = np.asarray([it["syn"] for it in items])
    layers, length = extract(model, tok, texts)

    result = {
        "model": args.model,
        "n_items": len(items),
        "n_meaning": int(np.unique(y_meaning).size),
        "n_form": int(np.unique(y_form).size),
        "n_layers": len(layers),
        "d": int(layers[0].shape[1]),
        "n_dir": args.n_dir,
        "length_range": [float(length.min()), float(length.max())],
        "per_layer": [],
    }

    for l, X in enumerate(layers):
        raw = layer_metrics(X, y_meaning, y_form, length, args.n_dir)
        resid_data = residualize(X, [length])
        resid = layer_metrics(resid_data, y_meaning, y_form, length, args.n_dir)
        cv_template = cv_group_metrics(X, y_meaning, groups_template, args.n_dir)
        cv_syn = cv_group_metrics(X, y_meaning, groups_syn, args.n_dir)
        result["per_layer"].append({
            "layer": l,
            "raw": raw,
            "len_resid": resid,
            "cv_template": cv_template,
            "cv_syn": cv_syn,
        })
        cm = cv_template["mean"]
        print(f"  L{l:02d}  ratio={raw['ratio']:.3f}  angle={raw['angle_meaning_form']:.1f}  "
              f"rho_m={raw['rho_meaning']:+.3f}  rho_f={raw['rho_form']:+.3f}  "
              f"auc_m={raw['auc_meaning']:.3f}  | cvT auc={cm['auc_proj']:.3f} "
              f"(raw {cm['auc_raw']:.3f}, form {cm['auc_formproj']:.3f})")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilgpt2")
    ap.add_argument("--n-dir", type=int, default=5)
    ap.add_argument("--max-items", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()
    dtype = str(next(model.parameters()).dtype).replace("torch.", "")
    print(f"[model] {args.model}  dtype={dtype}")

    result = run(model, tok, args)
    result["dtype"] = dtype

    out = Path(args.out) if args.out else \
        Path(__file__).resolve().parents[1] / "results" / f"separation_{model_slug(args.model)}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
