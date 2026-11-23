#!/usr/bin/env python
"""End-to-end H1 validation (the "step 0" of the document).

Checks, in order:
  1. determinism of extraction,
  2. synthetic positive control (Phi detects a planted spectral difference),
  3. model-level positive control (grammatical number, weak),
  4. test-retest reliability over paraphrase groups,
  5. within/between separation with a permutation test,
  6. length confound (R^2 and residual separation),
  7. untrained-model control.

Run:  uv run python scripts/run_h1.py --model distilgpt2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tc import controls
from tc.corpus import NUMBER_PAIRS, flat_paraphrases
from tc.metrics import hellinger, rdm
from tc.phi import phi_spec


def _pad_all(invs, d):
    return [np.pad(p, (0, d - p.size)) for p in invs]


def _synthetic_positive_control(n_rep=6, n=8, d=12, seed=0):
    """Two classes of V matrices with clearly different planted spectra."""
    rng = np.random.default_rng(seed)

    def make(decay):
        U, _ = np.linalg.qr(rng.standard_normal((n, d)))
        _, Vt = np.linalg.qr(rng.standard_normal((d, d)))
        s = np.exp(-decay * np.arange(min(n, d)))
        s = np.pad(s, (0, min(n, d) - s.size))
        return U @ np.diag(s[: min(n, d)]) @ Vt[: min(n, d), :]

    a = [make(0.3) for _ in range(n_rep)]
    b = [make(1.2) for _ in range(n_rep)]
    invs = _pad_all([phi_spec(x) for x in a + b], d)
    groups = [0] * n_rep + [1] * n_rep
    auc, p, _ = controls.permutation_test(rdm(invs), groups, n_perm=200)
    return auc, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilgpt2")
    ap.add_argument("--layer", type=int, default=-1, help="-1 = last layer")
    ap.add_argument("--n-perm", type=int, default=500)
    ap.add_argument("--out", default="results/h1_summary.json")
    args = ap.parse_args()

    from tc.extract import extract, last_layer, load, token_count

    model, tok = load(args.model)
    layer = args.layer if args.layer >= 0 else last_layer(model)
    report: dict = {"model": args.model, "layer": layer}
    print(f"[model] {args.model}  layer={layer}")

    # 1. determinism
    v1, a1 = extract("A river flows to the sea.", model, tok, layer)
    v2, a2 = extract("A river flows to the sea.", model, tok, layer)
    det = float(np.max(np.abs(v1 - v2)))
    report["determinism_maxdiff"] = det
    print(f"[1] determinism max|V1-V2| = {det:.3e}")

    # 2. synthetic positive control
    s_auc, s_p = _synthetic_positive_control()
    report["positive_control_synthetic_auc"] = s_auc
    report["positive_control_synthetic_p"] = s_p
    print(f"[2] synthetic positive control AUC = {s_auc:.3f} (p={s_p:.4f})")

    # 3. model-level positive control: grammatical number
    num_texts = [t for pair in NUMBER_PAIRS for t in pair]
    num_groups = [idx for _ in NUMBER_PAIRS for idx in (0, 1)]
    invs_num = _pad_all([phi_spec(extract(t, model, tok, layer)[0]) for t in num_texts], model.config.n_embd if hasattr(model.config, "n_embd") else model.config.hidden_size)
    num_auc = controls.separation_auc(rdm(invs_num), num_groups)
    report["positive_control_number_auc"] = num_auc
    print(f"[3] number-agreement positive control AUC = {num_auc:.3f} (weak)")

    # 4-6. paraphrase corpus
    texts, group_ids, group_names = flat_paraphrases()
    invs = [phi_spec(extract(t, model, tok, layer)[0]) for t in texts]
    d = max(p.size for p in invs)
    invs = _pad_all(invs, d)

    by_group: dict[int, list] = {}
    for inv, g in zip(invs, group_ids):
        by_group.setdefault(g, []).append(inv)
    group_lists = [by_group[g] for g in sorted(by_group)]
    rel_mean, rel_sb = controls.split_half_reliability(group_lists, hellinger)
    report["reliability_spearman"] = rel_mean
    report["reliability_spearman_brown"] = rel_sb
    print(f"[4] split-half reliability: r={rel_mean:.3f}  SB={rel_sb:.3f}")

    D = rdm(invs, hellinger)
    auc, pval, _ = controls.permutation_test(D, group_ids, n_perm=args.n_perm)
    report["separation_auc"] = auc
    report["separation_p"] = pval
    print(f"[5] within/between separation AUC = {auc:.3f} (p={pval:.4f})")

    lengths = np.array([token_count(t, tok) for t in texts], dtype=np.float64)
    lenD = np.abs(np.subtract.outer(lengths, lengths))
    r2 = controls.r2_explained(D, lenD)
    res = controls.residualize(controls.upper_tri(D), [controls.upper_tri(lenD)])
    Dr = np.zeros_like(D)
    iu = np.triu_indices(len(texts), 1)
    Dr[iu] = res
    Dr[(iu[1], iu[0])] = res
    auc_res = controls.separation_auc(Dr, group_ids)
    report["length_r2"] = r2
    report["separation_auc_residual_length"] = auc_res
    print(f"[6] length R^2 = {r2:.3f} ; residual AUC = {auc_res:.3f}")

    # 7. untrained control
    umodel, _ = load(args.model, untrained=True, seed=1)
    uinvs = [phi_spec(extract(t, umodel, tok, layer)[0]) for t in texts]
    uinvs = _pad_all(uinvs, d)
    u_auc = controls.separation_auc(rdm(uinvs, hellinger), group_ids)
    report["untrained_separation_auc"] = u_auc
    print(f"[7] untrained-model AUC = {u_auc:.3f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nsaved -> {out}")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
