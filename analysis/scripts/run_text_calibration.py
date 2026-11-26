#!/usr/bin/env python
"""Local calibration of TEXT-ONLY channels against white-box q (Qwen).

For tool/clothing/mammal: run (1) rating of the 20 annotation words (1-7) and
(2) a free list; parse them; compare -- rating Spearman vs the white-box
Spearman from experiment 010, and listed prototypes vs white-box/human tops.

Run:
    uv run python scripts/run_text_calibration.py --model Qwen/Qwen2.5-1.5B-Instruct
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tc.text_probe import list_prompt, parse_list, parse_rating, rating_prompt
from tc.lexicon import member_lemmas
from tc.typicality import load_norms

try:
    from nltk.stem import WordNetLemmatizer

    LEMMATIZER = WordNetLemmatizer()
except Exception:  # pragma: no cover
    LEMMATIZER = None

DEFAULT_DATA = Path(__file__).resolve().parents[1] / "data" / "typicality.tsv"
CATEGORIES = ["tool", "clothing", "mammal"]


def model_slug(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_")


def generate(model, tokenizer, prompt: str, max_new: int, sample: bool) -> str:
    messages = [{"role": "user", "content": prompt}]
    chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat, return_tensors="pt")
    kwargs = dict(max_new_tokens=max_new, do_sample=sample, pad_token_id=tokenizer.eos_token_id)
    if sample:
        kwargs.update(temperature=0.8, top_p=0.9)
    with torch.no_grad():
        out = model.generate(**inputs, **kwargs)
    return tokenizer.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)


def white_box(category: str, model: str):
    path = Path("results") / f"centrality_{model_slug(model)}_{category}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    if "centrality" in data:
        return data["centrality"]
    for frame in (data.get("frames") or {}).values():
        if "per_word" in frame:
            return frame
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--rating-tokens", type=int, default=140)
    ap.add_argument("--list-tokens", type=int, default=64)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    norms = load_norms(args.data)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()

    payload = {"model": args.model, "categories": {}}
    for category in CATEGORIES:
        rows = [r for r in norms if r["category"] == category]
        words = [r["item"] for r in rows]
        human = {r["item"]: float(r["typicality"]) for r in rows}
        members = member_lemmas(category, extra=words)
        wb = white_box(category, args.model)

        print(f"\n===== {category} =====")
        cat_out = {}

        # --- rating channel ---
        rprompt = rating_prompt(category, words, seed=0)
        rtext = generate(model, tokenizer, rprompt, args.rating_tokens, sample=False)
        ratings = parse_rating(rtext, words)
        present = [w for w in words if w in ratings]
        if len(present) >= 3:
            rho = float(spearmanr([ratings[w] for w in present],
                                  [human[w] for w in present])[0])
        else:
            rho = float("nan")
        print(f"  RATING parsed {len(ratings)}/{len(words)}")
        print(f"    output: {rtext.strip()[:240]!r}")
        print(f"    Spearman(rating, human) = {rho:+.3f}")
        if wb is not None:
            wb_q = {d["word"]: d["q"] for d in wb["per_word"]}
            wb_rho = spearmanr([wb_q[w] for w in words], [human[w] for w in words])[0]
            print(f"    white-box Spearman(q, human) = {wb_rho:+.3f}")
        cat_out["rating"] = {"rho": rho, "parsed": len(ratings), "output": rtext,
                             "white_box_rho": float(wb_rho) if wb else None}

        # --- list channel ---
        lprompt = list_prompt(category, n=10)
        ltext = generate(model, tokenizer, lprompt, args.list_tokens, sample=True)
        plist = parse_list(ltext, members, LEMMATIZER)
        human_top = sorted(words, key=lambda w: -human[w])[:3]
        ranks = {w: (plist["matched"].index(w) + 1) for w in human_top if w in plist["matched"]}
        print(f"  LIST matched {len(plist['matched'])} members, {len(plist['unmatched'])} unmatched")
        print(f"    output: {ltext.strip()[:240]!r}")
        print(f"    human top-3 {human_top} -> ranks in list: {ranks or 'absent'}")
        cat_out["list"] = {"matched": plist["matched"], "unmatched": plist["unmatched"],
                           "output": ltext, "human_top3_ranks": ranks}

        payload["categories"][category] = cat_out

    out = Path(args.out) if args.out else Path(
        f"results/text_calibration_{model_slug(args.model)}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
