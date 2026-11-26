#!/usr/bin/env python
"""Loop layers 2-4 (output of layer 4 -> input of layer 2) several times.

Re-enter the residual stream from layer 4 into layer 2, repeat the 2-4 cycle k
times, then read the next-token distribution (and greedy generation).  Supports
GPT-2 (``transformer.h``) and Qwen2 (``model.layers`` + RoPE).

Run:
    uv run python scripts/run_layer_loop.py --model distilgpt2
    uv run python scripts/run_layer_loop.py --model Qwen/Qwen2.5-1.5B-Instruct
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
from tc.centrality import compute_purity, topk_items
from tc.lexicon import member_lemmas

try:
    from nltk.stem import WordNetLemmatizer

    LEMMATIZER = WordNetLemmatizer()
except Exception:  # pragma: no cover
    LEMMATIZER = None

PROMPTS = ["Name one tool used for work: a", "A hand tool, such as a"]


def model_slug(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_")


def _causal_mask(seq: int, dtype) -> torch.Tensor:
    return torch.triu(torch.full((1, 1, seq, seq), float("-inf"), dtype=dtype), diagonal=1)


def _as_tensor(out):
    return out[0] if isinstance(out, (tuple, list)) else out


def forward_loop(model, tokenizer, text: str, repeats: int) -> np.ndarray:
    """Embedding -> layers 1-4 -> (layers 2-4) x repeats -> final norm/head."""
    ids = tokenizer(text, return_tensors="pt").input_ids
    seq = ids.shape[-1]
    pos = torch.arange(seq).unsqueeze(0)

    if hasattr(model, "transformer"):  # GPT-2
        t = model.transformer
        hidden = t.wte(ids) + t.wpe(pos)
        mask = _causal_mask(seq, hidden.dtype)
        init_blocks, loop_blocks, final_norm = t.h[:4], t.h[1:4], t.ln_f

        def run(block, h):
            return _as_tensor(block(h, attention_mask=mask))
    else:  # Qwen2
        m = model.model
        hidden = m.embed_tokens(ids)
        pos_emb = m.rotary_emb(hidden, pos)
        mask = _causal_mask(seq, hidden.dtype)
        init_blocks, loop_blocks, final_norm = m.layers[:4], m.layers[1:4], m.norm

        def run(block, h):
            return _as_tensor(block(h, attention_mask=mask, position_embeddings=pos_emb,
                                    position_ids=pos))

    for block in init_blocks:
        hidden = run(block, hidden)
    for _ in range(repeats):
        for block in loop_blocks:
            hidden = run(block, hidden)
    hidden = final_norm(hidden)
    with torch.no_grad():
        logits = model.lm_head(hidden[:, -1, :])[0]
    return torch.softmax(logits.double(), dim=-1).numpy()


def generate_loop(model, tokenizer, prompt: str, repeats: int, n_tokens: int) -> str:
    ids = tokenizer(prompt, return_tensors="pt").input_ids
    generated = []
    for _ in range(n_tokens):
        text = tokenizer.decode(ids[0], skip_special_tokens=True)
        probs = forward_loop(model, tokenizer, text, repeats)
        nxt = int(np.argmax(probs))
        generated.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]])], dim=1)
    return tokenizer.decode(generated, skip_special_tokens=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilgpt2")
    ap.add_argument("--category", default="tool")
    ap.add_argument("--repeats", default="0,1,2,3,5")
    ap.add_argument("--gen", type=int, default=8)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    members = member_lemmas(args.category)
    repeats = [int(r) for r in args.repeats.split(",") if r.strip()]

    print(f"\n##### {args.model} | category={args.category} | loop = output L4 -> input L2 #####")
    payload = {"model": args.model, "category": args.category, "prompts": {}}

    for prompt in PROMPTS:
        print(f"\n=== {prompt!r} ===")
        with torch.no_grad():
            real = torch.softmax(
                model(**tokenizer(prompt, return_tensors="pt")).logits[0, -1].double(), -1).numpy()
        real_top = topk_items(tokenizer, real, 10)
        print("  REAL full model: " + ", ".join(f"{t}({p:.3f})" for t, p in real_top))
        payload["prompts"][prompt] = {"real": real_top, "loops": {}}

        for k in repeats:
            probs = forward_loop(model, tokenizer, prompt, k)
            top = topk_items(tokenizer, probs, 10)
            purity = compute_purity(topk_items(tokenizer, probs, 50), members, LEMMATIZER)
            greedy = generate_loop(model, tokenizer, prompt, k, args.gen)
            print(f"  loop x{k}: purity={purity['purity']:.3f} | "
                  + ", ".join(f"{t}({p:.3f})" for t, p in top))
            print(f"            greedy: {greedy!r}")
            payload["prompts"][prompt]["loops"][str(k)] = {
                "top10": top, "purity": purity["purity"], "greedy": greedy,
            }

    out = Path(args.out) if args.out else Path(
        f"results/layer_loop_{model_slug(args.model)}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
