#!/usr/bin/env python
"""Freeze the WordNet slot-relation stimuli (experiment 020).

Builds the stimulus table once and writes ``data/slot_relations.tsv``; the model
run reads the frozen file instead of WordNet, so results do not depend on the
corpus version.

Run:
    uv run python scripts/build_slot_relations.py
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tc.slot_relations import build_stimuli, write_tsv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--max-per-relation", type=int, default=4)
    ap.add_argument("--min-zipf", type=float, default=4.0)
    ap.add_argument("--n-controls", type=int, default=4)
    args = ap.parse_args()

    items = build_stimuli(
        max_per_relation=args.max_per_relation,
        min_zipf=args.min_zipf,
        n_controls=args.n_controls,
    )
    out = Path(args.out) if args.out else \
        Path(__file__).resolve().parents[1] / "data" / "slot_relations.tsv"
    write_tsv(items, out)

    by_pos = Counter(it["pos"] for it in items)
    anchors = {(it["pos"], it["anchor_id"]) for it in items}
    print(f"[items] {len(items)}  anchors={len(anchors)}  by_pos={dict(by_pos)}")
    for pos in sorted(by_pos):
        sub = [it for it in items if it["pos"] == pos]
        counts = Counter(it["relation"] for it in sub)
        n_anchor = len({it["anchor_id"] for it in sub})
        print(f"  {pos}: anchors={n_anchor}  " +
              "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print(f"saved -> {out}")


if __name__ == "__main__":
    try:
        main()
    except LookupError as exc:
        print(f"WordNet data missing: {exc}\nTry: python -m nltk.downloader wordnet")
        raise SystemExit(1)
