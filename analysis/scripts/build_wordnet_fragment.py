#!/usr/bin/env python
"""Experiment 012 / arm b: build and freeze the WordNet hierarchy fragment.

Only common words are kept (the earlier broad-root tree was dominated by rare
technical terms, which confounds the read with frequency).  The fragment is
built from common concepts that have common relatives; every node's level still
comes from the WordNet hypernym graph.  The result is written to a TSV and then
read by ``run_hierarchy.py --arm wordnet``.

Protocol: ``../experiment-012-2026-11-26-hierarchy.md``.

Run:
    uv run python scripts/build_wordnet_fragment.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tc.hierarchy import fragment_concepts, wordnet_seed_fragment, write_fragment

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "data" / "wordnet_fragment.tsv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-zipf", type=float, default=4.0,
                    help="keep only words at least this common on the Zipf scale")
    ap.add_argument("--min-children", type=int, default=3,
                    help="a seed concept needs this many common hyponyms")
    ap.add_argument("--top-n", type=int, default=50000)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    fragment = wordnet_seed_fragment(min_zipf=args.min_zipf,
                                     min_children=args.min_children,
                                     top_n=args.top_n)
    write_fragment(fragment, args.out)
    concepts = fragment_concepts(fragment, min_children=args.min_children)

    print(f"min_zipf={args.min_zipf}  min_children={args.min_children}")
    print(f"nodes: {len(fragment['nodes'])}  roots: {len(fragment['roots'])}  "
          f"concepts(>= {args.min_children} children): {len(concepts)}")
    by_depth: dict[int, int] = {}
    for node in fragment["nodes"].values():
        by_depth[node["depth"]] = by_depth.get(node["depth"], 0) + 1
    print("nodes by depth:", dict(sorted(by_depth.items())))
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
