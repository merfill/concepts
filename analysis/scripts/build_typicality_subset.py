#!/usr/bin/env python
"""Build ``data/typicality.tsv`` from the THINGSplus mean-typicality table.

The raw file is ``typicality53_mean-ratings.tsv`` from the THINGS OSF project
(``jum2f``, folder ``03_category-level``).  Selection is deterministic and lives
in ``tc.typicality.select_subset``; nothing is hand-listed here.

Run:
    uv run python scripts/build_typicality_subset.py --raw /path/to/typicality53_mean-ratings.tsv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tc.typicality import (
    PER_CATEGORY,
    SELECTED_CATEGORIES,
    select_subset,
    write_norms,
)


def read_raw(path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            value = row.get("typicality_score", "")
            if value == "":
                continue
            rows.append({
                "category": row["category"].strip(),
                "item": row["Word"].replace("_", " ").strip(),
                "typicality": float(value),
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="typicality53_mean-ratings.tsv")
    ap.add_argument("--out",
                    default=str(Path(__file__).resolve().parents[1] / "data" / "typicality.tsv"))
    ap.add_argument("--categories", default=",".join(SELECTED_CATEGORIES))
    ap.add_argument("--per-category", type=int, default=PER_CATEGORY)
    args = ap.parse_args()

    categories = [c for c in args.categories.split(",") if c]
    raw = read_raw(args.raw)
    selected = select_subset(raw, categories=categories, per_category=args.per_category)
    write_norms(selected, args.out)

    counts = Counter(row["category"] for row in selected)
    print(f"raw rows: {len(raw)}")
    print(f"selected: {len(selected)} -> {args.out}")
    for category in categories:
        print(f"  {category:20s} {counts.get(category, 0)}")


if __name__ == "__main__":
    main()
