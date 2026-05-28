#!/usr/bin/env python3
"""
Build a results JSON slice for blind AB human eval: io, cot, tot only.

Excludes tot_astar, io_refine, and tot_refine (and any other keys not listed).
The slim subset matches the paper's Fig 5(b) AB comparison set; trimming
keeps the human-eval pipeline focused and avoids leaking the extended methods
to raters.

Example
-------
  python3 code/tot/tasks/creative-writing/export_human_eval_subset.py \\
    --source results/creative-writing/json_outputs/baseline.json \\
    --out results/creative-writing/human_eval/base_methods_only.json

Default source is baseline.json (aligned io / cot / tot runs).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SOURCE = ROOT / "results" / "creative-writing" / "json_outputs" / "baseline.json"
DEFAULT_OUT = ROOT / "results" / "creative-writing" / "human_eval" / "base_methods_only.json"
KEEP_METHODS = frozenset({"io", "cot", "tot"})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("Example")[0].strip())
    p.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Full results JSON.")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output path (subset JSON).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with args.source.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise SystemExit("Source must be a JSON object method -> list.")

    out = {k: v for k, v in data.items() if k in KEEP_METHODS}
    missing = KEEP_METHODS - set(out.keys())
    if missing:
        print(f"Warning: missing methods in source (skipped): {sorted(missing)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
        f.write("\n")

    counts = {k: len(v) if isinstance(v, list) else "?" for k, v in out.items()}
    print(f"Wrote {args.out}")
    print(f"  keys: {sorted(out.keys())}")
    print(f"  counts: {counts}")


if __name__ == "__main__":
    main()
