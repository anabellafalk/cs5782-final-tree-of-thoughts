#!/usr/bin/env python3
"""Pick best/worst passages per method for qualitative review.

Reads the full results JSON, sorts each method's outputs by judge score, and
writes both a machine-readable JSON (for downstream tools) and a human-friendly
markdown report. Used to spot-check whether high scores correspond to genuinely
good writing and whether failures cluster on certain sentence sets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
RESULTS_DIR = ROOT / "results" / "creative-writing"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export high/low scoring creative-writing examples for qualitative review."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=RESULTS_DIR / "json_outputs" / "all_with_astar.json",
        help="Results JSON from tot_creative_writing*.py",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=RESULTS_DIR / "json_outputs" / "showcase_examples.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=RESULTS_DIR / "showcase_examples.md",
        help="Output markdown path.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=3,
        help="How many best and worst examples to include.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=1400,
        help="Max passage chars shown in markdown.",
    )
    return parser.parse_args()


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + " ...[truncated]"


def _normalize_rows(raw: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    # Defensive parsing: results JSONs from older runs sometimes have missing
    # or stringified `score` fields. Silently skipping them is preferable to
    # crashing on a single bad row mid-report.
    rows: list[dict[str, Any]] = []
    for method, entries in raw.items():
        for e in entries:
            if "score" not in e:
                continue
            try:
                score = float(e["score"])
            except (TypeError, ValueError):
                continue
            rows.append(
                {
                    "method": method,
                    "id": e.get("id"),
                    "score": score,
                    "sentences": e.get("sentences", []),
                    "passage": e.get("passage", ""),
                }
            )
    return rows


def _build_showcase(rows: list[dict[str, Any]], top_n: int) -> dict[str, Any]:
    by_method: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_method.setdefault(r["method"], []).append(r)

    def picks(items: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        # Secondary sort by id makes ties deterministic across reruns so the
        # markdown report doesn't reshuffle examples on every regeneration.
        ranked = sorted(items, key=lambda r: (r["score"], r.get("id")))
        return {
            "lowest": ranked[:top_n],
            "highest": list(reversed(ranked[-top_n:])),
        }

    out: dict[str, Any] = {
        "overall": picks(rows),
        "by_method": {},
    }
    for method, items in sorted(by_method.items()):
        out["by_method"][method] = picks(items)
    return out


def _entry_to_md(e: dict[str, Any], max_chars: int) -> str:
    sid = e.get("id", "?")
    method = e.get("method", "")
    score = e.get("score", 0.0)
    endings = e.get("sentences", [])
    endings_txt = "\n".join(f"{i+1}. {s}" for i, s in enumerate(endings)) if endings else "(not available)"
    passage = _truncate(e.get("passage", ""), max_chars=max_chars)
    return (
        f"- id: `{sid}` | method: `{method}` | score: `{score:.4f}`\n"
        f"  endings:\n{endings_txt}\n\n"
        f"  passage:\n\n"
        f"  > {passage.replace(chr(10), chr(10) + '  > ')}\n"
    )


def _write_markdown(showcase: dict[str, Any], out_md: Path, max_chars: int) -> None:
    lines: list[str] = []
    lines.append("# Creative Writing Showcase: Best vs Worst\n")
    lines.append("## Overall Lowest Scoring\n")
    for e in showcase["overall"]["lowest"]:
        lines.append(_entry_to_md(e, max_chars))
    lines.append("\n## Overall Highest Scoring\n")
    for e in showcase["overall"]["highest"]:
        lines.append(_entry_to_md(e, max_chars))

    for method, block in showcase["by_method"].items():
        lines.append(f"\n## Method: `{method}` - Lowest\n")
        for e in block["lowest"]:
            lines.append(_entry_to_md(e, max_chars))
        lines.append(f"\n## Method: `{method}` - Highest\n")
        for e in block["highest"]:
            lines.append(_entry_to_md(e, max_chars))

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    raw = json.loads(args.input.read_text())
    if not isinstance(raw, dict):
        raise ValueError("Expected results JSON to be a dict of method -> entries.")

    rows = _normalize_rows(raw)
    if not rows:
        raise ValueError("No rows with numeric `score` found in input.")

    showcase = _build_showcase(rows, top_n=args.top_n)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(showcase, indent=2))
    _write_markdown(showcase, out_md=args.out_md, max_chars=args.max_chars)

    print(f"Wrote: {args.out_json}")
    print(f"Wrote: {args.out_md}")
    print(f"Rows processed: {len(rows)}")


if __name__ == "__main__":
    main()
