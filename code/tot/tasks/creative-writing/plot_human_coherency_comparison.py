#!/usr/bin/env python3
"""
Bar chart in the style of paper Fig. 5(b): CoT > ToT, Similar, ToT > CoT.

Reads merged_ratings.csv + pairs_key.json (same as human_pairwise_eval aggregate).
Reuses the aggregation helpers from `human_pairwise_eval.py` via runtime
import so the kappa/majority logic stays single-sourced.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]


def _load_aggregate_helpers():
    # Loaded by path rather than `import` because creative-writing/ isn't a
    # package (filename has a hyphen). Keeps both files runnable directly.
    path = ROOT / "tot" / "tasks" / "creative-writing" / "human_pairwise_eval.py"
    spec = importlib.util.spec_from_file_location("human_pairwise_eval", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def count_majority_outcomes(key_path: Path, ratings_path: Path) -> tuple[str, str, int, int, int]:
    """Returns (method_a, method_b, wins_a, wins_b, n_similar).

    Per-pair majority across raters is the unit of analysis (paper-style).
    Pairs where raters disagreed three ways are dropped, not bucketed — they
    don't belong in any of the three bars.
    """
    hpe = _load_aggregate_helpers()
    key_list = json.loads(key_path.read_text(encoding="utf-8"))
    key_by_pair = {int(r["pair_id"]): r for r in key_list}
    method_a = key_list[0]["method_a"]
    method_b = key_list[0]["method_b"]
    wins_a = wins_b = wins_sim = 0
    with ratings_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pid = int(row["pair_id"])
            krow = key_by_pair.get(pid)
            if not krow:
                continue
            c1 = hpe._choice_to_int(row.get("rater_1_choice", ""))
            c2 = hpe._choice_to_int(row.get("rater_2_choice", ""))
            c3 = hpe._choice_to_int(row.get("rater_3_choice", ""))
            if c1 is None and c2 is None and c3 is None:
                continue
            o1 = hpe._vote_to_outcome(c1, krow, method_a, method_b)
            o2 = hpe._vote_to_outcome(c2, krow, method_a, method_b)
            o3 = hpe._vote_to_outcome(c3, krow, method_a, method_b)
            maj = hpe._majority_outcome(o1, o2, o3)
            if maj is None:
                continue
            if maj == method_a:
                wins_a += 1
            elif maj == method_b:
                wins_b += 1
            else:
                wins_sim += 1
    return method_a, method_b, wins_a, wins_b, wins_sim


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--key", type=Path, required=True)
    p.add_argument("--ratings", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--title", default="(b) Human coherency comparison.")
    args = p.parse_args()

    ma, mb, tot_wins, cot_wins, n_sim = count_majority_outcomes(args.key, args.ratings)
    n = tot_wins + cot_wins + n_sim

    def pretty(m: str) -> str:
        aliases = {"tot": "ToT", "cot": "CoT", "io": "IO"}
        return aliases.get(m, m.upper())

    # Late import: matplotlib is slow to load and only needed for plotting.
    # The aggregation path above stays usable in headless / CI contexts.
    import matplotlib.pyplot as plt

    # Bar order matches paper Fig 5(b) exactly so side-by-side comparisons read
    # the same way: better-than-A, tied, better-than-B.
    labels = [f"{pretty(mb)} > {pretty(ma)}", "Similar", f"{pretty(ma)} > {pretty(mb)}"]
    values = [cot_wins, n_sim, tot_wins]
    colors = ["#c17f5c", "#8d9196", "#6fa572"]  # muted orange-brown, grey, sage green

    ymax = max(10, int(max(values) * 1.25) + 2)
    fig, ax = plt.subplots(figsize=(5.2, 4.0), dpi=150)
    x = range(len(labels))
    bars = ax.bar(x, values, color=colors, width=0.62, edgecolor="white", linewidth=0.8)

    for rect, v in zip(bars, values):
        if v <= 0:
            continue
        # Short bars need the label nudged so it doesn't fall below the axis;
        # very tall bars look best with the label visually centered.
        yc = max(v * 0.5, 0.55) if v < 3 else v * 0.5
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            min(yc, v - 0.08) if v >= 1 else yc,
            str(v),
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
            color="white",
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title(args.title, fontsize=11, pad=10)
    ax.set_ylim(0, ymax)
    ax.yaxis.grid(True, linestyle="-", alpha=0.35, color="#b0b0b0")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.text(0.5, 0.02, f"n = {n} passage pairs", ha="center", fontsize=9, color="#555")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"Wrote {args.out}")
    pa, pb = pretty(ma), pretty(mb)
    print(
        f"Summary: humans prefer {pa} over {pb} in {tot_wins} out of {n} passage pairs, "
        f"while prefer {pb} over {pa} in {cot_wins}, and {n_sim} similarly coherent."
    )


if __name__ == "__main__":
    main()
