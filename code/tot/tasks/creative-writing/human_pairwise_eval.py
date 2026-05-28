#!/usr/bin/env python3
"""
Pairwise human evaluation for creative writing (e.g. tot vs tot_astar).

Workflow
--------
1) generate  — build N comparison pairs, shuffle left/right to reduce position bias,
               export rater sheets + a private key for scoring.
2) aggregate — load completed ratings + key; report pairwise Cohen's kappa and majority vote.

Cohen's κ (two raters): substantial > 0.6, moderate 0.4–0.6, poor < 0.4
Use at least 2 raters for κ; with 3 raters we report κ for each pair of raters.

Example
-------
  python3 code/tot/tasks/creative-writing/human_pairwise_eval.py generate \\
    --input results/creative-writing/json_outputs/all_with_astar.json \\
    --method-a tot --method-b tot_astar \\
    --n-pairs 100 \\
    --out-dir results/creative-writing/human_eval

  # After filling rater_1_choice, rater_2_choice, rater_3_choice (1=left, 2=right, 3=similar):

  python3 code/tot/tasks/creative-writing/human_pairwise_eval.py aggregate \\
    --ratings results/creative-writing/human_eval/pairs_rater_template.csv \\
    --key results/creative-writing/human_eval/pairs_key.json
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_INPUT = ROOT / "results" / "creative-writing" / "json_outputs" / "all_with_astar.json"


def _cohen_kappa_binary(y1: list[int], y2: list[int]) -> float:
    """Cohen's kappa for two raters, labels in {0,1}. Same length, no NaNs.

    Reimplemented here (rather than pulling scikit-learn) to keep this script
    dependency-light — it has to run on graders' laptops without a full data
    stack. The 1e-12 epsilon avoids a divide-by-zero when both raters always
    vote the same way (chance agreement → 1.0 → undefined kappa).
    """
    n = len(y1)
    if n == 0 or len(y2) != n:
        raise ValueError("Rater lists must be non-empty and equal length.")
    agree = sum(1 for a, b in zip(y1, y2) if a == b)
    p_o = agree / n
    p1 = sum(y1) / n
    p2 = sum(y2) / n
    p_e = p1 * p2 + (1 - p1) * (1 - p2)
    if p_e >= 1.0 - 1e-12:
        return 1.0 if p_o == 1.0 else 0.0
    return (p_o - p_e) / (1 - p_e)


def _load_results(path: Path) -> dict[str, list[dict[str, Any]]]:
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Results JSON must be a dict: method -> list of entries.")
    return data


def _pair_index(
    results: dict[str, list[dict[str, Any]]],
    method_a: str,
    method_b: str,
) -> dict[Any, tuple[dict, dict]]:
    """task_id -> (entry_a, entry_b) for tasks present in both methods.

    Intersecting on task_id is essential: comparing different tasks would mix
    a method-quality signal with an input-difficulty signal, ruining the AB.
    """
    by_id_a = {e["id"]: e for e in results.get(method_a, []) if "id" in e}
    by_id_b = {e["id"]: e for e in results.get(method_b, []) if "id" in e}
    common = set(by_id_a) & set(by_id_b)
    return {tid: (by_id_a[tid], by_id_b[tid]) for tid in common}


def cmd_generate(args: argparse.Namespace) -> None:
    results = _load_results(args.input)
    index = _pair_index(results, args.method_a, args.method_b)
    if not index:
        raise SystemExit(f"No overlapping task ids between {args.method_a} and {args.method_b}.")

    task_ids = list(index.keys())
    random.seed(args.seed)
    random.shuffle(task_ids)
    task_ids = task_ids[: args.n_pairs]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    template_path = out_dir / "pairs_rater_template.csv"
    key_path = out_dir / "pairs_key.json"
    readme_path = out_dir / "INSTRUCTIONS.md"

    key_rows: list[dict[str, Any]] = []
    fieldnames = [
        "pair_id",
        "task_id",
        "passage_left",
        "passage_right",
        "rater_1_choice",
        "rater_2_choice",
        "rater_3_choice",
    ]

    with template_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for i, tid in enumerate(task_ids):
            ea, eb = index[tid]
            pa = ea.get("passage", "")
            pb = eb.get("passage", "")
            # Per-pair coin flip: humans show measurable position bias toward
            # the left passage. Random L/R assignment makes the bias wash out
            # across the dataset rather than systematically favoring one method.
            if random.random() < 0.5:
                left_m, right_m = args.method_a, args.method_b
                left_p, right_p = pa, pb
            else:
                left_m, right_m = args.method_b, args.method_a
                left_p, right_p = pb, pa

            pair_id = i + 1
            w.writerow(
                {
                    "pair_id": pair_id,
                    "task_id": tid,
                    "passage_left": left_p,
                    "passage_right": right_p,
                    "rater_1_choice": "",
                    "rater_2_choice": "",
                    "rater_3_choice": "",
                }
            )
            key_rows.append(
                {
                    "pair_id": pair_id,
                    "task_id": tid,
                    "left_method": left_m,
                    "right_method": right_m,
                    "method_a": args.method_a,
                    "method_b": args.method_b,
                    "sentences": ea.get("sentences", []),
                }
            )

    key_path.write_text(json.dumps(key_rows, indent=2), encoding="utf-8")

    readme_path.write_text(
        f"""# Pairwise human evaluation

## Setup
- **Raters:** at least 2 (ideally 3). Do not discuss choices until everyone has finished.
- **Task:** For each row, read **passage_left** and **passage_right**. Choose which passage is **better overall**
  for coherence / quality for this creative-writing task (same task_id; endings were fixed by the dataset).

## How to fill the CSV
- Open `pairs_rater_template.csv` in a spreadsheet.
- Each rater fills **only their column**:
  - `rater_1_choice`: **1** = left better, **2** = right better, **3** = similarly coherent (tie).
  - Same for `rater_2_choice`, `rater_3_choice`.
- Do not reorder rows or edit passages.

## Scoring
After all ratings are complete, run:

```bash
python3 code/tot/tasks/creative-writing/human_pairwise_eval.py aggregate \\
  --ratings {template_path.as_posix()} \\
  --key {key_path.as_posix()}
```

## Interpreting Cohen's κ (pairwise between raters)
- κ > 0.6: substantial agreement
- 0.4 ≤ κ ≤ 0.6: moderate
- κ < 0.4: poor (interpret human eval cautiously)

## Note
- `pairs_key.json` maps each row's left/right to `{args.method_a}` vs `{args.method_b}` for aggregation.
  Keep it private from raters if you want blind comparison.
""",
        encoding="utf-8",
    )

    print(f"Wrote {template_path}")
    print(f"Wrote {key_path}")
    print(f"Wrote {readme_path}")
    print(f"Pairs: {len(task_ids)} (from {len(index)} eligible tasks)")


def _choice_to_int(s: str) -> int | None:
    """1=left better, 2=right better, 3=similarly coherent (paper-style third option)."""
    s = str(s).strip()
    if not s:
        return None
    if s in ("1", "2", "3"):
        return int(s)
    raise ValueError(f"Invalid choice {s!r}; expected 1, 2, or 3.")


def _majority_vote(a: int | None, b: int | None, c: int | None) -> int | None:
    # Majority semantics:
    #   1 rater  -> that rater wins (single-rater pilots are still useful)
    #   2 raters -> must agree, else None (no tie-break)
    #   3 raters -> any 2-out-of-3 wins; 1-1-1 (all different) yields None
    # Returning None for ties is intentional: downstream reporting buckets them
    # as "inter-rater no majority" rather than silently picking a side.
    vals = [x for x in (a, b, c) if x is not None]
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    if len(vals) == 2:
        return vals[0] if vals[0] == vals[1] else None
    if vals[0] == vals[1] or vals[0] == vals[2]:
        return vals[0]
    if vals[1] == vals[2]:
        return vals[1]
    return None


def _vote_to_outcome(
    choice: int | None, krow: dict[str, Any], method_a: str, method_b: str
) -> str | None:
    """Map raw vote to method_a, method_b, or 'similar'."""
    if choice is None:
        return None
    if choice == 3:
        return "similar"
    if choice == 1:
        pref = krow["left_method"]
    elif choice == 2:
        pref = krow["right_method"]
    else:
        raise ValueError(f"Invalid vote {choice}")
    if pref not in (method_a, method_b):
        raise ValueError(f"Preferred method {pref!r} not in {{method_a, method_b}}")
    return pref


def _majority_outcome(o1: str | None, o2: str | None, o3: str | None) -> str | None:
    """Majority over {method_a, method_b, similar}; 2 raters must agree; 3 raters need ≥2."""
    vals = [x for x in (o1, o2, o3) if x is not None]
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    cnt = Counter(vals)
    best, nbest = cnt.most_common(1)[0]
    if len(vals) == 2:
        return best if nbest == 2 else None
    if nbest >= 2:
        return best
    return None


def _cohen_kappa_nominal(y1: list[int], y2: list[int], *, n_classes: int) -> float:
    """Cohen's κ for nominal labels in 0..n_classes-1 (same length).

    Needed alongside the binary version because the paper-style "similar"
    option creates a third class. Reporting both kappas separately lets us see
    whether disagreement comes from strict pref flips (binary) or from
    raters differing on what counts as "similar" (nominal).
    """
    n = len(y1)
    if n == 0 or len(y2) != n:
        raise ValueError("Rater lists must be non-empty and equal length.")
    conf = [[0] * n_classes for _ in range(n_classes)]
    for a, b in zip(y1, y2):
        if not (0 <= a < n_classes and 0 <= b < n_classes):
            raise ValueError("Labels out of range")
        conf[a][b] += 1
    p_o = sum(conf[i][i] for i in range(n_classes)) / n
    row_m = [sum(conf[i][j] for j in range(n_classes)) / n for i in range(n_classes)]
    col_m = [sum(conf[i][j] for i in range(n_classes)) / n for j in range(n_classes)]
    p_e = sum(row_m[k] * col_m[k] for k in range(n_classes))
    if p_e >= 1.0 - 1e-12:
        return 1.0 if p_o >= 1.0 - 1e-12 else 0.0
    return (p_o - p_e) / (1.0 - p_e)


def cmd_aggregate(args: argparse.Namespace) -> None:
    key_list = json.loads(Path(args.key).read_text(encoding="utf-8"))
    key_by_pair = {int(r["pair_id"]): r for r in key_list}

    rows: list[dict[str, str]] = []
    with Path(args.ratings).open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    skipped = sum(1 for row in rows if int(row["pair_id"]) not in key_by_pair)

    method_a = key_list[0]["method_a"]
    method_b = key_list[0]["method_b"]

    def _outcome_class(out: str) -> int:
        if out == method_a:
            return 0
        if out == method_b:
            return 1
        if out == "similar":
            return 2
        raise ValueError(out)

    # Two κ flavors below answer two different questions:
    #  - binary: of the strict-preference votes only, do raters agree on the
    #    winner? Drops all "similar" votes; matches the paper's strict κ.
    #  - nominal: include "similar" as a third class. Sensitive to whether
    #    raters draw the "obviously different" line in the same place.
    def aligned_binary(
        rows_in: list[dict[str, str]], col_a: str, col_b: str
    ) -> tuple[list[int], list[int]]:
        aa: list[int] = []
        bb: list[int] = []
        for row in rows_in:
            try:
                a = _choice_to_int(row.get(col_a, ""))
                b = _choice_to_int(row.get(col_b, ""))
            except ValueError:
                continue
            # Include only pairs where BOTH raters had a strict pref; a "similar"
            # from either side disqualifies the row from the binary kappa.
            if a is not None and b is not None and a in (1, 2) and b in (1, 2):
                aa.append(a - 1)
                bb.append(b - 1)
        return aa, bb

    def aligned_nominal(
        rows_in: list[dict[str, str]], col_a: str, col_b: str
    ) -> tuple[list[int], list[int]]:
        aa: list[int] = []
        bb: list[int] = []
        for row in rows_in:
            pid = int(row["pair_id"])
            krow = key_by_pair.get(pid)
            if not krow:
                continue
            try:
                ca = _choice_to_int(row.get(col_a, ""))
                cb = _choice_to_int(row.get(col_b, ""))
            except ValueError:
                continue
            if ca is None or cb is None:
                continue
            try:
                oa = _vote_to_outcome(ca, krow, method_a, method_b)
                ob = _vote_to_outcome(cb, krow, method_a, method_b)
            except ValueError:
                continue
            if oa is None or ob is None:
                continue
            aa.append(_outcome_class(oa))
            bb.append(_outcome_class(ob))
        return aa, bb

    print("-- Pairwise Cohen's κ (strict A vs B only: votes 1–2, coded 0=left / 1=right) --")
    for label, ca, cb in [
        ("rater_1 vs rater_2", "rater_1_choice", "rater_2_choice"),
        ("rater_1 vs rater_3", "rater_1_choice", "rater_3_choice"),
        ("rater_2 vs rater_3", "rater_2_choice", "rater_3_choice"),
    ]:
        y1, y2 = aligned_binary(rows, ca, cb)
        if len(y1) < 2:
            print(f"{label}: n={len(y1)} (need ≥2 paired strict prefs) — skip")
            continue
        k = _cohen_kappa_binary(y1, y2)
        print(f"{label}: n={len(y1)}, κ={k:.3f}")

    print(
        f"\n-- Pairwise Cohen's κ (nominal 3-way: {method_a}=0, {method_b}=1, similar=2) --"
    )
    for label, ca, cb in [
        ("rater_1 vs rater_2", "rater_1_choice", "rater_2_choice"),
        ("rater_1 vs rater_3", "rater_1_choice", "rater_3_choice"),
        ("rater_2 vs rater_3", "rater_2_choice", "rater_3_choice"),
    ]:
        y1, y2 = aligned_nominal(rows, ca, cb)
        if len(y1) < 2:
            print(f"{label}: n={len(y1)} (need ≥2 paired ratings) — skip")
            continue
        k = _cohen_kappa_nominal(y1, y2, n_classes=3)
        print(f"{label}: n={len(y1)}, κ={k:.3f}")

    wins_a = wins_b = wins_similar = inter_rater_tie = incomplete = 0
    for row in rows:
        pid = int(row["pair_id"])
        krow = key_by_pair.get(pid)
        if not krow:
            continue
        try:
            c1 = _choice_to_int(row.get("rater_1_choice", ""))
            c2 = _choice_to_int(row.get("rater_2_choice", ""))
            c3 = _choice_to_int(row.get("rater_3_choice", ""))
        except ValueError:
            incomplete += 1
            continue
        if c1 is None and c2 is None and c3 is None:
            incomplete += 1
            continue
        try:
            o1 = _vote_to_outcome(c1, krow, method_a, method_b)
            o2 = _vote_to_outcome(c2, krow, method_a, method_b)
            o3 = _vote_to_outcome(c3, krow, method_a, method_b)
        except ValueError:
            incomplete += 1
            continue
        maj = _majority_outcome(o1, o2, o3)
        if maj is None:
            inter_rater_tie += 1
            continue
        if maj == method_a:
            wins_a += 1
        elif maj == method_b:
            wins_b += 1
        else:
            wins_similar += 1

    n_pairs = len(rows)
    n_slotted = wins_a + wins_b + wins_similar
    print("\n-- Majority per pair (method_a / method_b / similarly coherent) --")
    print(f"method_a ({method_a}) wins: {wins_a}")
    print(f"method_b ({method_b}) wins: {wins_b}")
    print(f"similarly coherent (majority): {wins_similar}")
    print(f"inter-rater no majority: {inter_rater_tie}")
    print(f"incomplete rows (no usable votes): {incomplete}")
    if n_slotted:
        print(
            f"\nPaper-style line ({n_pairs} pairs): "
            f"humans prefer {method_a} over {method_b} in {wins_a} pairs, "
            f"{method_b} over {method_a} in {wins_b}, "
            f"and {wins_similar} similarly coherent (majority)."
        )
    if skipped:
        print(f"(skipped {skipped} rating rows with unknown pair_id)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pairwise human eval sheets + Cohen's kappa aggregation.")
    sub = parser.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="Build rater CSV + key from results JSON.")
    g.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    g.add_argument("--method-a", default="tot")
    g.add_argument("--method-b", default="tot_astar")
    g.add_argument("--n-pairs", type=int, default=100)
    g.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results" / "creative-writing" / "human_eval",
    )
    g.add_argument("--seed", type=int, default=42)
    g.set_defaults(func=cmd_generate)

    a = sub.add_parser("aggregate", help="Compute kappa + majority vote from filled CSV.")
    a.add_argument("--ratings", type=Path, required=True, help="Filled pairs_rater_template.csv")
    a.add_argument("--key", type=Path, required=True, help="pairs_key.json from generate")
    a.set_defaults(func=cmd_aggregate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
