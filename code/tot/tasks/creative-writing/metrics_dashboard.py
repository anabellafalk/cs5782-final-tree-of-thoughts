#!/usr/bin/env python3
"""Creative-writing metrics dashboard.

Reads `results/metrics_log.jsonl` (one row per task-run) and produces every
plot + table under `results/creative-writing/dashboard/`. The file is large
because each plot has hand-tuned styling for the report, but the overall
structure is:

  1. Constants + theme   — paper-aligned palette + axis ordering
  2. Slice helpers       — each `slice_*` isolates ONE ablation axis by
                           whitelisting result-file basenames + methods.
                           Slicing on basename rather than AxesConfig is what
                           keeps overlapping runs (e.g. baseline.json appears
                           in several slices) from polluting each other.
  3. Balance helpers     — downsample to equal n per method/bucket so plots
                           compare apples to apples (no method gets a bigger
                           sample by accident).
  4. Stats helpers       — ANOVA / paired-t / Welch + multiple-comparison
                           corrections. Reimplemented thinly so dashboards
                           stay reproducible without pinning a heavy stats lib.
  5. Plot functions      — one per panel. The `out_dir` arg roots all paths so
                           subdirectories stay parallel between the dashboard
                           tree and the paper's figure references.
  6. `main` dispatch     — runs every plot, in the order they appear in the
                           paper, then writes summary tables.

Run with `--clean` to delete prior outputs before rebuilding (safer than
overwriting because removed plots get cleaned up too). `--correlation-only`
short-circuits to just the tot↔tot_astar metric-delta correlations.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from scipy import stats
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[4]

# ── Theme (white background + sage / forest greens) ─────────────────────────────
C_WHITE = "#FFFFFF"
C_CREAM = "#FBFCF4"
C_SAGE_LIGHT = "#D5E3DB"
C_SAGE_MID = "#D9DED5"
C_GREEN_MUTED = "#8DAA9D"
C_GREEN_MD = "#5E8C76"
C_GREEN_DARK = "#445E4F"
C_FOREST = "#374836"
C_FOREST_DEEP = "#1B3328"
C_TEXT = C_FOREST_DEEP
C_TEXT_MUTED = C_GREEN_DARK
C_GRID = C_SAGE_MID
C_ACCENT_LINE = C_GREEN_MD
C_PAPER_LINE = C_FOREST
C_FALLBACK_SERIES = "#6B7F72"
CORR_CMAP = LinearSegmentedColormap.from_list(
    "forest_corr",
    [C_FOREST_DEEP, C_SAGE_LIGHT, C_WHITE, C_SAGE_MID, C_FOREST],
    N=256,
)

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update(
    {
        "figure.facecolor": C_WHITE,
        "axes.facecolor": C_WHITE,
        "axes.edgecolor": C_GRID,
        "axes.labelcolor": C_TEXT,
        "axes.titlecolor": C_TEXT,
        "xtick.color": C_TEXT_MUTED,
        "ytick.color": C_TEXT_MUTED,
        "text.color": C_TEXT,
        "grid.color": C_GRID,
        "grid.linewidth": 0.6,
        "legend.facecolor": C_CREAM,
        "legend.edgecolor": C_GRID,
        "font.family": "monospace",
        "axes.titlesize": 11,
        "axes.labelsize": 9,
    }
)

# Canonical display order for methods and prompt variants. All grouped plots
# must use these (via `method_order_present`) so legends and bar positions
# match across figures — readers can compare panels at a glance.
METHOD_ORDER = ["io", "cot", "tot", "tot_astar", "hybrid_tot", "io_refine", "tot_refine"]
PROMPT_ORDER = ["paper", "definition", "criteria"]
# Greens + reds / roses so line plots (k, n_votes) stay distinguishable on white backgrounds.
METHOD_PALETTE = {
    "io": "#1B3328",
    "cot": "#C44536",
    "tot": "#2F6B4A",
    "tot_astar": "#4A7BA7",
    "hybrid_tot": "#B86B00",
    "io_refine": "#E8A0A0",
    "tot_refine": "#7A1E2D",
}
METHOD_LINE_MARKERS: dict[str, str] = {
    "io": "o",
    "cot": "s",
    "tot": "^",
    "tot_astar": "D",
    "hybrid_tot": "P",
    "io_refine": "v",
    "tot_refine": "X",
}
DEFAULT_GROUP_BY = ["method", "k"]

DEFAULT_INPUT = ROOT / "results" / "metrics_log.jsonl"
DEFAULT_OUT_DIR = ROOT / "results" / "creative-writing" / "dashboard"
DEFAULT_RESULTS_JSON = ROOT / "results" / "creative-writing" / "json_outputs" / "all_with_astar.json"
DEFAULT_OVERVIEW_METHODS_JSON = ROOT / "results" / "creative-writing" / "json_outputs" / "BASELINE_minimal4o.json"

# ── Slice whitelists ────────────────────────────────────────────────────────────
# Each frozenset names the result-JSON basenames that are valid for a specific
# experimental question. Slicing this way (rather than by AxesConfig fields) is
# robust to ad-hoc reruns: as long as the run went into the right out_file,
# it ends up in the right slice. Adding a new ablation = pick a basename
# convention, drop it into the relevant set here, and the dashboard picks it up.
ASTAR_MERGE_JSON_NAME = "all_with_astar.json"
PLAN_METHODS_JSON_NAME = "plan_methods.json"
ASTAR_METHOD = "tot_astar"
SCORER_COMPARE_JSONS: frozenset[str] = frozenset({"baseline.json", "scorer4.1.json"})
PAPER_CORE_OUT_FILES: frozenset[str] = frozenset(
    {
        "baseline.json",
        "criteria.json",
        "def_4.1.json",
        "k_1.json",
        "k_3.json",
        "k_10.json",
        "n_votes_1.json",
        "n_votes_3.json",
        "n_votes_10.json",
        "score_definition.json",
        "scorer4.1.json",
    }
)
K_ABLATION_OUT_FILES: frozenset[str] = frozenset({"baseline.json", "k_1.json", "k_3.json", "k_10.json"})
N_VOTES_ABLATION_OUT_FILES: frozenset[str] = frozenset(
    {"baseline.json", "n_votes_1.json", "n_votes_3.json", "n_votes_10.json"}
)
SCORE_PROMPT_OUT_FILES: frozenset[str] = frozenset(
    {
        "baseline.json",
        "definition.json",
        "score_definition.json",
        "criteria.json",
    }
)
# Paper Table 5 methods. tot_astar / hybrid_tot are deliberately excluded here
# because they have their own dedicated slices/plots — mixing them into the
# paper-core would distort the mean-score comparison against published numbers.
PAPER_CORE_METHODS: frozenset[str] = frozenset({"io", "cot", "tot", "io_refine", "tot_refine"})
REFINE_METHODS: frozenset[str] = frozenset({"io_refine", "tot_refine"})
PLAN_METHODS_COMPARE_ORDER = ["tot", "tot_astar", "hybrid_tot"]
# Δ-correlations probe: "when A* beats ToT on the same task, does it also lower
# entropy/variance?" score_delta is the response, the Δ predictors are paired.
DEFAULT_CORRELATION_VARIABLES = [
    "delta_vote_entropy_plan",
    "delta_vote_entropy_passage",
    "delta_score_std",
]
CORRELATION_DELTA_LABELS: dict[str, str] = {
    "delta_vote_entropy_plan": "Δ plan vote entropy (A* − ToT)",
    "delta_vote_entropy_passage": "Δ passage vote entropy (A* − ToT)",
    "delta_score_std": "Δ score std (A* − ToT)",
}
DEFAULT_CORRELATION_PLOT_DIR = (
    ROOT / "results" / "creative-writing" / "dashboard" / "independent_exploration" / "astar" / "metric_delta_correlations"
)


# ── CLI ────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Creative-writing metrics dashboard: paper-core plots use a fixed whitelist of result JSONs "
            "and methods io/cot/tot/io_refine/tot_refine (balanced). "
            "tot↔tot_astar uses only all_with_astar.json (tot+tot_astar). "
            "tot↔tot_astar↔hybrid_tot uses only plan_methods.json. "
            "baseline.json↔scorer4.1.json comparison: analysis/ + tables/grouped_metrics_scorer_compare.* "
            "(paper-core methods; optional method×JSON balance). "
            "Optional correlation-only mode reads the metrics log."
        )
    )
    parser.add_argument(
        "--correlation-only",
        action="store_true",
        help="Only join tot vs tot_astar in metrics log, print correlations, optional --correlation-plot-dir.",
    )
    parser.add_argument(
        "--correlation-metrics-log",
        type=Path,
        default=DEFAULT_INPUT,
        help="Metrics JSONL for --correlation-only (default: same as --input default).",
    )
    parser.add_argument(
        "--correlation-out-file",
        default="creative-writing/json_outputs/all_with_astar.json",
        help="Filter log rows to this out_file for --correlation-only.",
    )
    parser.add_argument(
        "--correlation-prompt-variants",
        nargs="+",
        default=["paper", "definition", "criteria"],
        help="Score prompt variants for --correlation-only join.",
    )
    parser.add_argument(
        "--correlation-variables",
        nargs="+",
        default=DEFAULT_CORRELATION_VARIABLES,
        help="Columns to correlate with score_delta (default: metric deltas A*−ToT).",
    )
    parser.add_argument(
        "--correlation-plot-dir",
        type=Path,
        nargs="?",
        const=DEFAULT_CORRELATION_PLOT_DIR,
        default=None,
        help=(
            "With --correlation-only: write score_delta vs each predictor PNGs here. "
            "Use bare flag for default analysis/astar/metric_delta_correlations."
        ),
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="metrics_log.jsonl")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Dashboard output root.")
    parser.add_argument("--group-by", nargs="+", default=DEFAULT_GROUP_BY)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove prior dashboard plot dirs (baseline/, modifications/, independent_exploration/, tables/) before rebuilding.",
    )
    parser.add_argument(
        "--filter-methods",
        nargs="+",
        default=None,
        metavar="METHOD",
        help=(
            "Restrict the paper-core slice (whitelist JSONs) to these methods only. "
            "Does not affect tot↔tot_astar or plan_methods slices."
        ),
    )
    parser.add_argument(
        "--astar-merge-json-name",
        default=ASTAR_MERGE_JSON_NAME,
        metavar="FILENAME",
        help=(
            "Basename of results JSON for tot↔tot_astar-only plots and score-Δ correlations "
            f"(default: {ASTAR_MERGE_JSON_NAME})."
        ),
    )
    parser.add_argument(
        "--no-balance-methods",
        action="store_true",
        help=(
            "Disable equal-n downsampling on paper-core, all_with_astar, plan_methods, "
            "and baseline.json↔scorer4.1.json slices."
        ),
    )
    parser.add_argument(
        "--results-json",
        type=Path,
        default=None,
        help=(
            "Merged creative-writing results JSON (passage scores). "
            "If set, writes passage-level ToT vs tot_astar plots only into independent_exploration/astar_analysis/. "
            f"Typical: {DEFAULT_RESULTS_JSON.name}"
        ),
    )
    parser.add_argument(
        "--overview-methods-json",
        type=Path,
        default=DEFAULT_OVERVIEW_METHODS_JSON,
        help=(
            "Results JSON (method → list of entries with 'score') for overview/"
            "method_score_distribution.png and overview/method_mean_score_bar.png only. "
            f"Default: {DEFAULT_OVERVIEW_METHODS_JSON.name}. If missing, both use the paper-core metrics slice."
        ),
    )
    return parser.parse_args()


# ── Helpers ────────────────────────────────────────────────────────────────────


def ensure_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Combined `vote_entropy` lets plots show one entropy series per method
    # rather than two; skipna handles IO/CoT rows (no votes -> NaN).
    entropy_cols = [c for c in ["vote_entropy_plan", "vote_entropy_passage"] if c in df.columns]
    if entropy_cols:
        df = df.copy()
        df["vote_entropy"] = df[entropy_cols].mean(axis=1, skipna=True)
    return df


def out_file_basename(out_file: object | None) -> str:
    if out_file is None:
        return ""
    s = str(out_file).strip()
    return Path(s).name if s else ""


def slice_by_out_basenames(df: pd.DataFrame, basenames: frozenset[str] | set[str]) -> pd.DataFrame:
    if df.empty or "out_file" not in df.columns:
        return df.iloc[0:0].copy()
    return df[df["out_file"].map(lambda x: out_file_basename(x) in basenames)].copy()


def slice_astar_tot_compare(df: pd.DataFrame, merge_json_name: str) -> pd.DataFrame:
    """all_with_astar (or merge_name) rows, methods tot + tot_astar only."""
    s = slice_by_out_basenames(df, frozenset({merge_json_name}))
    return s[s["method"].isin(["tot", ASTAR_METHOD])].copy()


def slice_plan_methods_compare(df: pd.DataFrame) -> pd.DataFrame:
    """plan_methods.json rows: tot, tot_astar, hybrid_tot."""
    s = slice_by_out_basenames(df, frozenset({PLAN_METHODS_JSON_NAME}))
    return s[s["method"].isin(["tot", ASTAR_METHOD, "hybrid_tot"])].copy()


def slice_paper_core(df: pd.DataFrame) -> pd.DataFrame:
    """Whitelist JSONs + io/cot/tot/io_refine/tot_refine only (overview, ablations, etc.)."""
    s = slice_by_out_basenames(df, PAPER_CORE_OUT_FILES)
    return s[s["method"].isin(PAPER_CORE_METHODS)].copy()


def slice_scorer_baseline_vs_scorer41(df: pd.DataFrame) -> pd.DataFrame:
    """baseline.json vs scorer4.1.json only, paper-core methods (for scorer comparison)."""
    s = slice_by_out_basenames(df, SCORER_COMPARE_JSONS)
    return s[s["method"].isin(PAPER_CORE_METHODS)].copy()


def slice_k_ablation(df: pd.DataFrame) -> pd.DataFrame:
    """k ablations only: baseline + k_1/k_3/k_10, paper-core methods."""
    s = slice_by_out_basenames(df, K_ABLATION_OUT_FILES)
    return s[s["method"].isin(PAPER_CORE_METHODS)].copy()


def slice_n_votes_ablation(df: pd.DataFrame) -> pd.DataFrame:
    """n_votes ablations only: baseline + n_votes_1/n_votes_3/n_votes_10, paper-core methods."""
    s = slice_by_out_basenames(df, N_VOTES_ABLATION_OUT_FILES)
    return s[s["method"].isin(PAPER_CORE_METHODS)].copy()


def slice_score_prompt_compare(df: pd.DataFrame) -> pd.DataFrame:
    """Scoring prompt plots only: baseline + definition (+legacy score_definition) + criteria."""
    s = slice_by_out_basenames(df, SCORE_PROMPT_OUT_FILES)
    return s[s["method"].isin(PAPER_CORE_METHODS)].copy()


def balance_by_method_and_bucket(
    df: pd.DataFrame, method_col: str, bucket_col: str, *, random_state: int = 42
) -> pd.DataFrame:
    """Within each method, equalize row counts across bucket levels (downsample to min).

    Without this, an ablation with uneven coverage (e.g. k=1 has 100 tasks but
    k=10 only has 60) would show a spurious "k=10 looks worse" just because
    the smaller sample picks up more outlier means. The random_state mixes in
    method+bucket indices so different groups don't draw the same row indices.
    """
    if df.empty or method_col not in df.columns or bucket_col not in df.columns:
        return df
    out_parts: list[pd.DataFrame] = []
    for mi, (_, g) in enumerate(df.groupby(method_col, sort=False)):
        cnt = g.groupby(bucket_col, sort=False).size()
        if len(cnt) < 2 or int(cnt.min()) < 1:
            out_parts.append(g)
            continue
        n_min = int(cnt.min())
        acc: list[pd.DataFrame] = []
        for bi, (_, h) in enumerate(g.groupby(bucket_col, sort=False)):
            if len(h) > n_min:
                # Distinct seeds per (method, bucket) cell — using the same
                # seed would tend to pick the same row indices everywhere.
                acc.append(h.sample(n=n_min, random_state=random_state + mi * 1009 + bi * 503))
            else:
                acc.append(h)
        out_parts.append(pd.concat(acc, ignore_index=True))
    return pd.concat(out_parts, ignore_index=True)


def balance_method_counts(df: pd.DataFrame, *, random_state: int = 42) -> pd.DataFrame:
    """Downsample each method to the same count (min per-method n).

    Cross-method comparisons (e.g. tot vs tot_astar mean-score bars) need equal
    n; otherwise the method that happened to complete more tasks gets a tighter
    confidence interval, biasing significance tests. Skips downsampling when
    a method already has the minimum count (no-op for that group).
    """
    if df.empty or "method" not in df.columns:
        return df
    counts = df.groupby("method", sort=False).size()
    n_min = int(counts.min())
    if n_min < 1:
        return df
    parts: list[pd.DataFrame] = []
    for i, (_, g) in enumerate(df.groupby("method", sort=False)):
        if len(g) > n_min:
            parts.append(g.sample(n=n_min, random_state=random_state + i * 100_003))
        else:
            parts.append(g)
    return pd.concat(parts, ignore_index=True)


def augment_joined_with_metric_deltas(merged: pd.DataFrame) -> pd.DataFrame:
    """Add delta_* = (tot_astar − tot) for plan/passage entropy and score_std.

    Takes a frame already pivoted so the same task has tot_* and tot_astar_*
    columns side by side. Δ columns isolate the A* effect on each metric for
    that one task, which is what we correlate against score_delta.
    """
    if merged.empty:
        return merged
    out = merged.copy()
    mapping = (
        ("vote_entropy_plan", "delta_vote_entropy_plan"),
        ("vote_entropy_passage", "delta_vote_entropy_passage"),
        ("score_std", "delta_score_std"),
    )
    for base, dcol in mapping:
        a, b = f"{base}_tot_astar", f"{base}_tot"
        if a in out.columns and b in out.columns:
            out[dcol] = out[a] - out[b]
    return out


def grouped_path(out_dir: Path, subdir: str, fname: str) -> Path:
    p = out_dir / subdir
    p.mkdir(parents=True, exist_ok=True)
    return p / fname


def method_order_present(series: pd.Series) -> list[str]:
    # Preserves canonical METHOD_ORDER for known methods so panels stay aligned,
    # and appends unknowns alphabetically at the end (they'll show up the same
    # way across plots and won't crash the dashboard).
    present = set(series.dropna().astype(str).unique())
    return [m for m in METHOD_ORDER if m in present] + sorted(m for m in present if m not in METHOD_ORDER)


def method_line_markers_for_order(order: list[str]) -> list[str]:
    return [METHOD_LINE_MARKERS.get(m, "o") for m in order]


def _color_list(methods: list[str]) -> list[str]:
    return [METHOD_PALETTE.get(m, C_FALLBACK_SERIES) for m in methods]


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=C_WHITE, edgecolor="none")
    plt.close(fig)


def _stars(p: float) -> str:
    # Standard APA-style significance markers. `p != p` is the NaN check
    # without needing math.isnan (avoids a NumPy-vs-float discrepancy if `p`
    # comes back as a 0-d array from scipy).
    if p != p:
        return ""
    return "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))


def _format_p_short(p: float) -> str:
    if p != p:  # NaN
        return "—"
    if p < 1e-6:
        return f"{p:.1e}"
    if p < 0.0001:
        return f"{p:.0e}"
    return f"{p:.4g}"


def _one_way_anova_p(
    df: pd.DataFrame, group_col: str, value_col: str, *, levels: list[object] | None = None
) -> float:
    """Omnibus one-way ANOVA p across levels of group_col (treat levels as categorical)."""
    sub = df[[group_col, value_col]].dropna()
    if sub.empty:
        return float("nan")
    if levels is None:
        levels = sorted(sub[group_col].dropna().unique(), key=lambda x: (str(type(x).__name__), x))
    arrs = [
        sub.loc[sub[group_col] == g, value_col].to_numpy(dtype=float)
        for g in levels
        if (sub[group_col] == g).any()
    ]
    arrs = [a for a in arrs if len(a) > 0]
    if len(arrs) < 2:
        return float("nan")
    try:
        _, p = stats.f_oneway(*arrs)
        return float(p) if p == p and not np.isnan(p) else float("nan")
    except ValueError:
        return float("nan")


def _two_method_paired_and_welch(
    df: pd.DataFrame, method_col: str, value_col: str, m0: str, m1: str, *, task_id_col: str = "task_id"
) -> tuple[str | None, str | None]:
    """Return (paired_line, welch_line) for mean difference m1−m0; paired uses matched task_id if possible.

    We report BOTH because each answers a slightly different question:
      - paired-t (on per-task deltas): is m1 better than m0 on the same input?
        Higher power when methods share tasks (the common case).
      - Welch (unpaired): is m1's distribution different from m0's overall?
        Useful when task overlap is partial or absent.
    """
    sub = df[[method_col, value_col]].dropna()
    g0 = sub.loc[sub[method_col] == m0, value_col].to_numpy(dtype=float)
    g1 = sub.loc[sub[method_col] == m1, value_col].to_numpy(dtype=float)
    welch_line = None
    if len(g0) > 1 and len(g1) > 1:
        try:
            _, pw = stats.ttest_ind(g0, g1, equal_var=False)
            pw = float(pw)
            sw = _stars(pw)
            welch_line = f"Welch t (unpaired): p={_format_p_short(pw)}{(' ' + sw) if sw else ''}"
        except (ValueError, TypeError):
            pass
    paired_line = None
    if task_id_col in df.columns:
        w = df[[task_id_col, method_col, value_col]].dropna()
        pv = w.pivot_table(index=task_id_col, columns=method_col, values=value_col, aggfunc="first")
        if m0 in pv.columns and m1 in pv.columns:
            d = (pv[m1] - pv[m0]).dropna().to_numpy(dtype=float)
            d = d[np.isfinite(d)]
            if len(d) > 1 and np.std(d, ddof=1) > 1e-12:
                try:
                    _, pp = stats.ttest_1samp(d, 0.0)
                    pp = float(pp)
                    sp = _stars(pp)
                    paired_line = (
                        f"Paired t on Δ({m1}−{m0}), n={len(d)}: p={_format_p_short(pp)}{(' ' + sp) if sp else ''}"
                    )
                except (ValueError, TypeError):
                    pass
    return paired_line, welch_line


def _pairwise_welch_bonferroni_parts(
    df: pd.DataFrame, group_col: str, value_col: str, levels: list[str]
) -> list[str]:
    """One formatted string per unordered pair: Welch p with Bonferroni p_adj (m = #pairs).

    Bonferroni is intentionally conservative; with the small pair counts we
    produce here (≤ 21 pairs for 7 methods) it stays interpretable. We don't
    use Holm/BH because the dashboard reports raw and adjusted side by side
    anyway — readers can apply a less conservative correction if they prefer.
    """
    pairs: list[tuple[str, str]] = []
    for i, a in enumerate(levels):
        for b in levels[i + 1 :]:
            pairs.append((a, b))
    m_pairs = max(1, len(pairs))
    parts: list[str] = []
    for a, b in pairs:
        xa = df.loc[df[group_col] == a, value_col].dropna().to_numpy(dtype=float)
        xb = df.loc[df[group_col] == b, value_col].dropna().to_numpy(dtype=float)
        if len(xa) < 2 or len(xb) < 2:
            continue
        try:
            _, pr = stats.ttest_ind(xa, xb, equal_var=False)
            pr = float(pr)
        except (ValueError, TypeError, RuntimeError):
            continue
        if pr != pr:
            continue
        # Clamp at 1.0: Bonferroni can otherwise yield p_adj > 1 which is not
        # a valid probability and reads as a bug to readers.
        padj = min(1.0, pr * m_pairs)
        st = _stars(padj)
        parts.append(f"{a}–{b} p_adj={_format_p_short(padj)}{(' ' + st) if st else ''}")
    return parts


def _pairwise_welch_bonferroni_summary(
    df: pd.DataFrame, group_col: str, value_col: str, levels: list[str]
) -> str:
    """All unordered pairs on one line, ' | '-joined."""
    parts = _pairwise_welch_bonferroni_parts(df, group_col, value_col, levels)
    if not parts:
        return "(no pair with n≥2 both groups)"
    return " | ".join(parts)


def _omnibus_p_box(
    ax: plt.Axes, lines: list[str], *, x: float = 0.02, y: float = 0.98, fontsize: float = 7
) -> None:
    body = "\n".join(lines)
    if not body.strip():
        return
    ax.text(
        x,
        y,
        body,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=fontsize,
        color=C_FOREST_DEEP,
        bbox=dict(boxstyle="round,pad=0.32", facecolor=C_CREAM, edgecolor=C_GRID, alpha=0.94),
        zorder=20,
    )


def _stat_box(ax, r_p: float, p_p: float, r_s: float, p_s: float, n: int) -> None:
    text = (
        f"Pearson  r={r_p:.3f}  p={p_p:.3f} {_stars(p_p)}\n"
        f"Spearman ρ={r_s:.3f}  p={p_s:.3f} {_stars(p_s)}\n"
        f"n={n}"
    )
    ax.text(
        0.03,
        0.97,
        text,
        transform=ax.transAxes,
        fontsize=8.5,
        va="top",
        color=C_TEXT,
        bbox=dict(boxstyle="round,pad=0.4", facecolor=C_CREAM, edgecolor=C_GRID, alpha=0.95),
    )


def clean_dashboard_artifacts(out_dir: Path) -> None:
    """Remove plot dirs, all PNGs/CSVs/JSONs under out_dir, and grouped tables. Keeps README / .txt / GUIDE.md.

    Selective cleanup matters: deleting only generated plot files (and not the
    hand-written guides at the top of the tree) lets `--clean` run safely
    without losing documentation. We delete the parent subdirs at the end to
    purge any newly-orphaned subfolders left behind after PNG removal.
    """
    if not out_dir.is_dir():
        return
    for png in sorted(out_dir.rglob("*.png")):
        png.unlink()
    ie_astar = out_dir / "independent_exploration" / "astar"
    if ie_astar.is_dir():
        shutil.rmtree(ie_astar, ignore_errors=True)
    for name in (
        "baseline",
        "modifications",
        "independent_exploration",
        "tables",
    ):
        p = out_dir / name
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)


# ── Build grouped summary ──────────────────────────────────────────────────────


def build_grouped_table(df: pd.DataFrame, group_by: list[str]) -> pd.DataFrame:
    agg = (
        df.groupby(group_by, dropna=False)
        .agg(
            runs=("task_id", "count"),
            mean_score=("mean_score", "mean"),
            std_score=("mean_score", "std"),
            mean_cost_usd=("total_cost_usd", "mean"),
            total_cost_usd=("total_cost_usd", "sum"),
            mean_api_calls=("api_calls", "mean"),
            mean_prompt_tokens=("prompt_tokens", "mean"),
            mean_completion_tokens=("completion_tokens", "mean"),
        )
        .reset_index()
    )
    for col, digits in [
        ("mean_score", 4),
        ("std_score", 4),
        ("mean_cost_usd", 6),
        ("total_cost_usd", 6),
        ("mean_api_calls", 2),
    ]:
        agg[col] = agg[col].round(digits)
    return agg


# ── Plot functions ─────────────────────────────────────────────────────────────


def plot_method_score_distribution(
    df: pd.DataFrame, out_dir: Path, *, title_note: str | None = None
) -> None:
    """Violin + strip — shows full distribution, not just mean."""
    if "method" not in df.columns:
        return
    subset = df[["method", "mean_score"]].dropna()
    if subset.empty:
        return
    order = method_order_present(subset["method"])
    palette = {m: METHOD_PALETTE.get(m, C_FALLBACK_SERIES) for m in order}
    paper = {"io": 6.19, "cot": 6.93, "tot": 7.56, "io_refine": 7.67, "tot_refine": 7.91}

    fig, ax = plt.subplots(figsize=(11, 5))
    sns.violinplot(
        data=subset,
        x="method",
        y="mean_score",
        order=order,
        hue="method",
        hue_order=order,
        palette=palette,
        inner=None,
        alpha=0.35,
        ax=ax,
        legend=False,
    )
    for is_refine in (False, True):
        sub = subset[subset["method"].isin(REFINE_METHODS) if is_refine else ~subset["method"].isin(REFINE_METHODS)]
        if sub.empty:
            continue
        sns.stripplot(
            data=sub,
            x="method",
            y="mean_score",
            order=order,
            hue="method",
            hue_order=order,
            palette=palette,
            size=5.2 if is_refine else 3.5,
            marker="D" if is_refine else "o",
            alpha=0.82 if is_refine else 0.72,
            jitter=True,
            linewidth=0.45,
            edgecolor=C_FOREST,
            ax=ax,
            legend=False,
        )
    for i, method in enumerate(order):
        if method in paper:
            ax.hlines(paper[method], i - 0.4, i + 0.4, colors=C_PAPER_LINE, lw=1.5, linestyles="--")
    ax.plot([], [], color=C_PAPER_LINE, lw=1.5, linestyle="--", label="Paper target")
    ax.legend(fontsize=8)
    title = "Score Distribution by Method  (◇ = refine methods; violin + points, dashed = paper target)"
    if title_note:
        title = f"{title}\n{title_note}"
    ax.set_title(title)
    ax.set_xlabel("Method")
    ax.set_ylabel("Coherency score")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    _save(fig, grouped_path(out_dir, "baseline", "method_score_distribution.png"))


def plot_method_comparison_bar(
    df: pd.DataFrame, out_dir: Path, *, title_note: str | None = None
) -> None:
    """Vertical mean bar chart with 95% CI (1.96 × SEM). Solid fills, no hatching."""
    if "method" not in df.columns:
        return
    subset = df[["method", "mean_score"]].dropna()
    if subset.empty:
        return
    preferred_order = ["io", "cot", "tot", "io_refine", "tot_refine"]
    order = [m for m in preferred_order if m in set(subset["method"])]
    if not order:
        order = method_order_present(subset["method"])
    summary = subset.groupby("method")["mean_score"].agg(["mean", "std", "count"]).reindex(order).dropna()
    summary["ci95"] = summary["std"] / np.sqrt(summary["count"]) * 1.96
    # Solid colors: IO/IO+refine blue, CoT warm, ToT/ToT+refine green.
    paper_like_palette = {
        "io": "#4E79A7",
        "cot": "#C07A52",
        "tot": "#59A14F",
        "io_refine": "#4E79A7",
        "tot_refine": "#59A14F",
    }
    colors = [paper_like_palette.get(m, METHOD_PALETTE.get(m, C_FALLBACK_SERIES)) for m in summary.index]

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    x = np.arange(len(summary.index))
    bars = ax.bar(
        x,
        summary["mean"].to_numpy(),
        yerr=summary["ci95"].to_numpy(),
        color=colors,
        alpha=1.0,
        edgecolor="#3A3A3A",
        linewidth=1.2,
        error_kw=dict(ecolor="#2F2F2F", capsize=4, lw=1.5),
    )
    for bar, (_, row) in zip(bars, summary.iterrows()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            row["mean"] + row["ci95"] + 0.08,
            f"{row['mean']:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#333333",
            fontfamily="DejaVu Sans",
        )

    labels = {
        "io": "IO",
        "cot": "CoT",
        "tot": "ToT",
        "io_refine": "IO\n+refine",
        "tot_refine": "ToT\n+refine",
    }
    ax.set_xticks(x)
    ax.set_xticklabels(
        [labels.get(m, m) for m in summary.index],
        fontsize=11,
        color="#222222",
        fontfamily="DejaVu Sans",
    )
    ax.set_xlabel("")
    ax.set_ylabel("Mean coherency score", fontsize=11, fontfamily="DejaVu Sans")
    title = "Coherency scores"
    if title_note:
        title = f"{title}\n{title_note}"
    ax.set_title(title, fontsize=14, fontfamily="DejaVu Sans")
    ax.grid(False)
    if "tot" in order and "io_refine" in order:
        split_x = (order.index("tot") + order.index("io_refine")) / 2
        ax.axvline(split_x, color="#7A7A7A", linestyle="--", linewidth=2.0, alpha=0.9)
    lo = max(0.0, float((summary["mean"] - summary["ci95"]).min()) - 0.35)
    hi = min(10.0, float((summary["mean"] + summary["ci95"]).max()) + 0.85)
    ax.set_ylim(lo, hi)
    sns.despine(ax=ax, top=False, right=False)
    fig.tight_layout()
    _save(fig, grouped_path(out_dir, "baseline", "method_mean_score_bar.png"))


def plot_pairwise_score_gaps(df: pd.DataFrame, out_dir: Path) -> None:
    if "method" not in df.columns:
        return
    method_means = df.groupby("method")["mean_score"].mean()
    pairs = {
        "tot − io": ("tot", "io", 7.56 - 6.19),
        "tot − cot": ("tot", "cot", 7.56 - 6.93),
        "tot_refine − tot": ("tot_refine", "tot", 7.91 - 7.56),
        "io_refine − io": ("io_refine", "io", 7.67 - 6.19),
    }
    our_gaps, paper_gaps, labels = [], [], []
    for label, (m1, m2, paper_gap) in pairs.items():
        if m1 in method_means and m2 in method_means:
            our_gaps.append(method_means[m1] - method_means[m2])
            paper_gaps.append(paper_gap)
            labels.append(label)
    if not labels:
        return

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - w / 2, our_gaps, w, label="Ours", color=C_GREEN_MD, alpha=0.9)
    b2 = ax.bar(x + w / 2, paper_gaps, w, label="Paper", color=C_GREEN_MUTED, alpha=0.9)
    ax.axhline(0, color=C_TEXT_MUTED, lw=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10)
    ax.set_ylabel("Score gap")
    ax.set_title("Pairwise Score Gaps — Ours vs Paper Targets")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.2)
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.005,
            f"{h:+.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=C_TEXT_MUTED,
        )
    fig.tight_layout()
    _save(fig, grouped_path(out_dir, "baseline", "pairwise_score_gaps.png"))


def plot_astar_vs_tot_comparison(df: pd.DataFrame, out_dir: Path) -> None:
    """Side-by-side bars for tot vs tot_astar with t-test annotation."""
    if "method" not in df.columns:
        return
    metrics = {
        "mean_score": "Mean Score",
        "score_std": "Score Std Dev",
        "vote_entropy_plan": "H_plan",
        "vote_entropy_passage": "H_pass",
        "total_cost_usd": "Cost (USD)",
    }
    present = {k: v for k, v in metrics.items() if k in df.columns}
    subset = df[df["method"].isin(["tot", "tot_astar"])].copy()
    if subset.empty or not present:
        return

    n = len(present)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (col, label) in zip(axes, present.items()):
        cols_use = ["method", col] + (["task_id"] if "task_id" in subset.columns else [])
        data = subset[[c for c in cols_use if c in subset.columns]].dropna()
        order = [m for m in ["tot", "tot_astar"] if m in data["method"].values]
        colors = _color_list(order)
        summary = data.groupby("method")[col].agg(["mean", "std", "count"]).reindex(order)
        summary["ci95"] = summary["std"] / np.sqrt(summary["count"]) * 1.96

        bars = ax.bar(
            summary.index,
            summary["mean"],
            yerr=summary["ci95"],
            color=colors,
            alpha=0.85,
            edgecolor=C_FOREST,
            error_kw=dict(ecolor=C_TEXT_MUTED, capsize=5, lw=1.2),
            width=0.5,
        )

        y_top = max(summary["mean"] + summary["ci95"]) * 1.08
        ax.plot([0, 1], [y_top, y_top], color=C_TEXT_MUTED, lw=1)
        if len(order) == 2:
            pl, wl = _two_method_paired_and_welch(data, "method", col, order[0], order[1])
            p_bits = [x for x in (pl, wl) if x]
            if p_bits:
                ax.text(
                    0.5,
                    y_top * 1.012,
                    "\n".join(p_bits),
                    ha="center",
                    fontsize=6.2,
                    color=C_FOREST_DEEP,
                    va="bottom",
                )
        else:
            groups = [data[data["method"] == m][col].dropna().values for m in order]
            if len(groups) == 2 and len(groups[0]) > 1 and len(groups[1]) > 1:
                _, p = stats.ttest_ind(groups[0], groups[1], equal_var=False)
                st = _stars(p)
                ax.text(
                    0.5,
                    y_top * 1.015,
                    f"Welch t: p={_format_p_short(p)}{(' ' + st) if st else ''}",
                    ha="center",
                    fontsize=7.5,
                    color=C_FOREST_DEEP,
                )

        for bar, (idx, row) in zip(bars, summary.iterrows()):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + summary.loc[idx, "ci95"] * 0.1,
                f"{row['mean']:.3f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color=C_TEXT_MUTED,
            )

        ax.set_title(label, fontsize=9)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="y", alpha=0.2)

    fig.suptitle(
        "ToT vs ToT-A*  (mean ± 95% CI; brackets: paired t on matched task_id, else Welch)",
        fontsize=10,
        y=1.02,
    )
    fig.tight_layout()
    _save(fig, grouped_path(out_dir, "independent_exploration/astar_analysis", "astar_vs_tot_comparison.png"))


def plot_scorer_baseline_vs_scorer41(
    df: pd.DataFrame, out_dir: Path, *, balanced_method_json: bool = True
) -> None:
    """Grouped bars: mean coherency score only, baseline.json vs scorer4.1.json (paper-core methods)."""
    if df.empty or "results_json" not in df.columns or "mean_score" not in df.columns:
        return
    sub = df[df["results_json"].isin(SCORER_COMPARE_JSONS)].copy()
    sub = sub[sub["method"].isin(PAPER_CORE_METHODS)].dropna(subset=["mean_score"])
    if sub.empty or sub["results_json"].nunique() < 2:
        print("scorer_compare: skipped (need rows for both baseline.json and scorer4.1.json).")
        return
    hue_order = [x for x in ("baseline.json", "scorer4.1.json") if x in set(sub["results_json"].unique())]
    label_by_json: dict[str, str] = {}
    for j in hue_order:
        if "score_model_type" in sub.columns:
            sm = sub.loc[sub["results_json"] == j, "score_model_type"].dropna().astype(str).unique()
            tag = f" ({sm[0]})" if len(sm) == 1 else ""
        else:
            tag = ""
        label_by_json[j] = j.replace(".json", "") + tag
    sub["results_label"] = sub["results_json"].map(label_by_json)
    hue_labels = [label_by_json[j] for j in hue_order]
    order_m = method_order_present(sub["method"])

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ycol, ylabel = "mean_score", "Mean coherency score"
    d = sub[["method", "results_json", "results_label", ycol]].dropna()
    if d.empty:
        return
    pal: dict[str, str] = {}
    for j in hue_order:
        h = label_by_json[j]
        pal[h] = (
            METHOD_PALETTE.get("io", C_FOREST_DEEP)
            if "baseline" in j
            else METHOD_PALETTE.get("cot", C_GREEN_MD)
        )
    sns.barplot(
        data=d,
        x="method",
        y=ycol,
        hue="results_label",
        order=order_m,
        hue_order=hue_labels,
        palette=pal,
        estimator="mean",
        errorbar="ci",
        ax=ax,
    )
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Method")
    ax.grid(axis="y", alpha=0.2)
    ax.tick_params(axis="x", labelsize=8, rotation=12)
    ax.legend(title="Scorer", fontsize=8, title_fontsize=8, loc="upper left")

    bal_note = (
        "balanced n per method×JSON"
        if balanced_method_json
        else "raw row counts (no method×JSON balance)"
    )
    fig.suptitle(
        f"Scorer comparison: baseline.json vs scorer4.1.json  (paper-core methods; {bal_note})",
        fontsize=10,
        y=1.02,
    )
    fig.tight_layout()
    _save(fig, grouped_path(out_dir, "modifications/scoring_model", "baseline_vs_scorer4_1_by_method.png"))


def plot_plan_methods_three_way(df: pd.DataFrame, out_dir: Path) -> None:
    """tot vs tot_astar vs hybrid_tot (plan_methods.json metrics rows only)."""
    if "method" not in df.columns:
        return
    metrics = {
        "mean_score": "Mean Score",
        "score_std": "Score Std Dev",
        "vote_entropy_plan": "H_plan",
        "vote_entropy_passage": "H_pass",
        "total_cost_usd": "Cost (USD)",
    }
    subset = df[df["method"].isin(PLAN_METHODS_COMPARE_ORDER)].copy()
    if subset.empty:
        return
    present = {}
    for k, v in metrics.items():
        if k not in subset.columns:
            continue
        d = subset[["method", k]].dropna()
        if d["method"].nunique() >= 2:
            present[k] = v
    if not present:
        return

    panels: list[tuple[str, str, pd.DataFrame, pd.DataFrame]] = []
    for col, label in present.items():
        cols_m = ["method", col] + (["task_id"] if "task_id" in subset.columns else [])
        data = subset[[c for c in cols_m if c in subset.columns]].dropna()
        order_m = [m for m in PLAN_METHODS_COMPARE_ORDER if m in data["method"].unique()]
        summary = (
            data.groupby("method")[col]
            .agg(["mean", "std", "count"])
            .reindex(order_m)
            .dropna(subset=["count"])
        )
        if len(summary) >= 2:
            panels.append((col, label, summary, data))

    if not panels:
        return

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 6.2))
    if n == 1:
        axes = [axes]

    for ax, (col, label, summary, raw_m) in zip(axes, panels):
        summary = summary.copy()
        summary["ci95"] = summary["std"] / np.sqrt(summary["count"]) * 1.96
        colors = _color_list(list(summary.index))

        bars = ax.bar(
            summary.index,
            summary["mean"],
            yerr=summary["ci95"],
            color=colors,
            alpha=0.88,
            edgecolor=C_FOREST,
            error_kw=dict(ecolor=C_TEXT_MUTED, capsize=5, lw=1.2),
            width=0.55,
        )
        for bar, (idx, row) in zip(bars, summary.iterrows()):
            if idx == "hybrid_tot":
                bar.set_linewidth(2.4)
                bar.set_edgecolor(C_FOREST_DEEP)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + summary.loc[idx, "ci95"] * 0.1,
                f"{row['mean']:.3f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color=C_TEXT_MUTED,
            )
        methods_list = [m for m in PLAN_METHODS_COMPARE_ORDER if m in summary.index]
        if len(methods_list) >= 2 and col in raw_m.columns:
            p_an = _one_way_anova_p(raw_m, "method", col, levels=methods_list)
            st_an = _stars(p_an)
            pair_parts = _pairwise_welch_bonferroni_parts(raw_m, "method", col, methods_list)
            title_lines = [
                label,
                f"ANOVA p={_format_p_short(p_an)}{(' ' + st_an) if st_an else ''}",
            ]
            if pair_parts:
                title_lines.append("Pairwise Welch (Bonferroni p_adj):")
                title_lines.extend(pair_parts)
            else:
                title_lines.append("Pairwise: (need n≥2 per method)")
            ax.set_title("\n".join(title_lines), fontsize=6.8, pad=8)
        else:
            ax.set_title(label, fontsize=9)
        ax.tick_params(axis="x", labelsize=8, rotation=12)
        ax.grid(axis="y", alpha=0.2)

    fig.suptitle(
        "plan_methods.json — tot vs tot_astar vs hybrid_tot  (mean ± 95% CI; p-values in each panel title)",
        fontsize=10,
        y=1.02,
    )
    fig.tight_layout()
    _save(fig, grouped_path(out_dir, "independent_exploration/plan_methods", "three_method_comparison.png"))


def plot_score_delta_correlations(df: pd.DataFrame, out_dir: Path) -> None:
    """score_delta (astar − tot) vs per-task ToT variables — merge on full run keys (not task_id alone)."""
    if "method" not in df.columns:
        return
    key_cols = [
        c
        for c in [
            "task_id",
            "out_file",
            "score_prompt_variant",
            "k",
            "n_votes",
            "score_model_type",
        ]
        if c in df.columns
    ]
    if "task_id" not in key_cols:
        return

    extra = [
        c
        for c in (
            "mean_score",
            "vote_entropy_plan",
            "vote_entropy_passage",
            "score_std",
            "vote_parse_failures_plan",
        )
        if c in df.columns
    ]
    cols = list(dict.fromkeys(key_cols + extra))
    tot = df[df["method"] == "tot"][cols].drop_duplicates(subset=key_cols)
    astar = df[df["method"] == "tot_astar"][cols].drop_duplicates(subset=key_cols)
    merged = tot.merge(astar, on=key_cols, suffixes=("_tot", "_tot_astar"), how="inner")
    if merged.empty or len(merged) < 4:
        return
    merged["score_delta"] = merged["mean_score_tot_astar"] - merged["mean_score_tot"]
    merged = augment_joined_with_metric_deltas(merged)

    for col, label in CORRELATION_DELTA_LABELS.items():
        if col not in merged.columns:
            continue
        data = merged[["score_delta", col]].dropna()
        if len(data) < 4:
            continue
        if data[col].nunique() <= 1 or data["score_delta"].nunique() <= 1:
            continue
        r_p, p_p = pearsonr(data[col], data["score_delta"])
        r_s, p_s = spearmanr(data[col], data["score_delta"])
        n = len(data)

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.regplot(
            data=data,
            x=col,
            y="score_delta",
            scatter_kws={"alpha": 0.65, "s": 55, "color": C_GREEN_DARK, "edgecolor": C_FOREST, "linewidths": 0.4},
            line_kws={"color": C_ACCENT_LINE, "lw": 1.5},
            ax=ax,
        )
        ax.axhline(0, color=C_TEXT_MUTED, linestyle="--", lw=1)
        ax.axvline(0, color=C_TEXT_MUTED, linestyle="--", lw=1, alpha=0.7)
        _stat_box(ax, r_p, p_p, r_s, p_s, n)
        spp, sps = _stars(p_p), _stars(p_s)
        ax.set_title(
            f"score Δ vs {label}\n"
            f"Pearson p={_format_p_short(p_p)}{(' ' + spp) if spp else ''}  "
            f"Spearman p={_format_p_short(p_s)}{(' ' + sps) if sps else ''}",
            fontsize=8,
        )
        ax.set_xlabel(label)
        ax.set_ylabel("score_delta (mean score, A* − ToT)")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fname = f"score_delta_vs_{_safe_corr_filename(col)}.png"
        _save(fig, grouped_path(out_dir, "independent_exploration/astar_analysis", fname))


def plot_vote_entropy_vs_score(df: pd.DataFrame, out_dir: Path) -> None:
    specs = [
        ("vote_entropy_plan", "Plan", C_FOREST_DEEP, "-"),
        ("vote_entropy_passage", "Passage", "#C44536", "--"),
    ]
    present = [s for s in specs if s[0] in df.columns]
    if not present:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    p_lines = ["Correlation p-values (line-only fit):"]
    plotted = 0
    for entropy_col, step, color, ls in present:
        subset = df[[entropy_col, "mean_score"]].dropna()
        if len(subset) < 4:
            continue
        if subset[entropy_col].nunique() <= 1 or subset["mean_score"].nunique() <= 1:
            continue
        r_p, p_p = pearsonr(subset[entropy_col], subset["mean_score"])
        r_s, p_s = spearmanr(subset[entropy_col], subset["mean_score"])
        m_c, b_c = np.polyfit(subset[entropy_col], subset["mean_score"], 1)
        x_r = np.linspace(subset[entropy_col].min(), subset[entropy_col].max(), 200)
        ax.plot(
            x_r,
            m_c * x_r + b_c,
            color=color,
            lw=2.2,
            linestyle=ls,
            label=f"{step} line fit",
        )
        spp, sps = _stars(p_p), _stars(p_s)
        p_lines.append(
            f"  {step}: Pearson p={_format_p_short(p_p)}{(' ' + spp) if spp else ''}, "
            f"Spearman p={_format_p_short(p_s)}{(' ' + sps) if sps else ''} (n={len(subset)})"
        )
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return

    _omnibus_p_box(ax, p_lines, fontsize=7.2)
    ax.set_title("Vote Entropy (Plan + Passage) vs Mean Score")
    ax.set_xlabel("Vote entropy")
    ax.set_ylabel("Mean coherency score")
    ax.legend(title="Entropy source", loc="upper left", fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    _save(fig, grouped_path(out_dir, "modifications/correlations", "vote_entropy_plan_passage_vs_mean_score.png"))


def plot_score_std_vs_mean(df: pd.DataFrame, out_dir: Path) -> None:
    if "score_std" not in df.columns:
        return
    subset = df[["method", "score_std", "mean_score"]].dropna()
    if len(subset) < 4:
        return
    r_p, p_p = pearsonr(subset["score_std"], subset["mean_score"])
    r_s, p_s = spearmanr(subset["score_std"], subset["mean_score"])

    fig, ax = plt.subplots(figsize=(9, 5))
    order = method_order_present(subset["method"])
    palette = {m: METHOD_PALETTE.get(m, C_FALLBACK_SERIES) for m in order}
    for m in order:
        subm = subset[subset["method"] == m]
        if subm.empty:
            continue
        ref = m in REFINE_METHODS
        ax.scatter(
            subm["score_std"],
            subm["mean_score"],
            c=palette[m],
            label=m,
            alpha=0.72 if ref else 0.62,
            s=56 if ref else 42,
            marker="D" if ref else "o",
            edgecolors=C_FOREST,
            linewidths=0.45,
        )
    m_c, b_c = np.polyfit(subset["score_std"], subset["mean_score"], 1)
    x_r = np.linspace(subset["score_std"].min(), subset["score_std"].max(), 200)
    ax.plot(x_r, m_c * x_r + b_c, color=C_ACCENT_LINE, lw=1.5)
    _stat_box(ax, r_p, p_p, r_s, p_s, len(subset))
    ax.set_title("Score Std Dev vs Mean Score")
    ax.set_xlabel("Score std dev (across 5 judge calls)")
    ax.set_ylabel("Mean coherency score")
    ax.legend(title="method", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    _save(fig, grouped_path(out_dir, "modifications/correlations", "score_std_vs_mean_score.png"))

    def _plot_score_std_vs_mean_colored_by(
        src: pd.DataFrame,
        *,
        hue_col: str,
        filename: str,
        title_suffix: str,
        ordered_levels: list[str] | None = None,
    ) -> None:
        if hue_col not in src.columns:
            return
        d = src[["score_std", "mean_score", hue_col]].dropna()
        if len(d) < 4:
            return

        # Treat hue levels as categories for clearer legends (k / n_votes are discrete in this project).
        d = d.copy()
        d[hue_col] = d[hue_col].astype(str)
        if ordered_levels is not None:
            levels = [x for x in ordered_levels if x in set(d[hue_col].unique())]
        else:
            vals = sorted(d[hue_col].unique(), key=lambda x: (len(x), x))
            levels = vals
        if len(levels) < 2:
            return

        fig, ax = plt.subplots(figsize=(9, 5))
        if hue_col == "score_prompt_variant":
            prompt_pal = {"paper": C_FOREST_DEEP, "definition": C_GREEN_MD, "criteria": "#C44536"}
            pal = {k: prompt_pal.get(k, C_FALLBACK_SERIES) for k in levels}
        else:
            pal_list = sns.color_palette("viridis", n_colors=len(levels))
            pal = {lv: c for lv, c in zip(levels, pal_list)}

        sns.scatterplot(
            data=d,
            x="score_std",
            y="mean_score",
            hue=hue_col,
            hue_order=levels,
            palette=pal,
            s=54,
            alpha=0.75,
            edgecolor=C_FOREST,
            linewidth=0.45,
            ax=ax,
        )

        m_c, b_c = np.polyfit(d["score_std"], d["mean_score"], 1)
        x_r = np.linspace(d["score_std"].min(), d["score_std"].max(), 200)
        ax.plot(x_r, m_c * x_r + b_c, color=C_ACCENT_LINE, lw=1.5)

        rp, pp = pearsonr(d["score_std"], d["mean_score"])
        rs, ps = spearmanr(d["score_std"], d["mean_score"])
        _stat_box(ax, rp, pp, rs, ps, len(d))
        ax.set_title(f"Score Std Dev vs Mean Score\n(color = {title_suffix})")
        ax.set_xlabel("Score std dev (across 5 judge calls)")
        ax.set_ylabel("Mean coherency score")
        ax.legend(title=title_suffix, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        _save(fig, grouped_path(out_dir, "modifications/correlations", filename))

    overlay_cols = ["score_std", "mean_score"]
    for opt in ("score_prompt_variant", "k", "n_votes"):
        if opt in df.columns:
            overlay_cols.append(opt)
    subset_overlay = df[overlay_cols].dropna(subset=["score_std", "mean_score"])
    _plot_score_std_vs_mean_colored_by(
        subset_overlay,
        hue_col="score_prompt_variant",
        filename="score_std_vs_mean_score_by_prompt.png",
        title_suffix="scoring prompt",
        ordered_levels=PROMPT_ORDER,
    )
    _plot_score_std_vs_mean_colored_by(
        subset_overlay,
        hue_col="k",
        filename="score_std_vs_mean_score_by_k.png",
        title_suffix="k",
    )
    _plot_score_std_vs_mean_colored_by(
        subset_overlay,
        hue_col="n_votes",
        filename="score_std_vs_mean_score_by_n_votes.png",
        title_suffix="n_votes",
    )


def plot_compact_controlled_regression(df: pd.DataFrame, out_dir: Path) -> None:
    """Compact controlled-regression artifacts for key predictors vs mean_score."""
    required = {"mean_score", "method", "k", "n_votes", "score_prompt_variant"}
    if not required.issubset(df.columns):
        return

    data = df.copy()
    if "score_model_type" in data.columns:
        data["scorer_model"] = data["score_model_type"].fillna("unknown").astype(str)
    else:
        data["scorer_model"] = "unknown"

    predictors = [p for p in ("score_std", "vote_entropy_plan", "vote_entropy_passage") if p in data.columns]
    if not predictors:
        return

    rows: list[dict[str, float | str | int]] = []
    panel_data: list[tuple[str, pd.DataFrame, float, float, float, float]] = []
    controls = "C(score_prompt_variant) + C(method) + C(k) + C(n_votes) + C(scorer_model)"

    for pred in predictors:
        d = data[["mean_score", pred, "score_prompt_variant", "method", "k", "n_votes", "scorer_model"]].dropna()
        if len(d) < 12:
            continue
        model = smf.ols(f"mean_score ~ {pred} + {controls}", data=d).fit(cov_type="HC3")
        if pred not in model.params.index:
            continue
        beta = float(model.params[pred])
        se = float(model.bse[pred])
        pval = float(model.pvalues[pred])
        ci = model.conf_int().loc[pred].to_numpy(dtype=float)
        rows.append(
            {
                "predictor": pred,
                "coef": beta,
                "se_hc3": se,
                "p_value": pval,
                "ci95_low": float(ci[0]),
                "ci95_high": float(ci[1]),
                "n": int(model.nobs),
                "r_squared": float(model.rsquared),
            }
        )

        # Partial residual panel: residualize y and x on controls only.
        y_ctrl = smf.ols(f"mean_score ~ {controls}", data=d).fit()
        x_ctrl = smf.ols(f"{pred} ~ {controls}", data=d).fit()
        res = pd.DataFrame(
            {
                "x_resid": x_ctrl.resid,
                "y_resid": y_ctrl.resid,
            }
        )
        panel_data.append((pred, res, beta, pval, float(ci[0]), float(ci[1])))

    if not rows:
        return

    summary = pd.DataFrame(rows).sort_values("predictor")
    summary.to_csv(grouped_path(out_dir, "modifications/regression", "controlled_regression_summary.csv"), index=False)
    summary.to_json(
        grouped_path(out_dir, "modifications/regression", "controlled_regression_summary.json"),
        orient="records",
        indent=2,
    )

    # One compact coefficient plot ("one number" view per predictor).
    fig, ax = plt.subplots(figsize=(8.2, 3.9))
    y = np.arange(len(summary))
    ax.errorbar(
        summary["coef"].to_numpy(dtype=float),
        y,
        xerr=np.vstack(
            [
                (summary["coef"] - summary["ci95_low"]).to_numpy(dtype=float),
                (summary["ci95_high"] - summary["coef"]).to_numpy(dtype=float),
            ]
        ),
        fmt="o",
        color=C_FOREST_DEEP,
        ecolor=C_GREEN_MD,
        elinewidth=2.0,
        capsize=4,
    )
    ax.axvline(0.0, color=C_TEXT_MUTED, linestyle="--", linewidth=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels(summary["predictor"].tolist())
    ax.set_xlabel("Controlled coefficient on predictor (HC3 95% CI)")
    ax.set_title("Controlled regression: mean_score ~ predictor + controls")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    _save(fig, grouped_path(out_dir, "modifications/regression", "controlled_regression_coefficients.png"))

    # One compact multi-panel residual visualization.
    if panel_data:
        n_panels = len(panel_data)
        fig, axes = plt.subplots(1, n_panels, figsize=(5.0 * n_panels, 4.0), squeeze=False)
        for ax, (pred, res, beta, pval, ci_lo, ci_hi) in zip(axes[0], panel_data):
            sns.regplot(
                data=res,
                x="x_resid",
                y="y_resid",
                scatter_kws={
                    "alpha": 0.62,
                    "s": 28,
                    "color": C_GREEN_DARK,
                    "edgecolor": C_FOREST,
                    "linewidths": 0.25,
                },
                line_kws={"color": C_ACCENT_LINE, "lw": 1.8},
                ax=ax,
            )
            ax.axhline(0, color=C_TEXT_MUTED, ls="--", lw=0.8, alpha=0.8)
            ax.axvline(0, color=C_TEXT_MUTED, ls="--", lw=0.8, alpha=0.8)
            ax.set_title(pred)
            ax.set_xlabel(f"{pred} residual")
            ax.set_ylabel("mean_score residual")
            ax.text(
                0.02,
                0.98,
                f"b={beta:.3f}\np={_format_p_short(pval)}\nCI[{ci_lo:.3f}, {ci_hi:.3f}]",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=C_CREAM, edgecolor=C_GRID),
            )
            ax.grid(alpha=0.2)
        fig.suptitle("Partial residual view (controls removed)", y=1.03, fontsize=10)
        fig.tight_layout()
        _save(fig, grouped_path(out_dir, "modifications/regression", "controlled_regression_partial_residuals.png"))


def write_interactive_regression_explorer(df: pd.DataFrame, out_dir: Path) -> None:
    """Write a local interactive HTML to toggle controls and inspect coefficients."""
    required = {"mean_score", "method", "k", "n_votes", "score_prompt_variant"}
    if not required.issubset(df.columns):
        return

    data = df.copy()
    if "score_model_type" in data.columns:
        data["scorer_model"] = data["score_model_type"].fillna("unknown").astype(str)
    else:
        data["scorer_model"] = "unknown"

    predictor_list = [p for p in ("score_std", "vote_entropy_plan", "vote_entropy_passage") if p in data.columns]
    if not predictor_list:
        return

    control_map = {
        "prompt": "C(score_prompt_variant)",
        "method": "C(method)",
        "k": "C(k)",
        "n_votes": "C(n_votes)",
        "scorer_model": "C(scorer_model)",
    }
    control_ids = list(control_map.keys())

    rows: list[dict[str, object]] = []
    for pred in predictor_list:
        needed = ["mean_score", pred, "score_prompt_variant", "method", "k", "n_votes", "scorer_model"]
        d = data[needed].dropna()
        if len(d) < 12:
            continue
        for mask in range(1 << len(control_ids)):
            included = [cid for i, cid in enumerate(control_ids) if (mask & (1 << i))]
            terms = [pred] + [control_map[cid] for cid in included]
            formula = "mean_score ~ " + " + ".join(terms)
            try:
                m = smf.ols(formula, data=d).fit(cov_type="HC3")
            except Exception:
                continue
            if pred not in m.params.index:
                continue
            ci = m.conf_int().loc[pred].to_numpy(dtype=float)
            rows.append(
                {
                    "predictor": pred,
                    "controls": included,
                    "controls_key": "|".join(included) if included else "(none)",
                    "formula": formula,
                    "coef": float(m.params[pred]),
                    "se_hc3": float(m.bse[pred]),
                    "p_value": float(m.pvalues[pred]),
                    "ci95_low": float(ci[0]),
                    "ci95_high": float(ci[1]),
                    "n": int(m.nobs),
                    "r_squared": float(m.rsquared),
                }
            )

    if not rows:
        return
    out_json = grouped_path(out_dir, "modifications/regression", "interactive_regression_models.json")
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    payload = json.dumps(rows)
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Controlled Regression Explorer</title>
  <style>
    body {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; margin: 20px; color: #1B3328; }}
    .row {{ display: flex; gap: 18px; flex-wrap: wrap; align-items: center; margin-bottom: 12px; }}
    .controls {{ display: flex; gap: 12px; flex-wrap: wrap; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; max-width: 1120px; }}
    .card {{ border: 1px solid #D9DED5; border-radius: 8px; padding: 12px; background: #FBFCF4; }}
    .coef {{ font-size: 1.25rem; font-weight: 700; margin-bottom: 6px; }}
    .muted {{ color: #445E4F; font-size: 0.9rem; line-height: 1.35; }}
    .bar-wrap {{ margin-top: 10px; height: 18px; background: #E8EDE6; border-radius: 9px; position: relative; }}
    .zero {{ position: absolute; left: 50%; top: 0; bottom: 0; width: 1px; background: #374836; opacity: 0.6; }}
    .ci {{ position: absolute; top: 4px; height: 10px; background: #8DAA9D; border-radius: 6px; }}
    .pt {{ position: absolute; top: 2px; width: 14px; height: 14px; border-radius: 7px; background: #1B3328; transform: translateX(-7px); }}
    code {{ background: #F3F6EE; padding: 2px 5px; border-radius: 4px; }}
    svg {{ width: 100%; height: 330px; background: #FFFFFF; border: 1px solid #D9DED5; border-radius: 8px; }}
    .badge {{ display: inline-block; font-size: 0.82rem; padding: 3px 6px; border-radius: 999px; background: #EAF1EA; margin-right: 6px; }}
  </style>
</head>
<body>
  <h2>Controlled Regression Explorer</h2>
  <div class="muted">Model family: <code>mean_score ~ predictor + controls</code>. Checkboxes include/exclude controls in the model. Coefficient shown is the adjusted effect for the selected predictor.</div>
  <div class="row">
    <label>Predictor:
      <select id="predictor"></select>
    </label>
  </div>
  <div class="row">
    <span>Include controls:</span>
    <div class="controls">
      <label><input type="checkbox" id="c_prompt" checked /> prompt</label>
      <label><input type="checkbox" id="c_method" checked /> method</label>
      <label><input type="checkbox" id="c_k" checked /> k</label>
      <label><input type="checkbox" id="c_n_votes" checked /> n_votes</label>
      <label><input type="checkbox" id="c_scorer_model" checked /> scorer_model</label>
    </div>
  </div>

  <div class="grid">
    <div class="card">
      <div class="coef" id="coefText">—</div>
      <div class="muted" id="metaText"></div>
      <div class="muted" id="formulaText"></div>
      <div id="interpText" class="muted" style="margin-top:8px;"></div>
      <div class="bar-wrap">
        <div class="zero"></div>
        <div class="ci" id="ci"></div>
        <div class="pt" id="pt"></div>
      </div>
    </div>
    <div class="card">
      <div class="muted"><span class="badge">How to read</span> If CI crosses 0, effect may be uncertain. Checking a control means comparing rows while accounting for that variable's level shifts.</div>
      <div class="muted" style="margin-top:8px;"><span class="badge">Tip</span> Start with all controls checked, then uncheck one at a time to see sensitivity.</div>
      <div class="muted" id="controlsBadge" style="margin-top:8px;"></div>
    </div>
  </div>

  <h3 style="margin-top:16px;">Selected Predictor Across All Control Combinations</h3>
  <div class="muted" style="margin-bottom:8px;">Each point is one model variant. Horizontal line is its 95% CI; dashed vertical line is 0 effect.</div>
  <svg id="forest"></svg>

  <script>
    const rows = {payload};
    const predictorSel = document.getElementById('predictor');
    const ids = ['prompt','method','k','n_votes','scorer_model'];
    const cb = {{
      prompt: document.getElementById('c_prompt'),
      method: document.getElementById('c_method'),
      k: document.getElementById('c_k'),
      n_votes: document.getElementById('c_n_votes'),
      scorer_model: document.getElementById('c_scorer_model'),
    }};
    const coefText = document.getElementById('coefText');
    const metaText = document.getElementById('metaText');
    const formulaText = document.getElementById('formulaText');
    const interpText = document.getElementById('interpText');
    const controlsBadge = document.getElementById('controlsBadge');
    const ciEl = document.getElementById('ci');
    const ptEl = document.getElementById('pt');
    const forest = document.getElementById('forest');

    const predictors = [...new Set(rows.map(r => r.predictor))];
    predictors.forEach(p => {{
      const o = document.createElement('option');
      o.value = p; o.textContent = p;
      predictorSel.appendChild(o);
    }});

    function keyFromChecks() {{
      const included = ids.filter(id => cb[id].checked);
      return included.length ? included.join('|') : '(none)';
    }}

    function significanceLabel(p) {{
      if (p < 0.001) return 'very strong evidence (p<0.001)';
      if (p < 0.01) return 'strong evidence (p<0.01)';
      if (p < 0.05) return 'moderate evidence (p<0.05)';
      return 'weak/inconclusive evidence (p>=0.05)';
    }}

    function renderForest(pred, selectedKey) {{
      while (forest.firstChild) forest.removeChild(forest.firstChild);
      const arr = rows.filter(r => r.predictor === pred).sort((a,b) => a.coef - b.coef);
      if (!arr.length) return;
      const w = forest.clientWidth || 900;
      const h = 330;
      forest.setAttribute('viewBox', '0 0 ' + w + ' ' + h);
      const padL = 70, padR = 16, padT = 14, padB = 26;
      const minX = Math.min(...arr.map(r => r.ci95_low), 0);
      const maxX = Math.max(...arr.map(r => r.ci95_high), 0);
      const span = Math.max(maxX - minX, 1e-6);
      const x = v => padL + ((v - minX) / span) * (w - padL - padR);
      const y = i => padT + (i + 0.5) * ((h - padT - padB) / arr.length);

      const axis = document.createElementNS('http://www.w3.org/2000/svg','line');
      axis.setAttribute('x1', x(0)); axis.setAttribute('x2', x(0));
      axis.setAttribute('y1', padT); axis.setAttribute('y2', h - padB);
      axis.setAttribute('stroke', '#445E4F'); axis.setAttribute('stroke-dasharray', '4,3'); forest.appendChild(axis);

      arr.forEach((r, i) => {{
        const line = document.createElementNS('http://www.w3.org/2000/svg','line');
        line.setAttribute('x1', x(r.ci95_low)); line.setAttribute('x2', x(r.ci95_high));
        line.setAttribute('y1', y(i)); line.setAttribute('y2', y(i));
        line.setAttribute('stroke', '#8DAA9D'); line.setAttribute('stroke-width', '2'); forest.appendChild(line);
        const dot = document.createElementNS('http://www.w3.org/2000/svg','circle');
        dot.setAttribute('cx', x(r.coef)); dot.setAttribute('cy', y(i)); dot.setAttribute('r', '4');
        dot.setAttribute('fill', r.controls_key === selectedKey ? '#1B3328' : '#5E8C76');
        forest.appendChild(dot);
      }});
    }}

    function render() {{
      const pred = predictorSel.value;
      const key = keyFromChecks();
      const hit = rows.find(r => r.predictor === pred && r.controls_key === key);
      controlsBadge.textContent = 'Selected controls: ' + key;
      renderForest(pred, key);
      if (!hit) {{
        coefText.textContent = 'No model for this combination';
        metaText.textContent = '';
        formulaText.textContent = '';
        interpText.textContent = '';
        ciEl.style.width = '0';
        ptEl.style.left = '50%';
        return;
      }}
      coefText.textContent = 'coef(' + pred + ') = ' + hit.coef.toFixed(4);
      metaText.textContent = '95% CI [' + hit.ci95_low.toFixed(4) + ', ' + hit.ci95_high.toFixed(4) + '], p=' + hit.p_value.toExponential(2) + ', n=' + hit.n + ', R²=' + hit.r_squared.toFixed(3);
      formulaText.textContent = hit.formula;
      const dir = hit.coef > 0 ? 'positive' : (hit.coef < 0 ? 'negative' : 'near-zero');
      interpText.textContent = 'Interpretation: adjusted ' + dir + ' association; ' + significanceLabel(hit.p_value) + '.';

      const span = Math.max(Math.abs(hit.ci95_low), Math.abs(hit.ci95_high), Math.abs(hit.coef), 0.01) * 1.2;
      const toPct = x => 50 + (x / span) * 50;
      const lo = Math.max(0, Math.min(100, toPct(hit.ci95_low)));
      const hi = Math.max(0, Math.min(100, toPct(hit.ci95_high)));
      const pt = Math.max(0, Math.min(100, toPct(hit.coef)));
      ciEl.style.left = Math.min(lo, hi) + '%';
      ciEl.style.width = Math.abs(hi - lo) + '%';
      ptEl.style.left = pt + '%';
    }}

    predictorSel.addEventListener('change', render);
    ids.forEach(id => cb[id].addEventListener('change', render));
    predictorSel.value = predictors[0] || '';
    render();
  </script>
</body>
</html>
"""
    out_html = grouped_path(out_dir, "modifications/regression", "interactive_regression_explorer.html")
    with out_html.open("w", encoding="utf-8") as f:
        f.write(html)


def plot_k_ablation(df: pd.DataFrame, out_dir: Path) -> None:
    if "k" not in df.columns:
        return

    subset = df[["k", "mean_score", "method"]].dropna()
    if not subset.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        order = method_order_present(subset["method"])
        palette = {m: METHOD_PALETTE.get(m, C_FALLBACK_SERIES) for m in order}
        sns.lineplot(
            data=subset,
            x="k",
            y="mean_score",
            hue="method",
            hue_order=order,
            style="method",
            style_order=order,
            palette=palette,
            markers=method_line_markers_for_order(order),
            dashes=False,
            linewidth=2.35,
            estimator="mean",
            errorbar="ci",
            ax=ax,
        )
        ax.set_title("K vs Mean Score  (by method, 95% CI)")
        ax.set_xlabel("K (candidates)")
        ax.set_ylabel("Mean coherency score")
        ax.grid(alpha=0.25)
        k_levels = sorted(subset["k"].dropna().unique())
        p_lines = ["ANOVA mean_score ~ K (within method):"]
        for m in order:
            pm = _one_way_anova_p(subset[subset["method"] == m], "k", "mean_score", levels=k_levels)
            st = _stars(pm)
            p_lines.append(f"  {m}: p={_format_p_short(pm)}{' ' + st if st else ''}")
        _omnibus_p_box(ax, p_lines)
        fig.tight_layout()
        _save(fig, grouped_path(out_dir, "modifications/ablations", "k_vs_mean_score.png"))

    entropy_cols = [c for c in ["vote_entropy_plan", "vote_entropy_passage"] if c in df.columns]
    if entropy_cols:
        melted = (
            df[["k"] + entropy_cols]
            .melt(id_vars=["k"], value_vars=entropy_cols, var_name="step", value_name="entropy")
            .dropna()
        )
        melted["step"] = melted["step"].map({"vote_entropy_plan": "plan", "vote_entropy_passage": "passage"})
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.lineplot(
            data=melted,
            x="k",
            y="entropy",
            hue="step",
            palette={"plan": C_FOREST, "passage": "#E8A0A0"},
            style="step",
            markers={"plan": "o", "passage": "s"},
            dashes=False,
            linewidth=2.35,
            estimator="mean",
            errorbar="ci",
            ax=ax,
        )
        ax.set_title("K vs Vote Entropy — Plan vs Passage Step  (95% CI)")
        ax.set_xlabel("K (candidates)")
        ax.set_ylabel("Vote entropy (Shannon)")
        ax.grid(alpha=0.25)
        k_levels_e = sorted(melted["k"].dropna().unique())
        p_lines = ["ANOVA entropy ~ K:"]
        for step in ["plan", "passage"]:
            sub_s = melted[melted["step"] == step]
            if not sub_s.empty:
                pe = _one_way_anova_p(sub_s, "k", "entropy", levels=k_levels_e)
                ste = _stars(pe)
                p_lines.append(f"  {step}: p={_format_p_short(pe)}{' ' + ste if ste else ''}")
        _omnibus_p_box(ax, p_lines)
        fig.tight_layout()
        _save(fig, grouped_path(out_dir, "modifications/ablations", "k_vs_vote_entropy.png"))

    subset_cost = df[["k", "mean_score", "total_cost_usd"]].dropna()
    if not subset_cost.empty:
        grouped = (
            subset_cost.groupby("k")
            .agg(
                mean_score=("mean_score", "mean"),
                avg_cost=("total_cost_usd", "mean"),
            )
            .reset_index()
            .sort_values("k")
        )
        fig, ax1 = plt.subplots(figsize=(9, 5))
        ax2 = ax1.twinx()
        ax1.plot(grouped["k"], grouped["mean_score"], marker="o", color=C_FOREST_DEEP, lw=2, label="Mean score")
        ax2.plot(grouped["k"], grouped["avg_cost"], marker="s", color=C_GREEN_DARK, lw=2, label="Avg cost (USD)")
        ax1.set_xlabel("K (candidates)")
        ax1.set_ylabel("Mean score", color=C_FOREST_DEEP)
        ax2.set_ylabel("Avg cost (USD)", color=C_GREEN_DARK)
        ax1.tick_params(axis="y", labelcolor=C_FOREST_DEEP)
        ax2.tick_params(axis="y", labelcolor=C_GREEN_DARK)
        lines = ax1.get_lines() + ax2.get_lines()
        ax1.legend(lines, [l.get_label() for l in lines], loc="upper left", fontsize=8)
        ax1.set_title("K vs Score / Cost Tradeoff")
        ax1.grid(alpha=0.2)
        p_lines = []
        if len(subset_cost) >= 3 and subset_cost["k"].nunique() > 1:
            r_s, p_s = pearsonr(subset_cost["k"], subset_cost["mean_score"])
            r_c, p_c = pearsonr(subset_cost["k"], subset_cost["total_cost_usd"])
            sts, stc = _stars(p_s), _stars(p_c)
            p_lines = [
                "Pearson r(K, ·):",
                f"  mean_score: r={r_s:.3f} p={_format_p_short(p_s)}{' ' + sts if sts else ''}",
                f"  total_cost: r={r_c:.3f} p={_format_p_short(p_c)}{' ' + stc if stc else ''}",
                f"  n={len(subset_cost)}",
            ]
        if p_lines:
            _omnibus_p_box(ax1, p_lines, y=0.98)
        fig.tight_layout()
        _save(fig, grouped_path(out_dir, "modifications/ablations", "k_vs_cost_tradeoff.png"))


def plot_n_votes_ablation(df: pd.DataFrame, out_dir: Path) -> None:
    if "n_votes" not in df.columns:
        return

    subset = df[["n_votes", "mean_score", "method"]].dropna()
    if not subset.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        order = method_order_present(subset["method"])
        palette = {m: METHOD_PALETTE.get(m, C_FALLBACK_SERIES) for m in order}
        sns.lineplot(
            data=subset,
            x="n_votes",
            y="mean_score",
            hue="method",
            hue_order=order,
            style="method",
            style_order=order,
            palette=palette,
            markers=method_line_markers_for_order(order),
            dashes=False,
            linewidth=2.35,
            estimator="mean",
            errorbar="ci",
            ax=ax,
        )
        ax.set_title("N_votes vs Mean Score  (95% CI)")
        ax.set_xlabel("N_votes")
        ax.set_ylabel("Mean coherency score")
        ax.grid(alpha=0.25)
        nv_levels = sorted(subset["n_votes"].dropna().unique())
        p_lines = ["ANOVA mean_score ~ n_votes (within method):"]
        for m in order:
            pm = _one_way_anova_p(subset[subset["method"] == m], "n_votes", "mean_score", levels=nv_levels)
            st = _stars(pm)
            p_lines.append(f"  {m}: p={_format_p_short(pm)}{' ' + st if st else ''}")
        _omnibus_p_box(ax, p_lines)
        fig.tight_layout()
        _save(fig, grouped_path(out_dir, "modifications/ablations", "n_votes_vs_mean_score.png"))

    if "vote_entropy" in df.columns:
        subset_e = df[["n_votes", "vote_entropy", "method"]].dropna()
        if not subset_e.empty:
            fig, ax = plt.subplots(figsize=(9, 5))
            order = method_order_present(subset_e["method"])
            palette = {m: METHOD_PALETTE.get(m, C_FALLBACK_SERIES) for m in order}
            sns.lineplot(
                data=subset_e,
                x="n_votes",
                y="vote_entropy",
                hue="method",
                hue_order=order,
                style="method",
                style_order=order,
                palette=palette,
                markers=method_line_markers_for_order(order),
                dashes=False,
                linewidth=2.35,
                estimator="mean",
                errorbar="ci",
                ax=ax,
            )
            ax.set_title("N_votes vs Vote Entropy  (95% CI)")
            ax.set_xlabel("N_votes")
            ax.set_ylabel("Vote entropy")
            ax.grid(alpha=0.25)
            nv_levels_e = sorted(subset_e["n_votes"].dropna().unique())
            p_lines = ["ANOVA vote_entropy ~ n_votes (within method):"]
            for m in order:
                pm = _one_way_anova_p(
                    subset_e[subset_e["method"] == m], "n_votes", "vote_entropy", levels=nv_levels_e
                )
                st = _stars(pm)
                p_lines.append(f"  {m}: p={_format_p_short(pm)}{' ' + st if st else ''}")
            _omnibus_p_box(ax, p_lines)
            fig.tight_layout()
            _save(fig, grouped_path(out_dir, "modifications/ablations", "n_votes_vs_vote_entropy.png"))

    # Overlay entropy and mean score on dual y-axes (ToT-only when available)
    # to directly inspect potential divergence as n_votes increases.
    if "vote_entropy" in df.columns:
        overlay_cols = ["n_votes", "mean_score", "vote_entropy"]
        if "method" in df.columns:
            overlay_cols.append("method")
        overlay = df[overlay_cols].dropna()
        if not overlay.empty:
            method_note = "all methods pooled"
            if "method" in overlay.columns and "tot" in set(overlay["method"].astype(str).unique()):
                overlay = overlay[overlay["method"] == "tot"].copy()
                method_note = "ToT only"

            grouped = (
                overlay.groupby("n_votes")
                .agg(
                    mean_score_mean=("mean_score", "mean"),
                    mean_score_std=("mean_score", "std"),
                    mean_score_n=("mean_score", "count"),
                    vote_entropy_mean=("vote_entropy", "mean"),
                    vote_entropy_std=("vote_entropy", "std"),
                    vote_entropy_n=("vote_entropy", "count"),
                )
                .reset_index()
                .sort_values("n_votes")
            )
            if not grouped.empty and grouped["n_votes"].nunique() >= 2:
                grouped["mean_score_ci95"] = (
                    grouped["mean_score_std"].fillna(0.0) / np.sqrt(grouped["mean_score_n"].clip(lower=1)) * 1.96
                )
                grouped["vote_entropy_ci95"] = (
                    grouped["vote_entropy_std"].fillna(0.0)
                    / np.sqrt(grouped["vote_entropy_n"].clip(lower=1))
                    * 1.96
                )

                fig, ax1 = plt.subplots(figsize=(9, 5))
                ax2 = ax1.twinx()
                color_score = METHOD_PALETTE.get("tot", C_GREEN_MD)
                color_entropy = C_TEXT_MUTED

                ax1.errorbar(
                    grouped["n_votes"],
                    grouped["mean_score_mean"],
                    yerr=grouped["mean_score_ci95"],
                    color=color_score,
                    marker="o",
                    linewidth=2.4,
                    capsize=4,
                    label="Mean score (95% CI)",
                )
                ax2.errorbar(
                    grouped["n_votes"],
                    grouped["vote_entropy_mean"],
                    yerr=grouped["vote_entropy_ci95"],
                    color=color_entropy,
                    marker="D",
                    linewidth=2.2,
                    linestyle="--",
                    capsize=4,
                    label="Vote entropy (95% CI)",
                )

                ax1.set_title(
                    f"N_votes: vote entropy vs mean score overlay  ({method_note})"
                )
                ax1.set_xlabel("N_votes")
                ax1.set_ylabel("Mean coherency score", color=color_score)
                ax2.set_ylabel("Vote entropy", color=color_entropy)
                ax1.tick_params(axis="y", colors=color_score)
                ax2.tick_params(axis="y", colors=color_entropy)
                ax1.grid(alpha=0.2)
                ax1.axvline(5, color="#8A8A8A", linestyle=":", linewidth=1.8, alpha=0.9)

                h1, l1 = ax1.get_legend_handles_labels()
                h2, l2 = ax2.get_legend_handles_labels()
                ax1.legend(h1 + h2, l1 + l2, loc="best", fontsize=8)

                fig.tight_layout()
                _save(fig, grouped_path(out_dir, "modifications/ablations", "n_votes_entropy_vs_mean_score_overlay.png"))


def plot_score_prompt_comparison(df: pd.DataFrame, out_dir: Path) -> None:
    if "score_prompt_variant" not in df.columns:
        return

    subset = df[["score_prompt_variant", "mean_score", "method"]].dropna()
    if not subset.empty:
        fig, ax = plt.subplots(figsize=(11, 5))
        order_x = [p for p in PROMPT_ORDER if p in subset["score_prompt_variant"].unique()]
        order_h = method_order_present(subset["method"])
        palette = {m: METHOD_PALETTE.get(m, C_FALLBACK_SERIES) for m in order_h}
        sns.barplot(
            data=subset,
            x="score_prompt_variant",
            y="mean_score",
            hue="method",
            order=order_x,
            hue_order=order_h,
            palette=palette,
            estimator="mean",
            errorbar="ci",
            ax=ax,
        )
        ax.set_title("Scoring Prompt vs Mean Score  (by method, 95% CI)")
        ax.set_xlabel("Scoring prompt variant")
        ax.set_ylabel("Mean score")
        ax.legend(title="method", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        ax.grid(axis="y", alpha=0.2)
        p_lines = [
            "Pairwise prompts (within method): Welch t, p_adj = min(1, m·p_raw), m = #pairs.",
        ]
        for m in order_h:
            sm = subset[subset["method"] == m]
            pair_s = _pairwise_welch_bonferroni_summary(
                sm, "score_prompt_variant", "mean_score", order_x
            )
            p_lines.append(f"  {m}:")
            p_lines.append(f"      {pair_s}")
        _omnibus_p_box(ax, p_lines, fontsize=6.2)
        fig.tight_layout()
        _save(fig, grouped_path(out_dir, "modifications/score_prompt", "prompt_vs_mean_score.png"))

        subset_tot = subset[subset["method"] == "tot"].copy()
        if not subset_tot.empty:
            fig, ax = plt.subplots(figsize=(7.5, 4.6))
            order_x_tot = [p for p in PROMPT_ORDER if p in subset_tot["score_prompt_variant"].unique()]
            sns.barplot(
                data=subset_tot,
                x="score_prompt_variant",
                y="mean_score",
                order=order_x_tot,
                color=METHOD_PALETTE.get("tot", C_FALLBACK_SERIES),
                estimator="mean",
                errorbar="ci",
                ax=ax,
            )
            ax.set_title("Scoring Prompt vs Mean Score  (ToT only, 95% CI)")
            ax.set_xlabel("Scoring prompt variant")
            ax.set_ylabel("Mean score")
            ax.grid(axis="y", alpha=0.2)
            pair_s = _pairwise_welch_bonferroni_summary(
                subset_tot, "score_prompt_variant", "mean_score", order_x_tot
            )
            _omnibus_p_box(
                ax,
                [
                    "Pairwise prompts within ToT: Welch t, p_adj = min(1, m·p_raw), m = #pairs.",
                    f"  {pair_s}",
                ],
                fontsize=6.2,
            )
            fig.tight_layout()
            _save(fig, grouped_path(out_dir, "modifications/score_prompt", "prompt_vs_mean_score_tot_only.png"))

    if "score_std" in df.columns:
        cols_std = ["score_prompt_variant", "score_std"] + (["method"] if "method" in df.columns else [])
        subset_std = df[cols_std].dropna()
        if not subset_std.empty:
            order_x = [p for p in PROMPT_ORDER if p in subset_std["score_prompt_variant"].unique()]
            fig, ax = plt.subplots(figsize=(8, 4))
            _green_seq = [C_FOREST_DEEP, C_FOREST, C_GREEN_DARK, C_GREEN_MD, C_GREEN_MUTED]
            pal = sns.color_palette(_green_seq, n_colors=len(order_x))
            sns.barplot(
                data=subset_std,
                x="score_prompt_variant",
                y="score_std",
                hue="score_prompt_variant",
                order=order_x,
                hue_order=order_x,
                estimator="mean",
                errorbar="ci",
                palette=pal,
                ax=ax,
                legend=False,
            )
            ax.set_title("Scoring Prompt vs Score Std Dev\n(lower = more consistent scorer)")
            ax.set_xlabel("Scoring prompt variant")
            ax.set_ylabel("Score std dev (5 judge calls)")
            ax.grid(axis="y", alpha=0.2)
            for patch in ax.patches:
                h = patch.get_height()
                if not np.isnan(h) and h > 0:
                    ax.text(
                        patch.get_x() + patch.get_width() / 2,
                        h + 0.002,
                        f"{h:.4f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        color=C_TEXT_MUTED,
                    )
            pair_std = _pairwise_welch_bonferroni_summary(
                subset_std, "score_prompt_variant", "score_std", order_x
            )
            p_omni = _one_way_anova_p(subset_std, "score_prompt_variant", "score_std", levels=order_x)
            sto = _stars(p_omni)
            _omnibus_p_box(
                ax,
                [
                    "Pairwise prompts: Welch t, Bonferroni p_adj (same as mean-score note).",
                    f"  {pair_std}",
                    f"ANOVA omnibus: p={_format_p_short(p_omni)}{' ' + sto if sto else ''}",
                ],
                fontsize=6.5,
            )
            fig.tight_layout()
            _save(fig, grouped_path(out_dir, "modifications/score_prompt", "prompt_vs_score_std.png"))

            if "method" in subset_std.columns:
                subset_std_tot = subset_std[subset_std["method"] == "tot"].copy()
                if not subset_std_tot.empty:
                    order_x_tot = [p for p in PROMPT_ORDER if p in subset_std_tot["score_prompt_variant"].unique()]
                    fig, ax = plt.subplots(figsize=(7.5, 4.6))
                    sns.barplot(
                        data=subset_std_tot,
                        x="score_prompt_variant",
                        y="score_std",
                        order=order_x_tot,
                        color=METHOD_PALETTE.get("tot", C_FALLBACK_SERIES),
                        estimator="mean",
                        errorbar="ci",
                        ax=ax,
                    )
                    ax.set_title("Scoring Prompt vs Score Std Dev  (ToT only)")
                    ax.set_xlabel("Scoring prompt variant")
                    ax.set_ylabel("Score std dev (5 judge calls)")
                    ax.grid(axis="y", alpha=0.2)
                    for patch in ax.patches:
                        h = patch.get_height()
                        if not np.isnan(h) and h > 0:
                            ax.text(
                                patch.get_x() + patch.get_width() / 2,
                                h + 0.002,
                                f"{h:.4f}",
                                ha="center",
                                va="bottom",
                                fontsize=8,
                                color=C_TEXT_MUTED,
                            )
                    pair_std_tot = _pairwise_welch_bonferroni_summary(
                        subset_std_tot, "score_prompt_variant", "score_std", order_x_tot
                    )
                    p_omni_tot = _one_way_anova_p(
                        subset_std_tot, "score_prompt_variant", "score_std", levels=order_x_tot
                    )
                    sto_tot = _stars(p_omni_tot)
                    _omnibus_p_box(
                        ax,
                        [
                            "Pairwise prompts within ToT: Welch t, Bonferroni p_adj.",
                            f"  {pair_std_tot}",
                            f"ANOVA omnibus: p={_format_p_short(p_omni_tot)}{' ' + sto_tot if sto_tot else ''}",
                        ],
                        fontsize=6.5,
                    )
                    fig.tight_layout()
                    _save(fig, grouped_path(out_dir, "modifications/score_prompt", "prompt_vs_score_std_tot_only.png"))


def plot_cost_vs_score_pareto(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    subdir: str = "baseline",
    filename: str = "score_vs_cost_pareto.png",
    ttest_annotation: bool = False,
) -> None:
    if "method" not in df.columns or "mean_score" not in df.columns or "total_cost_usd" not in df.columns:
        return
    cols = ["method", "mean_score", "total_cost_usd"]
    if ttest_annotation and "task_id" in df.columns:
        cols = list(dict.fromkeys(cols + ["task_id"]))
    subset = df[cols].dropna()
    if subset.empty:
        return
    grouped = (
        subset.groupby("method")
        .agg(
            mean_score=("mean_score", "mean"),
            avg_cost=("total_cost_usd", "mean"),
        )
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    for _, row in grouped.iterrows():
        color = METHOD_PALETTE.get(row["method"], C_FALLBACK_SERIES)
        ax.scatter(row["avg_cost"], row["mean_score"], color=color, s=130, zorder=5, alpha=0.9)
        ax.annotate(
            row["method"],
            (row["avg_cost"], row["mean_score"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
            color=color,
        )
    ax.set_xlabel("Avg cost per task (USD)")
    ax.set_ylabel("Mean coherency score")
    title = "Score vs Cost — Pareto Frontier\n(upper-left = best value for money)"
    if subdir == "astar_analysis" and set(grouped["method"].unique()) <= {"tot", "tot_astar"}:
        title = "Score vs Cost — tot vs tot_astar (merge JSON slice only)"
    ax.set_title(title)
    ax.grid(alpha=0.25)

    if ttest_annotation:
        methods_g = sorted(grouped["method"].astype(str).unique().tolist())
        lines: list[str] = []
        if len(methods_g) == 2:
            m0, m1 = methods_g[0], methods_g[1]
            ps, _ = _two_method_paired_and_welch(subset, "method", "mean_score", m0, m1)
            pc, _ = _two_method_paired_and_welch(subset, "method", "total_cost_usd", m0, m1)
            if ps:
                lines.append("Mean score:")
                lines.append(f"  {ps}")
            if pc:
                lines.append("Total cost:")
                lines.append(f"  {pc}")
        elif len(methods_g) >= 3:
            order_pm = [m for m in PLAN_METHODS_COMPARE_ORDER if m in methods_g]
            if len(order_pm) >= 2:
                ps = _pairwise_welch_bonferroni_summary(subset, "method", "mean_score", order_pm)
                lines.append("Mean score Welch p_adj:")
                lines.append(f"  {ps}")
                if "total_cost_usd" in subset.columns:
                    pc = _pairwise_welch_bonferroni_summary(subset, "method", "total_cost_usd", order_pm)
                    lines.append("Cost Welch p_adj:")
                    lines.append(f"  {pc}")
        if lines:
            _omnibus_p_box(ax, lines, fontsize=6.2)

    fig.tight_layout()
    _save(fig, grouped_path(out_dir, subdir, filename))


def plot_correlation_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    num_cols = [
        c
        for c in [
            "mean_score",
            "score_std",
            "vote_entropy_plan",
            "vote_entropy_passage",
            "total_cost_usd",
            "api_calls",
            "k",
            "n_votes",
            "vote_parse_failures_plan",
            "vote_parse_failures_passage",
        ]
        if c in df.columns
    ]
    subset = df[num_cols].dropna()
    if len(subset) < 5 or len(num_cols) < 3:
        return
    corr = subset.corr(method="pearson")
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap=CORR_CMAP,
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        linecolor=C_GRID,
        annot_kws={"size": 7.5},
        ax=ax,
    )
    ax.set_title("Pearson Correlation Matrix  (lower triangle)", fontsize=11)
    fig.tight_layout()
    _save(fig, grouped_path(out_dir, "baseline", "correlation_heatmap.png"))


# ── Passage scores from results JSON (same styling as dashboard) ───────────────


def load_results_json_passage_scores_long(results_json: Path) -> pd.DataFrame:
    """Expand {method: [{score, ...}, ...]} to one row per task score (method, mean_score)."""
    with results_json.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return pd.DataFrame(columns=["method", "mean_score"])
    rows: list[dict[str, object]] = []
    for method, entries in data.items():
        if not isinstance(entries, list):
            continue
        for e in entries:
            if isinstance(e, dict) and "score" in e:
                rows.append({"method": str(method), "mean_score": float(e["score"])})
    return pd.DataFrame(rows)


def run_passage_json_plots(results_json: Path, astar_analysis_dir: Path) -> None:
    """Passage scores from results JSON: tot vs tot_astar only.

    Unlike the rest of the dashboard, these plots read the per-passage results
    JSON directly (not the metrics log). This is the data path for paired
    per-task analyses (scatter, delta-hist, paired-line) — they need the
    individual score values, not aggregated means.
    """
    baselines = ["tot"]
    methods = ["tot", ASTAR_METHOD]
    with results_json.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return

    rows: list[dict] = []
    for m in methods:
        for e in data.get(m, []):
            if "id" not in e or "score" not in e:
                continue
            rows.append({"method": m, "task_id": int(e["id"]), "score": float(e["score"])})
    long = pd.DataFrame(rows)
    if long.empty:
        print("Passage JSON plots skipped: no score rows for requested methods.")
        return

    # Only compare tasks both methods completed — otherwise a method that
    # happened to skip the harder tasks would get a free score boost.
    by_m = {m: set(long.loc[long["method"] == m, "task_id"]) for m in methods}
    common = set.intersection(*by_m.values()) if by_m else set()
    if not common:
        print(f"Passage JSON plots skipped: no common task ids across {methods}.")
        return

    long_c = long[long["task_id"].isin(common)].copy()
    bar_order = [m for m in baselines + [ASTAR_METHOD] if m in long_c["method"].unique()]
    astar_analysis_dir.mkdir(parents=True, exist_ok=True)

    palette_l = [METHOD_PALETTE.get(m, C_FALLBACK_SERIES) for m in bar_order]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(
        data=long_c,
        x="method",
        y="score",
        order=bar_order,
        hue="method",
        hue_order=bar_order,
        palette=palette_l,
        errorbar=("ci", 95),
        capsize=0.08,
        ax=ax,
        legend=False,
    )
    ax.set_title("Passage scores (ToT vs A*) — mean ± 95% CI across tasks")
    ax.set_xlabel("method")
    ax.set_ylabel("passage score")
    ax.grid(axis="y", alpha=0.2)
    wide_bar = long_c.pivot(index="task_id", columns="method", values="score")
    if "tot" in wide_bar.columns and ASTAR_METHOD in wide_bar.columns:
        d_bar = (wide_bar[ASTAR_METHOD] - wide_bar["tot"]).dropna().to_numpy(dtype=float)
        d_bar = d_bar[np.isfinite(d_bar)]
        if len(d_bar) > 1 and np.std(d_bar, ddof=1) > 1e-12:
            try:
                _, p_bar = stats.ttest_1samp(d_bar, 0.0)
                p_bar = float(p_bar)
                stb = _stars(p_bar)
                _omnibus_p_box(
                    ax,
                    [
                        f"Paired t: mean passage Δ ({ASTAR_METHOD}−tot) = 0",
                        f"  p={_format_p_short(p_bar)}{(' ' + stb) if stb else ''}  n={len(d_bar)}",
                    ],
                    fontsize=6.5,
                )
            except (ValueError, TypeError):
                pass
    fig.tight_layout()
    _save(fig, astar_analysis_dir / "passage_mean_by_method.png")

    wide = long_c.pivot(index="task_id", columns="method", values="score").reset_index()
    wide.columns.name = None
    wide = wide.rename(columns={c: f"score_{c}" for c in wide.columns if c != "task_id"})

    for b in baselines:
        sc_b, sc_a = f"score_{b}", f"score_{ASTAR_METHOD}"
        if sc_b not in wide.columns or sc_a not in wide.columns:
            continue
        x, y = wide[sc_b], wide[sc_a]
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        sns.scatterplot(
            x=x,
            y=y,
            s=55,
            alpha=0.75,
            color=METHOD_PALETTE.get(ASTAR_METHOD, C_GREEN_MD),
            edgecolor=C_FOREST,
            linewidths=0.4,
            ax=ax,
        )
        lo = float(min(x.min(), y.min()))
        hi = float(max(x.max(), y.max()))
        pad = (hi - lo) * 0.05 + 0.05
        lim = (lo - pad, hi + pad)
        ax.plot(lim, lim, color=C_TEXT_MUTED, ls="--", lw=1, alpha=0.8, label="y = x")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal", adjustable="box")
        if len(x) >= 3 and x.nunique() > 1 and y.nunique() > 1:
            r_p, p_p = pearsonr(x, y)
            r_s, p_s = spearmanr(x, y)
            _stat_box(ax, r_p, p_p, r_s, p_s, len(x))
            spp, sps = _stars(p_p), _stars(p_s)
            ax.set_title(
                f"Passage scores: {b} vs {ASTAR_METHOD} (n = {len(wide)})\n"
                f"Pearson p={_format_p_short(p_p)}{(' ' + spp) if spp else ''}  "
                f"Spearman p={_format_p_short(p_s)}{(' ' + sps) if sps else ''}",
                fontsize=8,
            )
        else:
            ax.set_title(f"Passage scores: {b} vs {ASTAR_METHOD} (n = {len(wide)})")
        ax.set_xlabel(f"{b} (passage score)")
        ax.set_ylabel(f"{ASTAR_METHOD} (passage score)")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        _save(fig, astar_analysis_dir / f"passage_scatter_{b}_vs_{ASTAR_METHOD}.png")

        delta = wide[sc_a] - wide[sc_b]
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(delta, bins=min(16, max(6, len(delta) // 3)), kde=True, color=C_GREEN_MD, ax=ax)
        ax.axvline(0, color=C_GRID, ls="-", lw=0.9)
        ax.axvline(delta.mean(), color=C_FOREST, ls="--", lw=1.2, label=f"mean Δ = {delta.mean():.3f}")
        d_arr = delta.dropna().to_numpy(dtype=float)
        d_arr = d_arr[np.isfinite(d_arr)]
        if len(d_arr) > 1 and np.std(d_arr, ddof=1) > 1e-12:
            try:
                _, p_d = stats.ttest_1samp(d_arr, 0.0)
                p_d = float(p_d)
                std = _stars(p_d)
                _omnibus_p_box(
                    ax,
                    [
                        "One-sample t: mean Δ = 0",
                        f"  p={_format_p_short(p_d)}{(' ' + std) if std else ''}  n={len(d_arr)}",
                    ],
                    fontsize=6.5,
                )
            except (ValueError, TypeError):
                pass
        ax.set_title(f"Δ passage score ({ASTAR_METHOD} − {b}) per task")
        ax.set_xlabel("Δ score")
        ax.set_ylabel("count")
        ax.legend()
        ax.grid(alpha=0.2)
        fig.tight_layout()
        _save(fig, astar_analysis_dir / f"passage_delta_hist_{ASTAR_METHOD}_minus_{b}.png")

        sub = wide[[sc_b, sc_a, "task_id"]].sort_values(sc_b)
        fig, ax = plt.subplots(figsize=(9, 4.5))
        xx = range(len(sub))
        ax.plot(xx, sub[sc_b].values, "o-", label=b, color=METHOD_PALETTE.get(b, C_FOREST_DEEP), alpha=0.9)
        ax.plot(xx, sub[sc_a].values, "o-", label=ASTAR_METHOD, color=METHOD_PALETTE.get(ASTAR_METHOD, C_GREEN_MD), alpha=0.9)
        d_pair = (sub[sc_a] - sub[sc_b]).to_numpy(dtype=float)
        d_pair = d_pair[np.isfinite(d_pair)]
        title_paired = f"Paired passage scores by task (sorted by {b})"
        if len(d_pair) > 1 and np.std(d_pair, ddof=1) > 1e-12:
            try:
                _, p_pr = stats.ttest_1samp(d_pair, 0.0)
                p_pr = float(p_pr)
                stp = _stars(p_pr)
                title_paired += (
                    f"\nPaired t (Δ=0): p={_format_p_short(p_pr)}{(' ' + stp) if stp else ''}  n={len(d_pair)}"
                )
            except (ValueError, TypeError):
                pass
        ax.set_title(title_paired)
        ax.set_xlabel("task rank")
        ax.set_ylabel("passage score")
        ax.legend()
        ax.grid(alpha=0.2)
        fig.tight_layout()
        _save(fig, astar_analysis_dir / f"passage_paired_{b}_vs_{ASTAR_METHOD}.png")

    print(f"Passage JSON plots (tot vs {ASTAR_METHOD}): {astar_analysis_dir} (n_tasks={len(common)})")


# ── Metrics-log correlation (formerly astar_comparison_analysis.py) ────────────


def load_metrics_log_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Metrics log not found: {path}")
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def build_tot_astar_joined_frame(
    rows: list[dict], out_file: str, prompt_variants: set[str]
) -> pd.DataFrame:
    """Pivot the long-form metrics log into wide (tot, tot_astar) pairs.

    Inner-join on `key_cols` (task_id + every AxesConfig field) so we only
    correlate truly paired runs — same task, same hyperparameters, just the
    two methods. Filtering by `out_file` keeps the join scoped to a single
    experiment file so re-runs in other contexts don't bleed in.
    """
    filtered = [
        r
        for r in rows
        if r.get("out_file") == out_file
        and str(r.get("score_prompt_variant", "")) in prompt_variants
        and r.get("method") in {"tot", "tot_astar"}
    ]
    if not filtered:
        return pd.DataFrame()

    df = pd.DataFrame(filtered)
    # Joining on all AxesConfig fields (k, n_votes, score_prompt_variant,
    # score_model_type) guarantees the pair came from the same configuration —
    # a mismatched k would silently change the comparison.
    key_cols = ["task_id", "out_file", "score_prompt_variant", "k", "n_votes", "score_model_type"]
    tot = df[df["method"] == "tot"].copy()
    astar = df[df["method"] == "tot_astar"].copy()

    keep_cols = key_cols + [
        "mean_score",
        "score_std",
        "vote_entropy_plan",
        "vote_entropy_passage",
        "vote_parse_failures_plan",
        "vote_parse_failures_passage",
        "score_parse_failures",
        "api_calls",
        "prompt_tokens",
        "completion_tokens",
        "total_cost_usd",
    ]
    tot = tot[[c for c in keep_cols if c in tot.columns]]
    astar = astar[[c for c in keep_cols if c in astar.columns]]

    joined = tot.merge(astar, on=key_cols, suffixes=("_tot", "_tot_astar"), how="inner")
    if joined.empty:
        return joined
    # Sign convention: positive score_delta = A* did better. Keep this consistent
    # — flipping it later would silently invert every correlation plot.
    joined["score_delta"] = joined["mean_score_tot_astar"] - joined["mean_score_tot"]
    return joined


def print_log_score_delta_correlations(df: pd.DataFrame, variables: list[str]) -> None:
    print(f"Matched rows: {len(df)}")
    if len(df) < 3:
        print("Need at least 3 matched rows for meaningful correlation.")
        return

    for var in variables:
        if var not in df.columns:
            print(f"{var}: missing column")
            continue
        sub = df[["score_delta", var]].dropna()
        if len(sub) < 3:
            print(f"{var}: insufficient non-null rows ({len(sub)})")
            continue
        if sub[var].nunique() <= 1 or sub["score_delta"].nunique() <= 1:
            print(f"{var}: skip (constant column or constant score_delta), n={len(sub)}")
            continue
        r, p = pearsonr(sub["score_delta"], sub[var])
        print(f"{var}: r={r:.3f}, p={p:.3g}, n={len(sub)}")


def print_log_score_delta_diagnostics(df: pd.DataFrame) -> None:
    if df.empty:
        return
    print("\nExtra diagnostics:")
    print(f"mean score_delta (tot_astar - tot): {df['score_delta'].mean():.4f}")
    print(f"median score_delta: {df['score_delta'].median():.4f}")
    win_rate = (df["score_delta"] > 0).mean()
    tie_rate = (df["score_delta"] == 0).mean()
    print(f"win rate (score_delta > 0): {win_rate:.2%}")
    print(f"tie rate (score_delta = 0): {tie_rate:.2%}")


def _safe_corr_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name).strip("_") or "var"


def write_log_correlation_delta_plots(df: pd.DataFrame, variables: list[str], plot_dir: Path) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    if df.empty or "score_delta" not in df.columns:
        print("Correlation plots skipped: empty joined frame.")
        return
    for var in variables:
        if var not in df.columns:
            continue
        sub = df[["score_delta", var]].dropna()
        if len(sub) < 3:
            print(f"plot {var}: skipped (n={len(sub)})")
            continue
        if sub[var].nunique() <= 1 or sub["score_delta"].nunique() <= 1:
            print(f"plot {var}: skipped (constant x or y)")
            continue
        r, p = pearsonr(sub["score_delta"], sub[var])
        xlab = CORRELATION_DELTA_LABELS.get(var, var)
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.regplot(
            data=sub,
            x=var,
            y="score_delta",
            scatter_kws={"alpha": 0.75, "s": 50, "color": C_GREEN_DARK, "edgecolor": C_FOREST, "linewidths": 0.4},
            line_kws={"color": C_ACCENT_LINE, "lw": 1.5},
            ax=ax,
        )
        ax.axhline(0, color=C_TEXT_MUTED, ls="--", lw=1)
        ax.axvline(0, color=C_TEXT_MUTED, ls="--", lw=1, alpha=0.7)
        ax.set_title(f"score Δ vs {xlab}\nPearson r = {r:.3f}, p = {p:.3g}, n = {len(sub)}")
        ax.set_xlabel(xlab)
        ax.set_ylabel("score_delta (mean score, A* − ToT)")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        out = plot_dir / f"score_delta_vs_{_safe_corr_filename(var)}.png"
        _save(fig, out)
        print(f"Wrote {out}")


def run_correlation_only(args: argparse.Namespace) -> None:
    rows = load_metrics_log_rows(args.correlation_metrics_log)
    df = build_tot_astar_joined_frame(
        rows,
        out_file=args.correlation_out_file,
        prompt_variants=set(args.correlation_prompt_variants),
    )
    df = augment_joined_with_metric_deltas(df)
    print_log_score_delta_correlations(df, list(args.correlation_variables))
    print_log_score_delta_diagnostics(df)
    if args.correlation_plot_dir is not None:
        write_log_correlation_delta_plots(df, list(args.correlation_variables), args.correlation_plot_dir)


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    if args.correlation_only:
        # Fast path for ad-hoc tot vs tot_astar exploration — skips all plotting
        # so reruns during analysis don't have to rebuild the whole dashboard.
        run_correlation_only(args)
        return

    out_dir = args.out_dir
    if args.clean:
        clean_dashboard_artifacts(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.input.exists():
        raise FileNotFoundError(f"Input not found: {args.input}")

    df_full = pd.read_json(args.input, lines=True)
    ensure_columns(df_full, args.group_by + ["mean_score", "total_cost_usd", "api_calls"])
    df_full = add_derived_columns(df_full)

    # Six slices, six experimental questions. Each is the input to a specific
    # subset of plots below; mixing them would silently combine ablations.
    merge_name = args.astar_merge_json_name
    df_paper_raw = slice_paper_core(df_full)
    df_astar_raw = slice_astar_tot_compare(df_full, merge_name)
    df_plan_raw = slice_plan_methods_compare(df_full)
    df_k_raw = slice_k_ablation(df_full)
    df_n_votes_raw = slice_n_votes_ablation(df_full)
    df_prompt_raw = slice_score_prompt_compare(df_full)

    if args.filter_methods:
        df_paper_raw = df_paper_raw[df_paper_raw["method"].isin(args.filter_methods)].copy()
        if df_paper_raw.empty:
            raise SystemExit(f"No rows left in paper-core slice after --filter-methods {args.filter_methods}.")
        df_k_raw = df_k_raw[df_k_raw["method"].isin(args.filter_methods)].copy()
        df_n_votes_raw = df_n_votes_raw[df_n_votes_raw["method"].isin(args.filter_methods)].copy()
        df_prompt_raw = df_prompt_raw[df_prompt_raw["method"].isin(args.filter_methods)].copy()

    # Balance by default: cross-method comparisons need equal n. The escape
    # hatch (--no-balance-methods) exists for diagnostic runs where you'd
    # rather see the raw row counts than the matched subset.
    if args.no_balance_methods:
        df_paper = df_paper_raw.copy()
        df_astar = df_astar_raw.copy()
        df_plan = df_plan_raw.copy()
        df_k = df_k_raw.copy()
        df_n_votes = df_n_votes_raw.copy()
        df_prompt = df_prompt_raw.copy()
    else:
        df_paper = balance_method_counts(df_paper_raw) if not df_paper_raw.empty else df_paper_raw
        df_astar = balance_method_counts(df_astar_raw) if not df_astar_raw.empty else df_astar_raw
        df_plan = balance_method_counts(df_plan_raw) if not df_plan_raw.empty else df_plan_raw
        df_k = balance_method_counts(df_k_raw) if not df_k_raw.empty else df_k_raw
        df_n_votes = balance_method_counts(df_n_votes_raw) if not df_n_votes_raw.empty else df_n_votes_raw
        df_prompt = balance_method_counts(df_prompt_raw) if not df_prompt_raw.empty else df_prompt_raw
        if not df_paper_raw.empty and not df_paper.empty:
            bc = df_paper_raw.groupby("method").size()
            ac = df_paper.groupby("method").size()
            print(
                "Paper-core (whitelist JSONs): balanced methods — "
                + ", ".join(f"{m}: {bc.get(m, 0)}→{ac.get(m, 0)}" for m in sorted(ac.index))
            )
        if not df_astar_raw.empty and not df_astar.empty:
            bc = df_astar_raw.groupby("method").size()
            ac = df_astar.groupby("method").size()
            print(
                f"{merge_name} (tot↔tot_astar): balanced — "
                + ", ".join(f"{m}: {bc.get(m, 0)}→{ac.get(m, 0)}" for m in sorted(ac.index))
            )
        if not df_plan_raw.empty and not df_plan.empty:
            bc = df_plan_raw.groupby("method").size()
            ac = df_plan.groupby("method").size()
            print(
                "plan_methods.json: balanced — "
                + ", ".join(f"{m}: {bc.get(m, 0)}→{ac.get(m, 0)}" for m in sorted(ac.index))
            )
        if not df_k_raw.empty and not df_k.empty:
            bc = df_k_raw.groupby("method").size()
            ac = df_k.groupby("method").size()
            print(
                "k-ablation JSONs: balanced — "
                + ", ".join(f"{m}: {bc.get(m, 0)}→{ac.get(m, 0)}" for m in sorted(ac.index))
            )
        if not df_n_votes_raw.empty and not df_n_votes.empty:
            bc = df_n_votes_raw.groupby("method").size()
            ac = df_n_votes.groupby("method").size()
            print(
                "n-votes ablation JSONs: balanced — "
                + ", ".join(f"{m}: {bc.get(m, 0)}→{ac.get(m, 0)}" for m in sorted(ac.index))
            )
        if not df_prompt_raw.empty and not df_prompt.empty:
            bc = df_prompt_raw.groupby("method").size()
            ac = df_prompt.groupby("method").size()
            print(
                "score-prompt JSONs: balanced — "
                + ", ".join(f"{m}: {bc.get(m, 0)}→{ac.get(m, 0)}" for m in sorted(ac.index))
            )

    grouped = build_grouped_table(df_paper, args.group_by)
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(tables_dir / "grouped_metrics.csv", index=False)
    grouped.to_json(tables_dir / "grouped_metrics.json", orient="records", indent=2)

    if not df_astar.empty:
        g_astar = build_grouped_table(df_astar, args.group_by)
        g_astar.to_csv(out_dir / "tables" / "grouped_metrics_tot_astar_compare.csv", index=False)
        g_astar.to_json(out_dir / "tables" / "grouped_metrics_tot_astar_compare.json", orient="records", indent=2)
    else:
        print(f"No rows for tot↔tot_astar comparison (expect {merge_name} with tot and tot_astar in the log).")

    if not df_plan.empty:
        g_plan = build_grouped_table(df_plan, args.group_by)
        g_plan.to_csv(out_dir / "tables" / "grouped_metrics_plan_methods.csv", index=False)
        g_plan.to_json(out_dir / "tables" / "grouped_metrics_plan_methods.json", orient="records", indent=2)
    else:
        print(f"No rows for plan_methods comparison (expect {PLAN_METHODS_JSON_NAME} with tot, tot_astar, hybrid_tot).")

    overview_json = args.overview_methods_json
    if overview_json.is_file():
        df_overview = load_results_json_passage_scores_long(overview_json)
        if df_overview.empty:
            print(
                f"Warning: no passage scores in {overview_json}; "
                "method_score_distribution + method_mean_score_bar use paper-core metrics."
            )
            plot_method_score_distribution(df_paper, out_dir)
            plot_method_comparison_bar(df_paper, out_dir)
        else:
            note = f"(passage scores: {overview_json.name})"
            plot_method_score_distribution(df_overview, out_dir, title_note=note)
            plot_method_comparison_bar(df_overview, out_dir)
    else:
        print(
            f"Warning: overview methods JSON not found ({overview_json}); "
            "method_score_distribution + method_mean_score_bar use paper-core metrics."
        )
        plot_method_score_distribution(df_paper, out_dir)
        plot_method_comparison_bar(df_paper, out_dir)
    plot_pairwise_score_gaps(df_paper, out_dir)
    plot_correlation_heatmap(df_paper, out_dir)
    plot_cost_vs_score_pareto(df_paper, out_dir)

    plot_astar_vs_tot_comparison(df_astar, out_dir)
    plot_score_delta_correlations(df_astar, out_dir)
    plot_cost_vs_score_pareto(
        df_astar,
        out_dir,
        subdir="independent_exploration/astar_analysis",
        filename="score_vs_cost_pareto_tot_vs_tot_astar.png",
        ttest_annotation=True,
    )

    plot_plan_methods_three_way(df_plan, out_dir)
    plot_cost_vs_score_pareto(
        df_plan,
        out_dir,
        subdir="independent_exploration/plan_methods",
        filename="score_vs_cost_pareto.png",
        ttest_annotation=True,
    )

    df_scorer_raw = slice_scorer_baseline_vs_scorer41(df_full)
    if not df_scorer_raw.empty:
        df_scorer = df_scorer_raw.copy()
        df_scorer["results_json"] = df_scorer["out_file"].map(out_file_basename)
        df_scorer = df_scorer[df_scorer["results_json"].isin(SCORER_COMPARE_JSONS)]
        if not args.no_balance_methods and not df_scorer.empty:
            bc = df_scorer.groupby(["method", "results_json"], sort=False).size()
            df_scorer = balance_by_method_and_bucket(df_scorer, "method", "results_json")
            ac = df_scorer.groupby(["method", "results_json"], sort=False).size()

            def _mi_cnt(s: pd.Series, m: str, j: str) -> int:
                key = (m, j)
                return int(s.loc[key]) if key in s.index else 0

            lines = []
            for m in sorted(df_scorer["method"].unique()):
                bits = [f"{j[:-5]} {_mi_cnt(bc, m, j)}→{_mi_cnt(ac, m, j)}" for j in ("baseline.json", "scorer4.1.json")]
                lines.append(f"{m}: " + ", ".join(bits))
            print("baseline.json vs scorer4.1.json: balanced method×JSON — " + "; ".join(lines))
        elif df_scorer.empty:
            pass
        else:
            print("baseline.json vs scorer4.1.json: using raw rows (--no-balance-methods).")

        if not df_scorer.empty and df_scorer["results_json"].nunique() >= 2:
            gb_scorer = list(dict.fromkeys(args.group_by + ["results_json"]))
            if "score_model_type" in df_scorer.columns:
                gb_scorer = list(dict.fromkeys(gb_scorer + ["score_model_type"]))
            g_scorer = build_grouped_table(df_scorer, gb_scorer)
            g_scorer.to_csv(out_dir / "tables" / "grouped_metrics_scorer_compare.csv", index=False)
            g_scorer.to_json(out_dir / "tables" / "grouped_metrics_scorer_compare.json", orient="records", indent=2)
            plot_scorer_baseline_vs_scorer41(
                df_scorer,
                out_dir,
                balanced_method_json=not args.no_balance_methods,
            )
        else:
            print(
                "scorer_compare: skipped tables/grouped_metrics_scorer_compare.* and plot (analysis/) "
                "(need both baseline.json and scorer4.1.json in the metrics log for paper-core methods)."
            )
    else:
        print("scorer_compare: no rows in baseline.json / scorer4.1.json paper-core slice.")

    plot_k_ablation(df_k, out_dir)
    plot_n_votes_ablation(df_n_votes, out_dir)

    plot_score_prompt_comparison(df_prompt, out_dir)

    plot_vote_entropy_vs_score(df_paper, out_dir)
    plot_score_std_vs_mean(df_paper, out_dir)
    plot_compact_controlled_regression(df_paper, out_dir)
    write_interactive_regression_explorer(df_paper, out_dir)

    results_json = args.results_json if args.results_json is not None else DEFAULT_RESULTS_JSON
    if results_json.is_file():
        run_passage_json_plots(results_json, out_dir / "independent_exploration" / "astar_analysis")
    elif args.results_json is not None:
        print(f"Warning: --results-json not found: {args.results_json}")
    elif not DEFAULT_RESULTS_JSON.is_file():
        print(f"Passage JSON plots skipped (default file not found): {DEFAULT_RESULTS_JSON}")

    print(f"\nTop {args.top_n} groups by mean_score:")
    print(grouped.sort_values("mean_score", ascending=False).head(args.top_n).to_string(index=False))
    print(f"\nDashboard written to: {out_dir}")


if __name__ == "__main__":
    main()
