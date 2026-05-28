"""Shared metrics data classes + jsonl logger.

Every solver writes one row per task to a single jsonl log (METRICS_LOG_PATH).
Filtering on `AxesConfig` (model, k, n_votes, prompt variant) is how the
dashboard isolates each ablation axis — keep this in sync with whatever knobs
get added to the ablation grid, otherwise the dashboard will mix runs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from math import log2
from pathlib import Path
from typing import Optional


@dataclass
class AxesConfig:
    """Identifies the experiment axis a row belongs to.

    Rows with different AxesConfigs are treated as different experiments by
    the dashboard. New ablation axes belong here so the filter logic stays in
    one place.
    """

    score_model_type: str
    k: int
    n_votes: int
    score_prompt_variant: str


@dataclass
class ScoreResult:
    mean: float
    std: float
    parse_failures: int
    parsed_scores: list[float]
    attempts: int


@dataclass
class VoteResult:
    entropy: float
    parse_failures: int
    tally: list[int]
    winner_index_1based: int
    attempts: int


@dataclass
class TaskMetrics:
    out_file: str
    method: str
    task_id: int
    mean_score: float
    score_std: float
    score_parse_failures: int
    score_attempts: int
    vote_entropy_plan: Optional[float]
    vote_entropy_passage: Optional[float]
    vote_parse_failures_plan: int
    vote_parse_failures_passage: int
    vote_attempts_plan: int
    vote_attempts_passage: int
    api_calls: int
    prompt_tokens: int
    completion_tokens: int
    total_cost_usd: float
    human_plan_choice: Optional[int] = None
    llm_plan_choice: Optional[int] = None
    human_llm_agree: Optional[bool] = None


def vote_entropy(tally: list[int]) -> float:
    """Shannon entropy (bits) over the vote distribution.

    Used as a confidence proxy in the dashboard: low entropy = judges agree on
    a winner, high entropy = the choice was effectively coin-flippy. Returns 0
    for empty tallies (no parsed votes) rather than raising, so a degenerate
    task doesn't blow up the whole batch.
    """
    total = sum(tally)
    if total <= 0:
        return 0.0
    entropy = 0.0
    for c in tally:
        if c <= 0:
            continue
        p = c / total
        entropy -= p * log2(p)
    return round(entropy, 6)


def append_metrics_jsonl(path: Path, axes: AxesConfig, task_metrics: TaskMetrics) -> None:
    # Append-only jsonl beats a single JSON blob here: concurrent solvers can
    # write without locking, and a crash mid-run leaves a partial-but-readable
    # log. The dashboard takes "latest row wins" per (axes, method, task_id).
    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **asdict(axes),
        **asdict(task_metrics),
    }
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def _avg(values: list[float]) -> float:
    return (sum(values) / len(values)) if values else 0.0


def build_method_summary(method: str, rows: list[TaskMetrics], axes: AxesConfig) -> dict:
    # Entropies filter out None so IO/CoT (no votes) don't artificially drag
    # the average toward zero — None means "not applicable", not "zero".
    mean_score = _avg([r.mean_score for r in rows])
    avg_score_std = _avg([r.score_std for r in rows])
    avg_vote_entropy_plan = _avg([r.vote_entropy_plan for r in rows if r.vote_entropy_plan is not None])
    avg_vote_entropy_passage = _avg([r.vote_entropy_passage for r in rows if r.vote_entropy_passage is not None])

    score_failures = sum(r.score_parse_failures for r in rows)
    score_attempts = sum(r.score_attempts for r in rows)
    vote_failures = sum(r.vote_parse_failures_plan + r.vote_parse_failures_passage for r in rows)
    vote_attempts = sum(r.vote_attempts_plan + r.vote_attempts_passage for r in rows)

    return {
        "method": method,
        **asdict(axes),
        "mean_score": round(mean_score, 4),
        "avg_score_std": round(avg_score_std, 4),
        "avg_vote_entropy_plan": round(avg_vote_entropy_plan, 4) if any(r.vote_entropy_plan is not None for r in rows) else None,
        "avg_vote_entropy_passage": round(avg_vote_entropy_passage, 4) if any(r.vote_entropy_passage is not None for r in rows) else None,
        "parse_failure_rate_score": round(score_failures / score_attempts, 6) if score_attempts else 0.0,
        "parse_failure_rate_vote": round(vote_failures / vote_attempts, 6) if vote_attempts else None,
        "total_api_calls": sum(r.api_calls for r in rows),
        "total_prompt_tokens": sum(r.prompt_tokens for r in rows),
        "total_completion_tokens": sum(r.completion_tokens for r in rows),
        "total_cost_usd": round(sum(r.total_cost_usd for r in rows), 6),
        "num_tasks": len(rows),
    }


def print_method_summary_table(summary: dict) -> None:
    def _fmt(v: Optional[float], digits: int = 4) -> str:
        if v is None:
            return "-"
        return f"{v:.{digits}f}"

    headers = ["method", "mean", "std", "H_plan", "H_pass", "pf_score", "pf_vote", "api_calls"]
    values = [
        str(summary["method"]),
        _fmt(summary["mean_score"]),
        _fmt(summary["avg_score_std"]),
        _fmt(summary["avg_vote_entropy_plan"]),
        _fmt(summary["avg_vote_entropy_passage"]),
        _fmt(summary["parse_failure_rate_score"], 6),
        _fmt(summary["parse_failure_rate_vote"], 6),
        str(summary["total_api_calls"]),
    ]
    widths = [max(len(h), len(v)) for h, v in zip(headers, values)]

    header_row = "  " + "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    divider_row = "  " + "  ".join("-" * w for w in widths)
    value_row = "  " + "  ".join(v.ljust(w) for v, w in zip(values, widths))

    print("\n-- Method Experiment Summary --")
    print(header_row)
    print(divider_row)
    print(value_row)

