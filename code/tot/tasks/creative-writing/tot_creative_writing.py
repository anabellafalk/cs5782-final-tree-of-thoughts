#!/usr/bin/env python3
"""
Tree of Thoughts: Creative Writing Task (Yao et al. 2023, §3.2).

Two ToT entry points coexist intentionally:
- `tot_solve` is the paper-faithful procedure (plan-vote then passage-vote, b=1)
  used for direct comparisons against published numbers. It's the path exercised
  by `--method tot` in this module's CLI.
- `CreativeWritingTask` exposes the same task to the generic `core.search`
  framework (BFS over plan/passage stages), so the same problem can be plugged
  into other search policies without rewriting prompts.

All model calls go through `core.llm.LLMClient`, which handles caching, cost
tracking, and provider quirks. Cache + cost logs live under the repo root so
multiple solvers (incl. tot_astar, hybrid_tot) share state.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import random
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from statistics import pstdev

from dotenv import load_dotenv

TOT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(TOT_ROOT) not in sys.path:
    sys.path.append(str(TOT_ROOT))

from core.llm import LLMClient
from core.search import bfs_search
from core.task import State, Task
from experiment_metrics import (
    AxesConfig,
    ScoreResult,
    TaskMetrics,
    VoteResult,
    append_metrics_jsonl,
    build_method_summary,
    print_method_summary_table,
    vote_entropy,
)

load_dotenv(REPO_ROOT / ".env")

# Hyperparameters. K, N_VOTES, and prompt variants are overridable from the CLI
# so we can sweep them in ablations without editing source.
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
# Separating SCORE_MODEL lets us cheaply judge with the same model used for
# generation, OR run a stronger judge (e.g. gpt-4.1) without re-running the
# expensive generation phase. Defaults to the gen model for parity with paper.
SCORE_MODEL = os.getenv("SCORE_MODEL", MODEL)
TEMPERATURE = 0.7
K = 5              # paper default: 5 candidates per ToT stage
N_VOTES = 5        # paper default: 5 vote samples to aggregate (mode wins)
N_SCORE = 5        # judges per final passage; averaged for stability
N_BASELINE = 10    # IO/CoT sample budget — picked to roughly match ToT's call count
MAX_REFINE = 5     # self-refine hard cap; we also early-exit when score stops improving
# Optional rate limit (seconds between requests). 0 disables — useful for providers
# without burst limits (OpenAI tier-2+) and avoids artificial slowdowns.
REQUEST_DELAY_SECONDS = float(os.getenv("REQUEST_DELAY_SECONDS", "0"))
# Judges run in parallel because they're independent and dominate wall-clock for ToT.
SCORE_WORKERS = int(os.getenv("SCORE_WORKERS", "8"))
SCORE_PROMPT_VARIANT = os.getenv("SCORE_PROMPT_VARIANT", "paper")
VOTE_PROMPT_VARIANT = os.getenv("VOTE_PROMPT_VARIANT", "paper")
PRINT_SCORE_THOUGHTS = os.getenv("PRINT_SCORE_THOUGHTS", "0") == "1"

ROOT = REPO_ROOT
DATA_PATH = ROOT / "data" / "sentences.json"
RESULTS_PATH = ROOT / "results"
CACHE_DIR = ROOT / ".llm_cache"
COST_LOG_PATH = RESULTS_PATH / "cost_log.jsonl"

# Two separate clients so a different scorer (e.g. gpt-4.1) can be A/B-tested
# without rebuilding the generation pipeline.
GEN_LLM = LLMClient(
    model=MODEL,
    provider="openai",
    cache_dir=str(CACHE_DIR),
    cost_log_path=str(COST_LOG_PATH),
)
JUDGE_LLM = LLMClient(
    model=SCORE_MODEL,
    provider="openai",
    cache_dir=str(CACHE_DIR),
    cost_log_path=str(COST_LOG_PATH),
)
_LAST_REQUEST_TS = 0.0
# Process-wide perf counters. Updated from parallel judge threads, so all
# mutations must hold `_PERF_LOCK`. We report two views: this-run totals and
# cumulative-across-runs totals (rebuilt by reading the metrics log).
_PERF = {
    "api_calls": 0,
    "api_time_s": 0.0,
    "rate_sleep_s": 0.0,
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_cost_usd": 0.0,
}
_RATE_LOCK = threading.Lock()   # serialises the inter-request delay across threads
_PERF_LOCK = threading.Lock()   # protects _PERF mutations
_PRINT_LOCK = threading.Lock()  # keeps interleaved judge prints readable


@dataclass
class SolveResult:
    passage: str
    score: float
    score_std: float
    score_parse_failures: int
    score_attempts: int
    vote_plan: Optional[VoteResult] = None
    vote_passage: Optional[VoteResult] = None


def _rate_limit() -> None:
    global _LAST_REQUEST_TS
    if REQUEST_DELAY_SECONDS <= 0:
        return
    with _RATE_LOCK:
        now = time.monotonic()
        wait = REQUEST_DELAY_SECONDS - (now - _LAST_REQUEST_TS)
        if wait > 0:
            with _PERF_LOCK:
                _PERF["rate_sleep_s"] += wait
            time.sleep(wait)
        _LAST_REQUEST_TS = time.monotonic()


def _record_api_time(elapsed_s: float, prompt_tokens: int, completion_tokens: int, cost_usd: float) -> None:
    with _PERF_LOCK:
        _PERF["api_calls"] += 1
        _PERF["api_time_s"] += elapsed_s
        _PERF["prompt_tokens"] += prompt_tokens
        _PERF["completion_tokens"] += completion_tokens
        _PERF["total_cost_usd"] += cost_usd


def _perf_snapshot() -> dict:
    with _PERF_LOCK:
        return dict(_PERF)


def _perf_delta(before: dict, after: dict) -> dict:
    keys = ("api_calls", "prompt_tokens", "completion_tokens", "total_cost_usd")
    return {k: after.get(k, 0) - before.get(k, 0) for k in keys}


def _load_cumulative_method_metrics(method: str, axes: AxesConfig, out_label: str) -> list[TaskMetrics]:
    # Latest-wins reduction by task_id: we may re-run a task and want the most
    # recent metrics row (jsonl is append-only). Filtering on AxesConfig keeps
    # ablation runs from polluting each other's summary.
    if not METRICS_LOG_PATH.exists():
        return []
    by_task: dict[int, TaskMetrics] = {}
    with open(METRICS_LOG_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("method") != method:
                continue
            row_out_file = row.get("out_file")
            # Backward compatibility: older metrics rows did not include out_file.
            # Treat missing out_file as matching the active output label.
            if row_out_file is not None and row_out_file != out_label:
                continue
            if row.get("score_model_type") != axes.score_model_type:
                continue
            if row.get("k") != axes.k or row.get("n_votes") != axes.n_votes:
                continue
            if row.get("score_prompt_variant") != axes.score_prompt_variant:
                continue
            by_task[row["task_id"]] = TaskMetrics(
                out_file=row_out_file or out_label,
                method=row["method"],
                task_id=row["task_id"],
                mean_score=row["mean_score"],
                score_std=row["score_std"],
                score_parse_failures=row["score_parse_failures"],
                score_attempts=row["score_attempts"],
                vote_entropy_plan=row["vote_entropy_plan"],
                vote_entropy_passage=row["vote_entropy_passage"],
                vote_parse_failures_plan=row["vote_parse_failures_plan"],
                vote_parse_failures_passage=row["vote_parse_failures_passage"],
                vote_attempts_plan=row["vote_attempts_plan"],
                vote_attempts_passage=row["vote_attempts_passage"],
                api_calls=row["api_calls"],
                prompt_tokens=row["prompt_tokens"],
                completion_tokens=row["completion_tokens"],
                total_cost_usd=row["total_cost_usd"],
            )
    return list(by_task.values())


def _load_cumulative_totals(methods: list[str], axes: AxesConfig, out_label: str) -> dict:
    # `seen` guards against double-counting a (method, task_id) if the same task
    # appears more than once after the latest-wins reduction above.
    totals = {"api_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_cost_usd": 0.0}
    seen: set[tuple[str, int]] = set()
    for method in methods:
        for row in _load_cumulative_method_metrics(method, axes, out_label):
            key = (method, row.task_id)
            if key in seen:
                continue
            seen.add(key)
            totals["api_calls"] += row.api_calls
            totals["prompt_tokens"] += row.prompt_tokens
            totals["completion_tokens"] += row.completion_tokens
            totals["total_cost_usd"] += row.total_cost_usd
    return totals

PLAN_PROMPT = """\
Write a brief plan for a coherent 4-paragraph passage where each paragraph \
ends with one of the following sentences (used in order):
1. {s1}
2. {s2}
3. {s3}
4. {s4}

Describe the theme, tone, and what each paragraph should cover."""

PASSAGE_PROMPT = """\
Write a coherent 4-paragraph passage following this plan:

{plan}

Each paragraph must end with its corresponding sentence (in order):
Paragraph 1 ends with: {s1}
Paragraph 2 ends with: {s2}
Paragraph 3 ends with: {s3}
Paragraph 4 ends with: {s4}

Output only the passage, no headers or extra commentary."""

VOTE_PROMPTS = {
    "paper": """\
Below are {k} candidate {kind}s for a creative writing task. \
Analyze each carefully, then conclude which is most promising.

{candidates}

After your analysis, end with exactly: "the best choice is {{i}}" \
where i is the number (1-{k}) of the best option.""",
    "criteria": """\
Below are {k} candidate {kind}s for a creative writing task.

Check each candidate across these criteria:
- Coherence: Does the writing flow logically and feel unified?
- Creativity: Is the writing original, engaging, and imaginative?
- Constraint Satisfaction: Does each paragraph naturally end with its required sentence?

{candidates}

After your analysis, end with exactly: "the best choice is {{i}}" \
where i is the number (1-{k}) of the highest total score.""",
}

SCORE_PROMPTS = {
    "paper": """\
Rate the coherency of the following passage on a 1-10 scale \
(1 = completely incoherent, 10 = perfectly coherent and well-written).

Passage:
{passage}

Briefly justify your rating, then end with exactly: "coherency score is {{n}}" \
where n is an INTEGER ONLY from 1 to 10.""",
    "definition": """\
Rate the passage's coherency from 1-10.
Coherence refers to the overall sense of unity among your ideas and clarity of your writing structure.

Passage:
{passage}

Briefly justify your rating, then end with exactly: "coherency score is {{n}}" \
where n is an INTEGER ONLY from 1 to 10.""",
    "criteria": """\
        Rate the coherency of the following passage on a 1-10 scale.

Check each criterion:
- Content & Unity: Excellent = clear topic, strong support | Good = mostly on topic | Fair = sometimes off-topic | Poor = off-topic
- Flow & Coherence: Excellent = ideas connect smoothly | Good = mostly clear | Fair = some jumpy ideas | Poor = no clear flow
- Organization: Excellent = clear structure throughout | Good = mostly organized | Fair = parts missing/out of order | Poor = no organization
- Grammar & Spelling: Excellent = few/no errors | Good = some small errors | Fair = frequent errors | Poor = many errors
- Vocabulary: Excellent = varied and interesting | Good = some variety | Fair = basic/repetitive | Poor = very limited

Passage:
{passage}

Instructions:
1. Rate each criterion as Poor, Fair, Good, or Excellent
2. Map your overall impression to a 1-10 score
3. Briefly justify your rating

End with exactly: "coherency score is {{n}}" where n is an INTEGER from 1 to 10.""",
}

PLAN_SCORE_PROMPT = """\
Rate this writing plan on a 1-10 scale for likely coherency and executability.

Plan:
{plan}

End with exactly: "coherency score is {{n}}" where n is an integer from 1 to 10."""

IO_PROMPT = """\
Write a coherent 4-paragraph passage where each paragraph ends with \
one of the following sentences (in order):
Paragraph 1 ends with: {s1}
Paragraph 2 ends with: {s2}
Paragraph 3 ends with: {s3}
Paragraph 4 ends with: {s4}

Output only the passage."""

COT_PROMPT = """\
Write a coherent 4-paragraph passage where each paragraph ends with \
one of the following sentences (in order):
Paragraph 1 ends with: {s1}
Paragraph 2 ends with: {s2}
Paragraph 3 ends with: {s3}
Paragraph 4 ends with: {s4}

First write a brief plan (theme, tone, paragraph structure). \
Then write the passage. Separate them with "---"."""

REFINE_PROMPT = """\
Below is a passage where each paragraph ends with one of the following \
sentences (in order):
1. {s1}
2. {s2}
3. {s3}
4. {s4}

Passage:
{passage}

Is this passage already perfectly coherent? If yes, output it unchanged. \
If not, output a refined version with greater coherency. \
Output only the passage."""


def _require_api_key() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY in environment.")


def sample(prompt: str, n: int, model_client: LLMClient = GEN_LLM, temperature: float = TEMPERATURE) -> list[str]:
    # Three sampling paths handle conflicting needs:
    #   (1) n=1 deterministic -> safe to cache (cheap rerun reproducibility)
    #   (2) OpenAI / Together with n>1 -> single batched request (fewer round trips)
    #   (3) Other providers (e.g. Groq) don't support n>1 -> fan out to n calls
    # Caching is force-disabled whenever stochasticity matters; otherwise repeated
    # "independent" samples would collapse to the same cached completion.
    use_cache = (n <= 1 and temperature == 0)

    if n <= 1:
        _rate_limit()
        t0 = time.perf_counter()
        resp = model_client.generate(
            prompt=prompt,
            temperature=temperature,
            n=1,
            max_tokens=1200,
            use_cache=use_cache,
        )
        _record_api_time(
            time.perf_counter() - t0,
            resp.prompt_tokens,
            resp.completion_tokens,
            resp.cost_usd,
        )
        return [(c or "").strip() for c in resp.completions]

    if getattr(model_client, "provider", None) in ("openai", "together"):
        _rate_limit()
        t0 = time.perf_counter()
        resp = model_client.generate(
            prompt=prompt,
            temperature=temperature,
            n=n,
            max_tokens=1200,
            use_cache=False,
        )
        _record_api_time(
            time.perf_counter() - t0,
            resp.prompt_tokens,
            resp.completion_tokens,
            resp.cost_usd,
        )
        return [(c or "").strip() for c in resp.completions]

    completions: list[str] = []
    for _ in range(n):
        _rate_limit()
        t0 = time.perf_counter()
        resp = model_client.generate(
            prompt=prompt,
            temperature=temperature,
            n=1,
            max_tokens=1200,
            use_cache=False,
        )
        _record_api_time(
            time.perf_counter() - t0,
            resp.prompt_tokens,
            resp.completion_tokens,
            resp.cost_usd,
        )
        completions.extend((c or "").strip() for c in resp.completions)
    return completions


def _score_passages_details(passages: list[str]) -> list[ScoreResult]:
    # Judges are I/O-bound and independent — parallelizing them is the single
    # biggest wall-clock win for IO/CoT baselines (N_BASELINE passages each).
    if len(passages) <= 1 or SCORE_WORKERS <= 1:
        return [score_passage_details(p) for p in passages]
    workers = min(SCORE_WORKERS, len(passages))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(score_passage_details, passages))


def parse_score(text: str) -> Optional[float]:
    # We rely on the prompt's "coherency score is N" sentinel rather than asking
    # for JSON, because small models are far more reliable at matching a free-form
    # template than at producing parseable JSON. Returns None on parse failure
    # so callers can count and report the failure rate.
    m = re.match(r".*coherency score is (\d+).*", text, re.IGNORECASE | re.DOTALL)
    return float(m.group(1)) if m else None


def extract_score_thoughts(text: str) -> str:
    # Strip the trailing "coherency score is N..." so only the judge's rationale
    # remains — used for --print-score-thoughts debugging output.
    cleaned = re.sub(r"coherency score is \d+.*$", "", text, flags=re.IGNORECASE | re.DOTALL)
    return cleaned.strip()


def parse_vote(text: str) -> Optional[int]:
    m = re.match(r".*best choice is .*(\d+).*", text, re.IGNORECASE | re.DOTALL)
    return int(m.group(1)) if m else None


def vote_tally(votes: list[Optional[int]], k: int) -> list[int]:
    counts = {i: 0 for i in range(1, k + 1)}
    for v in votes:
        if v is not None and v in counts:
            counts[v] += 1
    return [counts[i] for i in range(1, k + 1)]


def tally_votes(votes: list[Optional[int]], k: int) -> int:
    # Fallback to choice #1 if every vote failed to parse. Picking a winner
    # arbitrarily is preferable to crashing mid-experiment; the failure rate is
    # logged separately so degenerate runs are still visible in the metrics.
    tallies = vote_tally(votes, k)
    if all(c == 0 for c in tallies):
        return 1
    return tallies.index(max(tallies)) + 1


def vote_best(candidates: list[str], kind: str) -> tuple[str, VoteResult]:
    numbered = "\n\n".join(f"Choice {i + 1}:\n{c}" for i, c in enumerate(candidates))
    prompt_template = VOTE_PROMPTS.get(VOTE_PROMPT_VARIANT, VOTE_PROMPTS["paper"])
    prompt = prompt_template.format(k=len(candidates), kind=kind, candidates=numbered)
    raw_votes = sample(prompt, N_VOTES, model_client=GEN_LLM, temperature=0.2)
    parsed_votes = [parse_vote(v) for v in raw_votes]
    winner = tally_votes(parsed_votes, len(candidates))
    tally = vote_tally(parsed_votes, len(candidates))
    vote_result = VoteResult(
        entropy=vote_entropy(tally),
        parse_failures=sum(1 for v in parsed_votes if v is None),
        tally=tally,
        winner_index_1based=winner,
        attempts=len(parsed_votes),
    )
    return candidates[winner - 1], vote_result


def score_passage_details(passage: str) -> ScoreResult:
    # Low temperature for judges: we want stable, calibrated ratings, not creative
    # ones. N_SCORE judges are averaged to dampen judge variance (the dominant
    # source of noise in our metrics — see dashboard score_std analysis).
    prompt_template = SCORE_PROMPTS.get(SCORE_PROMPT_VARIANT, SCORE_PROMPTS["paper"])
    prompt = prompt_template.format(passage=passage)
    scores = []
    parse_failures = 0
    for i, raw in enumerate(sample(prompt, N_SCORE, model_client=JUDGE_LLM, temperature=0.2), start=1):
        s = parse_score(raw)
        if PRINT_SCORE_THOUGHTS:
            thoughts = extract_score_thoughts(raw)
            with _PRINT_LOCK:
                print(f"\n[score-judge {i}/{N_SCORE}] thoughts:")
                print(thoughts if thoughts else "(no explicit rationale text)")
                if s is None:
                    print("[score-judge result] score: parse_failed")
                else:
                    print(f"[score-judge result] score: {int(s)}")
        if s is None:
            parse_failures += 1
            continue
        # Clamp defends against judges that output 0 or 11+ despite the prompt;
        # rare with gpt-4o-mini but the metrics shouldn't be skewed by outliers.
        scores.append(min(max(s, 1.0), 10.0))
    avg = round(sum(scores) / len(scores), 4) if scores else 0.0
    std = round(pstdev(scores), 4) if len(scores) > 1 else 0.0
    return ScoreResult(
        mean=avg,
        std=std,
        parse_failures=parse_failures,
        parsed_scores=scores,
        attempts=N_SCORE,
    )


def score_passage(passage: str) -> float:
    return score_passage_details(passage).mean


def score_plan(plan: str) -> float:
    # Only 3 judges (vs N_SCORE for final passages): plan-level scoring is an
    # intermediate signal for `evaluate_states`, so the extra precision isn't
    # worth the cost. Used only by the generic `core.search` path, not paper-ToT.
    prompt = PLAN_SCORE_PROMPT.format(plan=plan)
    scores = []
    for raw in sample(prompt, 3, model_client=JUDGE_LLM, temperature=0.2):
        s = parse_score(raw)
        if s is not None:
            scores.append(min(max(s, 1.0), 10.0))
    return round(sum(scores) / len(scores), 4) if scores else 0.0


class CreativeWritingTask(Task):
    """`core.search`-compatible view of the task.

    Two-level tree: plan stage -> passage stage -> terminal. `tot_solve` below
    bypasses this class to follow the paper's exact procedure; this class exists
    so the same task can be driven by any policy in `core.search`.
    """

    max_depth = 2  # plan + passage stages; no further expansion past terminal

    def __init__(self, example: dict):
        self.example = example
        self.s1, self.s2, self.s3, self.s4 = example["sentences"]

    def get_input(self, idx: int) -> str:
        return " | ".join(self.example["sentences"])

    def propose_thoughts(self, state: State, n_propose: int) -> list[State]:
        stage = state.meta.get("stage", "plan")
        if stage == "plan":
            prompt = PLAN_PROMPT.format(s1=self.s1, s2=self.s2, s3=self.s3, s4=self.s4)
            plans = sample(prompt, n_propose)
            return [
                State(
                    text=p,
                    meta={"stage": "passage", "plan": p},
                )
                for p in plans
            ]

        if stage == "passage":
            plan = state.meta["plan"]
            prompt = PASSAGE_PROMPT.format(plan=plan, s1=self.s1, s2=self.s2, s3=self.s3, s4=self.s4)
            passages = sample(prompt, n_propose)
            return [
                State(
                    text=psg,
                    meta={"stage": "done", "plan": plan, "passage": psg},
                )
                for psg in passages
            ]

        return []

    def evaluate_states(self, states: list[State], n_eval: int) -> list[float]:
        values = []
        for s in states:
            stage = s.meta.get("stage")
            if stage == "passage":
                values.append(score_plan(s.text))
            elif stage == "done":
                values.append(score_passage(s.text))
            else:
                values.append(0.0)
        return values

    def is_terminal(self, state: State) -> bool:
        return state.meta.get("stage") == "done"

    def score_output(self, state: State, ground_truth=None) -> float:
        if not self.is_terminal(state):
            return 0.0
        return min(max(score_passage(state.text) / 10.0, 0.0), 1.0)


def tot_solve(x: dict) -> SolveResult:
    """Paper-faithful ToT: plan-vote -> passage-vote -> judge.

    Bypasses `core.search` on purpose: the paper's exact procedure (b=1 after
    each vote) is the experiment baseline, so we implement it directly to avoid
    any framework drift affecting the comparison.
    """
    s1, s2, s3, s4 = x["sentences"]

    plans = sample(PLAN_PROMPT.format(s1=s1, s2=s2, s3=s3, s4=s4), K)
    best_plan, plan_vote = vote_best(plans, "plan")

    passages = sample(
        PASSAGE_PROMPT.format(plan=best_plan, s1=s1, s2=s2, s3=s3, s4=s4),
        K,
    )
    best_passage, passage_vote = vote_best(passages, "passage")

    # Final score uses N_SCORE judges (vs the 3-judge `score_plan` mid-search):
    # this is the number reported in the paper-style summary table.
    score_details = score_passage_details(best_passage)
    return SolveResult(
        passage=best_passage,
        score=score_details.mean,
        score_std=score_details.std,
        score_parse_failures=score_details.parse_failures,
        score_attempts=score_details.attempts,
        vote_plan=plan_vote,
        vote_passage=passage_vote,
    )


def _io_raw(x: dict) -> tuple[list[str], list[ScoreResult]]:
    # Shared by io_solve and io_refine_solve so they consume the same passage pool.
    s1, s2, s3, s4 = x["sentences"]
    passages = sample(IO_PROMPT.format(s1=s1, s2=s2, s3=s3, s4=s4), N_BASELINE)
    score_details = _score_passages_details(passages)
    return passages, score_details


def io_solve(x: dict) -> SolveResult:
    # Report mean over all N_BASELINE samples (not the best) — matches the paper's
    # "average coherency" metric. We still keep the best passage in `passage` for
    # downstream qualitative inspection / human eval.
    passages, score_details = _io_raw(x)
    means = [s.mean for s in score_details]
    avg = sum(means) / len(means)
    score_std = round(pstdev(means), 4) if len(means) > 1 else 0.0
    best_idx = means.index(max(means))
    best = passages[best_idx]
    return SolveResult(
        passage=best,
        score=avg,
        score_std=score_std,
        score_parse_failures=sum(s.parse_failures for s in score_details),
        score_attempts=sum(s.attempts for s in score_details),
    )


def _extract_passage(cot_output: str) -> str:
    # CoT prompt asks for "plan --- passage". Some completions skip the marker;
    # treat the whole output as the passage in that case rather than dropping it.
    parts = cot_output.split("---")
    return parts[-1].strip() if len(parts) > 1 else cot_output.strip()


def cot_solve(x: dict) -> SolveResult:
    s1, s2, s3, s4 = x["sentences"]
    outputs = sample(COT_PROMPT.format(s1=s1, s2=s2, s3=s3, s4=s4), N_BASELINE)
    passages = [_extract_passage(o) for o in outputs]
    score_details = _score_passages_details(passages)
    means = [s.mean for s in score_details]
    avg = sum(means) / len(means)
    score_std = round(pstdev(means), 4) if len(means) > 1 else 0.0
    best_idx = means.index(max(means))
    best = passages[best_idx]
    return SolveResult(
        passage=best,
        score=avg,
        score_std=score_std,
        score_parse_failures=sum(s.parse_failures for s in score_details),
        score_attempts=sum(s.attempts for s in score_details),
    )


def io_refine_solve(x: dict) -> SolveResult:
    # Self-refine starting from a random IO sample — the paper's protocol picks
    # a random baseline (not the best) to isolate refinement's contribution.
    # Early-exit on the first non-improvement; otherwise the refiner can degrade
    # passages it has already polished.
    s1, s2, s3, s4 = x["sentences"]
    passages, _ = _io_raw(x)
    passage = random.choice(passages)
    score_detail = score_passage_details(passage)
    score = score_detail.mean
    for _ in range(MAX_REFINE):
        refined = sample(
            REFINE_PROMPT.format(s1=s1, s2=s2, s3=s3, s4=s4, passage=passage),
            1,
        )[0]
        refined_score_detail = score_passage_details(refined)
        new_score = refined_score_detail.mean
        if new_score > score:
            passage, score = refined, new_score
            score_detail = refined_score_detail
        else:
            break
    return SolveResult(
        passage=passage,
        score=score,
        score_std=score_detail.std,
        score_parse_failures=score_detail.parse_failures,
        score_attempts=score_detail.attempts,
    )


def tot_refine_solve(x: dict) -> SolveResult:
    s1, s2, s3, s4 = x["sentences"]
    base = tot_solve(x)
    passage, score = base.passage, base.score
    score_std = base.score_std
    score_parse_failures = base.score_parse_failures
    score_attempts = base.score_attempts
    for _ in range(MAX_REFINE):
        refined = sample(
            REFINE_PROMPT.format(s1=s1, s2=s2, s3=s3, s4=s4, passage=passage),
            1,
        )[0]
        refined_score_detail = score_passage_details(refined)
        new_score = refined_score_detail.mean
        if new_score > score:
            passage, score = refined, new_score
            score_std = refined_score_detail.std
            score_parse_failures = refined_score_detail.parse_failures
            score_attempts = refined_score_detail.attempts
        else:
            break
    return SolveResult(
        passage=passage,
        score=score,
        score_std=score_std,
        score_parse_failures=score_parse_failures,
        score_attempts=score_attempts,
        vote_plan=base.vote_plan,
        vote_passage=base.vote_passage,
    )


def tot_astar_solve(x: dict) -> SolveResult:
    """A* vote-selection ToT; implemented in tot_creative_writing_astar.py."""
    # Lazy import to keep this module self-contained for the basic methods —
    # astar pulls in the sibling file's prompts/constants.
    from tot_creative_writing_astar import astar_tot_solve

    return astar_tot_solve(x)


def load_data(n: int = 100) -> list[dict]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Data file not found at {DATA_PATH}.\nRun: python code/generate_data.py"
        )
    with open(DATA_PATH) as f:
        return json.load(f)[:n]


SOLVERS = {
    "io": io_solve,
    "cot": cot_solve,
    "tot": tot_solve,
    "io_refine": io_refine_solve,
    "tot_refine": tot_refine_solve,
    "tot_astar": tot_astar_solve,
}


METRICS_LOG_PATH = RESULTS_PATH / "metrics_log.jsonl"
SUMMARY_TABLE_PATH = RESULTS_PATH / "summary_table.json"


def run_method(method: str, data: list[dict], out_file: Path, out_label: str, axes: AxesConfig) -> dict:
    """Drive a solver over `data`, resumable: skips tasks already in `out_file`.

    Writes results JSON after every task so a kill-and-resume produces the same
    output as a single run. Each task also appends a row to METRICS_LOG_PATH for
    later dashboard analysis.
    """
    results: dict = {}
    if out_file.exists():
        with open(out_file) as f:
            results = json.load(f)
    if method not in results:
        results[method] = []

    # Resume by skipping already-completed tasks. Cheap and avoids re-paying
    # for API calls that have already been logged.
    start = len(results[method])
    solver = SOLVERS[method]
    n = len(data)
    method_metrics: list[TaskMetrics] = []

    print(f"\n-- {method.upper()} ({start}/{n} already done) --")
    for i, x in enumerate(data[start:], start=start):
        print(f"  [{i + 1:3d}/{n}] ", end="", flush=True)
        perf_before = _perf_snapshot()
        solve = solver(x)
        perf_after = _perf_snapshot()
        perf_delta = _perf_delta(perf_before, perf_after)

        results[method].append(
            {
                "id": x["id"],
                "sentences": x["sentences"],
                "passage": solve.passage,
                "score": solve.score,
            }
        )
        task_metrics = TaskMetrics(
            out_file=out_label,
            method=method,
            task_id=x["id"],
            mean_score=solve.score,
            score_std=solve.score_std,
            score_parse_failures=solve.score_parse_failures,
            score_attempts=solve.score_attempts,
            vote_entropy_plan=solve.vote_plan.entropy if solve.vote_plan else None,
            vote_entropy_passage=solve.vote_passage.entropy if solve.vote_passage else None,
            vote_parse_failures_plan=solve.vote_plan.parse_failures if solve.vote_plan else 0,
            vote_parse_failures_passage=solve.vote_passage.parse_failures if solve.vote_passage else 0,
            vote_attempts_plan=solve.vote_plan.attempts if solve.vote_plan else 0,
            vote_attempts_passage=solve.vote_passage.attempts if solve.vote_passage else 0,
            api_calls=perf_delta["api_calls"],
            prompt_tokens=perf_delta["prompt_tokens"],
            completion_tokens=perf_delta["completion_tokens"],
            total_cost_usd=perf_delta["total_cost_usd"],
        )
        method_metrics.append(task_metrics)
        append_metrics_jsonl(METRICS_LOG_PATH, axes, task_metrics)

        print(f"score={solve.score:.2f}")
        # Flush after every task: resumability + crash safety beat I/O cost
        # here (writes are tiny compared to the LLM round trip we just paid for).
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)

    scores = [e["score"] for e in results[method]]
    avg = sum(scores) / len(scores)
    print(f"  -> {method} average coherency: {avg:.2f} (n={len(scores)})")
    # Prefer cumulative rows so a resumed run reports stats over the full sample,
    # not just the tasks completed in this invocation.
    cumulative_rows = _load_cumulative_method_metrics(method, axes, out_label)
    method_summary = build_method_summary(method, cumulative_rows or method_metrics, axes)
    print_method_summary_table(method_summary)
    return method_summary


def print_summary(out_file: Path):
    # `target` reproduces the paper's Table 5 numbers so a quick visual diff
    # tells you whether the reimplementation is on the rails.
    if not out_file.exists():
        return
    with open(out_file) as f:
        results = json.load(f)
    target = {"io": 6.19, "cot": 6.93, "tot": 7.56, "io_refine": 7.67, "tot_refine": 7.91}
    print("\n-- Final Summary -----------------------------------------")
    print(f"  {'Method':<12}  {'Our avg':>8}  {'Paper':>8}")
    print(f"  {'-'*12}  {'-'*8}  {'-'*8}")
    for method, entries in results.items():
        if entries:
            scores = [e["score"] for e in entries]
            avg = sum(scores) / len(scores)
            paper = target.get(method, "-")
            paper_s = f"{paper:.2f}" if isinstance(paper, float) else paper
            print(f"  {method:<12}  {avg:>8.2f}  {paper_s:>8}")
    print()


def main():
    global K, N_VOTES, SCORE_PROMPT_VARIANT, VOTE_PROMPT_VARIANT, PRINT_SCORE_THOUGHTS
    _require_api_key()
    parser = argparse.ArgumentParser(description="ToT creative writing experiment")
    parser.add_argument(
        "--method",
        nargs="+",
        default=["all"],
        metavar="NAME",
        help=(
            "Which solver(s) to run. Use 'all' alone for every solver, or name one or more "
            "(e.g. tot tot_astar). Examples: --method tot tot_astar | --method all"
        ),
    )
    parser.add_argument("--n", type=int, default=100, help="Number of inputs")
    parser.add_argument("--out", default="creative_writing_results.json", help="Results filename")
    parser.add_argument("--k", type=int, default=K, help="Number of candidates for ToT voting steps")
    parser.add_argument("--n-votes", type=int, default=N_VOTES, help="Number of votes per vote prompt")
    parser.add_argument(
        "--score-prompt-variant",
        choices=["paper", "definition", "criteria"],
        default=SCORE_PROMPT_VARIANT,
        help="Scoring prompt template variant label",
    )
    parser.add_argument(
        "--vote-prompt-variant",
        choices=["paper", "criteria"],
        default=VOTE_PROMPT_VARIANT,
        help="Voting prompt template variant label",
    )
    parser.add_argument(
        "--print-score-thoughts",
        action="store_true",
        help="Print scorer rationale text for each judge call.",
    )
    args = parser.parse_args()

    K = args.k
    N_VOTES = args.n_votes
    SCORE_PROMPT_VARIANT = args.score_prompt_variant
    VOTE_PROMPT_VARIANT = args.vote_prompt_variant
    PRINT_SCORE_THOUGHTS = args.print_score_thoughts

    RESULTS_PATH.mkdir(exist_ok=True)
    out_file = RESULTS_PATH / args.out
    data = load_data(args.n)
    axes = AxesConfig(
        score_model_type=SCORE_MODEL,
        k=K,
        n_votes=N_VOTES,
        score_prompt_variant=SCORE_PROMPT_VARIANT,
    )

    requested = list(args.method)
    if requested == ["all"]:
        methods = list(SOLVERS.keys())
    elif "all" in requested:
        parser.error("Use '--method all' alone, or list specific methods (e.g. tot tot_astar), not both.")
    else:
        unknown = [m for m in requested if m not in SOLVERS]
        if unknown:
            parser.error(f"Unknown method(s): {unknown}. Choices: {', '.join(sorted(SOLVERS))}, all")
        methods = list(dict.fromkeys(requested))
    summary_rows = []
    for m in methods:
        summary_rows.append(run_method(m, data, out_file, args.out, axes))
    print_summary(out_file)
    with open(SUMMARY_TABLE_PATH, "w") as f:
        json.dump(summary_rows, f, indent=2)

    total_time = _PERF["api_time_s"] + _PERF["rate_sleep_s"]
    cumulative = _load_cumulative_totals(methods, axes, args.out)
    print("-- Runtime Profile ----------------------------------------")
    print(f"  API calls (run):        {_PERF['api_calls']}")
    print(f"  API time (s, run):      {_PERF['api_time_s']:.2f}")
    print(f"  Rate-limit (s, run):    {_PERF['rate_sleep_s']:.2f}")
    print(f"  Tracked total (s, run): {total_time:.2f}")
    print(f"  Prompt tokens (run):    {_PERF['prompt_tokens']}")
    print(f"  Completion tokens (run):{_PERF['completion_tokens']}")
    print(f"  Total cost (usd, run):  {_PERF['total_cost_usd']:.6f}")
    print("  --")
    print(f"  API calls (cumulative):        {cumulative['api_calls']}")
    print(f"  Prompt tokens (cumulative):    {cumulative['prompt_tokens']}")
    print(f"  Completion tokens (cumulative):{cumulative['completion_tokens']}")
    print(f"  Total cost (usd, cumulative):  {cumulative['total_cost_usd']:.6f}")


if __name__ == "__main__":
    main()
