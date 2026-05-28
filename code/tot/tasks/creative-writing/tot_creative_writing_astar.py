#!/usr/bin/env python3
"""
A* style plan/passage selection for creative writing (comparison vs `tot`).

`tot_creative_writing.py` registers `astar_tot_solve` as method `tot_astar` (lazy import).

Run **only** `tot_astar` via this file:

  python3 code/tot/tasks/creative-writing/tot_creative_writing_astar.py --n 20 --out creative-writing/json_outputs/astar.json

To run **io, cot, tot, … and tot_astar** together, use the main driver (default `--method all` includes `tot_astar`):

  python3 code/tot/tasks/creative-writing/tot_creative_writing.py --n 20 --out creative-writing/json_outputs/all.json --k 5 --n-votes 5 --score-prompt-variant paper

Plan phase (per candidate plan i, same K plans shown in both vote rounds):
  g(i) = 1 - (votes_i / N_VOTES)   from standard "most promising" vote prompt
  h(i) = 1 - (exec_votes_i / N_VOTES) from executability vote prompt
  f(i) = g(i) + h(i); pick argmin f(i)

Passage phase (h=0):
  g(j) = 1 - (votes_j / N_VOTES); pick argmin g(j) (max vote share)

Final passage is judged with the same N_SCORE pipeline as baseline tot.
Metrics use method name \"tot_astar\" for dashboard comparison vs \"tot\".
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Importing the sibling `tot_creative_writing` and `experiment_metrics` only
# works when this directory is on sys.path. We insert at position 0 so direct
# script execution (`python tot/tasks/creative-writing/tot_creative_writing_astar.py`)
# works even if the package isn't installed.
CW_DIR = Path(__file__).resolve().parent
if str(CW_DIR) not in sys.path:
    sys.path.insert(0, str(CW_DIR))

import tot_creative_writing as cw  # noqa: E402
from experiment_metrics import AxesConfig, VoteResult  # noqa: E402

PLAN_EXECUTABILITY_VOTE_PROMPT = """\
Below are {k} candidate plans for a creative writing task. Each plan should \
lead to a coherent 4-paragraph passage whose paragraphs end with these sentences (in order):

1. {s1}
2. {s2}
3. {s3}
4. {s4}

{candidates}

Analyze which plan is most likely to produce a coherent passage that satisfies \
these four ending constraints. After your analysis, end with exactly: "the best choice is {{i}}" \
where i is the number (1-{k}) of the best option."""

PASSAGE_COHERENCE_VOTE_PROMPT = """\
Below are {k} candidate passages for a creative writing task. \
Analyze each for overall coherence, then conclude which passage is most coherent.

{candidates}

After your analysis, end with exactly: "the best choice is {{i}}" \
where i is the number (1-{k}) of the best option."""


def _numbered_candidates(candidates: list[str]) -> str:
    return "\n\n".join(f"Choice {i + 1}:\n{c}" for i, c in enumerate(candidates))


def astar_tot_solve(x: dict) -> cw.SolveResult:
    s1, s2, s3, s4 = x["sentences"]
    k = cw.K
    plans = cw.sample(cw.PLAN_PROMPT.format(s1=s1, s2=s2, s3=s3, s4=s4), k)
    numbered_plans = _numbered_candidates(plans)

    # g(i): the standard ToT "most promising" vote (same prompt as baseline tot,
    # so g is directly comparable across methods).
    vote_template = cw.VOTE_PROMPTS.get(cw.VOTE_PROMPT_VARIANT, cw.VOTE_PROMPTS["paper"])
    prompt_std = vote_template.format(k=k, kind="plan", candidates=numbered_plans)
    raw_std = cw.sample(prompt_std, cw.N_VOTES, model_client=cw.GEN_LLM, temperature=0.2)
    parsed_std = [cw.parse_vote(v) for v in raw_std]
    tally_std = cw.vote_tally(parsed_std, k)

    # h(i): executability vote — a heuristic estimate of whether this plan can
    # actually be realized into a passage that satisfies the four ending
    # constraints. Cheap to compute (one extra vote round) and gives A* the
    # forward-looking signal it needs.
    prompt_exec = PLAN_EXECUTABILITY_VOTE_PROMPT.format(
        k=k, s1=s1, s2=s2, s3=s3, s4=s4, candidates=numbered_plans
    )
    raw_exec = cw.sample(prompt_exec, cw.N_VOTES, model_client=cw.GEN_LLM, temperature=0.2)
    parsed_exec = [cw.parse_vote(v) for v in raw_exec]
    tally_exec = cw.vote_tally(parsed_exec, k)

    # `or 1.0` guards against a misconfigured N_VOTES=0 producing a ZeroDivisionError.
    n = float(cw.N_VOTES) if cw.N_VOTES else 1.0

    def plan_f(idx: int) -> float:
        # Costs in [0, 2]: 0 = unanimous winner on both votes, 2 = no votes either.
        g_i = 1.0 - (tally_std[idx] / n)
        h_i = 1.0 - (tally_exec[idx] / n)
        return g_i + h_i

    # Tiebreak on idx makes the choice deterministic across reruns even when
    # multiple plans share the minimum f-cost (common with N_VOTES small).
    best_plan_idx = min(range(k), key=lambda i: (plan_f(i), i))
    best_plan = plans[best_plan_idx]

    # Logged tally is the standard "promising" vote so dashboards can compare
    # tot vs tot_astar on the same vote distribution; winner_index, however,
    # reflects the A* pick (which may diverge from plurality).
    vote_plan = VoteResult(
        entropy=cw.vote_entropy(tally_std),
        parse_failures=sum(1 for v in parsed_std if v is None),
        tally=tally_std,
        winner_index_1based=best_plan_idx + 1,
        attempts=len(parsed_std),
    )

    # Passage phase: terminal layer, so h=0 and A* degenerates to "pick by g".
    # We keep the same vote machinery (rather than scoring each passage with the
    # judge) because votes are 5x cheaper here and the judge pass still runs at
    # the end on the winner.
    passages = cw.sample(
        cw.PASSAGE_PROMPT.format(plan=best_plan, s1=s1, s2=s2, s3=s3, s4=s4),
        k,
    )
    numbered_passages = _numbered_candidates(passages)
    prompt_pass = PASSAGE_COHERENCE_VOTE_PROMPT.format(k=k, candidates=numbered_passages)
    raw_pass = cw.sample(prompt_pass, cw.N_VOTES, model_client=cw.GEN_LLM, temperature=0.2)
    parsed_pass = [cw.parse_vote(v) for v in raw_pass]
    tally_pass = cw.vote_tally(parsed_pass, k)

    def passage_g(idx: int) -> float:
        return 1.0 - (tally_pass[idx] / n)

    best_pass_idx = min(range(k), key=lambda j: (passage_g(j), j))
    best_passage = passages[best_pass_idx]

    vote_passage = VoteResult(
        entropy=cw.vote_entropy(tally_pass),
        parse_failures=sum(1 for v in parsed_pass if v is None),
        tally=tally_pass,
        winner_index_1based=best_pass_idx + 1,
        attempts=len(parsed_pass),
    )

    score_details = cw.score_passage_details(best_passage)
    return cw.SolveResult(
        passage=best_passage,
        score=score_details.mean,
        score_std=score_details.std,
        score_parse_failures=score_details.parse_failures,
        score_attempts=score_details.attempts,
        vote_plan=vote_plan,
        vote_passage=vote_passage,
    )


def main() -> None:
    cw._require_api_key()
    parser = argparse.ArgumentParser(
        description="A* vote-selection ToT (creative writing). Logs method tot_astar."
    )
    parser.add_argument("--n", type=int, default=100, help="Number of inputs")
    parser.add_argument(
        "--out",
        default="creative-writing/json_outputs/tot_astar_results.json",
        help="Results filename under results/",
    )
    parser.add_argument("--k", type=int, default=cw.K, help="Candidates per stage")
    parser.add_argument("--n-votes", type=int, default=cw.N_VOTES, help="Votes per vote prompt")
    parser.add_argument(
        "--score-prompt-variant",
        choices=["paper", "definition", "criteria"],
        default=cw.SCORE_PROMPT_VARIANT,
        help="Scoring prompt template variant label",
    )
    parser.add_argument(
        "--vote-prompt-variant",
        choices=["paper", "criteria"],
        default=cw.VOTE_PROMPT_VARIANT,
        help="Voting prompt template variant label",
    )
    parser.add_argument(
        "--print-score-thoughts",
        action="store_true",
        help="Forward to tot_creative_writing (scorer rationale).",
    )
    args = parser.parse_args()

    cw.K = args.k
    cw.N_VOTES = args.n_votes
    cw.SCORE_PROMPT_VARIANT = args.score_prompt_variant
    cw.VOTE_PROMPT_VARIANT = args.vote_prompt_variant
    cw.PRINT_SCORE_THOUGHTS = args.print_score_thoughts

    # Register so cw.run_method can look up "tot_astar" through SOLVERS.
    cw.SOLVERS["tot_astar"] = astar_tot_solve

    cw.RESULTS_PATH.mkdir(exist_ok=True)
    out_file = cw.RESULTS_PATH / args.out
    data = cw.load_data(args.n)
    axes = AxesConfig(
        score_model_type=cw.SCORE_MODEL,
        k=cw.K,
        n_votes=cw.N_VOTES,
        score_prompt_variant=cw.SCORE_PROMPT_VARIANT,
    )

    summary_row = cw.run_method("tot_astar", data, out_file, args.out, axes)
    cw.print_summary(out_file)
    with open(cw.SUMMARY_TABLE_PATH, "w") as f:
        json.dump([summary_row], f, indent=2)

    total_time = cw._PERF["api_time_s"] + cw._PERF["rate_sleep_s"]
    cumulative = cw._load_cumulative_totals(["tot_astar"], axes, args.out)
    print("-- Runtime Profile ----------------------------------------")
    print(f"  API calls (run):        {cw._PERF['api_calls']}")
    print(f"  API time (s, run):      {cw._PERF['api_time_s']:.2f}")
    print(f"  Rate-limit (s, run):    {cw._PERF['rate_sleep_s']:.2f}")
    print(f"  Tracked total (s, run): {total_time:.2f}")
    print(f"  Prompt tokens (run):    {cw._PERF['prompt_tokens']}")
    print(f"  Completion tokens (run):{cw._PERF['completion_tokens']}")
    print(f"  Total cost (usd, run):  {cw._PERF['total_cost_usd']:.6f}")
    print("  --")
    print(f"  API calls (cumulative):        {cumulative['api_calls']}")
    print(f"  Prompt tokens (cumulative):    {cumulative['prompt_tokens']}")
    print(f"  Completion tokens (cumulative):{cumulative['completion_tokens']}")
    print(f"  Total cost (usd, cumulative):  {cumulative['total_cost_usd']:.6f}")


if __name__ == "__main__":
    main()
