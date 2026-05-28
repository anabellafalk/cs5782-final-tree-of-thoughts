"""Command-line runner for Mini Crossword ToT DFS experiments."""
from __future__ import annotations

import argparse
import heapq
import itertools
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tot.core import LLMClient as ModelClient
from tot.core.task import State
from tot.tasks.crosswords import MiniCrosswordTask


class _MockResponse:
    def __init__(self, completions: list[str]):
        self.completions = completions
        self.model = "mock"
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cost_usd = 0.0
        self.cached = False
        self.raw = {}


class _MockModel:
    def generate(self, prompt: str, temperature: float = 0.7, n: int = 1, max_tokens: int = 1000, stop=None):
        if "Given the current status" in prompt or "Proposals:" in prompt:
            return _MockResponse(
                [
                    "\n".join(
                        [
                            "h1. sator (certain)",
                            "h2. arepo (certain)",
                            "h3. tenet (certain)",
                            "h4. opera (certain)",
                            "h5. rotas (certain)",
                        ]
                    )
                ]
            )
        if "Proposed answer:" in prompt:
            return _MockResponse(["sure"] * n)
        return _MockResponse(["sure"] * n)


def load_word_list(path: str | None) -> set[str] | None:
    if not path:
        return None
    words: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            word = line.strip().upper()
            if len(word) == 5 and word.isalpha():
                words.add(word)
    return words


def dfs_search_limited(
    task: MiniCrosswordTask,
    initial_state: State,
    n_propose: int,
    max_per_state: int,
    n_eval: int,
    prune_threshold: float,
    max_steps: int,
    prune: bool = True,
    stop_on_solution: bool = True,
    progress_every: int = 0,
    dfs_order: str = "value",
    method_generate: str = "propose",
    prompt_sample: str = "standard"
) -> tuple[State, dict[str, Any]]:
    """Paper-style DFS: sorted thoughts, pruning, backtracking, deepest output."""

    visited = 0
    pruned = 0
    best = initial_state
    first_terminal: State | None = None

    def visit(state: State) -> None:
        nonlocal visited, pruned, best, first_terminal
        if visited >= max_steps:
            return
        visited += 1
        if progress_every and visited % progress_every == 0:
            print(f"  visited={visited} depth={state.depth} best_depth={best.depth}")
        if state.depth > best.depth:
            best = state
        if task.is_terminal(state):
            if first_terminal is None:
                first_terminal = state
            best = state
            return

        if method_generate == "propose":
            candidates = task.propose_thoughts(state, n_propose=n_propose, n_select=max_per_state)
        elif method_generate == "sample":
            candidates = task.get_samples(state, n_sample=1, prompt_method=prompt_sample)
            # force termination
            first_terminal = candidates[0]
        for candidate in candidates:
            candidate.parent = state
            candidate.depth = state.depth + 1
        if not candidates:
            return

        values = task.evaluate_states(candidates, n_eval=n_eval)
        scored = list(zip(candidates, values))
        if dfs_order == "value":
            scored = sorted(scored, key=lambda item: item[1], reverse=True)
        for candidate, value in scored:
            if visited >= max_steps:
                return
            if stop_on_solution and first_terminal is not None:
                return
            candidate.value = value
            # Princeton's notebook records a candidate state before deciding
            # whether to recurse into its subtree. This matters for mini
            # crosswords because the final output is the deepest explored
            # state, even if a later value call prunes that branch.
            if candidate.depth > best.depth:
                best = candidate
            should_prune = value <= prune_threshold
            if "count" in candidate.meta:
                should_prune = candidate.meta["count"].get("impossible", 0) >= 1
            if prune and should_prune:
                pruned += 1
                continue
            visit(candidate)

    visit(initial_state)
    return first_terminal or best, {
        "visited": visited,
        "pruned": pruned,
        "terminal": first_terminal is not None,
        "max_per_state": max_per_state,
        "prune": prune,
        "dfs_order": dfs_order,
    }


def astar_search_limited(
    task: MiniCrosswordTask,
    initial_state: State,
    n_propose: int,
    max_per_state: int,
    n_eval: int,
    prune_threshold: float,
    max_steps: int,
    prune: bool = True,
    stop_on_solution: bool = True,
    progress_every: int = 0,
    depth_weight: float = 0.25,
    frontier_limit: int = 1000,
) -> tuple[State, dict[str, Any]]:
    """A*-style best-first ToT search using state value as the heuristic.

    The ToT paper leaves A* as future work. For crosswords, we maximize a
    priority score instead of minimizing path cost:
        priority = value(state) + depth_weight * filled_steps
    This keeps multiple partial boards alive instead of committing to one DFS
    branch after an early high-confidence word.
    """

    visited = 0
    pruned = 0
    pushed = 0
    best = initial_state
    best_score = task.score_output(initial_state)
    first_terminal: State | None = None
    counter = itertools.count()
    frontier: list[tuple[float, int, State]] = [(0.0, next(counter), initial_state)]
    seen: set[tuple[tuple[str, ...], tuple[int, ...]]] = set()

    while frontier and visited < max_steps:
        _, _, state = heapq.heappop(frontier)
        key = (tuple(state.meta.get("board", [])), tuple(state.meta.get("status", [])))
        if key in seen:
            continue
        seen.add(key)

        visited += 1
        state_score = task.score_output(state)
        if state.depth > best.depth or state_score > best_score:
            best = state
            best_score = state_score
        if progress_every and visited % progress_every == 0:
            print(f"  visited={visited} depth={state.depth} best_depth={best.depth} frontier={len(frontier)}")
        if task.is_terminal(state):
            first_terminal = state
            best = state
            if stop_on_solution:
                break
            continue

        candidates = task.propose_thoughts(state, n_propose=n_propose, n_select=max_per_state)
        for candidate in candidates:
            candidate.parent = state
            candidate.depth = state.depth + 1
        if not candidates:
            continue

        values = task.evaluate_states(candidates, n_eval=n_eval)
        for candidate, value in zip(candidates, values):
            candidate.value = value
            candidate_key = (tuple(candidate.meta.get("board", [])), tuple(candidate.meta.get("status", [])))
            if candidate_key in seen:
                continue
            should_prune = value <= prune_threshold
            if "count" in candidate.meta:
                should_prune = candidate.meta["count"].get("impossible", 0) >= 1
            if prune and should_prune:
                pruned += 1
                continue
            priority = value + depth_weight * candidate.depth
            heapq.heappush(frontier, (-priority, next(counter), candidate))
            pushed += 1

        if frontier_limit > 0 and len(frontier) > frontier_limit:
            frontier = heapq.nsmallest(frontier_limit, frontier)
            heapq.heapify(frontier)

    return first_terminal or best, {
        "visited": visited,
        "pruned": pruned,
        "terminal": first_terminal is not None,
        "max_per_state": max_per_state,
        "prune": prune,
        "search": "astar",
        "frontier_remaining": len(frontier),
        "pushed": pushed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Mini Crossword Tree-of-Thought search.")
    parser.add_argument("--data", default="data/crosswords_toy.json", help="Path to crossword JSON data.")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model name.")
    parser.add_argument("--mock-model", action="store_true", help="Use a deterministic fake model for smoke tests.")
    parser.add_argument("--oracle-proposals", action="store_true", help="Use ground-truth answers as proposals. Test mode only.")
    parser.add_argument(
        "--deterministic-eval",
        action="store_true",
        help="Use ground-truth consistency instead of value prompts. Intended for smoke tests only.",
    )
    parser.add_argument(
        "--proposal-only-eval",
        action="store_true",
        help="Rank states by proposal confidence without extra value prompts. Faster ablation.",
    )
    parser.add_argument(
        "--preset",
        choices=("paper", "budget", "debug"),
        default=None,
        help="Set common run sizes: paper=(8 proposal samples,3 candidates,100 steps), budget=(1,1,20), debug=(1,1,5).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of puzzles to run.")
    parser.add_argument("--start", type=int, default=0, help="Dataset index to start from.")
    parser.add_argument("--n-propose", type=int, default=5, help="Candidate words per clue.")
    parser.add_argument(
        "--max-per-state",
        type=int,
        default=None,
        help="Maximum ranked candidate actions to explore per DFS state. Defaults to --n-propose.",
    )
    parser.add_argument("--n-eval", type=int, default=3, help="Evaluation samples per state.")
    parser.add_argument(
        "--proposal-prompt",
        choices=("princeton", "fewshot", "strict"),
        default="princeton",
        help="Proposal prompt variant: original Princeton-style zero-shot, few-shot, or strict no-gibberish variant.",
    )
    parser.add_argument(
        "--value-prompt",
        choices=("princeton", "strict"),
        default="princeton",
        help="State value prompt variant.",
    )
    parser.add_argument(
        "--verify-proposals",
        action="store_true",
        help="Compatibility alias for --verify-mode filter.",
    )
    parser.add_argument(
        "--verify-mode",
        choices=("off", "boost", "filter", "penalize"),
        default="off",
        help="Use an extra clue-fit check for proposals: boost good answers, filter impossible answers, or penalize impossible answers.",
    )
    parser.add_argument(
        "--value-max-tokens",
        type=int,
        default=300,
        help="Maximum tokens for value-prompt responses. Higher values avoid truncated verdicts.",
    )
    parser.add_argument(
        "--strict-prompts",
        action="store_true",
        help="Convenience mode: strict proposal prompt, strict value prompt, and verifier boosting.",
    )
    parser.add_argument(
        "--word-list",
        default=None,
        help="Optional newline-delimited five-letter word list used to validate proposed crossword entries.",
    )
    parser.add_argument(
        "--dictionary-mode",
        choices=("off", "filter", "penalize"),
        default="off",
        help="Use --word-list to filter or penalize proposed words not present in the dictionary.",
    )
    parser.add_argument(
        "--no-dictionary-crossings",
        action="store_true",
        help="Only validate the proposed word itself, not any completed crossing words.",
    )

    parser.add_argument('--method_generate', type=str, choices=['sample', 'propose'], default='propose')
    parser.add_argument('--prompt_sample', type=str, choices=['standard', 'cot'], default='standard')

    parser.add_argument("--threshold", type=float, default=1.0, help="Value threshold for pruning search states.")
    parser.add_argument("--no-prune", action="store_true", help="Disable value-based DFS pruning, matching the paper ablation.")
    parser.add_argument(
        "--dfs-order",
        choices=("value", "proposal"),
        default="value",
        help="DFS candidate order. value uses this repo's value-sorted order; proposal matches Princeton's proposal-score order.",
    )
    parser.add_argument(
        "--search",
        choices=("dfs", "astar"),
        default="dfs",
        help="Search controller. dfs matches the paper crossword setup; astar is an experimental best-first variant.",
    )
    parser.add_argument("--astar-depth-weight", type=float, default=0.25, help="Depth bonus used by --search astar.")
    parser.add_argument("--frontier-limit", type=int, default=1000, help="Maximum queued states for --search astar; 0 disables trimming.")
    parser.add_argument("--max-steps", type=int, default=100, help="Maximum search states to visit.")
    parser.add_argument("--progress-every", type=int, default=0, help="Print DFS progress every N visited states.")
    parser.add_argument("--debug-proposals", action="store_true", help="Print top parsed proposals at each state.")
    parser.add_argument("--continue-after-solution", action="store_true", help="Keep DFS running after first terminal state.")
    parser.add_argument("--out", default="results/crosswords_results.jsonl", help="Output JSONL path.")
    args = parser.parse_args()
    if args.strict_prompts:
        args.proposal_prompt = "strict"
        args.value_prompt = "strict"
        if args.verify_mode == "off" and not args.verify_proposals:
            args.verify_mode = "boost"
    if args.preset == "paper":
        args.n_propose = 8
        args.max_per_state = 3
        args.n_eval = 1
        args.max_steps = 100
    elif args.preset == "budget":
        args.n_propose = 1
        args.max_per_state = 1
        args.n_eval = 1
        args.max_steps = 20
    elif args.preset == "debug":
        args.n_propose = 1
        args.max_per_state = 1
        args.n_eval = 1
        args.max_steps = 5
    if args.max_per_state is None:
        args.max_per_state = args.n_propose

    dataset = MiniCrosswordTask.load_dataset(args.data)
    word_list = load_word_list(args.word_list)
    if args.dictionary_mode != "off" and word_list is None:
        raise ValueError("--dictionary-mode requires --word-list")
    if args.start:
        dataset = dataset[args.start :]
    if args.limit is not None:
        dataset = dataset[: args.limit]

    model = _MockModel() if (args.mock_model or args.oracle_proposals) else ModelClient(model=args.model)
    task = MiniCrosswordTask(
        model=model,
        dataset=dataset,
        deterministic_eval=args.deterministic_eval,
        oracle_proposals=args.oracle_proposals,
        proposal_only_eval=args.proposal_only_eval,
        debug_proposals=args.debug_proposals,
        proposal_prompt=args.proposal_prompt,
        value_prompt=args.value_prompt,
        verify_mode=args.verify_mode,
        verify_proposals=args.verify_proposals,
        value_max_tokens=args.value_max_tokens,
        word_list=word_list,
        dictionary_mode=args.dictionary_mode,
        dictionary_validate_crossings=not args.no_dictionary_crossings,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    letter_scores = []
    word_scores = []
    game_scores = []

    with open(out_path, "w", encoding="utf-8") as f:
        for idx, example in enumerate(dataset):
            original_idx = args.start + idx
            initial = task.initial_state(idx)
            print(f"Starting {original_idx}: {example.puzzle_id}")
            try:
                if args.search == "astar":
                    best, search_info = astar_search_limited(
                        task,
                        initial,
                        n_propose=args.n_propose,
                        max_per_state=args.max_per_state,
                        n_eval=args.n_eval,
                        prune_threshold=args.threshold,
                        max_steps=args.max_steps,
                        prune=not args.no_prune,
                        stop_on_solution=not args.continue_after_solution,
                        progress_every=args.progress_every,
                        depth_weight=args.astar_depth_weight,
                        frontier_limit=args.frontier_limit,
                    )
                else:
                    best, search_info = dfs_search_limited(
                        task,
                        initial,
                        n_propose=args.n_propose,
                        max_per_state=args.max_per_state,
                        n_eval=args.n_eval,
                        prune_threshold=args.threshold,
                        max_steps=args.max_steps,
                        prune=not args.no_prune,
                        stop_on_solution=not args.continue_after_solution,
                        progress_every=args.progress_every,
                        dfs_order=args.dfs_order,
                        prompt_sample=args.prompt_sample,
                        method_generate=args.method_generate
                    )
                    search_info["search"] = "dfs"
            except Exception as exc:
                record = {
                    "idx": original_idx,
                    "puzzle_id": example.puzzle_id,
                    "error": type(exc).__name__,
                    "message": str(exc),
                }
                f.write(json.dumps(record) + "\n")
                f.flush()
                print(f"{original_idx}: {example.puzzle_id} ERROR {type(exc).__name__}: {exc}")
                break
            metrics = task.score_metrics(best)
            letter_scores.append(metrics["r_letter"])
            word_scores.append(metrics["r_word"])
            game_scores.append(metrics["r_game"])
            record = {
                "idx": original_idx,
                "puzzle_id": example.puzzle_id,
                **metrics,
                **search_info,
                "board": task.format_board(best.meta["board"]) if best is not None else "",
                "filled": best.meta.get("filled", {}) if best is not None else {},
                "trace": best.text if best is not None else "",
            }
            f.write(json.dumps(record) + "\n")
            f.flush()
            print(
                f"{original_idx}: {example.puzzle_id} "
                f"letters={metrics['r_letter']:.3f} words={metrics['r_word']:.3f} "
                f"game={metrics['r_game']:.0f} visited={search_info['visited']}"
            )

    denom = max(len(word_scores), 1)
    print(f"Average letters: {sum(letter_scores) / denom:.3f}")
    print(f"Average words: {sum(word_scores) / denom:.3f}")
    print(f"Solved games: {sum(game_scores):.0f}/{len(game_scores)}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
