"""
Search controllers used by all three tasks.

BFS (used by Game of 24 and Creative Writing): at each depth, generate
candidates from all current states, evaluate, keep top `breadth`.

DFS (used by Crosswords): explore deepest-first with backtracking, pruning
states whose value falls below a threshold.

These match Algorithms 1 and 2 from the ToT paper (Yao et al. 2023).
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

from .task import State, Task

logger = logging.getLogger(__name__)


def bfs_search(
    task: Task,
    initial_state: State,
    breadth: int = 5,
    n_propose: int = 1,
    n_eval: int = 3,
    max_depth: Optional[int] = None,
    log_callback: Optional[Callable[[int, list[State]], None]] = None,
) -> list[State]:
    """
    Breadth-first ToT search.

    At each depth: propose new candidates from non-terminal states, evaluate them,
    combine with carried-forward terminal states, and keep the top `breadth`.
    """
    max_depth = max_depth if max_depth is not None else task.max_depth
    current = [initial_state]

    for depth in range(max_depth):
        # Separate terminal states (carried forward) from those needing expansion
        candidates: list[State] = []
        terminal_states: list[State] = []
        for state in current:
            if task.is_terminal(state):
                terminal_states.append(state)
                continue
            new_states = task.propose_thoughts(state, n_propose=n_propose)
            for s in new_states:
                s.parent = state
                s.depth = depth + 1
            candidates.extend(new_states)

        # If no fresh candidates AND no terminal states, search is dead — halt
        if not candidates and not terminal_states:
            logger.warning(f"No candidates at depth {depth}; halting.")
            break

        # Evaluate fresh candidates only (terminal states keep their prior values)
        if candidates:
            values = task.evaluate_states(candidates, n_eval=n_eval)
            for state, v in zip(candidates, values):
                state.value = v

        # Combine and select top-b
        all_candidates = candidates + terminal_states
        all_candidates.sort(key=lambda s: s.value, reverse=True)
        current = all_candidates[:breadth]

        if log_callback is not None:
            log_callback(depth + 1, current)

        # Early stop: every kept state is terminal, no further work possible
        if all(task.is_terminal(s) for s in current):
            break

    current.sort(key=lambda s: s.value, reverse=True)
    return current


def dfs_search(
    task: Task,
    initial_state: State,
    n_propose: int = 5,
    n_eval: int = 3,
    prune_threshold: float = 3.0,
    max_depth: Optional[int] = None,
    log_callback: Optional[Callable[[State, str], None]] = None,
) -> Optional[State]:
    """
    Depth-first ToT search with pruning and backtracking.

    Used by the Crosswords task. Tries the highest-valued thought first, and
    if no completion below it succeeds, backtracks and tries the next-best.

    Args:
        prune_threshold: States with value < threshold are not explored.

    Returns:
        Best terminal state found, or None if search exhausts without solution.
    """
    max_depth = max_depth if max_depth is not None else task.max_depth
    best: Optional[State] = None

    def recurse(state: State) -> None:
        nonlocal best
        if log_callback is not None:
            log_callback(state, "visit")

        if task.is_terminal(state) or state.depth >= max_depth:
            score = task.score_output(state)
            if best is None or score > task.score_output(best):
                best = state
            return

        candidates = task.propose_thoughts(state, n_propose=n_propose)
        for c in candidates:
            c.parent = state
            c.depth = state.depth + 1
        if not candidates:
            return

        values = task.evaluate_states(candidates, n_eval=n_eval)
        scored = sorted(zip(candidates, values), key=lambda x: x[1], reverse=True)

        for cand, v in scored:
            if v < prune_threshold:
                if log_callback is not None:
                    log_callback(cand, "prune")
                continue
            cand.value = v
            recurse(cand)

    recurse(initial_state)
    return best