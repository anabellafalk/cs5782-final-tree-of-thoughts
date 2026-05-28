"""
Abstract Task interface.

Every task (Game24, Crosswords, CreativeWriting) subclasses Task and implements
five methods. The search controllers (BFS/DFS) call only through this interface,
which means new tasks plug in without touching search code.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class State:
    """
    A node in the ToT search tree.

    `text` is the partial solution so far (e.g. for Game of 24, the sequence
    of operations performed). `value` is filled in by the evaluator and used
    for pruning. `meta` is a free-form dict for task-specific data.
    """
    text: str
    value: float = 0.0
    depth: int = 0
    parent: "State | None" = field(default=None, repr=False)
    meta: dict[str, Any] = field(default_factory=dict)

    def trace(self) -> list["State"]:
        """Return the full path from root to this state."""
        path, cur = [], self
        while cur is not None:
            path.append(cur)
            cur = cur.parent
        return list(reversed(path))


class Task(ABC):
    """
    Contract every ToT task must implement.

    The four required methods correspond to the four components from the paper:
      1. Thought decomposition (handled implicitly by depth + propose)
      2. Thought generator → propose_thoughts
      3. State evaluator → evaluate_states
      4. Search algorithm → handled by BFS/DFS in search.py
    """

    # Maximum depth of the search tree for this task
    max_depth: int = 4

    @abstractmethod
    def get_input(self, idx: int) -> str:
        """Return the input string for example `idx` in the dataset."""

    @abstractmethod
    def propose_thoughts(self, state: State, n_propose: int) -> list[State]:
        """
        Generate candidate next-step states from the current state.

        For Game of 24, this generates possible next arithmetic operations.
        For Crosswords, this generates word fills for the next clue.
        For Creative Writing, this generates plan candidates or paragraph drafts.
        """

    @abstractmethod
    def evaluate_states(self, states: list[State], n_eval: int) -> list[float]:
        """
        Assign a numerical value to each state for pruning/ranking.

        Higher = more promising. Returns one float per input state.
        """

    @abstractmethod
    def is_terminal(self, state: State) -> bool:
        """Return True if `state` is a complete solution (no more thoughts to add)."""

    @abstractmethod
    def score_output(self, state: State, ground_truth: Any = None) -> float:
        """
        Score a terminal state against ground truth. Used only for evaluation.
        Returns a number in [0, 1] — typically 0/1 for correctness.
        """