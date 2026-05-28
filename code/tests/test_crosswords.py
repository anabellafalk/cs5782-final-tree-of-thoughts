"""Offline tests for the Mini Crossword task."""
from __future__ import annotations

from dataclasses import dataclass

from tot.core.search import dfs_search
from tot.core.task import State
from tot.tasks.crosswords import EMPTY, MiniCrosswordTask


@dataclass
class FakeResponse:
    completions: list[str]
    model: str = "fake"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    cached: bool = False
    raw: dict | None = None


class FakeModel:
    def generate(self, prompt: str, temperature: float = 0.7, n: int = 1, max_tokens: int = 1000, stop=None):
        if "Given the current status" in prompt or "Proposals:" in prompt:
            return FakeResponse(
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
            return FakeResponse(["sure"] * n)
        return FakeResponse(["sure"] * n)


def test_parse_and_place_candidate():
    dataset = MiniCrosswordTask.load_dataset("data/crosswords/crosswords_toy.json")
    task = MiniCrosswordTask(model=FakeModel(), dataset=dataset, deterministic_eval=True)
    state = task.initial_state(0)

    proposals = task.propose_thoughts(state, n_propose=2)

    assert proposals
    assert proposals[0].meta["filled"]["h1"] == "SATOR"
    assert task.format_board(proposals[0].meta["board"]).splitlines()[0] == "SATOR"


def test_score_completed_ground_truth():
    dataset = MiniCrosswordTask.load_dataset("data/crosswords/crosswords_toy.json")
    task = MiniCrosswordTask(model=FakeModel(), dataset=dataset)
    state = task.initial_state(0)
    board = [list(row) for row in ["SATOR", "AREPO", "TENET", "OPERA", "ROTAS"]]
    filled = {slot.slot_id: slot.answer for slot in dataset[0].slots}
    done = State(text="", meta={"idx": 0, "board": board, "filled": filled, "remaining": []})

    assert task.score_output(done) == 1.0


def test_judgment_extraction_prefers_last_line():
    dataset = MiniCrosswordTask.load_dataset("data/crosswords/crosswords_toy.json")
    task = MiniCrosswordTask(model=FakeModel(), dataset=dataset)

    assert task._extract_judgment("This might look likely.\nimpossible") == "impossible"


def test_strict_verifier_can_boost_proposals():
    dataset = MiniCrosswordTask.load_dataset("data/crosswords/crosswords_toy.json")
    task = MiniCrosswordTask(
        model=FakeModel(),
        dataset=dataset,
        proposal_prompt="princeton",
        value_prompt="strict",
        verify_mode="boost",
    )
    state = task.initial_state(0)

    proposals = task.propose_thoughts(state, n_propose=1)

    assert proposals
    assert proposals[0].meta["proposal_score"] > 1.0


def test_dictionary_filter_rejects_non_word_proposal():
    dataset = MiniCrosswordTask.load_dataset("data/crosswords/crosswords_toy.json")
    task = MiniCrosswordTask(
        model=FakeModel(),
        dataset=dataset,
        word_list={"AREPO", "TENET", "OPERA", "ROTAS"},
        dictionary_mode="filter",
    )
    state = task.initial_state(0)

    proposals = task.propose_thoughts(state, n_propose=1)

    assert proposals
    assert all(proposal.meta["proposal"] != "h1. sator" for proposal in proposals)


def test_dictionary_crossing_validation_rejects_completed_non_word():
    dataset = MiniCrosswordTask.load_dataset("data/crosswords/crosswords_toy.json")
    task = MiniCrosswordTask(
        model=FakeModel(),
        dataset=dataset,
        word_list={"SATOR"},
        dictionary_mode="filter",
    )
    state = task.initial_state(0)
    board = [EMPTY] * 25
    for idx in (5, 10, 15, 20):
        board[idx] = "X"
    state.meta["board"] = board

    assert not task._dictionary_accepts_action(state, "h1. sator")


def test_dfs_solves_literal_smoke_test():
    dataset = MiniCrosswordTask.load_dataset("data/crosswords/crosswords_toy.json")
    task = MiniCrosswordTask(model=FakeModel(), dataset=dataset)

    best = dfs_search(
        task,
        task.initial_state(0),
        n_propose=2,
        n_eval=1,
        prune_threshold=1.0,
        max_depth=len(dataset[0].slots),
    )

    assert best is not None
    assert task.score_output(best) == 1.0


def test_official_split_loads_and_oracle_solves_first_puzzle():
    dataset = MiniCrosswordTask.load_dataset("data/crosswords/crosswords_official_0_100_5_raw.json")
    task = MiniCrosswordTask(
        model=FakeModel(),
        dataset=dataset[:1],
        deterministic_eval=True,
        oracle_proposals=True,
    )

    best = dfs_search(
        task,
        task.initial_state(0),
        n_propose=1,
        n_eval=1,
        prune_threshold=1.0,
        max_depth=len(dataset[0].slots),
    )

    assert len(dataset) == 20
    assert best is not None
    assert task.score_metrics(best) == {"r_letter": 1.0, "r_word": 1.0, "r_game": 1.0}
