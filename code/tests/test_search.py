"""
Integration test: full BFS search on Game of 24 using a scripted mock LLM.
Verifies the search controller correctly orchestrates propose -> evaluate -> prune.
"""
import sys
from pathlib import Path
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tot.core.task import State
from tot.core.search import bfs_search
from tot.tasks.gameof24 import Game24Task


@dataclass
class FakeResponse:
    completions: list


class MockLLM:
    """Returns scripted responses based on what the prompt looks like."""
    def __init__(self):
        self.calls = []

    def generate(self, prompt, temperature=0.7, n=1, max_tokens=1000, stop=None):
        self.calls.append(prompt[:60])

        # Proposal request — return a few candidate next steps
        if "Possible next steps:" in prompt:
            # Use the puzzle "4 9 10 13" — propose realistic next moves
            if "4 9 10 13" in prompt:
                return FakeResponse(completions=[
                    "13 - 9 = 4 (left: 4 4 10)\n"
                    "10 - 4 = 6 (left: 6 9 13)\n"
                    "9 + 4 = 13 (left: 10 13 13)\n"
                ])
            if "4 4 10" in prompt:
                return FakeResponse(completions=[
                    "4 + 4 = 8 (left: 8 10)\n"
                    "10 - 4 = 6 (left: 4 6)\n"
                    "4 * 4 = 16 (left: 10 16)\n"
                ])
            if "4 6" in prompt and "left:" in prompt or "4 6" == prompt.strip().split()[-2:][-1]:
                return FakeResponse(completions=[
                    "4 * 6 = 24 (left: 24)\n"
                    "6 - 4 = 2 (left: 2)\n"
                ])
            return FakeResponse(completions=["1 + 1 = 2 (left: 2)\n"])

        # Final-answer generation
        if "complete the answer expression" in prompt:
            return FakeResponse(completions=["(13 - 9) * (10 - 4)"])

        # Value evaluation (return n samples, all "sure" for the right path)
        if "Evaluate if given numbers" in prompt:
            return FakeResponse(completions=["sure"] * n)
        if "give a judgement" in prompt:
            return FakeResponse(completions=["sure"] * n)

        return FakeResponse(completions=["impossible"] * n)


def test_bfs_runs_without_error():
    """Just verify the search loop executes and returns states."""
    llm = MockLLM()
    task = Game24Task(llm=llm, dataset=["4 9 10 13"])
    initial = State(text="", meta={"current": "4 9 10 13", "original_input": "4 9 10 13"})
    final = bfs_search(task, initial, breadth=2, n_propose=1, n_eval=1, max_depth=4)
    assert len(final) > 0, "BFS should return at least one state"
    assert llm.calls, "LLM should have been called"
    print(f"BFS made {len(llm.calls)} LLM calls and returned {len(final)} states")


def test_bfs_state_chain():
    """Verify states form a parent-child chain back to root."""
    llm = MockLLM()
    task = Game24Task(llm=llm, dataset=["4 9 10 13"])
    initial = State(text="", meta={"current": "4 9 10 13", "original_input": "4 9 10 13"})
    final = bfs_search(task, initial, breadth=2, n_propose=1, n_eval=1, max_depth=4)
    best = final[0]
    trace = best.trace()
    assert trace[0] is initial, "First state in trace should be the initial state"
    assert trace[-1] is best, "Last state in trace should be the final state"
    print(f"Trace length: {len(trace)} (depth {best.depth})")


if __name__ == "__main__":
    test_bfs_runs_without_error()
    test_bfs_state_chain()
    print("\nAll integration tests passed.")