"""
Sanity tests for Game of 24 task. Run with: pytest code/tests/test_game24.py

These tests validate parsing and verification without any LLM calls — so they
run instantly and catch the most common bugs (number-counting, sympy parsing).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tot.tasks.gameof24 import Game24Task


class FakeLLM:
    """Stub so we can instantiate Game24Task without a real API key."""
    def generate(self, *args, **kwargs):
        raise NotImplementedError("Should not be called in these tests")


def make_task():
    return Game24Task(llm=FakeLLM(), dataset=["4 9 10 13"])


def test_verify_correct_answer():
    t = make_task()
    assert t._verify_answer("(13 - 9) * (10 - 4) = 24", "4 9 10 13")
    assert t._verify_answer("4 * (9 - (13 - 10))", "4 9 10 13")  # No "= 24" trail


def test_verify_wrong_value():
    t = make_task()
    # Sums to 25, not 24
    assert not t._verify_answer("4 + 9 + 10 + 13 = 36", "4 9 10 13")


def test_verify_wrong_numbers():
    t = make_task()
    # Uses a 5 that wasn't in the input
    assert not t._verify_answer("5 * 4 + 4 = 24", "4 9 10 13")


def test_verify_reuses_number():
    t = make_task()
    # Uses 4 twice instead of using 9 and 10
    assert not t._verify_answer("4 * 4 + 13 - 5 = 24", "4 9 10 13")


def test_verify_division():
    t = make_task()
    # Genuine answer involving division
    assert t._verify_answer("(1 + 8 / 4) * 8 = 24", "1 4 8 8")


def test_parse_proposal_line():
    t = make_task()
    out = t._parse_proposal_line("8 / 2 = 4 (left: 4 8 14)", "2 8 8 14")
    assert out == "4 8 14"


def test_parse_proposal_invalid_numbers():
    t = make_task()
    # The proposed op uses a 7 not in the current set — should reject
    out = t._parse_proposal_line("7 + 2 = 9 (left: 8 9 14)", "2 8 8 14")
    assert out is None


def test_parse_proposal_strips_numbered_marker():
    """Llama 8B often outputs '3. 14 - 2 = 12 (left: 8 8 12)' — strip the prefix."""
    t = make_task()
    out = t._parse_proposal_line("3. 14 - 2 = 12 (left: 8 8 12)", "2 8 8 14")
    assert out == "8 8 12"


def test_parse_proposal_rejects_hallucinated_numbers():
    """Critical: if the puzzle is '1 1 4 6', '14 - 2 = 12 (left: 8 8 12)' must
    be rejected because there's no 14, no 8, etc. in the current set."""
    t = make_task()
    out = t._parse_proposal_line("14 - 2 = 12 (left: 8 8 12)", "1 1 4 6")
    assert out is None


def test_parse_proposal_rejects_inconsistent_left():
    """If the math doesn't add up (left set != current minus operands plus result),
    reject. Example: '4 - 1 = 3 (left: 1 2 3)' from puzzle '1 1 4 6' is wrong
    because '2' wasn't in the original set."""
    t = make_task()
    out = t._parse_proposal_line("4 - 1 = 3 (left: 1 2 3)", "1 1 4 6")
    assert out is None  # Should be "left: 1 1 3", not "1 2 3"


def test_parse_proposal_consistent_left():
    """Same operation but with the correct left set should pass.
    From '1 1 4 6', taking the 4 and one of the 1s leaves '1 6', plus the result 3."""
    t = make_task()
    out = t._parse_proposal_line("4 - 1 = 3 (left: 1 6 3)", "1 1 4 6")
    sorted_out = " ".join(sorted(out.split(), key=int))
    assert sorted_out == "1 3 6"


def test_propose_propagates_original_input():
    """Critical: every new state must carry original_input forward, otherwise
    the final-answer-generation step can't find it and returns nothing.
    This was a real bug that caused all ToT runs to report success=0 even
    when the search found correct paths."""
    # Use a fake LLM that returns one valid proposal
    class CaptureLLM:
        def generate(self, *args, **kwargs):
            class R: completions = ["1 * 4 = 4 (left: 1 6 4)"]
            return R()

    from tot.core import State
    from tot.tasks.gameof24 import Game24Task
    t = Game24Task(llm=CaptureLLM(), dataset=["1 1 4 6"])
    initial = State(
        text="",
        meta={"current": "1 1 4 6", "original_input": "1 1 4 6"},
    )
    new_states = t.propose_thoughts(initial, n_propose=1)
    assert len(new_states) >= 1
    for s in new_states:
        assert s.meta.get("original_input") == "1 1 4 6", (
            f"original_input not propagated; meta={s.meta}"
        )


def test_extract_judgment_sure():
    t = make_task()
    text = "10 + 14 = 24\nsure"
    assert t._extract_judgment(text) == "sure"


def test_extract_judgment_likely():
    t = make_task()
    text = "5 + 7 + 8 = 20\nI cannot obtain 24 now\nlikely"
    assert t._extract_judgment(text) == "likely"


def test_extract_judgment_impossible_fallback():
    t = make_task()
    # Even garbled output should default to impossible (safe default)
    text = "completely off topic"
    assert t._extract_judgment(text) == "impossible"

def test_verify_strips_latex_wrappers():
    """LaTeX-wrapped expressions should still verify correctly."""
    t = make_task()
    # gpt-4o-mini sometimes wraps in \( ... \)
    assert t._verify_answer("\\((13 - 9) * (10 - 4)\\) = 24", "4 9 10 13")
    # Or with $ ... $
    assert t._verify_answer("$(13 - 9) * (10 - 4)$ = 24", "4 9 10 13")
    # Plain expressions still work
    assert t._verify_answer("(13 - 9) * (10 - 4) = 24", "4 9 10 13")


def test_verify_does_not_strip_real_parens():
    """Critical: '(...)' parens that aren't LaTeX must be preserved."""
    t = make_task()
    # No backslash → not LaTeX → don't strip
    assert t._verify_answer("(13 - 9) * (10 - 4) = 24", "4 9 10 13")

if __name__ == "__main__":
    # Allow running directly
    import traceback
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {fn.__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
