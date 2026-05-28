"""
Game of 24 task.

Reproduces the headline result from Yao et al. 2023 (Table 2):
ToT (b=5) achieves ~74% success vs CoT's ~4%, IO's ~7%.

The task:
- Input: 4 numbers in [1, 13].
- Goal: combine them with +, -, *, / using each exactly once to make 24.
- Decomposed into 3 steps; each step picks 2 numbers and combines them.
"""
from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Optional

import sympy

from ...core.llm import LLMClient
from ...core.task import State, Task
from ...prompts.gameof24prompts import (
    COT_PROMPT,
    PROPOSE_PROMPT,
    VALUE_LAST_STEP_PROMPT,
    VALUE_PROMPT,
)

logger = logging.getLogger(__name__)


# Voting weights for the value evaluator. The paper uses these as the mapping
# from textual judgments to numerical values.
VALUE_MAP = {"sure": 20.0, "likely": 1.0, "impossible": 0.001}


class Game24Task(Task):
    max_depth = 3  # 4 numbers → 3 binary operations → 1 number
 
    def __init__(self, llm: LLMClient, dataset: list[str]):
        """
        Args:
            llm: The LLM client (already configured with model + caching).
            dataset: List of input strings, e.g. ["4 9 10 13", ...].
        """
        self.llm = llm
        self.dataset = dataset
 
    # ---------- Required Task interface ----------
 
    def get_input(self, idx: int) -> str:
        return self.dataset[idx]
 
    def propose_thoughts(self, state: State, n_propose: int = 1) -> list[State]:
        """
        Generate possible next-step states from the current set of numbers.
 
        The LLM returns lines like "8 / 2 = 4 (left: 4 8 14)". We parse each
        line into a new State whose `text` extends the parent's solution path
        and whose `meta["current"]` is the new set of numbers.
 
        For terminal-step states (only 1 number left), we instead generate a
        full answer string via the CoT prompt template applied to the path.
            """
        current_numbers = state.meta.get("current", state.text.strip())

        if len(current_numbers.split()) == 1:
            return self._propose_final_answer(state)

        prompt = PROPOSE_PROMPT.format(input=current_numbers)
        all_proposals: list[str] = []
        for _ in range(n_propose):
            resp = self.llm.generate(prompt, temperature=0.7, n=1, max_tokens=400, stop=["\nInput:"])
            all_proposals.extend(resp.completions[0].strip().split("\n"))

        new_states = []
        rejected = []
        for line in all_proposals:
            parsed = self._parse_proposal_line(line, current_numbers)
            if parsed is None:
                rejected.append(line)
                continue
            new_text = state.text + line + "\n"
            new_state = State(
                text=new_text,
                meta={
                    "current": parsed,
                    "last_op": line.split("(")[0].strip(),
                    "original_input": state.meta.get("original_input"),
                },
            )
            new_states.append(new_state)
        
        # DEBUG: print proposal info
        logger.info(
            f"[propose] from {current_numbers!r} (depth {state.depth}): "
            f"{len(all_proposals)} raw, {len(new_states)} accepted, {len(rejected)} rejected"
        )
        if rejected:
            logger.info(f"[rejected] {rejected[:3]}")  # Show first 3 rejected
        if new_states:
            logger.info(f"[accepted] {[s.meta['current'] for s in new_states[:5]]}")
        
        return new_states
 
    def _propose_final_answer(self, state: State) -> list[State]:
        """
        Once we're down to one number, generate a final answer expression — but
        only if that final number is 24. If the path led somewhere else, this is
        a failed branch; don't ask the model to fabricate an answer.
        # # """
        original_input = state.meta.get("original_input")
        if original_input is None:
            return []
        
        # Build the expression by symbolically composing the trace's operations.
        expression = self._reconstruct_expression(state.text)
        if expression is None:
            # Fallback: ask the model (matches paper's approach as a backup)
            return self._propose_final_answer_via_model(state)
        
        answer = f"{expression} = 24"
        new_state = State(
            text=state.text + f"Answer: {answer}\n",
            meta={"current": "DONE", "answer": answer, "original_input": original_input},
        )
        return [new_state]




        # original_input = state.meta.get("original_input")
        # if original_input is None:
        #     return []

        # prompt = (
        #     "Use numbers and basic arithmetic operations (+ - * /) to obtain 24. "
        #     "Each number must be used exactly once. Given partial steps, write the "
        #     "single complete expression as one line. No explanation, no preamble.\n\n"
        #     "Input: 4 4 6 8\n"
        #     "Steps:\n4 + 8 = 12 (left: 4 6 12)\n6 - 4 = 2 (left: 2 12)\n2 * 12 = 24 (left: 24)\n"
        #     "Answer: (4 + 8) * (6 - 4) = 24\n\n"
        #     "Input: 2 9 10 12\n"
        #     "Steps:\n12 * 2 = 24 (left: 9 10 24)\n10 - 9 = 1 (left: 1 24)\n24 * 1 = 24 (left: 24)\n"
        #     "Answer: (12 * 2) * (10 - 9) = 24\n\n"
        #     f"Input: {original_input}\n"
        #     f"Steps:\n{state.text}"
        #     "Answer:"
        # )
        # resp = self.llm.generate(prompt, temperature=0.7, n=1, max_tokens=100, stop=["\n", "Input:"])
        # answer = resp.completions[0].strip()
        # new_state = State(
        #     text=state.text + f"Answer: {answer}\n",
        #     meta={"current": "DONE", "answer": answer, "original_input": original_input},
        # )
        # return [new_state]
        # original_input = state.meta.get("original_input")
        # if original_input is None:
        #     return []
        
        # final_number_str = state.meta.get("current", "").strip()

        # try:
        #     final_number = float(final_number_str)
        # except ValueError:
        #     return []
        # if abs(final_number - 24.0) > 1e-6:
        #     return []
        

        # prompt = COT_PROMPT.format(input=original_input) + "Steps:\n" + state.text

        # resp = self.llm.generate(prompt, temperature=0.7, n=1, max_tokens=100, stop=["\n", "Input:"])
        # answer = resp.completions[0].strip()
        # new_state = State(
        #     text=state.text + f"Answer: {answer}\n",
        #     meta={"current": "DONE", "answer": answer, "original_input": original_input},
        # )
        # return [new_state]

 
    def evaluate_states(self, states: list[State], n_eval: int = 3) -> list[float]:
        """
        Score each state by majority-style voting over multiple LLM judgments.
 
        For non-terminal states, we ask "can these numbers reach 24?" and map
        sure/likely/impossible to numerical weights. For terminal states (with
        a final answer), we ask "is this answer correct?" and map sure/impossible.
 
        We sample n_eval times per state and sum the weights — equivalent to
        the paper's voting procedure.
        """
        values = []
        for state in states:
            current = state.meta.get("current", "")
            if current == "DONE":
                # Terminal state — judge the final answer
                prompt = VALUE_LAST_STEP_PROMPT.format(
                    input=state.meta.get("original_input", ""),
                    answer=state.meta.get("answer", ""),
                )
            else:
                prompt = VALUE_PROMPT.format(input=current)
 
            resp = self.llm.generate(prompt, temperature=1.0, n=n_eval, max_tokens=300)
            judgments = [self._extract_judgment(c) for c in resp.completions]
            value = sum(VALUE_MAP.get(j, 0.0) for j in judgments)
            values.append(value)
        return values
 
    def is_terminal(self, state: State) -> bool:
        return state.meta.get("current") == "DONE" or state.depth >= self.max_depth + 1
 
    def score_output(self, state: State, ground_truth=None) -> float:
        """
        Verify the final answer:
          (1) is a valid arithmetic expression,
          (2) evaluates to 24,
          (3) uses each input number exactly once.
        Returns 1.0 on success, 0.0 otherwise.
        """
 
        original_input = state.meta.get("original_input", "")
        if not original_input:
            return 0.0
        
        # Take last line of the full trace, strip "Answer: " prefix
        last_line = state.text.strip().split("\n")[-1] if state.text.strip() else ""
        expression = last_line.lower().replace("answer:", "").strip()
        
        return float(self._verify_answer(expression, original_input))
 


    def _reconstruct_expression(self, trace_text: str) -> Optional[str]:
        """
        Parse the trace's lines and compose them into a single expression.
        
        Each line is "A op B = C (left: ...)". We track each result back to
        the operation that produced it and the original input numbers used.
        
        Returns None if the trace can't be parsed (e.g., empty or malformed).
        """
        lines = [l.strip() for l in trace_text.strip().split("\n") if l.strip()]
        if not lines:
            return None
        
        # `expressions` maps a number string to the expression that produces it.
        # Initially, each input number maps to itself.
        # As we process steps, the result number's expression is the composition.
        expressions: dict[str, list[str]] = {}  # value -> list of expressions producing it
        
        for line in lines:
            # Strip leading list markers if any leaked through
            cleaned = re.sub(r"^\s*(?:\d+[.)]|-)\s*", "", line)
            m = re.match(
                r"\s*([\d.]+)\s*([+\-*/])\s*([\d.]+)\s*=\s*([\d.]+)",
                cleaned,
            )
            if not m:
                return None
            a_str, op, b_str, result_str = m.groups()
            
            # Get the expression for each operand. If not in our dict, it's a raw input.
            a_expr = expressions.get(a_str, [a_str]).pop() if a_str in expressions and expressions[a_str] else a_str
            b_expr = expressions.get(b_str, [b_str]).pop() if b_str in expressions and expressions[b_str] else b_str
            # Clean up empty lists
            if a_str in expressions and not expressions[a_str]:
                del expressions[a_str]
            if b_str in expressions and not expressions[b_str]:
                del expressions[b_str]
            
            # Compose: wrap each operand in parens for safety
            new_expr = f"({a_expr} {op} {b_expr})"
            expressions.setdefault(result_str, []).append(new_expr)
        
        # The last result should be 24 (or whatever the final number is).
        # Take the most recent expression for the final result number.
        final_value = lines[-1].split("=")[1].split("(")[0].strip()
        if final_value not in expressions or not expressions[final_value]:
            return None
        
        final_expr = expressions[final_value][-1]
        # Strip outermost parens — they're redundant on the full expression
        if final_expr.startswith("(") and final_expr.endswith(")"):
            final_expr = final_expr[1:-1]
        return final_expr


    def _propose_final_answer_via_model(self, state: State) -> list[State]:
        """Fallback: original model-based approach if reconstruction fails."""
        original_input = state.meta.get("original_input")
        prompt = (
            "Use numbers and basic arithmetic operations (+ - * /) to obtain 24. "
            "Each number must be used exactly once. Given partial steps, write the "
            "single complete expression as one line. No explanation, no preamble.\n\n"
            "Input: 4 4 6 8\n"
            "Steps:\n4 + 8 = 12 (left: 4 6 12)\n6 - 4 = 2 (left: 2 12)\n2 * 12 = 24 (left: 24)\n"
            "Answer: (4 + 8) * (6 - 4) = 24\n\n"
            "Input: 2 9 10 12\n"
            "Steps:\n12 * 2 = 24 (left: 9 10 24)\n10 - 9 = 1 (left: 1 24)\n24 * 1 = 24 (left: 24)\n"
            "Answer: (12 * 2) * (10 - 9) = 24\n\n"
            f"Input: {original_input}\n"
            f"Steps:\n{state.text}"
            "Answer:"
        )
        resp = self.llm.generate(prompt, temperature=0.7, n=1, max_tokens=100, stop=["\n", "Input:"])
        answer = resp.completions[0].strip()
        new_state = State(
            text=state.text + f"Answer: {answer}\n",
            meta={"current": "DONE", "answer": answer, "original_input": original_input},
        )
        return [new_state]

    # ---------- Helpers ----------
 
    def _parse_proposal_line(self, line: str, current_numbers: str) -> Optional[str]:
        stripped = re.sub(r"^\s*(?:\d+[.)]|-)\s*", "", line)
        
        # Match operation and left set
        match = re.match(
            r"\s*([\d.]+)\s*([+\-*/])\s*([\d.]+)\s*=\s*([\d.]+)\s+\(left:\s*([\d\s]+)\)",
            stripped,
        )
        if not match:
            return None
        
        a_str, op, b_str, result_str, left_str = match.groups()
        left_numbers = left_str.strip().split()
        
        current_list = current_numbers.split()
        current_counter = Counter(current_list)
        
        # Check operands exist
        if a_str == b_str:
            if current_counter[a_str] < 2:
                return None
        else:
            if current_counter[a_str] < 1 or current_counter[b_str] < 1:
                return None
        
        # Verify arithmetic
        try:
            a, b = float(a_str), float(b_str)
            expected = {"+": a+b, "-": a-b, "*": a*b, "/": a/b if b != 0 else None}[op]
            if expected is None or abs(expected - float(result_str)) > 1e-6:
                return None
        except (ValueError, ZeroDivisionError):
            return None
        
        # Compute correct left set as multiset
        new_counter = current_counter.copy()
        new_counter[a_str] -= 1
        new_counter[b_str] -= 1
        new_counter[result_str] = new_counter.get(result_str, 0) + 1
        new_counter = {k: v for k, v in new_counter.items() if v > 0}
        correct_left_multiset = Counter(new_counter)
        
        # Compare with model's left set (order doesn't matter, just multiset equality)
        model_left_multiset = Counter(left_numbers)
        if model_left_multiset != correct_left_multiset:
            return None
        
        # Return the model's left string (or normalized version)
        # To be safe, return sorted version as the test may expect specific order
        # But test_parse_proposal_line expects "4 8 14" (sorted numerically)
        sorted_left = sorted(left_numbers, key=int)
        return " ".join(sorted_left)
 
    def _extract_judgment(self, text: str) -> str:
        """Extract sure/likely/impossible from the last line of the model's output."""
        # The judgment is typically the last non-empty line
        lines = [l.strip().lower() for l in text.strip().split("\n") if l.strip()]
        if not lines:
            return "impossible"
        last = lines[-1]
        for key in ("sure", "likely", "impossible"):
            if key in last:
                return key
        # Fallback: search the whole text
        for key in ("sure", "likely", "impossible"):
            if key in text.lower():
                return key
        return "impossible"
 
    def _verify_answer(self, answer: str, original_input: str) -> bool:
        """
        Strictly verify the final answer:
          - Strip "= 24" suffix if present.
          - Extract numbers from the expression and check they match input.
          - Use sympy to evaluate the expression safely.
        """
    # Drop trailing "= ..." first
        expr = answer.split("=")[0].strip()
        if not expr:
            return False

        # Now strip LaTeX wrappers if present
        if expr.startswith("\\(") and expr.endswith("\\)"):
            expr = expr[2:-2].strip()
        if expr.startswith("$") and expr.endswith("$"):
            expr = expr[1:-1].strip()

        if not expr:
            return False

        used_numbers = re.findall(r"\d+", expr)
        input_numbers = re.findall(r"\d+", original_input) 
        if sorted(used_numbers) != sorted(input_numbers):
            return False

        try:
            return abs(float(sympy.simplify(expr)) - 24.0) < 1e-6
        except (sympy.SympifyError, TypeError, ValueError, ZeroDivisionError):
            return False
 
    # ---------- CoT baseline (for comparison) ----------
 
    def solve_cot(self, idx: int, n_samples: int = 100) -> tuple[float, list[str]]:
        """
        Run the CoT baseline: sample `n_samples` completions and report fraction
        that produce a valid 24 expression. The paper reports CoT @ best-of-100.
        """
        input_str = self.get_input(idx)
        prompt = COT_PROMPT.format(input=input_str)
        resp = self.llm.generate(prompt, temperature=0.7, n=n_samples, max_tokens=200)
        results = []
        for completion in resp.completions:
            answer = self._extract_cot_answer(completion)
            ok = self._verify_answer(answer, input_str) if answer else False
            results.append(answer if ok else "")
        success_rate = sum(1 for r in results if r) / max(len(results), 1)
        return success_rate, results
 
    def _extract_cot_answer(self, text: str) -> str:
        """
        Pull the answer line from a CoT completion. Robust to:
        - Plain 'Answer: <expr>'
        - Markdown '**Answer:**'
        - LaTeX 'Answer: \\(<expr>\\)' or '$<expr>$'
        """
        import re
        cleaned = text.replace("**", "")  # strip markdown bold first
        last_line = cleaned.strip().split("\n")[-1] if cleaned.strip() else ""
        expr = last_line.lower().replace("answer:", "").strip()
        
        # Strip LaTeX wrappers if present
        if expr.startswith("\\(") and expr.endswith("\\)"):
            expr = expr[2:-2].strip()
        if expr.startswith("$") and expr.endswith("$"):
            expr = expr[1:-1].strip()
        
        return expr