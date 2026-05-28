"""Mini Crossword task modeled on Princeton's Tree-of-Thought implementation.

The original code represents each crossword as a mutable environment with a
flat 25-cell board, ten answers, and clue statuses:
0 = unfilled, 1 = filled, 2 = filled then changed. The DFS rejects states with
status 2, which prevents overwriting existing commitments. This module keeps
that architecture while adapting it to this repo's generic State/Task interface
and model client.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from tot.core.task import State, Task
from tot.prompts.crosswordprompts import (
    COT_PROMPT,
    FEWSHOT_PROPOSE_PROMPT,
    PROPOSE_PROMPT,
    STANDARD_PROMPT,
    STRICT_PROPOSE_PROMPT,
    STRICT_VALUE_PROMPT,
    VALUE_PROMPT,
    VERIFY_PROMPT,
)


CONFIDENCE_MAP = {"certain": 1.0, "high": 0.5, "medium": 0.2, "low": 0.1}
EMPTY = "_"


@dataclass(frozen=True)
class CrosswordSlot:
    """A single across or down clue slot."""

    slot_id: str
    direction: str
    row: int
    col: int
    length: int
    clue: str
    answer: Optional[str] = None

    def cells(self) -> list[tuple[int, int]]:
        dr, dc = (0, 1) if self.direction == "across" else (1, 0)
        return [(self.row + dr * i, self.col + dc * i) for i in range(self.length)]


@dataclass
class CrosswordExample:
    """One 5x5 mini crossword example."""

    puzzle_id: str
    slots: list[CrosswordSlot]
    clues: list[str]
    board_gt: list[str]
    size: int = 5

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CrosswordExample":
        if "rows" in data:
            rows = [str(row).upper() for row in data["rows"]]
            board_gt = list("".join(rows))
            columns = ["".join(row[col] for row in rows) for col in range(5)]
            across_clues = data.get("across_clues", rows)
            down_clues = data.get("down_clues", columns)
            slots = []
            for row, (clue, answer) in enumerate(zip(across_clues, rows)):
                slots.append(
                    CrosswordSlot(
                        slot_id=f"h{row + 1}",
                        direction="across",
                        row=row,
                        col=0,
                        length=5,
                        clue=str(clue),
                        answer=answer,
                    )
                )
            for col, (clue, answer) in enumerate(zip(down_clues, columns)):
                slots.append(
                    CrosswordSlot(
                        slot_id=f"v{col + 1}",
                        direction="down",
                        row=0,
                        col=col,
                        length=5,
                        clue=str(clue),
                        answer=answer,
                    )
                )
            clues = [slot.clue for slot in slots]
            return cls(
                puzzle_id=str(data.get("id", "puzzle")),
                slots=slots,
                clues=clues,
                board_gt=board_gt,
                size=int(data.get("size", 5)),
            )
        slots = [
            CrosswordSlot(
                slot_id=str(item["id"]),
                direction=item["direction"].lower(),
                row=int(item["row"]),
                col=int(item["col"]),
                length=int(item["length"]),
                clue=item["clue"],
                answer=item.get("answer", None).upper() if item.get("answer") else None,
            )
            for item in data["slots"]
        ]
        rows = [slot.answer or (EMPTY * slot.length) for slot in slots if slot.direction == "across"]
        board_gt = list("".join(rows)) if len(rows) == 5 else [EMPTY] * 25
        clues = [slot.clue for slot in sorted(slots, key=lambda slot: (slot.direction != "across", slot.slot_id))]
        return cls(
            puzzle_id=str(data.get("id", "puzzle")),
            slots=slots,
            clues=clues,
            board_gt=board_gt,
            size=int(data.get("size", 5)),
        )

    @classmethod
    def from_official(cls, data: list[Any], idx: int) -> "CrosswordExample":
        clues, flat_board = data
        board = [ch.upper() for ch in flat_board]
        slots: list[CrosswordSlot] = []
        for row in range(5):
            slots.append(
                CrosswordSlot(
                    slot_id=f"h{row + 1}",
                    direction="across",
                    row=row,
                    col=0,
                    length=5,
                    clue=clues[row],
                    answer="".join(board[row * 5 : row * 5 + 5]),
                )
            )
        for col in range(5):
            slots.append(
                CrosswordSlot(
                    slot_id=f"v{col + 1}",
                    direction="down",
                    row=0,
                    col=col,
                    length=5,
                    clue=clues[5 + col],
                    answer="".join(board[col::5]),
                )
            )
        return cls(puzzle_id=f"official_{idx}", slots=slots, clues=list(clues), board_gt=board, size=5)


class MiniCrosswordsEnv:
    """Princeton-style mutable environment for one mini crossword."""

    def __init__(self, dataset: list[CrosswordExample]):
        self.dataset = dataset
        self.cache: dict[str, dict[str, float]] = {}
        self.prompt_status_cache: dict[str, str] = {}
        self.verify_cache: dict[str, str] = {}
        self.idx = 0
        self.data: list[str] = []
        self.board_gt: list[str] = []
        self.ans_gt: list[str] = []
        self.board: list[str] = []
        self.ans: list[str] = []
        self.status: list[int] = []
        self.steps = 0

    def __len__(self) -> int:
        return len(self.dataset)

    def reset(
        self,
        idx: int,
        board: list[str] | None = None,
        status: list[int] | None = None,
        steps: int | None = None,
    ) -> str:
        example = self.dataset[idx]
        self.idx = idx
        self.data = example.clues
        self.board_gt = example.board_gt
        self.board = [EMPTY] * 25
        self.ans = [EMPTY * 5] * 10
        self.ans_gt = self.get_ans(self.board_gt)
        self.status = [0] * 10
        self.steps = 0
        if board is not None:
            self.board = board.copy()
            self.ans = self.get_ans(self.board)
        if status is not None:
            self.status = status.copy()
        if steps is not None:
            self.steps = steps
        return self.render()

    def copy_from_state(self, state: State) -> None:
        self.reset(
            int(state.meta["idx"]),
            board=list(state.meta["board"]),
            status=list(state.meta["status"]),
            steps=int(state.meta["steps"]),
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "idx": self.idx,
            "board": self.board.copy(),
            "status": self.status.copy(),
            "steps": self.steps,
            "ans": self.ans.copy(),
            "actions": [],
        }

    def render_board(self) -> str:
        s = "Current Board:\n"
        for i in range(5):
            s += "".join(self.board[i * 5 : (i + 1) * 5]) + "\n"
        return s

    def render_clues(self, status: int | None = None) -> str:
        lines = []
        for i in range(5):
            if status is None or self.status[i] == status:
                lines.append(f"h{i + 1}. {self.data[i]}")
        for i in range(5, 10):
            if status is None or self.status[i] == status:
                lines.append(f"v{i - 4}. {self.data[i]}")
        return "\n".join(lines) + ("\n" if lines else "")

    def render_ans(self, status: int | None = None) -> str:
        lines = []
        for i in range(5):
            if status is None or self.status[i] == status:
                lines.append(f"h{i + 1}. {self.data[i]}: {self.ans[i]}")
        for i in range(5, 10):
            if status is None or self.status[i] == status:
                lines.append(f"v{i - 4}. {self.data[i]}: {self.ans[i]}")
        return "\n".join(lines) + ("\n" if lines else "")

    def render(self, status: bool = True) -> str:
        if status:
            return (
                self.render_board()
                + "\nUnfilled:\n"
                + self.render_ans(status=0)
                + "\nFilled:\n"
                + self.render_ans(status=1)
                + "\nChanged:\n"
                + self.render_ans(status=2)
            )
        return self.render_board() + "\n" + self.render_ans()

    def get_ans(self, board: list[str]) -> list[str]:
        ans = [""] * 10
        for i in range(5):
            ans[i] = "".join(board[i * 5 : (i + 1) * 5])
        for i in range(5):
            ans[i + 5] = "".join(board[i::5])
        return ans

    def step(self, action: str) -> tuple[str, bool, bool, dict[str, float]]:
        self.steps += 1
        action = action.split("\n")[-1].strip()
        parts = action.split(". ")
        if len(parts) != 2:
            return 'Invalid! Format should be like "h1. apple"', False, False, {}
        pos, word = parts
        word = word.strip().upper()

        if len(word) != 5:
            return "Invalid! Word should have 5 letters.", False, False, {}
        if pos.startswith("h"):
            idx = int(pos[1:]) - 1
            self.board[idx * 5 : (idx + 1) * 5] = list(word)
        elif pos.startswith("v"):
            idx = int(pos[1:]) - 1
            self.board[idx::5] = list(word)
            idx += 5
        else:
            return "Invalid! Position should be h1-h5 or v1-v5", False, False, {}

        new_ans = self.get_ans(self.board)
        self.status = [
            2 if any(letter != new_letter and letter != EMPTY for letter, new_letter in zip(ans, new_ans_item)) else old_status
            for old_status, ans, new_ans_item in zip(self.status, self.ans, new_ans)
        ]
        self.status[idx] = 1
        self.ans = new_ans
        r_all = self.board == self.board_gt
        r_letter = sum(a == b for a, b in zip(self.board, self.board_gt)) / 25
        r_word = sum(a == b for a, b in zip(self.ans, self.ans_gt)) / 10
        return self.render(), r_all, (r_all or self.steps >= 20), {
            "r_letter": r_letter,
            "r_word": r_word,
            "r_game": float(r_all),
        }


class MiniCrosswordTask(Task):
    """Task adapter backed by the Princeton-style crossword environment."""

    def __init__(
        self,
        model: Any,
        dataset: list[CrosswordExample],
        deterministic_eval: bool = False,
        oracle_proposals: bool = False,
        proposal_only_eval: bool = False,
        debug_proposals: bool = False,
        proposal_prompt: str = "princeton",
        value_prompt: str = "princeton",
        verify_mode: str = "off",
        verify_proposals: bool = False,
        value_max_tokens: int = 300,
        word_list: set[str] | None = None,
        dictionary_mode: str = "off",
        dictionary_validate_crossings: bool = True,
    ):
        self.model = model
        self.dataset = dataset
        self.env = MiniCrosswordsEnv(dataset)
        self.max_depth = 10
        self.deterministic_eval = deterministic_eval
        self.oracle_proposals = oracle_proposals
        self.proposal_only_eval = proposal_only_eval
        self.debug_proposals = debug_proposals
        self.proposal_prompt = proposal_prompt
        self.value_prompt = value_prompt
        self.verify_mode = "filter" if verify_proposals else verify_mode
        self.value_max_tokens = value_max_tokens
        self.word_list = {word.upper() for word in word_list} if word_list else None
        self.dictionary_mode = dictionary_mode
        self.dictionary_validate_crossings = dictionary_validate_crossings

    @staticmethod
    def load_dataset(path: str | Path) -> list[CrosswordExample]:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            raw = raw.get("puzzles", [raw])
        if raw and isinstance(raw[0], list):
            return [CrosswordExample.from_official(item, i) for i, item in enumerate(raw)]
        return [CrosswordExample.from_dict(item) for item in raw]

    def get_input(self, idx: int) -> str:
        self.env.reset(idx)
        return self.env.render_clues()

    def initial_state(self, idx: int = 0) -> State:
        self.env.reset(idx)
        meta = self.env.snapshot()
        meta["actions"] = []
        meta["filled"] = {}
        meta["remaining"] = [slot.slot_id for slot in self.dataset[idx].slots]
        return State(text=self.env.render(), meta=meta, depth=0)

    def propose_thoughts(self, state: State, n_propose: int, n_select: int | None = None) -> list[State]:
        self.env.copy_from_state(state)
        obs = self.env.render()
        candidates_to_scores = self._get_candidates_to_scores(state, obs, n_propose)
        ranked = sorted(candidates_to_scores.items(), key=lambda item: item[1], reverse=True)
        if self.debug_proposals:
            preview = ", ".join(f"{action.replace('. ', ':')}:{score:.2f}" for action, score in ranked[:8])
            print(f"  proposals depth={state.depth}: {preview or '(none)'}")

        max_candidates = n_propose if n_select is None else n_select
        states: list[State] = []
        tried = 0
        for action, score in ranked:
            if tried >= max_candidates:
                break
            self.env.copy_from_state(state)
            _, _, _, info = self.env.step(action)
            if any(status == 2 for status in self.env.status):
                continue
            tried += 1
            actions = list(state.meta.get("actions", [])) + [action]
            meta = self.env.snapshot()
            meta.update(
                {
                    "actions": actions,
                    "filled": self._filled_from_actions(actions),
                    "remaining": [slot.slot_id for slot in self.dataset[self.env.idx].slots if slot.slot_id not in self._filled_from_actions(actions)],
                    "proposal_score": score,
                    "proposal": action,
                    "info": info,
                }
            )
            states.append(State(text="\n".join(actions), meta=meta, depth=state.depth + 1))
        return states

    def get_samples(self, state: State, n_sample: int, prompt_method: str) -> list[State]:
        self.env.copy_from_state(state)
        obs = self.env.render_clues()
        if prompt_method == "standard":
            prompt = STANDARD_PROMPT.format(input=obs)
        elif prompt_method == "cot":
            prompt = COT_PROMPT.format(input=obs)
        resp = self.model.generate(prompt, temperature=0.7, n=n_sample, max_tokens=500)
        states: list[State] = []
        for completion in resp.completions:
            self.env.copy_from_state(state)
            raw_actions = self._parse_samples(completion)
            action = ""
            info = {}
            actions = []
            for raw_action in raw_actions:
                _, _, _, info = self.env.step(raw_action)
                if any(status == 2 for status in self.env.status):
                    break
                action = raw_action
                actions.append(action)
            meta = self.env.snapshot()
            meta.update(
                {
                    "actions": actions,
                    "filled": self._filled_from_actions(actions),
                    "remaining": [slot.slot_id for slot in self.dataset[self.env.idx].slots if slot.slot_id not in self._filled_from_actions(actions)],
                    "proposal": action,
                    "info": info,
                }
            )
            states.append(State(text="\n".join(actions), meta=meta, depth=state.depth+1))
        return states

    def _parse_samples(self, response: str):
        text = response.upper().replace("```", "")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        grid_rows = []
        for line in lines:
            letters = re.findall(r"[A-Z]", line)
            if not letters:
                continue
            if 1 <= len(letters) <= 6:
                row = "".join(letters[:5])
                row = row.ljust(5, "_")
                grid_rows.append(row)
        if not grid_rows:
            return []
        board = grid_rows[-5:]
        while len(board) < 5:
            board.append("_____")
        actions = []
        for i, row in enumerate(board):
            actions.append(f"h{i+1}. {row.lower()}")
        for j in range(5):
            col = "".join(board[i][j] for i in range(5))
            actions.append(f"v{j+1}. {col.lower()}")
        return actions

    def evaluate_states(self, states: list[State], n_eval: int) -> list[float]:
        values: list[float] = []
        for state in states:
            if any(status == 2 for status in state.meta["status"]):
                values.append(0.0)
                continue
            if self.deterministic_eval:
                values.append(self._deterministic_value(state))
                continue
            if self.proposal_only_eval:
                values.append(20.0 if self.is_terminal(state) else state.meta.get("proposal_score", 0.0))
                continue
            count = self.prompt_status(state, n_eval=n_eval)
            state.meta["count"] = count
            if count["impossible"] >= 1:
                values.append(0.0)
            else:
                values.append(count["sure"] * 20.0 + count["maybe"] + state.meta.get("proposal_score", 0.0))
        return values

    def prompt_status(self, state: State, n_eval: int = 1) -> dict[str, int]:
        self.env.copy_from_state(state)
        count = {"sure": 0, "maybe": 0, "impossible": 0}
        for ans, clue in zip(self.env.ans, self.env.data):
            if ans.count(EMPTY) >= 4:
                continue
            line = f'{clue}: {" ".join(ans.lower())}'
            prompt_template = STRICT_VALUE_PROMPT if self.value_prompt == "strict" else VALUE_PROMPT
            prompt = prompt_template.format(input=line)
            if prompt in self.env.prompt_status_cache:
                completions = [self.env.prompt_status_cache[prompt]]
            else:
                resp = self.model.generate(prompt, temperature=0.7, n=n_eval, max_tokens=self.value_max_tokens)
                completions = resp.completions
                if completions:
                    self.env.prompt_status_cache[prompt] = completions[0]
            for completion in completions:
                judgment = self._extract_judgment(completion)
                if judgment in count:
                    count[judgment] += 1
        return count

    def is_terminal(self, state: State) -> bool:
        return state.meta.get("steps", 0) >= 10 or state.meta.get("board") == self.dataset[int(state.meta["idx"])].board_gt

    def score_output(self, state: State, ground_truth: Any = None) -> float:
        return self.score_metrics(state)["r_word"]

    def score_metrics(self, state: State | None) -> dict[str, float]:
        if state is None:
            return {"r_letter": 0.0, "r_word": 0.0, "r_game": 0.0}
        example = self.dataset[int(state.meta["idx"])]
        board = self._flatten_board(state.meta["board"])
        ans = self.env.get_ans(board)
        ans_gt = self.env.get_ans(example.board_gt)
        r_letter = sum(a == b for a, b in zip(board, example.board_gt)) / 25
        r_word = sum(a == b for a, b in zip(ans, ans_gt)) / 10
        r_game = float(board == example.board_gt)
        return {"r_letter": r_letter, "r_word": r_word, "r_game": r_game}

    @staticmethod
    def format_board(board: list[str] | list[list[str]]) -> str:
        if board and isinstance(board[0], list):
            return "\n".join("".join(row) for row in board)  # compatibility for old tests
        flat = list(board)
        return "\n".join("".join(flat[i * 5 : (i + 1) * 5]) for i in range(5))

    @staticmethod
    def _flatten_board(board: list[str] | list[list[str]]) -> list[str]:
        if board and isinstance(board[0], list):
            return [ch for row in board for ch in row]
        return list(board)

    @staticmethod
    def format_filled(filled: dict[str, str]) -> str:
        if not filled:
            return "(none)"
        return "\n".join(f"{slot_id}: {word}" for slot_id, word in sorted(filled.items()))

    @staticmethod
    def format_clues(slots: Iterable[CrosswordSlot]) -> str:
        return "\n".join(f"{slot.slot_id}. {slot.clue}" for slot in slots)

    def board_words(self, state: State) -> dict[str, str]:
        example = self.dataset[int(state.meta["idx"])]
        ans = self.env.get_ans(list(state.meta["board"]))
        return {slot.slot_id: ans[i] for i, slot in enumerate(example.slots)}

    def _get_candidates_to_scores(self, state: State, obs: str, n_propose: int) -> dict[str, float]:
        candidates_to_scores: dict[str, float] = {}
        if self.oracle_proposals:
            for action in self._oracle_actions(state):
                candidates_to_scores[action] = candidates_to_scores.get(action, 0.0) + CONFIDENCE_MAP["certain"]
            return candidates_to_scores
        filled = self._filled_from_actions(list(state.meta.get("actions", [])))
        prompt_template = {
            "fewshot": FEWSHOT_PROPOSE_PROMPT,
            "strict": STRICT_PROPOSE_PROMPT,
        }.get(self.proposal_prompt, PROPOSE_PROMPT)
        prompt = prompt_template.format(status=obs)
        resp = self.model.generate(prompt, temperature=0.7, n=n_propose, max_tokens=350)
        for completion in resp.completions:
            parsed = self._parse_response(completion)
            if self.debug_proposals and not parsed:
                preview = completion.replace("\n", " | ")[:500]
                print(f"  raw unparsed proposal: {preview}")
            for action, score in parsed:
                slot_id = action.split(". ", 1)[0]
                if slot_id in filled:
                    continue
                dictionary_ok = self._dictionary_accepts_action(state, action)
                if not dictionary_ok:
                    if self.dictionary_mode == "filter":
                        continue
                    if self.dictionary_mode == "penalize":
                        score -= 1.0
                if self.verify_mode != "off":
                    verification = self._verify_action(state, action)
                    if self.verify_mode == "filter" and verification == "impossible":
                        continue
                    if verification == "sure":
                        score += 2.0
                    elif verification == "maybe":
                        score += 0.2
                    elif self.verify_mode == "penalize":
                        score -= 0.5
                candidates_to_scores[action] = candidates_to_scores.get(action, 0.0) + score
        return candidates_to_scores

    def _parse_response(self, response: str) -> list[tuple[str, float]]:
        parsed: list[tuple[str, float]] = []
        pattern = re.compile(r"^([hv][1-5])\. ([a-zA-Z]{5}) \((certain|high|medium|low)\).*$", re.IGNORECASE)
        for line in response.splitlines():
            match = pattern.match(line.strip())
            if not match:
                continue
            action = f"{match.group(1).lower()}. {match.group(2).lower()}"
            parsed.append((action, CONFIDENCE_MAP.get(match.group(3).lower(), 0.0)))
        return parsed

    def _dictionary_accepts_action(self, state: State, action: str) -> bool:
        if self.dictionary_mode == "off" or not self.word_list:
            return True
        parts = action.split(". ", 1)
        if len(parts) != 2:
            return False
        proposed = parts[1].strip().upper()
        if proposed not in self.word_list:
            return False
        if not self.dictionary_validate_crossings:
            return True

        board = list(state.meta["board"])
        pos = parts[0].lower()
        if pos.startswith("h"):
            row = int(pos[1:]) - 1
            board[row * 5 : (row + 1) * 5] = list(proposed)
        elif pos.startswith("v"):
            col = int(pos[1:]) - 1
            board[col::5] = list(proposed)
        else:
            return False

        for word in self.env.get_ans(board):
            if EMPTY not in word and word.upper() not in self.word_list:
                return False
        return True

    def _extract_judgment(self, text: str) -> str:
        lines = [line.strip().lower() for line in text.splitlines() if line.strip()]
        haystack = [lines[-1]] + lines[:-1] if lines else [text.lower()]
        for line in haystack:
            for label in ("sure", "maybe", "impossible"):
                if re.search(rf"\b{label}\b", line):
                    return label
        return "impossible"

    def _verify_action(self, state: State, action: str) -> str:
        parts = action.split(". ", 1)
        if len(parts) != 2:
            return "impossible"
        slot_id, answer = parts[0], parts[1].upper()
        clue = self._clue_for_slot(int(state.meta["idx"]), slot_id)
        if clue is None:
            return "impossible"
        prompt = VERIFY_PROMPT.format(clue=clue, answer=answer)
        if prompt in self.env.verify_cache:
            return self.env.verify_cache[prompt]
        resp = self.model.generate(prompt, temperature=0.0, n=1, max_tokens=300)
        judgment = self._extract_judgment(resp.completions[0] if resp.completions else "")
        self.env.verify_cache[prompt] = judgment
        return judgment

    def _clue_for_slot(self, idx: int, slot_id: str) -> str | None:
        for slot in self.dataset[idx].slots:
            if slot.slot_id == slot_id:
                return slot.clue
        return None

    def _deterministic_value(self, state: State) -> float:
        example = self.dataset[int(state.meta["idx"])]
        board = list(state.meta["board"])
        for i, ch in enumerate(board):
            if ch != EMPTY and ch != example.board_gt[i]:
                return 0.0
        return 20.0

    def _filled_from_actions(self, actions: list[str]) -> dict[str, str]:
        filled = {}
        for action in actions:
            parts = action.split(". ")
            if len(parts) == 2:
                filled[parts[0]] = parts[1].upper()
        return filled

    def _oracle_actions(self, state: State) -> list[str]:
        example = self.dataset[int(state.meta["idx"])]
        filled = self._filled_from_actions(list(state.meta.get("actions", [])))
        actions = []
        for slot in example.slots:
            if slot.slot_id not in filled and slot.answer:
                actions.append(f"{slot.slot_id}. {slot.answer.lower()}")
        return actions
