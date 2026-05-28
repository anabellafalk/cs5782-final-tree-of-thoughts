"""
Append-only JSONL log of every API call.

Use the analyze() method (or pandas) to summarize spending by model, by day,
by run. Critical for the report's "challenges" section — being able to say
"we spent $X across Y calls and ToT cost Z× more than CoT" is concrete and
demonstrates rigor.
"""
import json
import os
import time
from pathlib import Path


class CostTracker:
    def __init__(self, log_path: str):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, model: str, prompt_tokens: int, completion_tokens: int, cost_usd: float, **extra) -> None:
        record = {
            "timestamp": time.time(),
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost_usd": cost_usd,
            **extra,
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def total(self) -> dict:
        """Return total spend and call count grouped by model."""
        if not self.log_path.exists():
            return {}
        totals = {}
        with open(self.log_path) as f:
            for line in f:
                rec = json.loads(line)
                m = rec["model"]
                if m not in totals:
                    totals[m] = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}
                totals[m]["calls"] += 1
                totals[m]["prompt_tokens"] += rec["prompt_tokens"]
                totals[m]["completion_tokens"] += rec["completion_tokens"]
                totals[m]["cost_usd"] += rec["cost_usd"]
        return totals