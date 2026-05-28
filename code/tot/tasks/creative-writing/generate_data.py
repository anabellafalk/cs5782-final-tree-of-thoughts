#!/usr/bin/env python3
"""Generate 100 inputs of 4 random sentences each (Yao et al. 2023, §3.2).

Reproduces the paper's protocol: draw sentences independently (so they are not
thematically related) then shuffle into groups of 4. The seeded shuffle (42)
makes the generated dataset reproducible across runs even though the LLM calls
themselves are uncached and stochastic.

Usage:
    python code/generate_data.py
"""

import json
import os
import random
import re
import sys
from pathlib import Path
from dotenv import load_dotenv

TOT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(TOT_ROOT) not in sys.path:
    sys.path.append(str(TOT_ROOT))

from core.llm import LLMClient

load_dotenv(REPO_ROOT / ".env")
MODEL    = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
ROOT     = REPO_ROOT
OUT_PATH = ROOT / "data" / "sentences.json"
CACHE_DIR = ROOT / ".llm_cache"
COST_LOG_PATH = ROOT / "results" / "cost_log.jsonl"
N_INPUTS = 100
llm      = LLMClient(
    model=MODEL,
    provider="openai",
    cache_dir=str(CACHE_DIR),
    cost_log_path=str(COST_LOG_PATH),
)

PROMPT = """\
Generate 10 random sentences. Each should be a standalone sentence \
that could end a paragraph — varied in topic, tone, and style. \
Do not make them related to each other.

Output exactly 10 sentences, one per line, numbered 1–10. Nothing else."""


def generate_batch() -> list[str]:
    # temperature=1.0 + uncached: maximize diversity. Caching would collapse 40
    # repeated calls into the same 10 sentences.
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY in environment.")
    resp = llm.generate(
        prompt=PROMPT,
        temperature=1.0,
        n=1,
        max_tokens=400,
        use_cache=False,
    )
    sentences = []
    text = (resp.completions[0] if resp.completions else "").strip()
    for line in text.splitlines():
        # Strip "1.", "2)", etc. — models inconsistently prefix numbered lists.
        line = re.sub(r"^\d+[\.\)]\s*", "", line.strip())
        if line:
            sentences.append(line)
    # Hard cap to 10 in case the model ignored the instruction and over-produced.
    return sentences[:10]


def main():
    OUT_PATH.parent.mkdir(exist_ok=True)

    print("Generating 400 random sentences...")
    pool = []
    for i in range(40):
        # Retry whole batches that came up short — easier than partial fills,
        # and rare enough that the extra calls don't matter.
        batch = generate_batch()
        while len(batch) < 10:
            batch = generate_batch()
        pool.extend(batch)
        print(f"  [{len(pool)}/400]")

    # Fixed seed so the grouping is reproducible. The pool itself varies across
    # runs (uncached generation), but anyone with the same pool gets the same
    # 4-tuples — useful when comparing experiments on identical inputs.
    random.seed(42)
    random.shuffle(pool)

    data = [
        {"id": i, "sentences": pool[i * 4: i * 4 + 4]}
        for i in range(N_INPUTS)
    ]

    with open(OUT_PATH, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Done. Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
