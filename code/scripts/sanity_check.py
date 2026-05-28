"""
Sanity check: confirm the LLM client can talk to your provider.

Usage:
    python code/scripts/sanity_check.py --model llama-3.3-70b-versatile
    python code/scripts/sanity_check.py --model ollama/llama3.1:70b
    python code/scripts/sanity_check.py --model gpt-4o-mini

Run this BEFORE running the full pipeline. It costs essentially nothing and
catches the 90% of bugs that are "wrong API key" or "wrong model name".
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tot.core import LLMClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model name to test")
    args = parser.parse_args()

    print(f"Testing model: {args.model}")
    llm = LLMClient(model=args.model, cache_dir=None, cost_log_path=None)
    print(f"Provider inferred: {llm.provider}")

    # 1. Trivial completion
    print("\n[1/3] Asking 'What is 2+2?'...")
    resp = llm.generate("What is 2+2? Answer with just the number.", temperature=0.0, n=1, max_tokens=20)
    print(f"  Response: {resp.completions[0]!r}")
    print(f"  Tokens: prompt={resp.prompt_tokens}, completion={resp.completion_tokens}")
    print(f"  Cost: ${resp.cost_usd:.6f}")

    # 2. Check that it follows few-shot format (this is what actually matters for ToT)
    print("\n[2/3] Testing few-shot format following...")
    prompt = (
        "Output exactly one word on the final line: sure, likely, or impossible.\n"
        "10 14\n10 + 14 = 24\nsure\n"
        "1 1\n1 + 1 = 2\n1 * 1 = 1\nimpossible\n"
        "5 7 8\n"
    )
    resp = llm.generate(prompt, temperature=0.7, n=1, max_tokens=100)
    last_line = resp.completions[0].strip().split("\n")[-1].strip().lower()
    print(f"  Full response:\n  ---\n  {resp.completions[0]}\n  ---")
    print(f"  Last line: {last_line!r}")
    if last_line in ("sure", "likely", "impossible"):
        print("  ✓ Follows format")
    else:
        print("  ✗ Does NOT follow format — your value extraction may need work")

    # 3. Multiple samples (n>1)
    print("\n[3/3] Testing n=3 sampling...")
    resp = llm.generate("Say 'hello' once.", temperature=0.7, n=3, max_tokens=20)
    print(f"  Got {len(resp.completions)} completions")
    for i, c in enumerate(resp.completions):
        print(f"    [{i}] {c[:50]!r}")

    print("\nAll sanity checks complete. If everything looks right, you're ready to run the full pipeline.")


if __name__ == "__main__":
    main()
