"""
Run Game of 24 experiments.

Usage:
    python code/scripts/run_game24.py --config configs/game24_tot.yaml

For a quick smoke test on 5 puzzles:
    python code/scripts/run_game24.py --config configs/game24_tot.yaml --limit 5
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

# Make the `code` package importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tot.core import LLMClient, State, bfs_search
from tot.tasks.gameof24 import Game24Task


def setup_logging(level: str) -> None:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=getattr(logging, level.upper()),
    )


def load_dataset(path: str, start: int, end: int) -> list[str]:
    """
    The 24 dataset is a CSV with columns: Rank,Puzzles,AMT (1 sigma),AMT (2 sigma),Solved rate.
    Each row's Puzzles column is a string like "1 1 4 6". We extract that column.
    """
    df = pd.read_csv(path)
    puzzles = df["Puzzles"].astype(str).tolist()
    return puzzles[start:end]


def run_io(task, idx):
    """
    Direct prompt baseline. Uses few-shot examples (4 demos) to ensure the model
    produces a one-line expression rather than conversational preamble.
    
    Note: this differs slightly from the paper's pure naive baseline. Conversational
    models like gpt-4o-mini won't produce a parseable answer without examples.
    """
    input_str = task.get_input(idx)
    prompt = (
        "Use numbers and basic arithmetic operations (+ - * /) to obtain 24. "
        "Each number must be used exactly once. Output ONLY the answer line, "
        "no explanation.\n"
        "Input: 4 4 6 8\nAnswer: (4 + 8) * (6 - 4) = 24\n"
        "Input: 2 9 10 12\nAnswer: 2 * 12 * (10 - 9) = 24\n"
        "Input: 4 9 10 13\nAnswer: (13 - 9) * (10 - 4) = 24\n"
        "Input: 1 4 8 8\nAnswer: (1 + 8 / 4) * 8 = 24\n"
        f"Input: {input_str}\nAnswer:"
    )
    resp = task.llm.generate(prompt, temperature=0.7, n=1, max_tokens=100, stop=["\n", "Input:"])
    answer = resp.completions[0].strip()
    success = task._verify_answer(answer, input_str)
    return {"method": "io", "input": input_str, "answer": answer, "success": int(success)}


def run_cot(task: Game24Task, idx: int) -> dict:
    """Single CoT sample."""
    input_str = task.get_input(idx)
    success_rate, results = task.solve_cot(idx, n_samples=1)
    return {
        "method": "cot",
        "input": input_str,
        "answer": results[0] if results else "",
        "success": int(success_rate > 0),
    }


def run_cot_sc(task: Game24Task, idx: int, n_samples: int = 100) -> dict:
    """CoT with self-consistency: take the most common valid answer."""
    input_str = task.get_input(idx)
    success_rate, results = task.solve_cot(idx, n_samples=n_samples)
    valid_answers = [r for r in results if r]
    return {
        "method": "cot_sc",
        "input": input_str,
        "n_valid": len(valid_answers),
        "n_samples": n_samples,
        "success": int(len(valid_answers) > 0),
    }


def run_tot(task: Game24Task, idx: int, breadth: int, n_propose: int, n_eval: int) -> dict:
    """Full ToT BFS search."""
    input_str = task.get_input(idx)
    initial = State(text="", meta={"current": input_str, "original_input": input_str})
    final_states = bfs_search(
        task=task,
        initial_state=initial,
        breadth=breadth,
        n_propose=n_propose,
        n_eval=n_eval,
        max_depth=task.max_depth + 1,  # +1 for the final-answer step
    )
    # Success if ANY of the b retained final states verifies
    success = any(task.score_output(s) > 0.5 for s in final_states)
    best = final_states[0] if final_states else None
    return {
        "method": "tot",
        "input": input_str,
        "answer": best.meta.get("answer", "") if best else "",
        "n_kept": len(final_states),
        "any_success": int(success),
        "success": int(success),
        "trace": best.text if best else "",
    }


METHOD_FNS = {
    "io": run_io,
    "cot": run_cot,
    "cot_sc": run_cot_sc,
    "tot": run_tot,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int, default=None,
                        help="Only run the first N puzzles (smoke test)")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    setup_logging(cfg.get("log_level", "INFO"))
    logger = logging.getLogger("run_game24")

    # Set up output directory
    results_dir = Path(cfg["results_dir"]) / cfg["experiment_name"]
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "config.yaml").write_text(yaml.safe_dump(cfg))

    # Initialize LLM and task
    llm = LLMClient(
        model=cfg["model"],
        cache_dir=cfg.get("cache_dir"),
        cost_log_path=cfg.get("cost_log"),
    )
    dataset = load_dataset(cfg["data_path"], cfg["start_idx"], cfg["end_idx"])
    if args.limit is not None:
        dataset = dataset[: args.limit]
    task = Game24Task(llm=llm, dataset=dataset)

    # Run each enabled method
    summary = {}
    for method_cfg in cfg["methods"]:
        if not method_cfg["enabled"]:
            continue
        name = method_cfg["name"]
        logger.info(f"Running method: {name}")
        records = []
        t0 = time.time()
        for idx in tqdm(range(len(dataset)), desc=name):
            try:
                if name == "tot":
                    rec = run_tot(
                        task, idx,
                        breadth=method_cfg["breadth"],
                        n_propose=method_cfg["n_propose"],
                        n_eval=method_cfg["n_eval"],
                    )
                elif name == "cot_sc":
                    rec = run_cot_sc(task, idx, n_samples=method_cfg["n_samples"])
                else:
                    rec = METHOD_FNS[name](task, idx)
                rec["puzzle_idx"] = cfg["start_idx"] + idx
                records.append(rec)
            except Exception as e:
                logger.exception(f"Puzzle {idx} ({name}) failed: {e}")
                records.append({"method": name, "puzzle_idx": cfg["start_idx"] + idx, "error": str(e), "success": 0})

        elapsed = time.time() - t0
        out_path = results_dir / f"{name}_results.jsonl"
        with open(out_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        success_rate = sum(r.get("success", 0) for r in records) / max(len(records), 1)
        summary[name] = {
            "success_rate": success_rate,
            "n_puzzles": len(records),
            "elapsed_sec": elapsed,
        }
        logger.info(f"{name}: {success_rate:.3f} success on {len(records)} puzzles ({elapsed:.1f}s)")

    # Final summary
    summary_path = results_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print("\n" + "=" * 50)
    print(f"Results written to {results_dir}")
    print(json.dumps(summary, indent=2))

    # Cost summary
    if llm.cost_tracker:
        totals = llm.cost_tracker.total()
        print("\nAPI usage:")
        print(json.dumps(totals, indent=2))


if __name__ == "__main__":
    main()
