# Tree of Thoughts Reimplementation

This repository contains a CS 4782 reimplementation and extension study of *Tree of Thoughts: Deliberate Problem Solving with Large Language Models* by Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffiths, Yuan Cao, and Karthik Narasimhan.

## Introduction

Tree of Thoughts (ToT) improves on standard left-to-right prompting by letting a language model generate, evaluate, and search over multiple intermediate reasoning steps called thoughts. This project studies that idea on the paper's main task settings: Game of 24, Creative Writing, and Mini Crossword.

## Chosen Result

We aimed to reproduce the paper's task-level gains from structured search over direct prompting, focusing on Game of 24 success rate, Creative Writing coherence, and Mini Crossword letter/word/game accuracy.

For Mini Crossword, we specifically targeted Section 4.3 / Table 3, where ToT outperformed IO and Chain-of-Thought prompting on 20 selected 5x5 crossword games.

## GitHub Contents

- `code/tot/`: main implementation package for core search, task definitions, prompts, and model clients.
- `code/scripts/`: command-line runners for Game of 24 and Mini Crossword.
- `code/tests/`: offline tests for Mini Crossword behavior.
- `code/tot/tasks/creative-writing/`: self-contained Creative Writing runners and analysis scripts.
- `data/`: task inputs for Game of 24, Creative Writing, and Mini Crossword.
- `results/`: logs, JSON outputs, tables, figures, and dashboards.
- `poster/` and `report/`: final presentation and report materials.

## Reimplementation Details

The main practical challenge was reproducing a GPT-4-era paper with a weaker and cheaper model. We intentionally used `gpt-4o-mini` because stronger models are expensive, and part of our goal was to test whether ToT can be adapted for students and small research groups with limited API budgets.

Game of 24 uses BFS-style ToT search over arithmetic expression states and compares IO, CoT, self-consistency, and ToT success rates.

Creative Writing follows the paper's four-ending-sentence setup: IO, CoT, ToT, IO+Refine, and ToT+Refine generate four-paragraph passages, then coherence is scored on a 1-10 scale. We added optimization analyses for `k`, vote count, scoring prompts, A* ToT, and Hybrid ToT to study whether search and evaluation choices could compensate for a weaker model.

Mini Crossword implements DFS-style ToT with backtracking and pruning. Each state stores a partial 5x5 board, filled clues, remaining clues, and action history; evaluation reports letter accuracy, word accuracy, and fully solved games. Because weaker proposals made pruning more fragile, we added mock/oracle modes, stricter validation, and paper/budget/debug presets to make the pipeline easier to test and adapt.

## Reproduction Steps

Install dependencies:

```powershell
pip install -r requirements.txt
```

Set an OpenAI API key for model-backed runs:

```powershell
$env:OPENAI_API_KEY="your_key_here"
```

Run tests:

```powershell
python -m pytest code\tests -q -p no:cacheprovider
```

Run Game of 24:

```powershell
python code\scripts\run_game24.py --config code\configs\game24_tot.yaml --limit 5
```

Run Creative Writing:

```powershell
python code\tot\tasks\creative-writing\tot_creative_writing.py --method all --n 5 --out creative_writing_results.json --k 5 --n-votes 5
```

Run Mini Crossword:

```powershell
python code\scripts\run_crosswords.py --data data\crosswords\crosswords_toy.json --mock-model --n-propose 2 --n-eval 1 --threshold 1.0
python code\scripts\run_crosswords.py --data data\crosswords\crosswords_official_0_100_5_raw.json --model gpt-4o-mini --preset paper --out results\crosswords\openai_full.jsonl
```

Only the tests and `--mock-model` Mini Crossword smoke test run without an API key. Game of 24, Creative Writing, and real Mini Crossword reproduction use OpenAI `gpt-4o-mini` by default and require `OPENAI_API_KEY`; no GPU is required.

## Results/Insights

Creative Writing results are in `results/creative-writing/`. In the baseline run, ToT-Refine and ToT produced the strongest model-scored outputs, but scores were higher and less separated than the paper's GPT-4 results; human pairwise comparison still preferred ToT over CoT in 14/20 cases.

Mini Crossword results are in `results/crosswords/`. Our ToT without pruning was close to the paper's reported Table 3 result: 0.63 letter accuracy, 0.36 word accuracy, and 0.05 game accuracy, compared with the paper's 0.65, 0.41, and 0.05.

Across tasks, the main takeaway is that ToT works best when intermediate states can be proposed and evaluated meaningfully. Model strength, pruning quality, vote count, and scoring prompt design all materially affect reproducibility.

## Conclusion

This project reimplemented Tree of Thoughts across Game of 24, Creative Writing, and Mini Crossword while adding lower-cost `gpt-4o-mini` runs, debugging modes, and analysis dashboards.

The biggest lesson was that ToT is not just a search wrapper: task-specific state design, evaluation prompts, pruning rules, and scorer calibration determine whether the search tree actually improves results.

## References

- Yao, S., Yu, D., Zhao, J., Shafran, I., Griffiths, T. L., Cao, Y., & Narasimhan, K. (2023). *Tree of Thoughts: Deliberate Problem Solving with Large Language Models*. arXiv:2305.10601.
- Original Tree of Thoughts repository and task datasets.

## Acknowledgements

This project was completed for CS 4782 at Cornell University by Aarsha Joshi, Aileen Huang, Bella Falkenberg, and Ena Kovac.
