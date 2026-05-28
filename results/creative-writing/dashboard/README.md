# Creative writing results dashboard

See **`GUIDE.md`** for the full annotated layout and file descriptions.

Single entrypoint: `code/tot/tasks/creative-writing/metrics_dashboard.py`
(reads `results/metrics_log.jsonl`; result JSONs live in `results/creative-writing/json_outputs/`)

## Quick commands

**Full rebuild:**
```bash
python3 code/tot/tasks/creative-writing/metrics_dashboard.py --clean
```

**ToT vs A\* only** (adds passage-level plots in `independent_exploration/astar_analysis/`):
```bash
python3 code/tot/tasks/creative-writing/metrics_dashboard.py --clean \
  --filter-methods tot tot_astar \
  --results-json results/creative-writing/json_outputs/all_with_astar.json
```

**A\* score-delta correlations** (prints Pearson r for score Δ vs metric Δ; optional PNGs under `independent_exploration/astar/`):
```bash
python3 code/tot/tasks/creative-writing/metrics_dashboard.py --correlation-only \
  --correlation-prompt-variants paper \
  --correlation-plot-dir \
  | tee results/creative-writing/dashboard/independent_exploration/correlation_logs/astar_correlations_paper_only.txt
```

**Qualitative examples:**
```bash
python3 code/tot/tasks/creative-writing/showcase_examples.py \
  --input results/creative-writing/json_outputs/all_with_astar.json
```

## Layout (top-level)

| Folder | Contents |
|--------|----------|
| `baseline/` | Method score bars, distribution, heatmap, score gaps, cost Pareto |
| `modifications/` | Hyperparameter & scorer variations (`ablations/`, `score_prompt/`, `scoring_model/`, `correlations/`, `regression/`) |
| `independent_exploration/` | A\* and planning method experiments (`astar_analysis/`, `plan_methods/`, `correlation_logs/`) |
| `tables/` | Aggregated metrics CSV/JSON (all comparison slices) |
