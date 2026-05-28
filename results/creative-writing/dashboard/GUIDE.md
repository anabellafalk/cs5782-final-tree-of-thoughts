# Dashboard Guide

All plots are generated from `results/metrics_log.jsonl` by running:

```bash
python3 code/tot/tasks/creative-writing/metrics_dashboard.py --clean
```

Result JSONs live in `results/creative-writing/json_outputs/`.

---

## Folder structure

### `baseline/`
Core method comparison plots — the first thing to look at.

| File | What it shows |
|------|---------------|
| `method_mean_score_bar.png` | Mean coherency score per method (IO, CoT, ToT, IO+refine, ToT+refine) with 95% CI |
| `method_score_distribution.png` | Per-method score distributions |
| `correlation_heatmap.png` | Pearson correlation matrix across all numeric metrics |
| `pairwise_score_gaps.png` | Score gaps between method pairs |
| `score_vs_cost_pareto.png` | Score vs average cost per task — Pareto frontier view |

---

### `modifications/`
Results from varying hyperparameters and scoring setup.

#### `modifications/ablations/`
K (candidates per stage) and N_votes sweeps.

| File | What it shows |
|------|---------------|
| `k_vs_mean_score.png` | Effect of k on mean coherency score |
| `k_vs_vote_entropy.png` | Effect of k on vote entropy |
| `k_vs_cost_tradeoff.png` | Score vs cost tradeoff across k values |
| `n_votes_vs_mean_score.png` | Effect of n_votes on mean score |
| `n_votes_vs_vote_entropy.png` | Effect of n_votes on vote entropy |
| `n_votes_entropy_vs_mean_score_overlay.png` | Entropy vs mean score overlay across n_votes |

#### `modifications/score_prompt/`
How the scoring prompt variant (paper / definition / criteria) affects results.

| File | What it shows |
|------|---------------|
| `prompt_vs_mean_score.png` | Mean score by scoring prompt variant |
| `prompt_vs_mean_score_tot_only.png` | Same, ToT only |
| `prompt_vs_score_std.png` | Score std dev by scoring prompt variant |
| `prompt_vs_score_std_tot_only.png` | Same, ToT only |

#### `modifications/scoring_model/`
Effect of using a different judge model (gpt-4o-mini vs gpt-4.1).

| File | What it shows |
|------|---------------|
| `baseline_vs_scorer4_1_by_method.png` | Mean score comparison: baseline scorer vs scorer4.1, by method |

#### `modifications/correlations/`
Relationships between internal metrics (vote entropy, score std dev) and mean score.

| File | What it shows |
|------|---------------|
| `vote_entropy_plan_passage_vs_mean_score.png` | Plan + passage vote entropy vs mean score (combined) |
| `vote_entropy_plan_vs_mean_score.png` | Plan-stage vote entropy vs mean score |
| `vote_entropy_passage_vs_mean_score.png` | Passage-stage vote entropy vs mean score |
| `score_std_vs_mean_score.png` | Score std dev (judge disagreement) vs mean score, by method |

#### `modifications/regression/`
Controlled regression analysis: do vote entropy / score std predict mean score after controlling for method, k, n_votes, prompt?

| File | What it shows |
|------|---------------|
| `controlled_regression_coefficients.png` | Forest plot of regression coefficients with HC3 95% CIs |
| `controlled_regression_partial_residuals.png` | Partial residual plots (controls removed) |
| `controlled_regression_summary.csv/.json` | Full regression table (coef, SE, p-value, CI, n, R²) |
| `tot_score_std_partial_effect.png` | Partial effect of score_std on mean_score (ToT only) |
| `interactive_regression_explorer.html` | Interactive HTML — toggle controls, inspect coefficients in browser |
| `interactive_regression_models.json` | Model data backing the interactive explorer |

---

### `independent_exploration/`
Experiments that go beyond the base methods — A\* plan selection and hybrid planning.

#### `independent_exploration/astar_analysis/`
ToT vs ToT-A\* comparison (A\* selects the best plan via a two-stage vote heuristic).

| File | What it shows |
|------|---------------|
| `astar_vs_tot_comparison.png` | Side-by-side mean score bar |
| `score_delta_vs_delta_score_std.png` | Score Δ (A\*−ToT) vs Δ score std — strongest predictor |
| `score_delta_vs_delta_vote_entropy_plan.png` | Score Δ vs Δ plan vote entropy |
| `score_delta_vs_delta_vote_entropy_passage.png` | Score Δ vs Δ passage vote entropy |
| `score_vs_cost_pareto_tot_vs_tot_astar.png` | Score vs cost, tot vs tot_astar only |
| `passage_*.png` | Passage-level paired comparisons (requires `--results-json`) |

#### `independent_exploration/plan_methods/`
Three-way comparison: standard ToT vote, A\*, and hybrid planning.

| File | What it shows |
|------|---------------|
| `three_method_comparison.png` | Bar chart of all three planning methods |
| `score_vs_cost_pareto.png` | Score vs cost for all three |

#### `independent_exploration/correlation_logs/`
Saved text output from `--correlation-only` mode (A\*−ToT metric delta Pearson correlations).

| File | What it covers |
|------|----------------|
| `astar_correlations_paper_only.txt` | Paper prompt variant only |
| `astar_correlations_all_prompts.txt` | All scoring prompt variants combined |

---

### `tables/`
Aggregated metrics tables (CSV + JSON). One file per comparison slice.

| File | Slice |
|------|-------|
| `grouped_metrics.csv/.json` | Paper-core (io / cot / tot / io_refine / tot_refine) |
| `grouped_metrics_tot_astar_compare.csv/.json` | tot vs tot_astar |
| `grouped_metrics_plan_methods.csv/.json` | tot vs tot_astar vs hybrid_tot |
| `grouped_metrics_scorer_compare.csv/.json` | baseline.json vs scorer4.1.json |

---

## Regeneration commands

**Full rebuild:**
```bash
python3 code/tot/tasks/creative-writing/metrics_dashboard.py --clean
```

**ToT vs A\* with passage-level plots:**
```bash
python3 code/tot/tasks/creative-writing/metrics_dashboard.py --clean \
  --filter-methods tot tot_astar \
  --results-json results/creative-writing/json_outputs/all_with_astar.json
```

**A\* score-delta correlations (prints + saves PNGs):**
```bash
python3 code/tot/tasks/creative-writing/metrics_dashboard.py --correlation-only \
  --correlation-prompt-variants paper \
  --correlation-plot-dir \
  | tee results/creative-writing/dashboard/independent_exploration/correlation_logs/astar_correlations_paper_only.txt
```
