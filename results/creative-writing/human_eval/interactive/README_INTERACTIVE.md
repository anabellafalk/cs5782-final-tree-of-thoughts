# Interactive pairwise judging

1. Each rater opens `index.html` in a **desktop browser** (Chrome / Firefox / Safari).
   - Same folder must contain `pairs_embed.js` (do not rename).
2. Select **Rater ID** (1, 2, or 3). Vote for all pairs; use **Export my votes (JSON)** when finished.
3. Send exported files to the organizer.
4. Merge + κ:

```bash
python3 code/tot/tasks/creative-writing/human_pairwise_interactive.py merge \
  --key results/creative-writing/human_eval/interactive/pairs_key.json \
  --rater-exports rater_1_votes.json rater_2_votes.json rater_3_votes.json \
  --out-csv merged_ratings.csv

python3 code/tot/tasks/creative-writing/human_pairwise_eval.py aggregate \
  --ratings merged_ratings.csv \
  --key pairs_key.json
```

Blind protocol: raters never see `pairs_key.json` (maps A/B to `tot` / `cot`).

Generated **20** pairs.
