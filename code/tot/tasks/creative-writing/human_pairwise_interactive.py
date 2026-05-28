#!/usr/bin/env python3
"""
Interactive HTML pairwise judging for final passages (human eval only).

Static-site variant of `human_pairwise_eval.py generate`. Two reasons to have
both:
- This version is friendlier for non-technical raters (no spreadsheet, no
  CSV format confusion, keyboard shortcuts, auto-save in localStorage).
- It writes nothing during rating — the page is fully self-contained, so
  raters can work offline and email back a JSON export.

The `aggregate` step is still done by `human_pairwise_eval.py` after the
exports are merged here, so kappa / majority logic stays single-sourced.

Generates a small static site you open in a browser (file:// or any static host).
Each rater picks rater ID (1–3), votes independently; progress saves in localStorage.
Votes: **1** = left (A) better, **2** = right (B) better, **3** = similarly coherent (paper-style third option).
Export JSON when done; merge exports into CSV for human_pairwise_eval.py aggregate.

Example
-------
  python3 code/tot/tasks/creative-writing/human_pairwise_interactive.py generate \\
    --input results/creative-writing/json_outputs/all_with_astar.json \\
    --out-dir results/creative-writing/human_eval/interactive

  # Open in browser:
  #   results/creative-writing/human_eval/interactive/index.html

  python3 code/tot/tasks/creative-writing/human_pairwise_interactive.py merge \\
    --key results/creative-writing/human_eval/interactive/pairs_key.json \\
    --rater-exports r1.json r2.json r3.json \\
    --out-csv results/creative-writing/human_eval/interactive/merged_ratings.csv

  python3 code/tot/tasks/creative-writing/human_pairwise_eval.py aggregate \\
    --ratings results/creative-writing/human_eval/interactive/merged_ratings.csv \\
    --key results/creative-writing/human_eval/interactive/pairs_key.json
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_INPUT = ROOT / "results" / "creative-writing" / "json_outputs" / "all_with_astar.json"


def _load_results(path: Path) -> dict[str, list[dict[str, Any]]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Results JSON must be a dict: method -> list of entries.")
    return data


def _pair_index(
    results: dict[str, list[dict[str, Any]]],
    method_a: str,
    method_b: str,
) -> dict[Any, tuple[dict, dict]]:
    by_id_a = {e["id"]: e for e in results.get(method_a, []) if "id" in e}
    by_id_b = {e["id"]: e for e in results.get(method_b, []) if "id" in e}
    common = set(by_id_a) & set(by_id_b)
    return {tid: (by_id_a[tid], by_id_b[tid]) for tid in common}


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Creative writing — pairwise passage judgment</title>
  <style>
    :root {
      font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
      --bg: #0f1419;
      --card: #1a2332;
      --text: #e7ecf3;
      --muted: #8b9cb3;
      --accent: #3d8bfd;
      --good: #3fb950;
      --border: #30363d;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0; background: var(--bg); color: var(--text);
      min-height: 100vh; line-height: 1.5;
    }
    header {
      padding: 1rem 1.25rem; border-bottom: 1px solid var(--border);
      background: var(--card); position: sticky; top: 0; z-index: 10;
    }
    h1 { margin: 0; font-size: 1.1rem; font-weight: 600; }
    .sub { color: var(--muted); font-size: 0.85rem; margin-top: 0.35rem; }
    main { max-width: 1200px; margin: 0 auto; padding: 1rem 1.25rem 3rem; }
    .controls {
      display: flex; flex-wrap: wrap; gap: 0.75rem; align-items: center;
      margin-bottom: 1rem;
    }
    label { color: var(--muted); font-size: 0.85rem; }
    select, button, input[type="file"] {
      background: var(--card); color: var(--text); border: 1px solid var(--border);
      padding: 0.45rem 0.75rem; border-radius: 6px; font-size: 0.9rem;
    }
    button.primary {
      background: var(--accent); border-color: var(--accent); color: #fff; font-weight: 600;
      cursor: pointer;
    }
    button.primary:hover { filter: brightness(1.08); }
    button.secondary { cursor: pointer; }
    button.good { background: var(--good); border-color: var(--good); color: #0d1117; font-weight: 600; cursor: pointer; }
    .progress { color: var(--muted); font-size: 0.9rem; }
    .bar {
      height: 6px; background: var(--border); border-radius: 3px; overflow: hidden; margin-top: 0.35rem;
    }
    .bar > div { height: 100%; background: var(--accent); transition: width 0.2s; }
    .pair-meta { margin-bottom: 1rem; color: var(--muted); font-size: 0.9rem; }
    .panels {
      display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;
    }
    @media (max-width: 800px) { .panels { grid-template-columns: 1fr; } }
    .panel {
      background: var(--card); border: 1px solid var(--border); border-radius: 10px;
      padding: 1rem; min-height: 200px;
    }
    .panel h2 {
      margin: 0 0 0.75rem; font-size: 0.95rem; color: var(--muted); font-weight: 600;
    }
    .passage {
      white-space: pre-wrap; word-break: break-word; font-size: 0.95rem;
      max-height: 55vh; overflow-y: auto;
    }
    .vote-row {
      display: flex; flex-wrap: wrap; gap: 0.75rem; margin-top: 1.25rem; align-items: center;
    }
    button.vote-btn {
      min-width: 12rem; padding: 0.7rem 1.1rem; font-size: 1rem; font-weight: 600;
      border-width: 2px; border-radius: 8px;
      transition: transform 0.12s ease, box-shadow 0.12s ease, border-color 0.12s ease, background 0.12s ease;
    }
    button.vote-btn:hover:not(.selected) { transform: scale(1.02); }
    /* Unselected: neutral so “selected” pops */
    button#btnLeft:not(.selected) { background: #21262d; border-color: #444; color: #e6edf3; }
    button#btnRight:not(.selected) { background: #21262d; border-color: #444; color: #e6edf3; }
    button#btnLeft.selected {
      background: linear-gradient(160deg, #238636, #2ea043); border-color: #56d364; color: #fff;
      box-shadow: 0 0 0 4px rgba(63, 185, 80, 0.45); transform: scale(1.03);
    }
    button#btnRight.selected {
      background: linear-gradient(160deg, #1f6feb, #388bfd); border-color: #79c0ff; color: #fff;
      box-shadow: 0 0 0 4px rgba(56, 139, 253, 0.45); transform: scale(1.03);
    }
    button#btnSimilar:not(.selected) { background: #21262d; border-color: #444; color: #e6edf3; }
    button#btnSimilar.selected {
      background: linear-gradient(160deg, #6e7681, #8b949e); border-color: #c9d1d9; color: #fff;
      box-shadow: 0 0 0 4px rgba(139, 148, 158, 0.35); transform: scale(1.03);
    }
    .panels.similar-picked .panel { opacity: 0.88; }
    .panel.panel-chosen-a {
      border-color: #3fb950;
      box-shadow: inset 0 0 0 2px rgba(63, 185, 80, 0.35);
    }
    .panel.panel-chosen-b {
      border-color: #388bfd;
      box-shadow: inset 0 0 0 2px rgba(56, 139, 253, 0.35);
    }
    .hint { color: var(--muted); font-size: 0.8rem; }
    .status { margin-top: 0.75rem; font-size: 0.85rem; }
    .error { color: #f85149; }
    footer {
      margin-top: 2rem; padding-top: 1rem; border-top: 1px solid var(--border);
      color: var(--muted); font-size: 0.8rem;
    }
  </style>
</head>
<body>
  <header>
    <h1>Pairwise passage judgment (human only)</h1>
    <div class="sub">Do not discuss with other raters until everyone finishes. For each pair: choose <strong>A</strong> or <strong>B</strong> if one passage is clearly more coherent, or <strong>Similarly coherent</strong> if they are about the same (paper-style third option). Left/right assignment is randomized per pair.</div>
  </header>
  <main>
    <div class="controls">
      <label>Rater ID <select id="raterId"><option value="1">1</option><option value="2">2</option><option value="3">3</option></select></label>
      <button type="button" class="secondary" id="btnSave">Save reminder</button>
      <button type="button" class="secondary" id="btnExport">Export my votes (JSON)</button>
      <label>Import session <input type="file" id="importFile" accept="application/json" /></label>
    </div>
    <div class="progress">
      <span id="progText"></span>
      <div class="bar"><div id="progBar"></div></div>
    </div>
    <p class="pair-meta" id="pairMeta"></p>
    <div class="panels" id="panelRow">
      <div class="panel" id="leftPanel">
        <h2>Passage A (left)</h2>
        <div class="passage" id="leftText"></div>
      </div>
      <div class="panel" id="rightPanel">
        <h2>Passage B (right)</h2>
        <div class="passage" id="rightText"></div>
      </div>
    </div>
    <div class="vote-row">
      <button type="button" class="vote-btn" id="btnLeft" aria-pressed="false">A is better</button>
      <button type="button" class="vote-btn" id="btnRight" aria-pressed="false">B is better</button>
      <button type="button" class="vote-btn" id="btnSimilar" aria-pressed="false">Similarly coherent</button>
      <button type="button" class="secondary" id="btnClear">Clear vote for this pair</button>
      <span class="hint">Shortcuts: <kbd>1</kbd> A · <kbd>2</kbd> B · <kbd>3</kbd> similar</span>
    </div>
    <div class="vote-row">
      <button type="button" class="secondary" id="btnPrev">← Previous</button>
      <button type="button" class="primary" id="btnNext">Next →</button>
      <button type="button" class="secondary" id="btnFirstMissing">Jump to first missing</button>
    </div>
    <p class="status" id="status"></p>
    <footer>
      Cohen’s κ interpretation (pairwise between raters): κ &gt; 0.6 substantial; 0.4–0.6 moderate; &lt; 0.4 poor.
      After all raters export, run <code>human_pairwise_interactive.py merge</code> then <code>human_pairwise_eval.py aggregate</code>.
    </footer>
  </main>
  <script src="pairs_embed.js"></script>
  <script>
(function() {
  const STORAGE_PREFIX = "cw_pairwise_v2_rater_";
  let idx = 0;

  function storageKey() {
    const r = document.getElementById("raterId").value;
    return STORAGE_PREFIX + r;
  }

  function loadVotes() {
    try {
      const raw = localStorage.getItem(storageKey());
      return raw ? JSON.parse(raw) : {};
    } catch (e) { return {}; }
  }

  function saveVotes(votes) {
    localStorage.setItem(storageKey(), JSON.stringify(votes));
  }

  function pairCount() {
    return (window.CW_PAIRS || []).length;
  }

  function currentPair() {
    return window.CW_PAIRS[idx];
  }

  function render() {
    const pairs = window.CW_PAIRS || [];
    if (!pairs.length) {
      document.getElementById("status").textContent = "No pairs loaded.";
      document.getElementById("status").classList.add("error");
      return;
    }
    const p = pairs[idx];
    document.getElementById("leftText").textContent = p.left || "";
    document.getElementById("rightText").textContent = p.right || "";
    document.getElementById("pairMeta").textContent = "Pair " + p.pair_id + " of " + pairs.length + " · task_id " + p.task_id;
    const votes = loadVotes();
    const c = votes[String(p.pair_id)];
    const pct = ((idx + 1) / pairs.length) * 100;
    document.getElementById("progBar").style.width = pct + "%";
    const done = Object.keys(votes).length;
    document.getElementById("progText").textContent = "Progress: " + done + " / " + pairs.length + " voted (this rater)";
    const st = document.getElementById("status");
    st.classList.remove("error");
    const btnL = document.getElementById("btnLeft");
    const btnR = document.getElementById("btnRight");
    const btnS = document.getElementById("btnSimilar");
    const leftPanel = document.getElementById("leftPanel");
    const rightPanel = document.getElementById("rightPanel");
    const panelRow = document.getElementById("panelRow");
    btnL.classList.remove("selected");
    btnR.classList.remove("selected");
    btnS.classList.remove("selected");
    leftPanel.classList.remove("panel-chosen-a");
    rightPanel.classList.remove("panel-chosen-b");
    panelRow.classList.remove("similar-picked");
    btnL.setAttribute("aria-pressed", "false");
    btnR.setAttribute("aria-pressed", "false");
    btnS.setAttribute("aria-pressed", "false");
    if (c === 1) {
      st.textContent = "Your vote: A (left) is better.";
      btnL.classList.add("selected");
      btnL.setAttribute("aria-pressed", "true");
      leftPanel.classList.add("panel-chosen-a");
      btnL.textContent = "✓  A is better  (your choice)";
      btnR.textContent = "B is better";
      btnS.textContent = "Similarly coherent";
    } else if (c === 2) {
      st.textContent = "Your vote: B (right) is better.";
      btnR.classList.add("selected");
      btnR.setAttribute("aria-pressed", "true");
      rightPanel.classList.add("panel-chosen-b");
      btnR.textContent = "✓  B is better  (your choice)";
      btnL.textContent = "A is better";
      btnS.textContent = "Similarly coherent";
    } else if (c === 3) {
      st.textContent = "Your vote: similarly coherent (about the same quality).";
      btnS.classList.add("selected");
      btnS.setAttribute("aria-pressed", "true");
      panelRow.classList.add("similar-picked");
      btnL.textContent = "A is better";
      btnR.textContent = "B is better";
      btnS.textContent = "✓  Similarly coherent  (your choice)";
    } else {
      st.textContent = "Choose A, B, or Similarly coherent.";
      btnL.textContent = "A is better";
      btnR.textContent = "B is better";
      btnS.textContent = "Similarly coherent";
    }
  }

  function setVote(choice) {
    const p = currentPair();
    if (!p) return;
    const votes = loadVotes();
    votes[String(p.pair_id)] = choice;
    saveVotes(votes);
    render();
  }

  function clearVote() {
    const p = currentPair();
    if (!p) return;
    const votes = loadVotes();
    delete votes[String(p.pair_id)];
    saveVotes(votes);
    render();
  }

  function exportJson() {
    const votes = loadVotes();
    const rater = document.getElementById("raterId").value;
    const payload = {
      format: "cw_pairwise_export_v2",
      rater_id: parseInt(rater, 10),
      votes: votes,
      vote_legend: { "1": "left better", "2": "right better", "3": "similarly coherent" }
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "rater_" + rater + "_votes.json";
    a.click();
    URL.revokeObjectURL(a.href);
  }

  document.getElementById("raterId").addEventListener("change", () => { idx = 0; render(); });
  document.getElementById("btnLeft").addEventListener("click", () => setVote(1));
  document.getElementById("btnRight").addEventListener("click", () => setVote(2));
  document.getElementById("btnSimilar").addEventListener("click", () => setVote(3));
  document.getElementById("btnClear").addEventListener("click", clearVote);
  document.getElementById("btnPrev").addEventListener("click", () => {
    idx = Math.max(0, idx - 1);
    render();
  });
  document.getElementById("btnNext").addEventListener("click", () => {
    idx = Math.min(pairCount() - 1, idx + 1);
    render();
  });
  document.getElementById("btnFirstMissing").addEventListener("click", () => {
    const votes = loadVotes();
    const pairs = window.CW_PAIRS || [];
    for (let i = 0; i < pairs.length; i++) {
      if (votes[String(pairs[i].pair_id)] === undefined) {
        idx = i;
        render();
        return;
      }
    }
    document.getElementById("status").textContent = "All pairs have a vote.";
  });
  document.getElementById("btnExport").addEventListener("click", exportJson);
  document.getElementById("btnSave").addEventListener("click", () => {
    alert("Votes auto-save in this browser. Use Export to share your file with the organizer.");
  });
  document.getElementById("importFile").addEventListener("change", (ev) => {
    const f = ev.target.files[0];
    if (!f) return;
    const r = new FileReader();
    r.onload = () => {
      try {
        const data = JSON.parse(r.result);
        const votes = data.votes || data;
        if (typeof votes !== "object") throw new Error("bad format");
        localStorage.setItem(storageKey(), JSON.stringify(votes));
        if (data.rater_id) document.getElementById("raterId").value = String(data.rater_id);
        render();
      } catch (e) {
        alert("Import failed: " + e.message);
      }
    };
    r.readAsText(f);
  });
  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
    if (e.key === "1") setVote(1);
    if (e.key === "2") setVote(2);
    if (e.key === "3") setVote(3);
  });

  render();
})();
  </script>
</body>
</html>
"""


def cmd_generate(args: argparse.Namespace) -> None:
    results = _load_results(args.input)
    index = _pair_index(results, args.method_a, args.method_b)
    if not index:
        raise SystemExit(f"No overlapping task ids between {args.method_a} and {args.method_b}.")

    task_ids = list(index.keys())
    # Local Random instance (not the global) so this script doesn't perturb
    # any other randomness in the parent process and reruns with the same
    # seed produce the exact same pair set + L/R assignments.
    rnd = random.Random(args.seed)
    rnd.shuffle(task_ids)
    task_ids = task_ids[: args.n_pairs]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    public_pairs: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []

    for i, tid in enumerate(task_ids):
        ea, eb = index[tid]
        pa = ea.get("passage", "")
        pb = eb.get("passage", "")
        if rnd.random() < 0.5:
            left_m, right_m = args.method_a, args.method_b
            left_p, right_p = pa, pb
        else:
            left_m, right_m = args.method_b, args.method_a
            left_p, right_p = pb, pa

        pair_id = i + 1
        public_pairs.append(
            {
                "pair_id": pair_id,
                "task_id": tid,
                "left": left_p,
                "right": right_p,
            }
        )
        key_rows.append(
            {
                "pair_id": pair_id,
                "task_id": tid,
                "left_method": left_m,
                "right_method": right_m,
                "method_a": args.method_a,
                "method_b": args.method_b,
                "sentences": ea.get("sentences", []),
            }
        )

    # Pairs are embedded as a JS file (not fetched) so the page works from
    # `file://` without CORS. Only the *public* fields go here — left_method /
    # right_method live in pairs_key.json and must stay hidden from raters.
    embed_path = out_dir / "pairs_embed.js"
    embed_path.write_text(
        "window.CW_PAIRS = " + json.dumps(public_pairs, ensure_ascii=False) + ";\n",
        encoding="utf-8",
    )
    (out_dir / "pairs_key.json").write_text(json.dumps(key_rows, indent=2), encoding="utf-8")
    (out_dir / "index.html").write_text(HTML_TEMPLATE, encoding="utf-8")
    readme = out_dir / "README_INTERACTIVE.md"
    readme.write_text(
        f"""# Interactive pairwise judging

1. Each rater opens `index.html` in a **desktop browser** (Chrome / Firefox / Safari).
   - Same folder must contain `pairs_embed.js` (do not rename).
2. Select **Rater ID** (1, 2, or 3). For each pair choose **A better**, **B better**, or **Similarly coherent** (votes 1 / 2 / 3). Export when finished.
3. Send exported files to the organizer.
4. Merge + κ:

```bash
python3 code/tot/tasks/creative-writing/human_pairwise_interactive.py merge \\
  --key {readme.parent / "pairs_key.json"} \\
  --rater-exports rater_1_votes.json rater_2_votes.json rater_3_votes.json \\
  --out-csv merged_ratings.csv

python3 code/tot/tasks/creative-writing/human_pairwise_eval.py aggregate \\
  --ratings merged_ratings.csv \\
  --key pairs_key.json
```

Blind protocol: raters never see `pairs_key.json` (maps A/B to `{args.method_a}` / `{args.method_b}`).

Generated **{len(public_pairs)}** pairs.
""",
        encoding="utf-8",
    )

    print(f"Wrote {out_dir / 'index.html'}")
    print(f"Wrote {embed_path}")
    print(f"Wrote {out_dir / 'pairs_key.json'}")
    print(f"Wrote {readme}")
    print(f"Pairs: {len(public_pairs)}")


def cmd_merge(args: argparse.Namespace) -> None:
    # Output CSV must match the schema `human_pairwise_eval.py aggregate` expects:
    # one row per pair_id, rater_{1,2,3}_choice columns, blanks for missing votes.
    # That contract is what lets us share the aggregator across both UIs.
    key_path = Path(args.key)
    key_rows = json.loads(key_path.read_text(encoding="utf-8"))
    pair_ids = [int(r["pair_id"]) for r in key_rows]
    max_id = max(pair_ids)

    rater_votes: dict[int, dict[str, int]] = {}
    for path in args.rater_exports:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        rid = int(data.get("rater_id", 0))
        votes = data.get("votes")
        if not isinstance(votes, dict) or rid < 1:
            raise SystemExit(f"Bad export file: {path}")
        # Validate every vote up-front: bad exports should fail loudly here
        # rather than producing a CSV that silently breaks the aggregator.
        norm: dict[str, int] = {}
        for k, v in votes.items():
            vv = int(v)
            if vv not in (1, 2, 3):
                raise SystemExit(f"Invalid vote {k}:{v} in {path} (expected 1=left, 2=right, 3=similar)")
            norm[str(k)] = vv
        rater_votes[rid] = norm

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "pair_id",
        "task_id",
        "rater_1_choice",
        "rater_2_choice",
        "rater_3_choice",
    ]
    id_to_task = {int(r["pair_id"]): r["task_id"] for r in key_rows}

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for pid in range(1, max_id + 1):
            row = {
                "pair_id": pid,
                "task_id": id_to_task.get(pid, ""),
                "rater_1_choice": rater_votes.get(1, {}).get(str(pid), ""),
                "rater_2_choice": rater_votes.get(2, {}).get(str(pid), ""),
                "rater_3_choice": rater_votes.get(3, {}).get(str(pid), ""),
            }
            w.writerow(row)

    print(f"Wrote {out_csv} ({max_id} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive HTML pairwise human judging.")
    sub = parser.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="Write index.html + pairs_embed.js + pairs_key.json")
    g.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    g.add_argument("--method-a", default="tot")
    g.add_argument("--method-b", default="tot_astar")
    g.add_argument("--n-pairs", type=int, default=100)
    g.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results" / "creative-writing" / "human_eval" / "interactive",
    )
    g.add_argument("--seed", type=int, default=42)
    g.set_defaults(func=cmd_generate)

    m = sub.add_parser("merge", help="Merge rater JSON exports into CSV for aggregate.")
    m.add_argument("--key", type=Path, required=True)
    m.add_argument("--rater-exports", nargs="+", required=True, help="Exported JSON files (rater_*_votes.json)")
    m.add_argument("--out-csv", type=Path, required=True)
    m.set_defaults(func=cmd_merge)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
