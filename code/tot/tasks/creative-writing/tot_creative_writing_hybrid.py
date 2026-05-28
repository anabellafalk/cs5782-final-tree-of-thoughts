#!/usr/bin/env python3
"""
Hybrid ToT runner for creative-writing.

The `hybrid_tot` method swaps the plan-selection vote for a human pick, then
runs the rest of the ToT pipeline unchanged. We also run a *silent* LLM vote
on the same plans and log it alongside the human choice so we can measure
human↔LLM agreement after the fact without affecting the produced passage.

Two UIs for the human selection:
- CLI prompt (default; works over SSH/tmux).
- A small local HTTP server with a browser form (`--plan-ui html`). The server
  is bound to 127.0.0.1 with a per-process random token so the page can't be
  read by other local users without inspecting this process.
"""

from __future__ import annotations

import argparse
import html
import json
import secrets
import threading
import webbrowser
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs

import tot_creative_writing as cw
from experiment_metrics import AxesConfig, TaskMetrics, append_metrics_jsonl, build_method_summary, print_method_summary_table


@dataclass
class SolveResult:
    passage: str
    score: float
    score_std: float
    score_parse_failures: int
    score_attempts: int
    vote_plan: Optional[object] = None
    vote_passage: Optional[object] = None
    human_plan_choice: Optional[int] = None
    llm_plan_choice: Optional[int] = None
    human_llm_agree: Optional[bool] = None


def _coerce_result(base: cw.SolveResult) -> SolveResult:
    # The local SolveResult has extra human/LLM-choice fields. This shim lets
    # us reuse cw.* solvers unchanged when running them through this file's
    # `run_method` loop (so io / cot / tot / etc. all share one writer).
    return SolveResult(
        passage=base.passage,
        score=base.score,
        score_std=base.score_std,
        score_parse_failures=base.score_parse_failures,
        score_attempts=base.score_attempts,
        vote_plan=base.vote_plan,
        vote_passage=base.vote_passage,
    )


def _human_select_plan(task_idx_1: int, total_tasks: int, sentences: list[str], plans: list[str]) -> int:
    k = len(plans)
    while True:
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"TASK [{task_idx_1}/{total_tasks}] — Human Plan Selection")
        print("Required ending sentences:")
        print(f"  1. {sentences[0]}")
        print(f"  2. {sentences[1]}")
        print(f"  3. {sentences[2]}")
        print(f"  4. {sentences[3]}")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        for i, plan in enumerate(plans, start=1):
            print(f"Plan {i}:")
            print(plan)
            if i < k:
                print("\n---")
            else:
                print()
        raw = input(f"Enter the number of the best plan (1-{k}): ").strip()
        try:
            choice = int(raw)
        except ValueError:
            print(f"Invalid input '{raw}'. Please enter an integer from 1 to {k}.")
            continue
        if 1 <= choice <= k:
            return choice
        print(f"Out of range: {choice}. Please enter an integer from 1 to {k}.")


class HtmlPlanSelector:
    """Local one-rater HTML form bound to 127.0.0.1.

    The token gate isn't authentication so much as a CSRF/wrong-window guard:
    it ensures the browser tab we opened is the one submitting (other local
    apps shouldn't be poking at this port). We use a threading event so the
    solver thread blocks cleanly until the browser POSTs a choice.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True):
        self.host = host
        self.port = port
        self.open_browser = open_browser
        self._token = secrets.token_urlsafe(16)
        self._lock = threading.Lock()
        # _ready toggles per task: cleared before each `choose`, set when a
        # valid POST arrives. Lets the solver thread `.wait()` without polling.
        self._ready = threading.Event()
        self._selection: Optional[int] = None
        self._current: Optional[dict] = None
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/?token={self._token}"

    def start(self) -> None:
        selector = self

        class Handler(BaseHTTPRequestHandler):
            def _write(self, body: str, status: int = 200) -> None:
                self.send_response(status)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(body.encode("utf-8"))

            def do_GET(self) -> None:  # noqa: N802
                if f"token={selector._token}" not in self.path:
                    self._write("<h3>Unauthorized</h3>", status=403)
                    return
                self._write(selector.render_page())

            def do_POST(self) -> None:  # noqa: N802
                if f"token={selector._token}" not in self.path:
                    self._write("<h3>Unauthorized</h3>", status=403)
                    return
                try:
                    n = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    n = 0
                body = self.rfile.read(n).decode("utf-8", errors="ignore")
                form = parse_qs(body)
                raw = (form.get("choice") or [""])[0].strip()
                with selector._lock:
                    current = selector._current
                    k = int(current["k"]) if current else 0
                try:
                    choice = int(raw)
                except ValueError:
                    self._write(selector.render_page(error=f"Invalid input '{html.escape(raw)}'. Enter 1-{k}."))
                    return
                if not (1 <= choice <= k):
                    self._write(selector.render_page(error=f"Out of range: {choice}. Enter 1-{k}."))
                    return
                with selector._lock:
                    selector._selection = choice
                selector._ready.set()
                self._write(selector.render_page(message=f"Saved selection: Plan {choice}. You can return to terminal."))

            def log_message(self, format: str, *args) -> None:  # noqa: A003
                # Suppress BaseHTTPRequestHandler's default stderr access log —
                # it would interleave with the solver's progress output.
                return

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        print(f"[hybrid_tot] HTML plan picker: {self.url}")
        if self.open_browser:
            try:
                webbrowser.open(self.url)
            except Exception:
                pass

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def choose(self, task_idx_1: int, total_tasks: int, sentences: list[str], plans: list[str]) -> int:
        with self._lock:
            self._current = {
                "task_idx_1": task_idx_1,
                "total_tasks": total_tasks,
                "sentences": list(sentences),
                "plans": list(plans),
                "k": len(plans),
            }
            self._selection = None
        self._ready.clear()
        print(f"[hybrid_tot] Waiting for browser selection at {self.url}")
        self._ready.wait()
        with self._lock:
            selection = self._selection
        if selection is None:
            raise RuntimeError("Missing HTML selection after wait.")
        return selection

    def render_page(self, error: str = "", message: str = "") -> str:
        with self._lock:
            current = self._current
        if not current:
            body = "<p>Waiting for the next task...</p>"
        else:
            plans_html = []
            for i, p in enumerate(current["plans"], start=1):
                plans_html.append(
                    f"<section class='plan'><h3>Plan {i}</h3><pre>{html.escape(p)}</pre></section>"
                )
            body = f"""
            <h2>TASK [{current['task_idx_1']}/{current['total_tasks']}] — Human Plan Selection</h2>
            <h3>Required ending sentences</h3>
            <ol>
              <li>{html.escape(current['sentences'][0])}</li>
              <li>{html.escape(current['sentences'][1])}</li>
              <li>{html.escape(current['sentences'][2])}</li>
              <li>{html.escape(current['sentences'][3])}</li>
            </ol>
            {''.join(plans_html)}
            <form method="post" action="/submit?token={self._token}">
              <label for="choice"><b>Best plan number (1-{current['k']}):</b></label>
              <input type="number" id="choice" name="choice" min="1" max="{current['k']}" required />
              <button type="submit">Submit Choice</button>
            </form>
            """
        alert = ""
        if error:
            alert = f"<p class='error'>{error}</p>"
        elif message:
            alert = f"<p class='ok'>{message}</p>"
        return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Hybrid ToT Plan Picker</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 24px; background: #111; color: #eee; }}
    pre {{ white-space: pre-wrap; background: #1a1a1a; border: 1px solid #333; padding: 12px; border-radius: 8px; }}
    .plan {{ margin: 16px 0; }}
    input, button {{ font-size: 16px; padding: 8px; }}
    button {{ margin-left: 8px; }}
    .error {{ color: #ff7070; }}
    .ok {{ color: #9effa5; }}
  </style>
</head>
<body>
  {alert}
  {body}
</body>
</html>"""


_HTML_SELECTOR: Optional[HtmlPlanSelector] = None


def hybrid_tot_solve(x: dict, task_idx_1: int, total_tasks: int) -> SolveResult:
    s1, s2, s3, s4 = x["sentences"]
    plans = cw.sample(cw.PLAN_PROMPT.format(s1=s1, s2=s2, s3=s3, s4=s4), cw.K)

    # The CLI fallback runs when no HTML server was started — keeps this method
    # usable in headless / SSH environments without any extra flags.
    if _HTML_SELECTOR is not None:
        human_choice = _HTML_SELECTOR.choose(task_idx_1, total_tasks, [s1, s2, s3, s4], plans)
    else:
        human_choice = _human_select_plan(task_idx_1, total_tasks, [s1, s2, s3, s4], plans)
    human_plan = plans[human_choice - 1]

    # Silent LLM vote: we discard its winner pick (the human's choice is used
    # downstream) but log the full tally so we can compute agreement rates and
    # keep vote_entropy comparable to baseline ToT.
    _, llm_plan_vote = cw.vote_best(plans, "plan")
    llm_choice = llm_plan_vote.winner_index_1based

    passages = cw.sample(
        cw.PASSAGE_PROMPT.format(plan=human_plan, s1=s1, s2=s2, s3=s3, s4=s4),
        cw.K,
    )
    best_passage, passage_vote = cw.vote_best(passages, "passage")
    score_details = cw.score_passage_details(best_passage)
    return SolveResult(
        passage=best_passage,
        score=score_details.mean,
        score_std=score_details.std,
        score_parse_failures=score_details.parse_failures,
        score_attempts=score_details.attempts,
        vote_plan=llm_plan_vote,
        vote_passage=passage_vote,
        human_plan_choice=human_choice,
        llm_plan_choice=llm_choice,
        human_llm_agree=(human_choice == llm_choice),
    )


def run_method(method: str, data: list[dict], out_file: Path, out_label: str, axes: AxesConfig) -> dict:
    results: dict = {}
    if out_file.exists():
        with open(out_file) as f:
            results = json.load(f)
    if method not in results:
        results[method] = []

    start = len(results[method])
    solver = SOLVERS[method]
    n = len(data)
    method_metrics: list[TaskMetrics] = []

    print(f"\n-- {method.upper()} ({start}/{n} already done) --")
    for i, x in enumerate(data[start:], start=start):
        print(f"  [{i + 1:3d}/{n}] ", end="", flush=True)
        perf_before = cw._perf_snapshot()

        # hybrid_tot has an extra (task_idx, total) signature so the UI can
        # show "TASK 12/100" while the human reads plans. Other solvers stay
        # on the simpler (x) signature shared with the main runner.
        if method == "hybrid_tot":
            solve = solver(x, i + 1, n)
        else:
            solve = _coerce_result(solver(x))

        perf_after = cw._perf_snapshot()
        perf_delta = cw._perf_delta(perf_before, perf_after)

        results[method].append(
            {
                "id": x["id"],
                "sentences": x["sentences"],
                "passage": solve.passage,
                "score": solve.score,
                "human_plan_choice": solve.human_plan_choice,
                "llm_plan_choice": solve.llm_plan_choice,
                "human_llm_agree": solve.human_llm_agree,
            }
        )
        task_metrics = TaskMetrics(
            out_file=out_label,
            method=method,
            task_id=x["id"],
            mean_score=solve.score,
            score_std=solve.score_std,
            score_parse_failures=solve.score_parse_failures,
            score_attempts=solve.score_attempts,
            vote_entropy_plan=solve.vote_plan.entropy if solve.vote_plan else None,
            vote_entropy_passage=solve.vote_passage.entropy if solve.vote_passage else None,
            vote_parse_failures_plan=solve.vote_plan.parse_failures if solve.vote_plan else 0,
            vote_parse_failures_passage=solve.vote_passage.parse_failures if solve.vote_passage else 0,
            vote_attempts_plan=solve.vote_plan.attempts if solve.vote_plan else 0,
            vote_attempts_passage=solve.vote_passage.attempts if solve.vote_passage else 0,
            api_calls=perf_delta["api_calls"],
            prompt_tokens=perf_delta["prompt_tokens"],
            completion_tokens=perf_delta["completion_tokens"],
            total_cost_usd=perf_delta["total_cost_usd"],
            human_plan_choice=solve.human_plan_choice,
            llm_plan_choice=solve.llm_plan_choice,
            human_llm_agree=solve.human_llm_agree,
        )
        method_metrics.append(task_metrics)
        append_metrics_jsonl(cw.METRICS_LOG_PATH, axes, task_metrics)

        print(f"score={solve.score:.2f}")
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)

    scores = [e["score"] for e in results[method]]
    avg = sum(scores) / len(scores)
    print(f"  -> {method} average coherency: {avg:.2f} (n={len(scores)})")
    cumulative_rows = cw._load_cumulative_method_metrics(method, axes, out_label)
    method_summary = build_method_summary(method, cumulative_rows or method_metrics, axes)
    print_method_summary_table(method_summary)
    return method_summary


SOLVERS = {
    "io": cw.io_solve,
    "cot": cw.cot_solve,
    "tot": cw.tot_solve,
    "hybrid_tot": hybrid_tot_solve,
    "io_refine": cw.io_refine_solve,
    "tot_refine": cw.tot_refine_solve,
    "tot_astar": cw.tot_astar_solve,
}


def main() -> None:
    global _HTML_SELECTOR
    cw._require_api_key()
    parser = argparse.ArgumentParser(description="ToT creative writing experiment (with hybrid_tot)")
    parser.add_argument(
        "--method",
        nargs="+",
        default=["all"],
        choices=list(SOLVERS.keys()) + ["all"],
        metavar="NAME",
        help="Which solver(s) to run. Use 'all' alone, or one/more named methods.",
    )
    parser.add_argument("--n", type=int, default=100, help="Number of inputs")
    parser.add_argument("--out", default="creative_writing_results.json", help="Results filename")
    parser.add_argument("--k", type=int, default=cw.K, help="Number of candidates for ToT voting steps")
    parser.add_argument("--n-votes", type=int, default=cw.N_VOTES, help="Number of votes per vote prompt")
    parser.add_argument(
        "--score-prompt-variant",
        choices=["paper", "definition", "criteria"],
        default=cw.SCORE_PROMPT_VARIANT,
        help="Scoring prompt template variant label",
    )
    parser.add_argument(
        "--vote-prompt-variant",
        choices=["paper", "criteria"],
        default=cw.VOTE_PROMPT_VARIANT,
        help="Voting prompt template variant label",
    )
    parser.add_argument(
        "--print-score-thoughts",
        action="store_true",
        help="Print scorer rationale text for each judge call.",
    )
    parser.add_argument(
        "--plan-ui",
        choices=["cli", "html"],
        default="cli",
        help="Human plan selection UI for hybrid_tot.",
    )
    parser.add_argument(
        "--ui-port",
        type=int,
        default=8765,
        help="Port for --plan-ui html local server.",
    )
    parser.add_argument(
        "--no-open-browser",
        action="store_true",
        help="With --plan-ui html, do not auto-open browser.",
    )
    args = parser.parse_args()

    cw.K = args.k
    cw.N_VOTES = args.n_votes
    cw.SCORE_PROMPT_VARIANT = args.score_prompt_variant
    cw.VOTE_PROMPT_VARIANT = args.vote_prompt_variant
    cw.PRINT_SCORE_THOUGHTS = args.print_score_thoughts

    cw.RESULTS_PATH.mkdir(exist_ok=True)
    out_file = cw.RESULTS_PATH / args.out
    data = cw.load_data(args.n)
    axes = AxesConfig(
        score_model_type=cw.SCORE_MODEL,
        k=cw.K,
        n_votes=cw.N_VOTES,
        score_prompt_variant=cw.SCORE_PROMPT_VARIANT,
    )

    requested = list(args.method)
    if requested == ["all"]:
        methods = list(SOLVERS.keys())
    elif "all" in requested:
        parser.error("Use '--method all' alone, or list specific methods.")
    else:
        methods = list(dict.fromkeys(requested))

    # Only spin up the HTTP server when it'll actually be used; otherwise
    # `hybrid_tot` falls back to the CLI prompt and pure-LLM methods skip it
    # entirely. `try/finally` guarantees we close the socket on Ctrl-C.
    if args.plan_ui == "html" and "hybrid_tot" in methods:
        _HTML_SELECTOR = HtmlPlanSelector(port=args.ui_port, open_browser=not args.no_open_browser)
        _HTML_SELECTOR.start()

    summary_rows = []
    try:
        for m in methods:
            summary_rows.append(run_method(m, data, out_file, args.out, axes))
    finally:
        if _HTML_SELECTOR is not None:
            _HTML_SELECTOR.stop()
            _HTML_SELECTOR = None
    cw.print_summary(out_file)
    with open(cw.SUMMARY_TABLE_PATH, "w") as f:
        json.dump(summary_rows, f, indent=2)

    total_time = cw._PERF["api_time_s"] + cw._PERF["rate_sleep_s"]
    cumulative = cw._load_cumulative_totals(methods, axes, args.out)
    print("-- Runtime Profile ----------------------------------------")
    print(f"  API calls (run):        {cw._PERF['api_calls']}")
    print(f"  API time (s, run):      {cw._PERF['api_time_s']:.2f}")
    print(f"  Rate-limit (s, run):    {cw._PERF['rate_sleep_s']:.2f}")
    print(f"  Tracked total (s, run): {total_time:.2f}")
    print(f"  Prompt tokens (run):    {cw._PERF['prompt_tokens']}")
    print(f"  Completion tokens (run):{cw._PERF['completion_tokens']}")
    print(f"  Total cost (usd, run):  {cw._PERF['total_cost_usd']:.6f}")
    print("  --")
    print(f"  API calls (cumulative):        {cumulative['api_calls']}")
    print(f"  Prompt tokens (cumulative):    {cumulative['prompt_tokens']}")
    print(f"  Completion tokens (cumulative):{cumulative['completion_tokens']}")
    print(f"  Total cost (usd, cumulative):  {cumulative['total_cost_usd']:.6f}")


if __name__ == "__main__":
    main()
