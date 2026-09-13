# vllm-benchmark MCP server

## Description
Wraps this repo's own async streaming benchmark harness (`scripts/run_bench.py`)
and analysis logic (`scripts/analyze.py`) as MCP tools, so an agent (or a
human, via `mcp dev server.py`) can launch a benchmark run, read back a
percentile summary, and check its cost breakdown — without shelling out by
hand each time.

## Tools
- `list_runs()` — every run already on disk, grouped by variant, with which
  files exist and when each was last touched.
- `get_run_summary(variants)` — p50/p95/p99 TTFT, aggregate throughput, and
  warmup-cluster exclusion for one or more variants (reuses
  `analyze.summarize()`'s real logic instead of re-deriving percentile math).
- `get_run_cost(variant)` — the `$/1k tokens` / tokens-per-dollar breakdown,
  if the run was recorded with a hardware cost.
- `run_benchmark(...)` — runs `run_bench.py` against a live vLLM server and
  returns its summary + cost inline.
- `get_trace_history(limit)` — the most recent tool-call traces (newest
  first): every tool's duration/success/error, plus `run_benchmark`'s own
  per-stage breakdown (`smoke_test` / `subprocess_run` / `summarize`).

## Tracing
Every tool call is traced to `traces.jsonl` (gitignored, operational data —
see `tracing.py`): a `@traced` decorator on all five tools records a
uniform duration/success/error line, and `run_benchmark` additionally uses
a `StageTimer` to record which of its three real internal phases actually
took the time. Deliberately not a full OpenTelemetry/Jaeger stack — just
enough structure to answer "trace every step, show a latency breakdown by
stage" for a tool-calling agent.

## Why this design
Modeled on the same lifecycle-guardrail pattern as an internal Locust
load-test MCP server built at a previous job: validate before committing to
the real thing, and never silently clobber existing results. Applied here to
a from-scratch personal tool with no proprietary internals.

## Core rules
1. **Smoke test before a real run.** `run_benchmark` fires one
   `max_tokens=1` completion against the target server first; if that fails,
   it refuses to launch the full run rather than burning the whole duration
   on a server that was never going to answer.
2. **Never silently overwrite a variant.** If `runs/<variant>.csv` already
   exists, `run_benchmark` returns an error instead of clobbering it.
3. **Reuse, don't re-derive.** Percentile/warmup-cluster-exclusion math
   always goes through `scripts/analyze.py`'s real `summarize()`, via a thin
   subprocess helper (`summarize_helper.py`) run under the harness's own
   Python 3.9 venv, since `analyze.py` needs pandas/matplotlib that this
   server's own (newer) Python environment doesn't carry.
4. **Bounded execution.** `run_benchmark` runs under a 900s subprocess
   timeout rather than hanging forever on a stuck server.

## Setup
```bash
cd mcp_server
/opt/homebrew/bin/python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/mcp dev server.py   # opens the MCP Inspector to call tools by hand
```

`run_benchmark` and `get_run_summary`/`get_run_cost` shell out to the
harness's *existing* venv (`../venv/bin/python3`) to actually run
`run_bench.py`/`analyze.py` — this server's own `.venv` only needs the `mcp`
and `httpx` packages listed above.
