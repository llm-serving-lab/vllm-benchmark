"""
vllm-benchmark MCP server
=========================
Wraps this repo's own async streaming benchmark harness (scripts/run_bench.py)
and analysis logic (scripts/analyze.py) as MCP tools, so an agent (or a human,
via `mcp dev server.py`) can launch a benchmark run and read back a real
percentile summary and cost breakdown without shelling out by hand.

Deliberately modeled on the same lifecycle-guardrail pattern as an internal
Locust load-test MCP server built at a previous job (smoke-test before
committing to a real run; never silently clobber an existing result) --
applied here to a from-scratch personal tool with no proprietary internals.

Every tool call is traced (see tracing.py): a uniform duration/success
record for all five tools, plus a per-stage latency breakdown specifically
for run_benchmark's three real internal phases (smoke test, subprocess run,
summarize). `get_trace_history` reads that log back.
See mcp_server/README.md for the full design writeup.
"""
import json
import subprocess
import time
from pathlib import Path
from typing import Optional

import httpx
from mcp.server.mcpserver import MCPServer

from tracing import StageTimer, read_trace_history, traced

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "runs"
RUN_BENCH = REPO_ROOT / "scripts" / "run_bench.py"
SUMMARIZE_HELPER = Path(__file__).resolve().parent / "summarize_helper.py"
# analyze.py needs pandas + matplotlib, which live in the harness's own venv
# (a separate, older Python than this MCP server runs under) -- shell out to
# it rather than duplicating percentile/warmup-exclusion math here.
HARNESS_PYTHON = REPO_ROOT / "venv" / "bin" / "python3"

mcp = MCPServer("vllm-benchmark")


def _summarize(variants: list[str]) -> dict:
    paths = [str(RUNS_DIR / f"{v}.csv") for v in variants]
    proc = subprocess.run(
        [str(HARNESS_PYTHON), str(SUMMARIZE_HELPER), *paths],
        capture_output=True, text=True, timeout=60,
    )
    if proc.returncode != 0:
        return {"error": "summarize_helper.py failed", "stderr": proc.stderr[-2000:]}
    return json.loads(proc.stdout)


@mcp.tool()
@traced
def list_runs() -> list[dict]:
    """List benchmark runs already on disk, grouped by variant name, with which files exist and when each was last modified."""
    variants: dict[str, dict] = {}
    for f in sorted(RUNS_DIR.glob("*")):
        if f.name.endswith("_cost.json"):
            variant, kind = f.name[: -len("_cost.json")], "cost"
        elif f.suffix == ".csv":
            variant, kind = f.stem, "csv"
        elif f.suffix == ".jsonl":
            variant, kind = f.stem, "jsonl"
        else:
            continue
        entry = variants.setdefault(variant, {
            "variant": variant, "has_csv": False, "has_jsonl": False,
            "has_cost": False, "modified": 0.0,
        })
        entry[f"has_{kind}"] = True
        entry["modified"] = max(entry["modified"], f.stat().st_mtime)
    return sorted(variants.values(), key=lambda r: r["modified"], reverse=True)


@mcp.tool()
@traced
def get_run_summary(variants: list[str]) -> dict:
    """Summarize one or more existing runs by variant name: p50/p95/p99 TTFT, aggregate throughput, warmup-cluster exclusion. Reuses scripts/analyze.py's real logic. Pass several variants to compare a saturation curve."""
    return _summarize(variants)


@mcp.tool()
@traced
def get_run_cost(variant: str) -> dict:
    """Return the cost_per_1k_tokens / tokens_per_dollar breakdown for a run, if it was recorded with a hardware cost."""
    p = RUNS_DIR / f"{variant}_cost.json"
    if not p.exists():
        return {"error": f"no cost data recorded for '{variant}' -- rerun with cost_hardware_usd set to enable it"}
    return json.loads(p.read_text())


@mcp.tool()
@traced
def run_benchmark(
    model: str,
    dataset: str = "datasets/prompts.jsonl",
    concurrency: int = 1,
    variant: Optional[str] = None,
    max_samples: Optional[int] = None,
    repeat: int = 1,
    base_url: str = "http://localhost:8000/v1",
    api_key: Optional[str] = None,
    cost_hardware_usd: Optional[float] = None,
) -> dict:
    """Run scripts/run_bench.py against a live vLLM server and return its summary.

    Refuses to launch the full run if a one-request smoke test against the
    server fails, and refuses to silently overwrite an existing variant --
    the same two guardrails (validate before committing, never clobber
    silently) as the Locust-portal MCP tool this design is modeled on.
    """
    timer = StageTimer("run_benchmark")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    timer.start("smoke_test")
    try:
        # 130s, not 15s: this project's own Phase 1 investigation documented a
        # genuine cold-start stall (MLX's async_eval blocking on a native
        # condition_variable::wait) on the *first* request after any fresh
        # server start, observed up to 121s in one recorded case. A truly
        # dead server (wrong port, nothing listening) fails almost instantly
        # with a connection error regardless of this value -- the long
        # timeout only matters for the real-but-slow case, not the dead one.
        resp = httpx.post(
            f"{base_url}/chat/completions",
            json={"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1},
            headers=headers,
            timeout=130.0,
        )
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        # The server answered but rejected the request -- a different failure
        # than "unreachable" (wrong api_key/model name, not a dead server).
        timer.finish(False, f"smoke_test_http_{e.response.status_code}")
        return {
            "error": f"smoke test got HTTP {e.response.status_code} -- server is reachable but rejected the request (check api_key / model name)",
            "detail": str(e),
        }
    except Exception as e:
        timer.finish(False, "smoke_test_unreachable")
        return {"error": "smoke test failed -- server not reachable, refusing to launch the full run", "detail": str(e)}

    if variant is None:
        variant = f"mcp-run-{int(time.time())}"
    elif (RUNS_DIR / f"{variant}.csv").exists():
        timer.variant = variant
        timer.finish(False, "variant_already_exists")
        return {"error": f"variant '{variant}' already has a run on disk -- pass a different name or delete it first"}
    timer.variant = variant

    dataset_path = REPO_ROOT / dataset
    if not dataset_path.exists():
        timer.finish(False, "dataset_not_found")
        return {"error": f"dataset not found: {dataset_path}"}

    cmd = [
        str(HARNESS_PYTHON), str(RUN_BENCH),
        "--base-url", base_url,
        "--model", model,
        "--dataset", str(dataset_path),
        "--concurrency", str(concurrency),
        "--variant", variant,
        "--repeat", str(repeat),
    ]
    if api_key is not None:
        cmd += ["--api-key", api_key]
    if max_samples is not None:
        cmd += ["--max-samples", str(max_samples)]
    if cost_hardware_usd is not None:
        cmd += ["--cost-hardware-usd", str(cost_hardware_usd)]

    timer.start("subprocess_run")
    try:
        # run_bench.py writes to a *relative* "runs" path (its own line 435),
        # resolved against whatever directory the process runs from -- not
        # relative to the script itself. Without an explicit cwd here, it
        # inherits wherever the MCP server was launched from (e.g.
        # mcp_server/, per the README's own setup instructions), silently
        # writing output to the wrong place. Pin it to REPO_ROOT so the
        # file always lands where _summarize()'s absolute RUNS_DIR expects it.
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900, cwd=str(REPO_ROOT))
    except subprocess.TimeoutExpired:
        timer.finish(False, "subprocess_timeout")
        return {"error": f"run_bench.py timed out after 900s for variant '{variant}'"}

    result = {"variant": variant, "exit_code": proc.returncode, "stdout_tail": proc.stdout[-2000:]}
    if proc.returncode != 0:
        result["error"] = "run_bench.py exited non-zero"
        result["stderr_tail"] = proc.stderr[-2000:]
        timer.finish(False, "run_bench_nonzero_exit")
        return result

    timer.start("summarize")
    if (RUNS_DIR / f"{variant}.csv").exists():
        result["summary"] = _summarize([variant]).get("summary")
    cost_path = RUNS_DIR / f"{variant}_cost.json"
    if cost_path.exists():
        result["cost"] = json.loads(cost_path.read_text())
    timer.finish(True)
    return result


@mcp.tool()
def get_trace_history(limit: int = 20) -> list[dict]:
    """Return the most recent tool-call traces (newest first): tool name, duration, success/error, and -- for run_benchmark -- a per-stage latency breakdown (smoke_test / subprocess_run / summarize)."""
    return read_trace_history(limit)


if __name__ == "__main__":
    mcp.run()
