#!/usr/bin/env python3
"""
run_bench.py
============
Async, streaming benchmark harness for a local vLLM OpenAI-compatible server.

Unlike a non-streaming harness (which can only measure total request time),
this script times each individual SSE chunk as it arrives, so it can report:

    ttft_ms   — time to first token   (time to first streamed chunk)
    tpot_ms   — time per output token (mean gap between chunks after the first)
    itl_ms    — inter-token latency list (per-chunk gaps; tpot is their mean)

It also polls vLLM's Prometheus /metrics endpoint immediately after each
request completes, to capture scheduler/KV-cache state at that point in time:

    vllm_num_requests_running
    vllm_num_requests_waiting
    vllm_kv_cache_usage_perc

Note (Apple Silicon / Metal): vLLM does not expose a GPU compute utilization
percentage metric the way DCGM does on NVIDIA GPUs. On this hardware, GPU
utilization has to come from a separate `powermetrics` / asitop sample taken
alongside the run — that's a documented gap, not a bug in this script. See
README.md "Known limitations" for the plan to close it.

Usage:
    python scripts/run_bench.py \\
        --base-url http://localhost:8000/v1 \\
        --api-key local-vllm-key \\
        --model mlx-community/Mistral-7B-Instruct-v0.3-4bit \\
        --dataset datasets/prompts.jsonl \\
        --concurrency 1 \\
        --variant concurrency1

For saturation-curve runs at higher concurrency, add --repeat to cycle the
dataset enough times to get a meaningful sample size for p95/p99 (a 5-prompt
dataset run once gives only 5 data points per level -- not enough to compute
a real p99):

    python scripts/run_bench.py \\
        ... --concurrency 25 --repeat 20 --variant concurrency25
"""

import argparse
import asyncio
import csv
import json
import re
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Optional

import httpx


# -----------------------------
# Data model
# -----------------------------

@dataclass
class RequestResult:
    request_id: str
    prompt: str
    ttft_ms: Optional[float]
    tpot_ms: Optional[float]
    itl_ms_list: List[float] = field(default_factory=list)
    total_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    tokens_per_sec: float = 0.0
    error: Optional[str] = None
    # scheduler / KV-cache snapshot, taken right after this request completes
    num_requests_running: Optional[float] = None
    num_requests_waiting: Optional[float] = None
    kv_cache_usage_perc: Optional[float] = None
    # Diagnostic timing, added after an unexplained 36s TTFT outlier at
    # concurrency=1 (2026-08-02) that the vLLM server's own log did not
    # corroborate as real prefill time. queue_wait_ms measures time from
    # task creation (i.e. when run_benchmark builds the task list) to
    # semaphore acquisition -- if this is large, the delay is harness-side
    # scheduling/contention, not the server. connect_wait_ms measures
    # semaphore acquisition to the HTTP stream actually opening -- if this
    # is large, the delay is in connection setup, not server-side prefill.
    queue_wait_ms: Optional[float] = None
    connect_wait_ms: Optional[float] = None


# -----------------------------
# Prometheus /metrics scraping
# -----------------------------

# vLLM 0.26.x exposes metrics under the `vllm:` namespace in Prometheus text
# format, e.g.:
#   vllm:num_requests_running{model_name="..."} 1.0
#   vllm:num_requests_waiting{model_name="..."} 2.0
#   vllm:kv_cache_usage_perc{model_name="..."} 0.0512
METRIC_PATTERNS = {
    "num_requests_running": re.compile(r"^vllm:num_requests_running(\{[^}]*\})?\s+([0-9.eE+-]+)", re.MULTILINE),
    "num_requests_waiting": re.compile(r"^vllm:num_requests_waiting(\{[^}]*\})?\s+([0-9.eE+-]+)", re.MULTILINE),
    "kv_cache_usage_perc": re.compile(r"^vllm:kv_cache_usage_perc(\{[^}]*\})?\s+([0-9.eE+-]+)", re.MULTILINE),
}


async def scrape_metrics(client: httpx.AsyncClient, metrics_url: str) -> dict:
    """Fetch and parse the subset of vLLM Prometheus metrics we care about.

    Returns a dict with None values for any metric not found (older/newer
    vLLM versions do rename these — see README for the version note).
    """
    out = {k: None for k in METRIC_PATTERNS}
    try:
        resp = await client.get(metrics_url, timeout=5.0)
        resp.raise_for_status()
        text = resp.text
    except Exception:
        return out

    for key, pattern in METRIC_PATTERNS.items():
        match = pattern.search(text)
        if match:
            try:
                out[key] = float(match.group(2))
            except ValueError:
                pass
    return out


# -----------------------------
# Dataset loading
# -----------------------------

def load_dataset(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def expand_dataset(rows: List[dict], repeat: int) -> List[dict]:
    """Cycle through `rows` `repeat` times to reach a larger request volume.

    A 5-prompt dataset run once gives 5 samples per concurrency level, which
    is not enough to compute a meaningful p95/p99 (p99 of 5 points is just
    the max). --repeat cycles the same prompt set N times so a concurrency=25
    or 50 run has enough in-flight volume to make the percentiles meaningful.

    Each repeated copy gets a distinct request_id (e.g. "3" -> "3-r2" on the
    second pass) so results stay traceable back to which prompt and which
    pass produced them — original ids are left untouched on the first pass
    for backward compatibility with existing run outputs.
    """
    if repeat <= 1:
        return rows
    expanded = []
    for rep in range(1, repeat + 1):
        for row in rows:
            copy = dict(row)
            if rep > 1:
                copy["id"] = f"{row.get('id', '')}-r{rep}"
            expanded.append(copy)
    return expanded


# -----------------------------
# Single streaming request
# -----------------------------

async def run_one_request(
    client: httpx.AsyncClient,
    base_url: str,
    api_key: str,
    model: str,
    metrics_url: str,
    row: dict,
    created_at: Optional[float] = None,
) -> RequestResult:
    prompt_id = str(row.get("id", ""))
    prompt_text = row.get("prompt") or row.get("text") or ""

    result = RequestResult(
        request_id=prompt_id,
        prompt=prompt_text,
        ttft_ms=None,
        tpot_ms=None,
    )

    # Time from task creation to this point = time spent waiting on the
    # semaphore (i.e. queued behind other in-flight requests at this
    # concurrency level). Recorded even though at concurrency=1 it should
    # be ~0 for a solitary request -- if it isn't, that's the diagnostic
    # signal itself.
    acquired_at = time.perf_counter()
    if created_at is not None:
        result.queue_wait_ms = (acquired_at - created_at) * 1000.0

    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt_text}],
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    chunk_times: List[float] = []
    output_text_chunks: List[str] = []
    usage = None

    start = time.perf_counter()
    try:
        async with client.stream(
            "POST",
            f"{base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=120.0,
        ) as resp:
            connected_at = time.perf_counter()
            result.connect_wait_ms = (connected_at - start) * 1000.0
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                now = time.perf_counter()
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue

                if obj.get("usage"):
                    usage = obj["usage"]

                choices = obj.get("choices") or []
                if choices:
                    delta = choices[0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        chunk_times.append(now)
                        output_text_chunks.append(content)
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"
        result.total_ms = (time.perf_counter() - start) * 1000.0
        return result

    end = time.perf_counter()
    result.total_ms = (end - start) * 1000.0

    if chunk_times:
        result.ttft_ms = (chunk_times[0] - start) * 1000.0
        if len(chunk_times) > 1:
            gaps = [
                (chunk_times[i] - chunk_times[i - 1]) * 1000.0
                for i in range(1, len(chunk_times))
            ]
            result.itl_ms_list = gaps
            result.tpot_ms = sum(gaps) / len(gaps)
        else:
            result.itl_ms_list = []
            result.tpot_ms = None

    if usage:
        result.input_tokens = int(usage.get("prompt_tokens", 0))
        result.output_tokens = int(usage.get("completion_tokens", 0))
    else:
        # Fallback: approximate output tokens by chunk count if usage wasn't sent
        result.output_tokens = len(chunk_times)

    if result.total_ms > 0:
        result.tokens_per_sec = result.output_tokens / (result.total_ms / 1000.0)

    # Snapshot scheduler state right after this request completes.
    metrics = await scrape_metrics(client, metrics_url)
    result.num_requests_running = metrics["num_requests_running"]
    result.num_requests_waiting = metrics["num_requests_waiting"]
    result.kv_cache_usage_perc = metrics["kv_cache_usage_perc"]

    return result


# -----------------------------
# Concurrency-bounded run
# -----------------------------

async def run_benchmark(
    base_url: str,
    api_key: str,
    model: str,
    dataset_path: Path,
    concurrency: int,
    variant: str,
    max_samples: Optional[int] = None,
    repeat: int = 1,
) -> Path:
    rows = load_dataset(dataset_path)
    rows = expand_dataset(rows, repeat)
    if max_samples is not None:
        rows = rows[:max_samples]

    metrics_url = base_url.rsplit("/v1", 1)[0] + "/metrics"

    semaphore = asyncio.Semaphore(concurrency)
    results: List[RequestResult] = []

    async with httpx.AsyncClient() as client:

        async def bound_run(row, created_at):
            async with semaphore:
                return await run_one_request(
                    client, base_url, api_key, model, metrics_url, row,
                    created_at=created_at,
                )

        # created_at is stamped when the task list is built, i.e. before any
        # semaphore slot is available -- this is what makes queue_wait_ms a
        # true measure of harness-side queueing rather than 0 by construction.
        now = time.perf_counter()
        tasks = [bound_run(row, now) for row in rows]
        for coro in asyncio.as_completed(tasks):
            r = await coro
            results.append(r)
            if r.error:
                status = "ERR"
            elif r.ttft_ms is not None:
                status = f"{r.ttft_ms:.0f}ms TTFT"
                if r.queue_wait_ms is not None and r.queue_wait_ms > 50:
                    status += f" (queue_wait={r.queue_wait_ms:.0f}ms)"
            else:
                status = "n/a"
            print(f"  [{r.request_id}] {status}")

    runs_dir = Path("runs")
    runs_dir.mkdir(exist_ok=True)
    out_path = runs_dir / f"{variant}.csv"

    fieldnames = [
        "request_id", "prompt", "ttft_ms", "tpot_ms", "total_ms",
        "queue_wait_ms", "connect_wait_ms",
        "input_tokens", "output_tokens", "tokens_per_sec",
        "num_requests_running", "num_requests_waiting", "kv_cache_usage_perc",
        "error",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = asdict(r)
            row.pop("itl_ms_list", None)
            writer.writerow(row)

    # Also dump full ITL lists to JSONL, since per-chunk gap lists don't fit
    # cleanly in a CSV cell.
    jsonl_path = runs_dir / f"{variant}.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(asdict(r)) + "\n")

    n_ok = sum(1 for r in results if not r.error)
    n_err = sum(1 for r in results if r.error)
    print(f"\nSaved {len(results)} results ({n_ok} ok, {n_err} errors) to {out_path}")
    print(f"Full per-chunk detail: {jsonl_path}")
    return out_path


# -----------------------------
# CLI
# -----------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Async streaming benchmark harness for a local vLLM server."
    )
    p.add_argument("--base-url", default="http://localhost:8000/v1",
                    help="vLLM OpenAI-compatible base URL (default: http://localhost:8000/v1)")
    p.add_argument("--api-key", default="local-vllm-key",
                    help="API key to send (vLLM doesn't validate it locally, but the header is required)")
    p.add_argument("--model", required=True,
                    help="Model name as passed to `vllm serve`")
    p.add_argument("--dataset", required=True, type=Path,
                    help="Path to a JSONL dataset with {id, prompt} rows")
    p.add_argument("--concurrency", type=int, default=1,
                    help="Number of concurrent in-flight requests (default: 1)")
    p.add_argument("--variant", default="run",
                    help="Name used for output files, e.g. concurrency1, concurrency10")
    p.add_argument("--max-samples", type=int, default=None,
                    help="Optional cap on number of dataset rows to use (applied after --repeat expansion)")
    p.add_argument("--repeat", type=int, default=1,
                    help="Cycle through the dataset this many times before running, to get enough "
                         "volume for meaningful p95/p99 at higher concurrency levels (default: 1, no repeat)")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"Running against {args.base_url} | model={args.model} | "
          f"concurrency={args.concurrency} | repeat={args.repeat}")
    asyncio.run(
        run_benchmark(
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            dataset_path=args.dataset,
            concurrency=args.concurrency,
            variant=args.variant,
            max_samples=args.max_samples,
            repeat=args.repeat,
        )
    )


if __name__ == "__main__":
    main()
