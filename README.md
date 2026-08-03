# vllm-benchmark

Async, streaming benchmark harness for a locally-served vLLM instance
(built and validated against **vllm-metal on an M4 Mac mini**, see
[vllm-apple-silicon-serving](https://github.com/rap7239/vllm-apple-silicon-serving)
for the deployment side of this project).

This is a from-scratch harness, not a port of an earlier non-streaming
OpenAI-API lab harness ([llm-benchmark-module1](https://github.com/rap7239/llm-benchmark-module1)).
That harness measured `ttft_ms` as total request latency because it made
non-streaming calls — a reasonable simplification for a hosted-API teaching
lab, but not sufficient for measuring real prefill/decode behavior against a
locally-served vLLM instance. This harness streams every request and times
each token chunk individually, so TTFT, TPOT, and per-chunk inter-token
latency (ITL) are each real, independently-measured numbers.

## What it measures

Per request, streamed via Server-Sent Events (`stream: true`):

| Field | Meaning |
|---|---|
| `ttft_ms` | Time from request sent to first streamed token chunk — dominated by prompt processing ("prefill") |
| `tpot_ms` | Mean time between subsequent chunks — decode speed |
| `itl_ms_list` (JSONL only) | Every individual inter-chunk gap, not just the mean |
| `total_ms` | Full request wall-clock time |
| `input_tokens` / `output_tokens` | From the API's `usage` object (requires `stream_options.include_usage`) |
| `tokens_per_sec` | `output_tokens / (total_ms / 1000)` |

Immediately after each request completes, the harness also scrapes vLLM's
built-in Prometheus `/metrics` endpoint and records a snapshot of:

- `num_requests_running` (`vllm:num_requests_running`)
- `num_requests_waiting` (`vllm:num_requests_waiting`)
- `kv_cache_usage_perc` (`vllm:kv_cache_usage_perc`)

## Known limitation: GPU utilization on Apple Silicon

vLLM does not export a GPU compute-utilization percentage metric — that
normally comes from NVIDIA's DCGM exporter, which has no Metal/Apple Silicon
equivalent. On this hardware, GPU utilization has to be sampled separately
via `sudo powermetrics` or `asitop` alongside a benchmark run. This harness
does not attempt to fake that number; `gpu_util_pct` is intentionally absent
from the output schema until a Metal-native sampling script is added as a
follow-up.

## Quick start

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# In another terminal, make sure vLLM is already serving:
#   ./scripts/serve.sh   (from vllm-apple-silicon-serving)

python scripts/run_bench.py \
  --base-url http://localhost:8000/v1 \
  --api-key local-vllm-key \
  --model mlx-community/Mistral-7B-Instruct-v0.3-4bit \
  --dataset datasets/prompts.jsonl \
  --concurrency 1 \
  --variant concurrency1

# Repeat at higher concurrency levels to build a saturation curve:
python scripts/run_bench.py ... --concurrency 5  --variant concurrency5
python scripts/run_bench.py ... --concurrency 10 --variant concurrency10

python scripts/analyze.py runs/concurrency1.csv runs/concurrency5.csv runs/concurrency10.csv
```

## Saturation curve: dataset size + `--repeat`

`datasets/prompts.jsonl` originally had 5 hand-written vLLM-concept prompts
(PagedAttention, TTFT/TPOT, etc). Run once, that's 5 data points per
concurrency level — not enough to compute a meaningful p95/p99 (the p99 of 5
numbers is just the largest one).

Rather than solve this with `--repeat` alone, the dataset was merged with
the 15 real customer-support prompts from an earlier project
([`llm-benchmark-module1`](https://github.com/rap7239/llm-benchmark-module1),
ids `s1`-`s15`), bringing the base set to 20 prompts. This gives more
realistic traffic-shape diversity than repeating 5 synthetic prompts alone —
short questions, varied topics, a known "regression canary" prompt
(`s15`, the payment-failure one, was the slowest prompt in that project's
own benchmark run). `--repeat N` still exists and is still needed at higher
concurrency levels, but now cycles a 20-prompt base instead of 5, so fewer
repeat passes are needed to reach the same total volume — more variety per
request batch.

Repeated copies get a suffixed `id` (e.g. `s3-r2`) so results stay traceable
back to which prompt and which pass produced them.

Rule of thumb used for this project: request count should be at least `4x`
the concurrency level, so p95 has ~20+ samples above it and p99 isn't just
the single slowest request in the batch.

```bash
# concurrency=1  -> repeat=2   (40 requests; low bar, mostly a smoke test)
python scripts/run_bench.py --dataset datasets/prompts.jsonl \
  --concurrency 1  --repeat 2 --variant concurrency1

# concurrency=5  -> repeat=2   (40 requests)
python scripts/run_bench.py --dataset datasets/prompts.jsonl \
  --concurrency 5  --repeat 2 --variant concurrency5

# concurrency=10 -> repeat=3   (60 requests)
python scripts/run_bench.py --dataset datasets/prompts.jsonl \
  --concurrency 10 --repeat 3 --variant concurrency10

# concurrency=25 -> repeat=5   (100 requests)
python scripts/run_bench.py --dataset datasets/prompts.jsonl \
  --concurrency 25 --repeat 5 --variant concurrency25

# concurrency=50 -> repeat=5   (100 requests, 2x concurrency minimum)
python scripts/run_bench.py --dataset datasets/prompts.jsonl \
  --concurrency 50 --repeat 5 --variant concurrency50

python scripts/analyze.py runs/concurrency1.csv runs/concurrency5.csv \
  runs/concurrency10.csv runs/concurrency25.csv runs/concurrency50.csv
```

## Warmup-cluster handling in `analyze.py`

The Phase 1 Task 3 log entry documented a cold-start TTFT outlier (MLX
Metal-shader JIT compile cost landing on the first request processed).
Running the actual concurrency 1/5/10/25/50 sweep showed this cost isn't
paid just once per server process — it's paid once per distinct *batch
size* the server executes for the first time. At `--concurrency N`, the
first N requests submitted all get a semaphore slot instantly (no queueing)
but all pay a shared one-time compile cost together, producing a tight
cluster of N requests with near-identical, much-higher TTFT than the rest
of the run. This was confirmed directly against the server's own log
(`Running: N reqs` holding steady, no stalls) at concurrency=5 and 10, and
the pattern held exactly (cluster size == concurrency level) through 25 and
50 as well — see `PHASE1_LOG.md` for the full investigation.

`analyze.py` now detects and excludes this cluster automatically before
computing percentiles (a request is warmup if its `queue_wait_ms` is
near-zero AND its `ttft_ms` is more than 5x the file's steady-state median
— both conditions required, and the exclusion is printed to the console,
never silent). Without this, concurrency=50's reported p50 would land
around 72 seconds instead of the real ~1.3 second steady-state figure,
since half that run's requests sat in the warmup cluster.

## Actual sweep results (2026-08-03, M4 Mac mini, Mistral-7B-Instruct-v0.3-4bit, bf16)

| Concurrency | TTFT p50 | TTFT p95 | TTFT p99 | Aggregate tok/s | Warmup excluded |
|---|---|---|---|---|---|
| 1  | 273ms | 364ms   | 545ms   | 282.4 | 1 |
| 5  | 482ms | 588ms   | 638ms   | 93.6  | 5 |
| 10 | 728ms | 785ms   | 864ms   | 100.3 | 10 |
| 25 | 760ms | 1,031ms | 1,557ms | 147.5 | 25 |
| 50 | 1,348ms | 2,147ms | 2,170ms | 80.2 | 50 |

TTFT climbs gradually through concurrency 25, then p95/p99 both bend
sharply upward at 50 — the non-linear tail-latency inflection point this
task exists to find. Aggregate throughput (total output tokens / wall-clock
span — the real "is the server doing more total work" number, as opposed to
per-request average tokens/sec, which naturally declines with concurrency
even when the server is doing fine) peaks at concurrency=25 and *drops*
at 50, below even the concurrency=5 level — a second, independent signal
that 50 has crossed past this server's practical concurrency ceiling on
this hardware. Full chart: `charts/saturation_curve.png`.

## Repo structure

```
vllm-benchmark/
├── datasets/
│   └── prompts.jsonl        # 20 prompts (5 vLLM-concept + 15 real support-chat), cycled via --repeat for volume
├── runs/                    # output CSV + JSONL per run (gitignored)
├── charts/                  # generated saturation/throughput charts
├── scripts/
│   ├── run_bench.py         # the harness — run this
│   └── analyze.py           # summary stats + chart generator
├── requirements.txt
└── README.md
```
