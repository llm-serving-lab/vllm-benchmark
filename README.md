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

## Repo structure

```
vllm-benchmark/
├── datasets/
│   └── prompts.jsonl        # small prompt set for saturation-curve runs
├── runs/                    # output CSV + JSONL per run (gitignored)
├── charts/                  # generated saturation/throughput charts
├── scripts/
│   ├── run_bench.py         # the harness — run this
│   └── analyze.py           # summary stats + chart generator
├── requirements.txt
└── README.md
```
