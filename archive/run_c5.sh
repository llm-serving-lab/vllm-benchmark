#!/usr/bin/env bash
set -euo pipefail
cd ~/Documents/vllm-benchmark
./venv/bin/python3 scripts/run_bench.py \
  --model mlx-community/Mistral-7B-Instruct-v0.3-4bit \
  --dataset datasets/prompts.jsonl \
  --concurrency 5 \
  --max-samples 10 \
  --variant refresher_c5 \
  --no-gpu-metrics
