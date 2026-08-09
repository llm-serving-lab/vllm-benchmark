#!/usr/bin/env bash
set -euo pipefail
cd ~/Documents/vllm-benchmark

echo "--- throwaway warmup request (absorbing first-batch stall) ---"
./venv/bin/python3 scripts/run_bench.py \
  --model mlx-community/Mistral-7B-Instruct-v0.3-4bit \
  --dataset datasets/prompts.jsonl \
  --concurrency 1 \
  --max-samples 1 \
  --variant burst_warmup_throwaway \
  --no-gpu-metrics

echo "--- real baseline measurement ---"
./venv/bin/python3 scripts/run_bench.py \
  --model mlx-community/Mistral-7B-Instruct-v0.3-4bit \
  --dataset datasets/prompts.jsonl \
  --concurrency 5 \
  --max-samples 10 \
  --variant burst_baseline \
  --no-gpu-metrics
