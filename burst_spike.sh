#!/usr/bin/env bash
set -euo pipefail
cd ~/Documents/vllm-benchmark

echo "--- throwaway warmup request (absorbing first-batch stall) ---"
./venv/bin/python3 scripts/run_bench.py \
  --model mlx-community/Mistral-7B-Instruct-v0.3-4bit \
  --dataset datasets/prompts.jsonl \
  --concurrency 1 \
  --max-samples 1 \
  --variant burst_spike_warmup_throwaway \
  --no-gpu-metrics

echo "--- real burst: 50 concurrent requests ---"
./venv/bin/python3 scripts/run_bench.py \
  --model mlx-community/Mistral-7B-Instruct-v0.3-4bit \
  --dataset datasets/prompts.jsonl \
  --concurrency 50 \
  --repeat 3 \
  --max-samples 50 \
  --variant burst_spike \
  --no-gpu-metrics
