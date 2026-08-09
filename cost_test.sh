#!/usr/bin/env bash
set -euo pipefail
cd ~/Documents/vllm-benchmark

echo "--- warmup at concurrency=5 (same as real test, zero gap before it) ---"
./venv/bin/python3 scripts/run_bench.py \
  --model mlx-community/Mistral-7B-Instruct-v0.3-4bit \
  --dataset datasets/prompts.jsonl \
  --concurrency 5 \
  --max-samples 5 \
  --variant cost_test_clean_warmup \
  --no-gpu-metrics

echo "--- real cost-tracked run (should be clean now) ---"
./venv/bin/python3 scripts/run_bench.py \
  --model mlx-community/Mistral-7B-Instruct-v0.3-4bit \
  --dataset datasets/prompts.jsonl \
  --concurrency 5 \
  --max-samples 10 \
  --variant cost_test_clean \
  --cost-hardware-usd 2000 \
  --cost-lifetime-years 2 \
  --cost-electricity-per-kwh 0.15
