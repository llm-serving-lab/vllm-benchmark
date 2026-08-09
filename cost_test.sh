#!/usr/bin/env bash
set -euo pipefail
cd ~/Documents/vllm-benchmark
./venv/bin/python3 scripts/run_bench.py \
  --model mlx-community/Mistral-7B-Instruct-v0.3-4bit \
  --dataset datasets/prompts.jsonl \
  --concurrency 5 \
  --max-samples 10 \
  --variant cost_test \
  --cost-hardware-usd 2000 \
  --cost-lifetime-years 2 \
  --cost-electricity-per-kwh 0.15
