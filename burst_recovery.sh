#!/usr/bin/env bash
set -euo pipefail
cd ~/Documents/vllm-benchmark

# Fires the same single canary prompt (dataset row 1, deterministic since
# run_bench.py doesn't shuffle) every 5s, 24 times (~2 minutes), so every
# probe is otherwise identical -- only elapsed-time-since-burst varies.
for i in $(seq 1 24); do
  echo "--- probe $i ($(date +%H:%M:%S)) ---"
  ./venv/bin/python3 scripts/run_bench.py \
    --model mlx-community/Mistral-7B-Instruct-v0.3-4bit \
    --dataset datasets/prompts.jsonl \
    --concurrency 1 \
    --max-samples 1 \
    --variant "burst_recovery_${i}" \
    --no-gpu-metrics
  sleep 5
done
