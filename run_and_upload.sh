#!/bin/sh
# Runs the benchmark exactly as before (writes to the PVC-mounted runs/
# directory, same as always), then additionally uploads a copy to MinIO --
# the durable, off-node copy. The PVC copy still happens too; this doesn't
# replace it, it demonstrates the same output landing in two places with
# two very different durability guarantees.
set -e

python3 scripts/run_bench.py "$@"

mc alias set local http://minio:9000 "$MINIO_ROOT_USER" "$MINIO_ROOT_PASSWORD"
mc cp runs/*.csv runs/*.jsonl local/bench-results/
echo "Uploaded results to MinIO bucket local/bench-results/"
