#!/bin/sh
# Runs the benchmark exactly as before (writes to the PVC-mounted runs/
# directory, same as always), then additionally uploads a copy to the
# S3-compatible object store (Garage) -- the durable, off-node copy. The
# PVC copy still happens too; this doesn't replace it, it demonstrates the
# same output landing in two places with two very different durability
# guarantees.
#
# Upload client: scripts/upload_results.py (boto3). Previously MinIO's `mc`,
# whose download URL is gone (HTTP 410) -- see PHASE2_LOG.md, 2026-10
# maintenance entry.
set -e

python3 scripts/run_bench.py "$@"

python3 scripts/upload_results.py
