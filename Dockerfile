# Phase 2 task #5: packages this repo's benchmark harness so it can run as
# a Kubernetes Job (see ../vllm-apple-silicon-serving/k8s/benchmark-job.yaml)
# inside the same `kind` cluster as the vLLM server, rather than as a local
# CLI script. Lives in this repo (not vllm-apple-silicon-serving) because
# it's this repo's own source code being packaged, same reasoning as why
# the vLLM server Dockerfile lives in the deployment repo instead of vLLM's
# own upstream source.
FROM python:3.12-slim

WORKDIR /app

# Only httpx is needed to run the benchmark itself -- pandas/matplotlib
# (also in requirements.txt) are for analyze.py's offline plotting, done
# after pulling results back out of the cluster, not inside the Job.
RUN pip install --no-cache-dir httpx>=0.27.0

# MinIO's own client (mc), a static binary -- used by run_and_upload.sh to
# push results into the object store after the benchmark finishes. arm64
# because the kind node (and this Mac) is Apple Silicon.
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    curl -sSL -o /usr/local/bin/mc https://dl.min.io/client/mc/release/linux-arm64/mc && \
    chmod +x /usr/local/bin/mc && \
    apt-get purge -y curl && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

COPY scripts/run_bench.py scripts/run_bench.py
COPY datasets/prompts.jsonl datasets/prompts.jsonl
COPY run_and_upload.sh run_and_upload.sh
RUN chmod +x run_and_upload.sh

# CLI flags (--concurrency, --variant, etc.) are supplied by the Job spec's
# `args`, not baked in here, so the same image can run different benchmark
# configurations without a rebuild. run_and_upload.sh runs run_bench.py
# with those args, then uploads the results to MinIO.
ENTRYPOINT ["./run_and_upload.sh"]
