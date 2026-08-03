#!/usr/bin/env python3
"""
analyze.py
==========
Summarize one or more run_bench.py output CSVs (typically one per
concurrency level) into a p50/p95/p99 table and a saturation chart.

Usage:
    python scripts/analyze.py runs/concurrency1.csv runs/concurrency5.csv ...

Warmup-cluster exclusion
------------------------
The first batch of requests submitted at any concurrency level N pay a
one-time MLX/Metal shader-compile cost for that batch size the first time
the server executes it (documented in PHASE1_LOG.md, entries 2026-08-02 and
2026-08-03: confirmed via direct server-log cross-reference at concurrency
5 and 10, and by consistent cluster-size matching at 25 and 50). These
requests have near-zero queue_wait_ms (they got a semaphore slot
instantly, since they're among the first N submitted) but a TTFT that is
many multiples of every other request in the same run.

Without excluding them, percentile math is badly distorted -- at
concurrency=50, exactly half the run (50 of 100 requests) sat in this
cluster at ~72,000ms TTFT, which would make the reported p50 itself land
around 72 seconds and completely misrepresent real steady-state behavior.

Detection is statistical, not a hardcoded time threshold (thresholds like
"72000ms" would silently stop working the moment the model, hardware, or
prompt set changes): a request is treated as warmup-cluster if its
queue_wait_ms is below WARMUP_QUEUE_WAIT_THRESHOLD_MS (near-instant
semaphore acquisition) AND its ttft_ms is more than WARMUP_TTFT_MULTIPLIER
times the median ttft_ms of all *other* requests in the same file. Both
conditions are required so a merely-fast-queued request with normal TTFT
isn't misclassified.
"""

import sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
CHART_DIR = ROOT / "charts"
CHART_DIR.mkdir(exist_ok=True)

WARMUP_QUEUE_WAIT_THRESHOLD_MS = 100.0
WARMUP_TTFT_MULTIPLIER = 5.0


def concurrency_from_filename(path: Path) -> str:
    return path.stem


def split_warmup_cluster(ok: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a file's successful requests into (steady_state, warmup_cluster).

    Returns (steady_state_df, warmup_df). If queue_wait_ms isn't present
    (pre-fix CSVs from before 2026-08-02) or there's nothing that looks
    like a cluster, warmup_df is empty and steady_state_df is everything --
    i.e. this is a strict no-op on older data, not a forced exclusion.
    """
    if "queue_wait_ms" not in ok.columns or ok.empty:
        return ok, ok.iloc[0:0]

    instant = ok["queue_wait_ms"].notna() & (ok["queue_wait_ms"] < WARMUP_QUEUE_WAIT_THRESHOLD_MS)
    if not instant.any():
        return ok, ok.iloc[0:0]

    # Median TTFT of requests NOT in the instant-queue group anchors what
    # "normal" looks like for this file, so the multiplier isn't thrown off
    # by the cluster itself.
    baseline = ok.loc[~instant, "ttft_ms"]
    if baseline.empty:
        # Every request had near-zero queue wait (e.g. concurrency=1 with a
        # single request) -- nothing to compare against, so don't guess.
        return ok, ok.iloc[0:0]

    baseline_median = baseline.median()
    is_warmup = instant & (ok["ttft_ms"] > WARMUP_TTFT_MULTIPLIER * baseline_median)

    return ok.loc[~is_warmup], ok.loc[is_warmup]


def summarize(paths: list[Path]) -> pd.DataFrame:
    rows = []
    for path in paths:
        df = pd.read_csv(path)
        ok = df[df["error"].isna()]
        if ok.empty:
            print(f"WARNING: {path} has no successful requests, skipping")
            continue

        steady, warmup = split_warmup_cluster(ok)
        if not warmup.empty:
            print(f"{path}: excluded {len(warmup)} warmup-cluster request(s) "
                  f"from percentile math (near-instant queue_wait_ms, TTFT "
                  f">{WARMUP_TTFT_MULTIPLIER:.0f}x steady-state median -- "
                  f"see PHASE1_LOG.md for why)")

        if steady.empty:
            print(f"WARNING: {path} has no non-warmup requests left after "
                  f"exclusion, skipping")
            continue

        # Aggregate throughput = total output tokens across every steady-state
        # request, divided by the wall-clock span those requests actually
        # occupied (last request's completion minus first request's start,
        # approximated here via total_ms + queue_wait_ms per request isn't
        # directly comparable across overlapping requests -- so this uses
        # total output tokens / (max total_ms among steady requests), which
        # approximates "how much real work got done per second of wall clock
        # once the server was warm and running at this concurrency level."
        # This is the number that answers "is the server doing more total
        # work at higher concurrency," which avg_tokens_per_sec (a per-request
        # average) does NOT answer -- per-request throughput naturally drops
        # as more requests share one GPU even when aggregate throughput rises.
        total_output_tokens = steady["output_tokens"].sum()
        wall_clock_span_s = steady["total_ms"].max() / 1000.0 if len(steady) else 0
        aggregate_tokens_per_sec = (
            total_output_tokens / wall_clock_span_s if wall_clock_span_s > 0 else None
        )

        rows.append({
            "variant": concurrency_from_filename(path),
            "n": len(df),
            "n_ok": len(ok),
            "n_err": len(df) - len(ok),
            "n_warmup_excluded": len(warmup),
            "ttft_p50_ms": steady["ttft_ms"].quantile(0.50),
            "ttft_p95_ms": steady["ttft_ms"].quantile(0.95),
            "ttft_p99_ms": steady["ttft_ms"].quantile(0.99),
            "tpot_p50_ms": steady["tpot_ms"].quantile(0.50),
            "total_p50_ms": steady["total_ms"].quantile(0.50),
            "total_p95_ms": steady["total_ms"].quantile(0.95),
            "total_p99_ms": steady["total_ms"].quantile(0.99),
            # queue_wait_ms didn't exist before the 2026-08-02 harness fix;
            # older CSVs won't have the column, so guard with .get-style access.
            "queue_wait_p95_ms": steady["queue_wait_ms"].quantile(0.95) if "queue_wait_ms" in steady else None,
            # Per-request average -- naturally declines with concurrency as
            # requests share one GPU. NOT the same as "server got slower."
            "avg_tokens_per_sec_per_request": steady["tokens_per_sec"].mean(),
            # Total work done per second of wall clock -- the real "did
            # throughput improve at this concurrency level" answer.
            "aggregate_tokens_per_sec": aggregate_tokens_per_sec,
            "avg_kv_cache_usage_perc": steady["kv_cache_usage_perc"].mean(),
            "max_num_requests_waiting": steady["num_requests_waiting"].max(),
        })
    return pd.DataFrame(rows)


def print_summary(summary: pd.DataFrame):
    print("\n" + "=" * 78)
    print("  VLLM BENCHMARK SUMMARY")
    print("=" * 78)
    print(summary.to_string(index=False))
    print("=" * 78 + "\n")


def build_chart(summary: pd.DataFrame, out_path: Path):
    # Three panels, not two: TTFT (this task's actual metric) and total_ms
    # (includes queueing, scales very differently) were previously plotted
    # on one shared axis, where total_p99_ms's ~120,000ms scale flattened the
    # TTFT lines (max ~2,200ms) into an invisible line at the bottom of the
    # chart -- exactly backwards for a task whose deliverable is the TTFT
    # p50/p95/p99 curve. Split onto separate axes so both are readable.
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.plot(summary["variant"], summary["ttft_p50_ms"], marker="o", label="TTFT p50")
    ax.plot(summary["variant"], summary["ttft_p95_ms"], marker="o", label="TTFT p95")
    ax.plot(summary["variant"], summary["ttft_p99_ms"], marker="o", label="TTFT p99")
    ax.set_title("TTFT vs concurrency (warmup-cluster excluded)")
    ax.set_ylabel("ms")
    ax.legend(fontsize=8)
    ax.grid(linestyle="--", alpha=0.4)

    ax = axes[1]
    ax.plot(summary["variant"], summary["total_p50_ms"], marker="o", label="Total p50")
    ax.plot(summary["variant"], summary["total_p95_ms"], marker="o", label="Total p95")
    ax.plot(summary["variant"], summary["total_p99_ms"], marker="o", label="Total p99")
    ax.set_title("Total request time vs concurrency")
    ax.set_ylabel("ms")
    ax.legend(fontsize=8)
    ax.grid(linestyle="--", alpha=0.4)

    ax = axes[2]
    ax.plot(summary["variant"], summary["aggregate_tokens_per_sec"], marker="o",
            color="green", label="Aggregate (total work/sec)")
    ax.plot(summary["variant"], summary["avg_tokens_per_sec_per_request"], marker="o",
            color="orange", linestyle="--", label="Per-request average")
    ax.set_title("Throughput vs run")
    ax.set_ylabel("tokens/sec")
    ax.legend(fontsize=8)
    ax.grid(linestyle="--", alpha=0.4)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Chart saved -> {out_path}")


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python scripts/analyze.py runs/<file1>.csv [runs/<file2>.csv ...]")

    paths = [Path(p) for p in sys.argv[1:]]
    for p in paths:
        if not p.exists():
            sys.exit(f"File not found: {p}")

    summary = summarize(paths)
    if summary.empty:
        sys.exit("No successful runs to summarize.")

    print_summary(summary)
    build_chart(summary, CHART_DIR / "saturation_curve.png")


if __name__ == "__main__":
    main()
