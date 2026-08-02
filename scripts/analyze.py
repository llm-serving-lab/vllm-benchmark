#!/usr/bin/env python3
"""
analyze.py
==========
Summarize one or more run_bench.py output CSVs (typically one per
concurrency level) into a p50/p95/p99 table and a saturation chart.

Usage:
    python scripts/analyze.py runs/concurrency1.csv runs/concurrency5.csv ...
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


def concurrency_from_filename(path: Path) -> str:
    return path.stem


def summarize(paths: list[Path]) -> pd.DataFrame:
    rows = []
    for path in paths:
        df = pd.read_csv(path)
        ok = df[df["error"].isna()]
        if ok.empty:
            print(f"WARNING: {path} has no successful requests, skipping")
            continue
        rows.append({
            "variant": concurrency_from_filename(path),
            "n": len(df),
            "n_ok": len(ok),
            "n_err": len(df) - len(ok),
            "ttft_p50_ms": ok["ttft_ms"].quantile(0.50),
            "ttft_p95_ms": ok["ttft_ms"].quantile(0.95),
            "ttft_p99_ms": ok["ttft_ms"].quantile(0.99),
            "tpot_p50_ms": ok["tpot_ms"].quantile(0.50),
            "total_p50_ms": ok["total_ms"].quantile(0.50),
            "total_p95_ms": ok["total_ms"].quantile(0.95),
            "total_p99_ms": ok["total_ms"].quantile(0.99),
            "avg_tokens_per_sec": ok["tokens_per_sec"].mean(),
            "avg_kv_cache_usage_perc": ok["kv_cache_usage_perc"].mean(),
            "max_num_requests_waiting": ok["num_requests_waiting"].max(),
        })
    return pd.DataFrame(rows)


def print_summary(summary: pd.DataFrame):
    print("\n" + "=" * 78)
    print("  VLLM BENCHMARK SUMMARY")
    print("=" * 78)
    print(summary.to_string(index=False))
    print("=" * 78 + "\n")


def build_chart(summary: pd.DataFrame, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(summary["variant"], summary["ttft_p50_ms"], marker="o", label="TTFT p50")
    ax.plot(summary["variant"], summary["ttft_p95_ms"], marker="o", label="TTFT p95")
    ax.plot(summary["variant"], summary["total_p99_ms"], marker="o", label="Total p99")
    ax.set_title("Latency vs run")
    ax.set_ylabel("ms")
    ax.legend(fontsize=8)
    ax.grid(linestyle="--", alpha=0.4)

    ax = axes[1]
    ax.plot(summary["variant"], summary["avg_tokens_per_sec"], marker="o", color="green")
    ax.set_title("Throughput vs run")
    ax.set_ylabel("tokens/sec")
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
