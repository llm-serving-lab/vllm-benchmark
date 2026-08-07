#!/usr/bin/env python3
"""
idle_recovery_test.py
======================
Targeted reproduction test for a finding first observed during Task 7
(2026-08-04): a request following a period where the vLLM engine went fully
idle (Running: 0, Waiting: 0) showed a ~37s TTFT spike, similar in size to
the documented cold-start/warmup-cluster costs but occurring mid-run on an
already-warm server, at a point with no new execution shape (concurrency=1
throughout, same as many earlier requests in the same run).

This script isolates the one variable at a time: send a request, wait a
controlled idle period, send another request, and report both TTFTs plus
the idle duration. Run this multiple times with different idle durations
to characterize where (if anywhere) a threshold exists.

Usage:
    python scripts/idle_recovery_test.py --idle-seconds 35
    python scripts/idle_recovery_test.py --idle-seconds 10
    python scripts/idle_recovery_test.py --idle-seconds 60
"""

import argparse
import asyncio
import json
import time

import httpx


async def single_request(client: httpx.AsyncClient, base_url: str, api_key: str,
                          model: str, prompt: str) -> float:
    """Send one streaming request, return ttft_ms."""
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    start = time.perf_counter()
    ttft_ms = None
    async with client.stream("POST", f"{base_url}/chat/completions",
                              headers=headers, json=payload, timeout=120.0) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line or not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            choices = obj.get("choices") or []
            if choices and choices[0].get("delta", {}).get("content") and ttft_ms is None:
                ttft_ms = (time.perf_counter() - start) * 1000.0
    return ttft_ms


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default="local-vllm-key")
    parser.add_argument("--model", required=True)
    parser.add_argument("--idle-seconds", type=float, required=True,
                         help="Idle gap to insert between the warm-up request and the test request")
    args = parser.parse_args()

    async with httpx.AsyncClient() as client:
        print(f"[1/3] Sending warm-up request (establishes a known-warm baseline)...")
        warmup_ttft = await single_request(
            client, args.base_url, args.api_key, args.model,
            "What is PagedAttention in one sentence?"
        )
        print(f"      warm-up ttft_ms = {warmup_ttft:.1f}")

        print(f"[2/3] Idling for {args.idle_seconds:.1f}s "
              f"(server should show Running: 0, Waiting: 0 during this window "
              f"-- watch the serve.sh terminal now)...")
        await asyncio.sleep(args.idle_seconds)

        print(f"[3/3] Sending post-idle test request...")
        test_ttft = await single_request(
            client, args.base_url, args.api_key, args.model,
            "Explain the difference between TTFT and TPOT."
        )
        print(f"      post-idle ttft_ms = {test_ttft:.1f}")

        print()
        print(f"RESULT: idle_seconds={args.idle_seconds:.1f} "
              f"warmup_ttft_ms={warmup_ttft:.1f} post_idle_ttft_ms={test_ttft:.1f} "
              f"ratio={test_ttft/warmup_ttft:.2f}x")


if __name__ == "__main__":
    asyncio.run(main())
