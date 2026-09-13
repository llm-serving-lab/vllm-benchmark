"""
Lightweight tool-call tracing for this MCP server.

This is the master plan's Phase 4 "agent/tool-call tracing" item, applied to
a real, already-built agent-callable tool surface instead of a from-scratch
toy example. Deliberately NOT a full OpenTelemetry/Jaeger stack -- that's
separate, larger scope. Just enough structure to answer "trace every step,
show a latency breakdown by stage" for a tool-calling agent, which is the
actual thing worth being able to demonstrate.

Two layers, matching how real tracing systems split the work:
- `traced` -- a decorator applied uniformly to all four tools, recording
  one JSONL record per call (tool name, duration, success/error). This is
  the auto-instrumentation layer: every tool gets it for free.
- `StageTimer` -- a manual span helper for the one tool complex enough to
  have real internal structure (`run_benchmark`'s smoke test / subprocess
  run / summarize steps). This is the manual-instrumentation layer, the
  same split OTel itself uses (auto-instrumentation plus hand-added spans
  where the auto layer alone wouldn't show enough).
"""
import functools
import json
import time
from pathlib import Path

TRACE_LOG = Path(__file__).resolve().parent / "traces.jsonl"


def _write(record: dict) -> None:
    record.setdefault("ts", time.time())
    with TRACE_LOG.open("a") as f:
        f.write(json.dumps(record) + "\n")


def traced(fn):
    """Wrap an MCP tool so every call writes one JSONL trace record.

    A tool's own return value decides success/failure the same way a
    caller would read it: a dict with an "error" key counts as a failed
    call even though no Python exception was raised (matches how every
    tool in this server reports its own guardrail refusals).
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        start = time.time()
        try:
            result = fn(*args, **kwargs)
            failed = isinstance(result, dict) and "error" in result
            _write({
                "tool": fn.__name__,
                "duration_ms": round((time.time() - start) * 1000, 1),
                "success": not failed,
                "error": result.get("error") if failed else None,
            })
            return result
        except Exception as e:
            _write({
                "tool": fn.__name__,
                "duration_ms": round((time.time() - start) * 1000, 1),
                "success": False,
                "error": f"{type(e).__name__}: {e}",
            })
            raise
    return wrapper


class StageTimer:
    """Manual per-stage span recorder for one tool call.

    Not a full OTel span (no parent/child tree, no distributed context,
    no exporter) -- just enough to show which stage of a multi-step tool
    call actually took the time, which is the real, demonstrable thing.
    """

    def __init__(self, tool: str):
        self.tool = tool
        self.variant = None
        self.stages: list[dict] = []
        self._stage_name = None
        self._stage_start = None

    def start(self, name: str) -> None:
        self._flush_current()
        self._stage_name = name
        self._stage_start = time.time()

    def _flush_current(self) -> None:
        if self._stage_name is not None:
            self.stages.append({
                "stage": self._stage_name,
                "duration_ms": round((time.time() - self._stage_start) * 1000, 1),
            })
        self._stage_name = None

    def finish(self, success: bool, error: str = None) -> None:
        self._flush_current()
        _write({
            "tool": self.tool,
            "variant": self.variant,
            "success": success,
            "error": error,
            "total_ms": round(sum(s["duration_ms"] for s in self.stages), 1),
            "stages": self.stages,
        })


def read_trace_history(limit: int = 20) -> list[dict]:
    """Return the most recent `limit` trace records, newest first."""
    if not TRACE_LOG.exists():
        return []
    lines = TRACE_LOG.read_text().splitlines()
    records = [json.loads(line) for line in lines[-limit:]]
    return list(reversed(records))
