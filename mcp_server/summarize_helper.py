#!/usr/bin/env python3
"""Thin JSON wrapper around analyze.summarize().

Run under the harness's own venv (scripts/run_bench.py's Python), not the
MCP server's, since analyze.py needs pandas + matplotlib that the server's
newer Python environment doesn't carry -- see mcp_server/README.md.
"""
import contextlib
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import analyze  # noqa: E402


def main():
    paths = [Path(p) for p in sys.argv[1:]]
    missing = [str(p) for p in paths if not p.exists()]
    existing = [p for p in paths if p.exists()]
    if not existing:
        print(json.dumps({"error": "no matching CSVs found", "missing": missing}))
        return
    # summarize() prints warmup-cluster-exclusion diagnostics to stdout as a
    # side effect -- fine for the CLI tool it was written for, but it would
    # corrupt this script's JSON-on-stdout contract with the MCP server.
    # Capture and fold them into the JSON instead of discarding them.
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        df = analyze.summarize(existing)
    out = {"summary": df.to_dict(orient="records")}
    if missing:
        out["missing"] = missing
    notes = captured.getvalue().strip()
    if notes:
        out["notes"] = notes
    print(json.dumps(out))


if __name__ == "__main__":
    main()
