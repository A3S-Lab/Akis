"""Compare Akis complete envelopes with the frozen Needle engine calls.

The oracle is tests/e2e/cases/needle_engine.json, captured from pinned
library 993d617e. This script does not load that library or the SDK.
"""
import hashlib
import json
import pathlib
import subprocess
import sys

from live_e2e import MODEL_SHA, ROOT, TOOL

ORACLE = ROOT / "tests/e2e/cases/needle_engine.json"
LIBRARY = ROOT / ".cache/libneedle3.dylib"


def digest(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def calls(value):
    return [{"name": item["name"], "arguments": item["arguments"]} for item in value]


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: python3 tools/needle_e2e.py MODEL BINARY")
    model, binary = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    oracle = json.loads(ORACLE.read_text())
    if oracle["model_sha256"] != MODEL_SHA or digest(model) != MODEL_SHA:
        print(json.dumps({"status": "blocked", "reason": "model sha256 differs from the frozen envelope"}))
        return 2
    if not LIBRARY.is_file() or digest(LIBRARY) != oracle["library_sha256"]:
        print(json.dumps({"status": "blocked", "reason": "pinned native library is missing or differs"}))
        return 2
    rows = []
    for case in oracle["cases"]:
        request = json.dumps(
            {"input": case["input"], "tools": [TOOL], "max_tokens": 128},
            ensure_ascii=False,
        )
        proc = subprocess.run(
            [str(binary), "complete", str(model)],
            input=request,
            capture_output=True,
            text=True,
            timeout=120,
        )
        row = {"id": case["id"]}
        if proc.returncode != 0 or proc.stderr or not proc.stdout:
            row.update(status="failed", error=proc.stderr.strip() or f"exit {proc.returncode}")
        else:
            value = json.loads(proc.stdout)
            actual = {
                "function_calls": calls(value.get("function_calls") or []),
                "suppressed_calls": calls(value.get("suppressed_calls") or []),
                "ungrounded": list(value.get("ungrounded") or []),
            }
            expected = {
                "function_calls": calls(case["function_calls"]),
                "suppressed_calls": calls(case["suppressed_calls"]),
                "ungrounded": list(case["ungrounded"]),
            }
            row["actual"] = actual
            row["status"] = "passed" if actual == expected else "failed"
            if row["status"] == "failed":
                row["expected"] = expected
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    summary = {
        "passed": sum(row["status"] == "passed" for row in rows),
        "failed": sum(row["status"] == "failed" for row in rows),
        "binary_sha256": digest(binary),
        "model_sha256": MODEL_SHA,
        "library_sha256": oracle["library_sha256"],
        "oracle_sha256": digest(ORACLE),
    }
    print(json.dumps(summary), flush=True)
    return int(summary["failed"] != 0)


if __name__ == "__main__":
    sys.exit(main())
