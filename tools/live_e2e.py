"""Real-model process tests. All inputs are synthetic; failures are never retried here."""
import argparse
import hashlib
import json
import math
import pathlib
import platform
import struct
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODEL_SHA = "c9d915eca282ed42d1a09b143b592adb4cc6744ffe2d294adf5cfc5548170c38"
TOOL = {"name": "set_lights", "description": "Turn lights on or off", "parameters": {
    "type": "object", "properties": {"room": {"type": "string"}, "on": {"type": "boolean"}},
    "required": ["room", "on"]}}


def digest(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def sort_json(value):
    """serde_json object keys are stored in byte order. Arrays keep their order."""
    if isinstance(value, dict):
        return {key: sort_json(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [sort_json(item) for item in value]
    return value


def model_prompt(user, tools):
    rendered = []
    for tool in tools:
        rendered.append({
            "name": tool["name"],
            "description": tool["description"],
            "parameters": sort_json(tool["parameters"]),
        })
    tools_json = json.dumps(rendered, ensure_ascii=False, separators=(",", ":"))
    return (
        "<|im_start|>user\n<tools>"
        + tools_json
        + "</tools>\n"
        + user
        + "<|im_end|>\n<|im_start|>assistant\n"
    )


def token_record(ids):
    raw = b"".join(struct.pack("<I", i) for i in ids)
    return hashlib.sha256(b"a3s-power-token-ids-v1\0" + raw).hexdigest(), len(raw), len(ids)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=pathlib.Path)
    parser.add_argument("--binary", type=pathlib.Path, default=ROOT / "target/release/a3s-akis")
    parser.add_argument("--case", help="Run one exact case ID for failure triage")
    args = parser.parse_args()
    binary, model = str(args.binary.resolve()), str(args.model.resolve())
    assert digest(model) == MODEL_SHA, "unexpected model: this campaign is pinned"
    cases_path = ROOT / "tests/e2e/cases/tool_calls.json"
    rows = []

    def command(cmd, *tail, request=None, error=None):
        proc = subprocess.run([binary, cmd, model, *tail], input=request,
                              capture_output=True, text=True, timeout=120)
        if error is not None:
            assert proc.returncode != 0 and not proc.stdout and error in proc.stderr, proc
            return None
        assert proc.returncode == 0 and not proc.stderr, proc.stderr
        return json.loads(proc.stdout)

    def case(name, run):
        if args.case and args.case != name:
            return
        start = time.perf_counter()
        row = {"id": name}
        try:
            row["observed"] = run()
            row["status"] = "passed"
        except (AssertionError, ValueError, subprocess.SubprocessError) as exc:
            row.update(status="failed", error=str(exc))
        row["seconds"] = round(time.perf_counter() - start, 6)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    def inspect():
        value = command("inspect")
        assert value["sha256"] == MODEL_SHA and value["tensors"] == 581
        assert value["geometry"]["width"] == 768
        return value
    case("inspect", inspect)

    def tokenize():
        text = "你好 🦀 café e\u0301\n<tool_call>"
        value = command("tokenize", text)
        assert value["text"] == text, value
        assert all(type(i) is int and 0 <= i < 8192 for i in value["ids"])
        return value
    case("tokenize-unicode", tokenize)

    def embedding():
        value = command("embed", "turn on the kitchen lights")
        assert value and all(math.isfinite(x) for x in value)
        norm = math.sqrt(sum(x*x for x in value))
        assert abs(norm - 1) < 1e-5, norm
        assert command("embed", "turn on the kitchen lights") == value
        return {"dimensions": len(value), "norm": norm}
    case("embedding-deterministic-normalized", embedding)

    for fixture in json.loads(cases_path.read_text()):
        def completion(fixture=fixture):
            request = {"input": fixture["input"], "tools": [TOOL], "max_tokens": 128}
            value = command("complete", request=json.dumps(request, ensure_ascii=False))
            calls, gen = value["function_calls"], value["generation"]
            assert calls == fixture["expected"], {"expected": fixture["expected"], "actual": calls}
            assert gen["finished"] and 0 < len(gen["token_ids"]) <= 128
            assert 0 <= gen["head_score"] <= 1 and 0 <= gen["mean_token_probability"] <= 1
            receipt = gen["receipt"]
            assert receipt["schema"] == "a3s.power.embedded-execution-receipt.v1"
            assert receipt["runtime"]["device"] == "cpu"
            assert receipt["runtime"]["name"] == "a3s-power-native"
            assert receipt["model"]["weightsSha256"] == MODEL_SHA
            tokenized = command("tokenize", model_prompt(fixture["input"], [TOOL]))
            bound = token_record([2, *tokenized["ids"]])
            assert (
                receipt["input"]["sha256"],
                receipt["input"]["byteLength"],
                receipt["input"]["itemCount"],
            ) == bound, "input receipt digest"
            changed = [2, *tokenized["ids"]]
            changed[-1] ^= 1
            assert token_record(changed)[0] != bound[0]
            output = token_record(gen["token_ids"])
            assert (
                receipt["output"]["sha256"],
                receipt["output"]["byteLength"],
                receipt["output"]["itemCount"],
            ) == output, "output receipt digest"
            changed_output = list(gen["token_ids"])
            changed_output[-1] ^= 1
            assert token_record(changed_output)[0] != output[0]
            return {"calls": calls, "tokens": len(gen["token_ids"]), "head_score": gen["head_score"],
                    "receipt": receipt}
        case(fixture["id"], completion)

    case("invalid-stdin", lambda: command("complete", request="{", error="EOF while parsing"))
    case("unknown-command", lambda: command("unknown", error="unknown command"))
    def closed_pipe():
        proc = subprocess.Popen([binary, "inspect", model], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        proc.stdout.close()
        try:
            code = proc.wait(timeout=30)
            error = proc.stderr.read().decode()
            assert code == 1 and "Broken pipe" in error and "panicked" not in error, (code, error)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            proc.stderr.close()
    case("closed-output-pipe", closed_pipe)
    case("zero-generation-bound", lambda: command("generate", "hello", "0", error="generation bound"))
    case("truncated-completion", lambda: command("complete", request=json.dumps({
        "input": "turn on the kitchen lights", "tools": [TOOL], "max_tokens": 1}), error="exhausted"))
    invalid_tool = json.loads(json.dumps(TOOL))
    invalid_tool["parameters"]["properties"]["room"]["pattern"] = ".*"
    case("unsupported-schema", lambda: command("complete", request=json.dumps({
        "input": "hello", "tools": [invalid_tool], "max_tokens": 8}), error="unsupported tool schema"))
    if not rows:
        parser.error("unknown case ID")
    summary = {"platform": platform.platform(), "machine": platform.machine(),
               "model_sha256": MODEL_SHA, "binary_sha256": digest(binary),
               "cases_sha256": digest(cases_path), "passed": sum(r["status"] == "passed" for r in rows),
               "failed": sum(r["status"] == "failed" for r in rows)}
    print(json.dumps(summary), flush=True)
    return int(summary["failed"] != 0)


if __name__ == "__main__":
    sys.exit(main())
