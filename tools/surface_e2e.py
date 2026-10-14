"""One process chain for the seven CLI commands on the pinned archive.

Usage: python3 tools/surface_e2e.py MODEL BINARY
"""
import hashlib
import json
import math
import pathlib
import subprocess
import sys

from live_e2e import MODEL_SHA, TOOL


def digest(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def run(binary, model, command, *tail, request=None):
    proc = subprocess.run(
        [binary, command, model, *tail],
        input=request,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if proc.returncode != 0 or proc.stderr or not proc.stdout:
        raise SystemExit(proc.stderr.strip() or f"{command} exit {proc.returncode}")
    return json.loads(proc.stdout)


def finite(values, label):
    if not values or not all(math.isfinite(item) for item in values):
        raise SystemExit(f"non-finite {label}")


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: python3 tools/surface_e2e.py MODEL BINARY")
    model, binary = sys.argv[1], sys.argv[2]
    if digest(model) != MODEL_SHA:
        print(json.dumps({"status": "blocked", "reason": "model sha256 differs"}))
        return 2
    info = run(binary, model, "inspect")
    geometry = info["geometry"]
    assert info["sha256"] == MODEL_SHA and info["tensors"] == 581
    assert geometry["context"] == 8192 and geometry["layers"] == 20
    assert geometry["width"] == 768 and geometry["vocab"] == 8192
    empty_logits = run(binary, model, "logits", "")
    assert empty_logits["ids"] == [2] and len(empty_logits["logits"]) == 8192
    finite(empty_logits["logits"], "empty logits")
    assert run(binary, model, "logits", "") == empty_logits
    empty_layers = run(binary, model, "layers", "")
    assert empty_layers["ids"] == [2] and len(empty_layers["layers"]) == 20
    for index, row in enumerate(empty_layers["layers"]):
        assert len(row) == 768, index
        finite(row, f"empty layer {index}")
    assert len(empty_layers["quantized_hidden"]) == 768
    finite(empty_layers["quantized_hidden"], "empty quantized hidden")
    assert run(binary, model, "layers", "") == empty_layers
    empty_embed = run(binary, model, "embed", "")
    assert len(empty_embed) == 3072
    finite(empty_embed, "empty embedding")
    assert run(binary, model, "embed", "") == empty_embed
    tokens = run(binary, model, "tokenize", "hello")
    assert tokens["text"] == "hello" and tokens["ids"]
    expected_ids = [2, *tokens["ids"]]
    logits = run(binary, model, "logits", "hello")
    assert logits["ids"] == expected_ids and len(logits["logits"]) == 8192
    finite(logits["logits"], "logits")
    layers = run(binary, model, "layers", "hello")
    assert layers["ids"] == expected_ids and len(layers["layers"]) == 20
    for index, row in enumerate(layers["layers"]):
        assert len(row) == 768, index
        finite(row, f"layer {index}")
    assert len(layers["quantized_hidden"]) == 768
    finite(layers["quantized_hidden"], "quantized hidden")
    first = run(binary, model, "embed", "hello")
    second = run(binary, model, "embed", "hello")
    assert first == second and len(first) == 3072
    finite(first, "embedding")
    norm = math.sqrt(sum(item * item for item in first))
    assert abs(norm - 1) < 1e-5, norm
    generated = run(binary, model, "generate", "hello", "4")
    # This prompt does not emit EOS inside four tokens. The bound must stay visible.
    assert generated["finished"] is False and len(generated["token_ids"]) == 4
    assert 0 <= generated["head_score"] <= 1
    repeated = run(binary, model, "generate", "hello", "4")
    assert repeated["token_ids"] == generated["token_ids"]
    assert repeated["head_score"] == generated["head_score"]
    request = json.dumps(
        {"input": "turn on the kitchen lights", "tools": [TOOL], "max_tokens": 128},
        ensure_ascii=False,
    )
    complete = run(binary, model, "complete", request=request)
    calls = complete["function_calls"]
    assert calls == [{"name": "set_lights", "arguments": {"on": True, "room": "kitchen"}}]
    assert len(complete["generation"]["token_ids"]) == 19
    assert f"{complete['generation']['head_score']:.7g}" == "0.8660065"
    again = run(binary, model, "complete", request=request)
    assert again["function_calls"] == calls
    assert again["generation"]["token_ids"] == complete["generation"]["token_ids"]
    assert again["generation"]["head_score"] == complete["generation"]["head_score"]
    print(json.dumps({
        "status": "passed",
        "commands": ["inspect", "tokenize", "logits", "layers", "embed", "generate", "complete"],
        "binary_sha256": digest(binary),
        "model_sha256": MODEL_SHA,
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
