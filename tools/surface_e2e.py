"""One process chain for the seven CLI commands on the pinned archive.

Usage: python3 tools/surface_e2e.py MODEL BINARY
"""
import hashlib
import json
import math
import pathlib
import subprocess
import sys

from live_e2e import MODEL_SHA, TOOL, model_prompt, token_record


# Last matching rule wins. File and process access stay open; network does not.
OFFLINE = "(version 1)(allow default)(deny network*)"


def digest(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def launch(binary, args, request=None):
    try:
        return subprocess.run(
            ["sandbox-exec", "-p", OFFLINE, binary, *args],
            input=request,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except FileNotFoundError as exc:
        raise SystemExit("sandbox-exec is required to prove offline inference") from exc


def run(binary, model, command, *tail, request=None):
    proc = launch(binary, [command, model, *tail], request)
    if proc.returncode != 0 or proc.stderr or not proc.stdout:
        raise SystemExit(proc.stderr.strip() or f"{command} exit {proc.returncode}")
    return json.loads(proc.stdout)


def finite(values, label):
    if not values or not all(math.isfinite(item) for item in values):
        raise SystemExit(f"non-finite {label}")


PROOF_FIELDS = {
    "signature",
    "attestation",
    "accelerator",
    "microbatch",
    "shapeProfile",
    "timestamp",
}


def exact_keys(value, expected, label):
    if set(value) != set(expected):
        raise SystemExit(f"{label} keys {sorted(value)}")


def forbid_fields(value, banned):
    if isinstance(value, dict):
        found = banned.intersection(value)
        if found:
            raise SystemExit(f"unexpected fields {sorted(found)}")
        for item in value.values():
            forbid_fields(item, banned)
    elif isinstance(value, list):
        for item in value:
            forbid_fields(item, banned)


def assert_bare(value, keys):
    exact_keys(value, keys, "bare result")
    forbid_fields(value, PROOF_FIELDS | {"receipt"})


def assert_receipt(receipt, input_ids, output_ids):
    exact_keys(receipt, ["schema", "model", "runtime", "input", "output"], "receipt")
    forbid_fields(receipt, PROOF_FIELDS)
    if receipt["schema"] != "a3s.power.embedded-execution-receipt.v1":
        raise SystemExit(receipt["schema"])
    exact_keys(receipt["model"], ["family", "revision", "weightsSha256"], "model")
    exact_keys(receipt["runtime"], ["name", "version", "device"], "runtime")
    if receipt["model"]["family"] != "needle3" or receipt["model"]["weightsSha256"] != MODEL_SHA:
        raise SystemExit("receipt model identity")
    if receipt["runtime"]["name"] != "a3s-power-native" or receipt["runtime"]["device"] != "cpu":
        raise SystemExit("receipt runtime identity")
    for side, ids in (("input", input_ids), ("output", output_ids)):
        exact_keys(receipt[side], ["representation", "sha256", "byteLength", "itemCount"], side)
        digest, size, count = token_record(ids)
        if (
            receipt[side]["representation"] != "token-ids"
            or (receipt[side]["sha256"], receipt[side]["byteLength"], receipt[side]["itemCount"])
            != (digest, size, count)
        ):
            raise SystemExit(f"{side} receipt digest")
    changed = list(output_ids)
    changed[-1] ^= 1
    if token_record(changed)[0] == receipt["output"]["sha256"]:
        raise SystemExit("changed token matched the output digest")


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: python3 tools/surface_e2e.py MODEL BINARY")
    model, binary = sys.argv[1], sys.argv[2]
    if digest(model) != MODEL_SHA:
        print(json.dumps({"status": "blocked", "reason": "model sha256 differs"}))
        return 2
    missing = launch(binary, ["inspect", str(pathlib.Path(model).with_name("missing-akis.cact"))])
    if (
        missing.returncode == 0
        or missing.stdout
        or "panicked" in missing.stderr
        or "No such file" not in missing.stderr
        or "embedded-execution-receipt" in missing.stderr
    ):
        raise SystemExit(missing.stderr.strip() or "missing model did not fail closed")
    info = run(binary, model, "inspect")
    assert_bare(info, ["sha256", "tensors", "geometry"])
    geometry = info["geometry"]
    assert info["sha256"] == MODEL_SHA and info["tensors"] == 581
    assert geometry["context"] == 8192 and geometry["layers"] == 20
    assert geometry["width"] == 768 and geometry["vocab"] == 8192
    empty_logits = run(binary, model, "logits", "")
    assert_bare(empty_logits, ["ids", "logits"])
    assert empty_logits["ids"] == [2] and len(empty_logits["logits"]) == 8192
    finite(empty_logits["logits"], "empty logits")
    assert run(binary, model, "logits", "") == empty_logits
    empty_layers = run(binary, model, "layers", "")
    assert_bare(empty_layers, ["ids", "layers", "quantized_hidden"])
    assert empty_layers["ids"] == [2] and len(empty_layers["layers"]) == 20
    for index, row in enumerate(empty_layers["layers"]):
        assert len(row) == 768, index
        finite(row, f"empty layer {index}")
    assert len(empty_layers["quantized_hidden"]) == 768
    finite(empty_layers["quantized_hidden"], "empty quantized hidden")
    assert run(binary, model, "layers", "") == empty_layers
    empty_embed = run(binary, model, "embed", "")
    forbid_fields(empty_embed, PROOF_FIELDS | {"receipt"})
    assert isinstance(empty_embed, list) and len(empty_embed) == 3072
    finite(empty_embed, "empty embedding")
    assert run(binary, model, "embed", "") == empty_embed
    tokens = run(binary, model, "tokenize", "hello")
    assert_bare(tokens, ["ids", "text"])
    assert tokens["text"] == "hello" and tokens["ids"]
    expected_ids = [2, *tokens["ids"]]
    logits = run(binary, model, "logits", "hello")
    assert_bare(logits, ["ids", "logits"])
    assert logits["ids"] == expected_ids and len(logits["logits"]) == 8192
    finite(logits["logits"], "logits")
    layers = run(binary, model, "layers", "hello")
    assert_bare(layers, ["ids", "layers", "quantized_hidden"])
    assert layers["ids"] == expected_ids and len(layers["layers"]) == 20
    for index, row in enumerate(layers["layers"]):
        assert len(row) == 768, index
        finite(row, f"layer {index}")
    assert len(layers["quantized_hidden"]) == 768
    finite(layers["quantized_hidden"], "quantized hidden")
    first = run(binary, model, "embed", "hello")
    forbid_fields(first, PROOF_FIELDS | {"receipt"})
    second = run(binary, model, "embed", "hello")
    assert first == second and len(first) == 3072
    finite(first, "embedding")
    norm = math.sqrt(sum(item * item for item in first))
    assert abs(norm - 1) < 1e-5, norm
    generated = run(binary, model, "generate", "hello", "4")
    exact_keys(
        generated,
        [
            "text",
            "token_ids",
            "input_tokens",
            "finished",
            "mean_token_probability",
            "head_score",
            "receipt",
        ],
        "generate",
    )
    forbid_fields(generated, PROOF_FIELDS)
    assert_receipt(generated["receipt"], [2, *tokens["ids"]], generated["token_ids"])
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
    exact_keys(
        complete,
        ["function_calls", "suppressed_calls", "ungrounded", "generation"],
        "complete",
    )
    forbid_fields(complete, PROOF_FIELDS)
    exact_keys(
        complete["generation"],
        [
            "text",
            "token_ids",
            "input_tokens",
            "finished",
            "mean_token_probability",
            "head_score",
            "receipt",
        ],
        "generation",
    )
    calls = complete["function_calls"]
    assert calls == [{"name": "set_lights", "arguments": {"on": True, "room": "kitchen"}}]
    assert len(complete["generation"]["token_ids"]) == 19
    assert f"{complete['generation']['head_score']:.7g}" == "0.8660065"
    again = run(binary, model, "complete", request=request)
    assert again["function_calls"] == calls
    assert again["generation"]["token_ids"] == complete["generation"]["token_ids"]
    assert again["generation"]["head_score"] == complete["generation"]["head_score"]
    assert receipt_digests(binary, model, complete) == receipt_digests(binary, model, again)
    print(json.dumps({
        "status": "passed",
        "commands": ["inspect", "tokenize", "logits", "layers", "embed", "generate", "complete"],
        "binary_sha256": digest(binary),
        "model_sha256": MODEL_SHA,
    }))
    return 0


def receipt_digests(binary, model, body):
    receipt = body["generation"]["receipt"]
    tokenized = run(binary, model, "tokenize", model_prompt("turn on the kitchen lights", [TOOL]))
    assert_receipt(receipt, [2, *tokenized["ids"]], body["generation"]["token_ids"])
    bound = token_record([2, *tokenized["ids"]])
    assert (
        receipt["input"]["sha256"],
        receipt["input"]["byteLength"],
        receipt["input"]["itemCount"],
    ) == bound
    output = token_record(body["generation"]["token_ids"])
    assert (
        receipt["output"]["sha256"],
        receipt["output"]["byteLength"],
        receipt["output"]["itemCount"],
    ) == output
    changed = [2, *tokenized["ids"]]
    changed[-1] ^= 1
    assert token_record(changed)[0] != bound[0]
    return bound[0], output[0]


if __name__ == "__main__":
    sys.exit(main())
