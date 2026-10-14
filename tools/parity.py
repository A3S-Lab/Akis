"""Independent parity against the pinned upstream JAX forward, using identical CQ bytes.

Development only. Usage: python tools/parity.py MODEL REFERENCE_DIR AKIS_BINARY
REFERENCE_DIR contains upstream architecture.py, export.py and quantize.py.
"""
import importlib
import json
import math
import pathlib
import struct
import subprocess
import sys
import time
import types
import argparse

import jax
import jax.numpy as jnp
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("model")
parser.add_argument("reference")
parser.add_argument("binary")
parser.add_argument("--boundaries", action="store_true", help="Also verify 255/256/257 and 1023/1024/1025 tokens")
args = parser.parse_args()
path, source, binary = args.model, args.reference, args.binary
required = [
    pathlib.Path(path), pathlib.Path(binary),
    pathlib.Path(source) / "architecture.py",
    pathlib.Path(source) / "export.py",
    pathlib.Path(source) / "quantize.py",
]
missing = [str(item) for item in required if not item.is_file()]
if missing:
    raise SystemExit("missing required file: " + ", ".join(missing))
package = types.ModuleType("needle_reference")
package.__path__ = [str(pathlib.Path(source).resolve())]
sys.modules[package.__name__] = package
arch = importlib.import_module("needle_reference.architecture")
export = importlib.import_module("needle_reference.export")
quant = importlib.import_module("needle_reference.quantize")
raw = pathlib.Path(path).read_bytes()
codebook = np.frombuffer(raw, "<f4", count=28, offset=196).copy()
original_codebook = export._cq_codebook_np
export._cq_codebook_np = lambda bits, group: (
    codebook[(1 << bits) - 4:(1 << (bits + 1)) - 4]
    if bits in (2, 3, 4) and group == 128 else original_codebook(bits, group)
)
g, tensors = export.read_export(path)
tok = export.RefTokenizer(export.parse_tokenizer_blob(tensors[-1]))
params = {"embedding": {"embedding": tensors[0]}}
blocks = []
i = 1
for layer in range(g["num_layers"]):
    def take():
        global i
        value = tensors[i]
        i += 1
        return value
    block = {"ZCRMSNorm_0": {"scale": take()}}
    attention = {key: {"kernel": take().T} for key in ("q_proj", "k_proj", "v_proj")}
    if g["qkv_conv_taps"]:
        for key in ("q_taps", "k_taps", "v_taps"):
            attention[key] = take()
    for key in ("q_norm", "k_norm"):
        attention[key] = {"scale": take()}
    for key in ("gate_proj", "out_proj"):
        attention[key] = {"kernel": take().T}
    block["self_attn"] = attention
    block["post_attn_norm"] = {"scale": take()}
    block["attn_gate"] = take().reshape(())
    block["pre_hada_norm"] = {"scale": take()}
    block["hadamard_mlp"] = {key: take() for key in (
        "d1", "d2", "b2", "d3", "d4", "w1a", "w1b", "w2a", "w2b", "w3a", "w3b", "cond_v", "cond_u")}
    blocks.append(block)
params["stack"] = {"layers": {"block": jax.tree_util.tree_map(lambda *a: np.stack(a), *blocks)}}
for key in ("mhc_a_pre", "mhc_a_post", "mhc_a_res", "mhc_b_pre", "mhc_b_post", "mhc_b_res"):
    params["stack"][key] = tensors[i]
    i += 1
for key in ("mhc_phi_pre", "mhc_phi_post", "mhc_phi_res"):
    value = tensors[i]
    params["stack"][key] = value.reshape(g["num_layers"], -1, value.shape[-1]).transpose(0, 2, 1)
    i += 1
# The upstream reference derives permutations from its fixed seeds; verify the container agrees.
p1, p2 = arch._hada_perms(g["hada_n"])
np.testing.assert_array_equal(p1, tensors[i])
np.testing.assert_array_equal(p2, tensors[i + 1])
i += 2
for site in range(len(g["engram_layers"])):
    params[f"engrams_{site}"] = {
        "embedding": tensors[i].reshape(g["num_engram_tables"], g["engram_slots"], g["engram_sub_dim"]),
        "key_proj": {"kernel": tensors[i + 1].T},
        "value_proj": {"kernel": tensors[i + 2].T}, "taps": tensors[i + 3],
    }
    i += 4
params["stack"]["final_norm"] = {"scale": tensors[i]}
heads = {}
if i + 2 < len(tensors):
    codes = tensors[i + 1].astype(int).tolist()
    i += 2
    for code in codes:
        heads[code] = tensors[i:i + 6]
        i += 7 if code == 3 else 6
cfg = arch.TransformerConfig(**g, dtype="float32", flash=False, remat=False,
                            engram_heads=g["num_engram_tables"] // len(g["engram_orders"]))
quant.configure_deploy(act_bits=8, kv_bits=8)
model = arch.SimpleAttentionNetwork(cfg)
params = jax.tree_util.tree_map(jnp.asarray, params)

def reference_logits(ids, quantize):
    tokens = jnp.asarray([ids])
    positions = jnp.arange(len(ids))
    mask = arch.make_causal_mask(len(ids))
    if g["kv_window"]:
        mask &= ((positions[:, None] - positions[None, :]) < g["kv_window"])[None, None]
    expected = np.asarray(model.apply({"params": params}, tokens, mask=mask, quant=quantize))[0, -1]
    return np.asarray(expected, np.float32)

def compare(expected, found, case, **extra):
    expected = np.asarray(expected, np.float32)
    found = np.asarray(found, np.float32)
    if expected.shape != found.shape or not np.all(np.isfinite(expected)) or not np.all(np.isfinite(found)):
        raise SystemExit(f"non-finite or mismatched comparison: {case}")
    denom = float(np.linalg.norm(expected) * np.linalg.norm(found))
    if denom == 0.0:
        raise SystemExit(f"zero vector: {case}")
    cosine = float(np.dot(expected, found) / denom)
    row = {"case": case, "cosine": cosine,
           "max_absolute_error": float(np.max(np.abs(expected - found))),
           "top1_agreement": bool(int(expected.argmax()) == int(found.argmax())),
           "reference_top": int(expected.argmax()), "akis_top": int(found.argmax()), **extra}
    print(json.dumps(row), flush=True)
    if cosine <= 0.999 or not row["top1_agreement"]:
        raise SystemExit(json.dumps(row))
    return row

def akis(command, text):
    proc = subprocess.run([binary, command, path, text], capture_output=True, text=True)
    if proc.returncode != 0 or not proc.stdout:
        detail = proc.stderr.strip() or proc.stdout.strip() or command
        raise SystemExit(f"akis {command} failed: {detail}")
    return json.loads(proc.stdout)

def repeated(piece, total_tokens):
    body = total_tokens - 1
    text = piece * body
    ids = tok.encode(text)
    if len(ids) != body:
        raise SystemExit(f"{piece!r} did not encode to {body} tokens")
    return text

def measure_logits(text, total_tokens):
    if sys.platform != "darwin":
        raise SystemExit("peak RSS measurement requires macOS /usr/bin/time -l")
    started = time.perf_counter()
    proc = subprocess.run(["/usr/bin/time", "-l", binary, "logits", path, text],
                          capture_output=True, text=True)
    elapsed = time.perf_counter() - started
    if proc.returncode != 0 or not proc.stdout:
        detail = proc.stderr.strip() or "measured logits failed"
        raise SystemExit(detail)
    rss = [int(line.split()[0]) for line in proc.stderr.splitlines()
           if line.strip().endswith("maximum resident set size")]
    if len(rss) != 1:
        raise SystemExit("time -l did not report maximum resident set size")
    actual = json.loads(proc.stdout)
    if len(actual["ids"]) != total_tokens or not all(math.isfinite(x) for x in actual["logits"]):
        raise SystemExit(f"measured logits were not a finite {total_tokens}-token result")
    return actual, rss[0], elapsed

cases = [
    ("hello", "hello"),
    ("zh-lights", "你好，关闭客厅的灯。"),
    ("tool-prompt", '<|im_start|>user\n<tools>[{"name":"set_lights","description":"Turn lights on or off","parameters":{"type":"object","properties":{"room":{"type":"string"},"on":{"type":"boolean"}},"required":["room","on"]}}]</tools>\nturn on the kitchen lights<|im_end|>\n<|im_start|>assistant\n'),
]
if args.boundaries:
    for length in [255, 256, 257, 1023, 1024, 1025]:
        cases.append((f"boundary-{length}", repeated("<think>", length)))
long_cases = [
    ("long-1280-think", repeated("<think>", 1280)),
    ("long-1536-tool-call", repeated("<tool_call>", 1536)),
    ("long-2048-user", repeated("user", 2048)),
]
results = []
for case, text in cases:
    actual = akis("logits", text)
    ids = [2] + tok.encode(text)
    if actual["ids"] != ids:
        raise SystemExit(f"tokenizer mismatch: {case}")
    results.append(compare(reference_logits(ids, True), actual["logits"], case, tokens=len(ids)))

layer_text = "hello"
layer_actual = akis("layers", layer_text)
layer_ids = [2] + tok.encode(layer_text)
if layer_actual["ids"] != layer_ids:
    raise SystemExit("tokenizer mismatch: per-layer")
cells = np.asarray(model.apply(
    {"params": params}, jnp.asarray([layer_ids]), quant=True, window=g["kv_window"] or 0,
    method=model.hidden_cells))
reference_layers = np.asarray(cells[0, -1, 1:], np.float32)
found_layers = np.asarray(layer_actual["layers"], np.float32)
if found_layers.shape != reference_layers.shape:
    raise SystemExit(f"layer shape mismatch: {found_layers.shape} vs {reference_layers.shape}")
for index, (expected, found) in enumerate(zip(reference_layers, found_layers)):
    results.append(compare(expected, found, f"layer-{index:02d}", tokens=len(layer_ids)))
scale = np.asarray(params["stack"]["final_norm"]["scale"], np.float32)
last = reference_layers[-1]
rms = np.sqrt(np.mean(last.astype(np.float32) ** 2) + 1e-6)
normed = (1.0 + scale) * last / rms
quantized = np.asarray(quant.fake_quant_act(jnp.asarray(normed)), np.float32)
results.append(compare(
    quantized, layer_actual["quantized_hidden"], "quantization-activation", tokens=len(layer_ids)))
quant_text = "the"
quant_actual = akis("logits", quant_text)
quant_ids = [2] + tok.encode(quant_text)
if quant_actual["ids"] != quant_ids:
    raise SystemExit("tokenizer mismatch: quantization-logits")
deployed = reference_logits(quant_ids, True)
float_path = reference_logits(quant_ids, False)
quant_row = compare(
    deployed, quant_actual["logits"], "quantization-logits", tokens=len(quant_ids),
    float_path_max_absolute_error=float(np.max(np.abs(deployed - float_path))),
    float_path_top1_agreement=bool(int(deployed.argmax()) == int(float_path.argmax())))
results.append(quant_row)

# Exported head weights are already CQ-decoded: call upstream pooling directly
# instead of re-quantizing the matrices with the training straight-through path.
def pooled(ids, code):
    tokens = jnp.asarray([ids])
    cells = model.apply({"params": params}, tokens, quant=True, window=g["kv_window"],
                        method=model.hidden_cells)
    probes, gain, query, bias, _, _ = heads[code]
    probes = probes.reshape(g["num_layers"] + 1, gain.shape[1], g["d_model"])
    return np.asarray(arch.probe_pool(cells, tokens != 0, jnp.asarray(probes),
                                     jnp.asarray(gain), jnp.asarray(query),
                                     jnp.asarray(bias), dtype=jnp.float32))[0]

if 1 not in heads and 2 not in heads:
    raise SystemExit("pinned artifact is missing embedding and confidence heads")
code = 1 if 1 in heads else 2
actual = np.asarray(akis("embed", "hello"), np.float32)
expected = pooled([2] + tok.encode("hello"), code)
if code == 1:
    expected = heads[1][4] @ expected
expected = expected / np.sqrt(np.sum(expected**2) + 1e-12)
results.append(compare(
    expected, actual, "embedding" if code == 1 else "confidence-pool-embedding",
    dimensions=int(actual.shape[0])))
if 2 not in heads:
    raise SystemExit("pinned artifact is missing the confidence head")
proc = subprocess.run([binary, "generate", path, "hello", "4"], capture_output=True, text=True)
if proc.returncode != 0 or not proc.stdout:
    raise SystemExit(proc.stderr.strip() or "akis generate failed")
generation = json.loads(proc.stdout)
pool = pooled([2] + tok.encode("hello") + generation["token_ids"], 2)
logit = float((heads[2][4] @ pool + heads[2][5]).item())
expected_score = float(jax.nn.sigmoid(logit))
actual_score = generation["head_score"]
if actual_score is None or not math.isfinite(actual_score):
    raise SystemExit("confidence head did not return a finite score")
absolute_error = abs(expected_score - actual_score)
head_row = {"case": "confidence-head", "reference": expected_score, "akis": actual_score,
            "absolute_error": absolute_error, "top1_agreement": absolute_error < 0.01}
print(json.dumps(head_row), flush=True)
if absolute_error >= 0.01:
    raise SystemExit(json.dumps(head_row))
results.append(head_row)

for case, text in long_cases[:-1]:
    actual = akis("logits", text)
    ids = [2] + tok.encode(text)
    if actual["ids"] != ids:
        raise SystemExit(f"tokenizer mismatch: {case}")
    results.append(compare(reference_logits(ids, True), actual["logits"], case, tokens=len(ids)))
longest_case, longest_text = long_cases[-1]
longest_ids = [2] + tok.encode(longest_text)
measured, peak_rss, elapsed = measure_logits(longest_text, len(longest_ids))
if measured["ids"] != longest_ids:
    raise SystemExit(f"tokenizer mismatch: {longest_case}")
results.append(compare(
    reference_logits(longest_ids, True), measured["logits"], longest_case, tokens=len(longest_ids),
    maximum_resident_bytes=peak_rss, seconds=elapsed,
    retained_1025_token_peak_rss_bytes=536035328))
sample_text = repeated("<think>", 1025)
sample, sample_rss, sample_seconds = measure_logits(sample_text, 1025)
sample_ids = [2] + tok.encode(sample_text)
if sample["ids"] != sample_ids:
    raise SystemExit("tokenizer mismatch: budget-1025-think")
results.append(compare(
    reference_logits(sample_ids, True), sample["logits"], "budget-1025-think", tokens=1025,
    maximum_resident_bytes=sample_rss, seconds=sample_seconds,
    retained_1025_token_peak_rss_bytes=536035328))
print(json.dumps({"passed": len(results)}), flush=True)
