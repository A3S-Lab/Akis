"""Independent parity against the pinned upstream JAX forward, using identical CQ bytes.

Development only. Usage: python tools/parity.py MODEL REFERENCE_DIR AKIS_BINARY
REFERENCE_DIR contains upstream architecture.py, export.py and quantize.py.
"""
import importlib
import json
import pathlib
import struct
import subprocess
import sys
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

cases = ["hello", "你好，关闭客厅的灯。", '<|im_start|>user\n<tools>[{"name":"set_lights","description":"Turn lights on or off","parameters":{"type":"object","properties":{"room":{"type":"string"},"on":{"type":"boolean"}},"required":["room","on"]}}]</tools>\nturn on the kitchen lights<|im_end|>\n<|im_start|>assistant\n']
if args.boundaries:
    for length in [255,256,257,1023,1024,1025]:
        prefix = len(tok.encode("<think>"))-1
        text = "<think>" * (length-1-prefix)
        assert len(tok.encode(text))+1 == length
        cases.append(text)
results = []
for text in cases:
    actual = json.loads(subprocess.check_output([binary, "logits", path, text], text=True))
    ids = [2] + tok.encode(text)
    assert actual["ids"] == ids, "tokenizer mismatch"
    tokens = jnp.asarray([ids])
    positions = jnp.arange(len(ids))
    mask = arch.make_causal_mask(len(ids))
    if g["kv_window"]:
        mask &= ((positions[:, None] - positions[None, :]) < g["kv_window"])[None, None]
    expected = np.asarray(model.apply({"params": params}, tokens, mask=mask, quant=True))[0, -1]
    found = np.asarray(actual["logits"], np.float32)
    cosine = float(np.dot(expected, found) / (np.linalg.norm(expected) * np.linalg.norm(found)))
    row = {"tokens": len(ids), "cosine": cosine, "max_absolute_error": float(np.max(np.abs(expected - found))),
           "reference_top": int(expected.argmax()), "akis_top": int(found.argmax())}
    print(json.dumps(row), flush=True)
    assert cosine > 0.999, row
    assert expected.argmax() == found.argmax(), row
    results.append(row)
print(json.dumps({"passed": len(results), "comparisons": results}), flush=True)

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

if 1 in heads or 2 in heads:
    code = 1 if 1 in heads else 2
    actual = np.asarray(json.loads(subprocess.check_output(
        [binary, "embed", path, "hello"], text=True)), np.float32)
    expected = pooled([2] + tok.encode("hello"), code)
    if code == 1:
        expected = heads[1][4] @ expected
    expected = expected / np.sqrt(np.sum(expected**2) + 1e-12)
    cosine = float(np.dot(actual, expected) / (np.linalg.norm(actual)*np.linalg.norm(expected)))
    print(json.dumps({"head": "embedding" if code == 1 else "confidence-pool-fallback",
                      "dimensions": len(actual), "cosine": cosine}), flush=True)
    assert cosine > 0.999

if 2 in heads:
    generation = json.loads(subprocess.check_output([binary, "generate", path, "hello", "4"], text=True))
    pool = pooled([2] + tok.encode("hello") + generation["token_ids"], 2)
    logit = float((heads[2][4] @ pool + heads[2][5]).item())
    expected = float(jax.nn.sigmoid(logit))
    actual = generation["head_score"]
    print(json.dumps({"head": "confidence", "reference": expected, "akis": actual,
                      "absolute_error": abs(expected-actual)}), flush=True)
    assert abs(expected - actual) < 0.01
