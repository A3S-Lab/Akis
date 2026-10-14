# Longer-context, per-layer and quantization parity

Date: 2026-10-14 Asia/Shanghai. Parent: `8f7e8fd`.
Status: the new numerical cases passed inside the existing state budget. Full G1 and a Needle replacement are **not** certified.

## What changed

`layers MODEL TEXT` returns the last-token lane mean after every layer and the activation-quantized final hidden. Ordinary logits, embedding and generation do not capture those vectors, so their residency stays on the previous path.

`tools/parity.py` is still the only numerical oracle. A missing model, binary, or upstream `architecture.py` / `export.py` / `quantize.py` exits non-zero. The run now records, on the pinned Needle 3 artifact:

- the retained short prompts `hello`, `zh-lights`, and `tool-prompt`
- exact lengths 1280 (`<think>`), 1536 (`<tool_call>`), and 2048 (`user`), plus a fresh 1025-token (`<think>`) measurement
- last-token lane means for all 20 layers on `hello`, compared with upstream `hidden_cells` at `quant=True` and the artifact KV window
- the A8 final hidden against `fake_quant_act` of the final norm, and deployed logits against both `quant=True` and `quant=False` on `the`
- the retained confidence-pool embedding and confidence head

Prompts are repetitions of stable tokenizer pieces. The CLI prepends BOS, and the script rejects a piece that does not encode to the requested body length. Each vector row stores cosine, max absolute error, and top-1 agreement. The confidence head stores absolute error. The threshold is unchanged: cosine above 0.999 and identical top-1, or absolute error below 0.01 for the scalar head.

## Executed evidence

| Check | Result | File |
| --- | --- | --- |
| fmt / clippy all targets, warnings denied | Passed | certification.log |
| Rust tests | 36 library passed, 2 CLI signal passed, 0 failed | certification.log |
| Independent parity, first run | 31 passed, 0 failed | numerical-parity.jsonl |
| Independent parity, second run | Same 31 case ids and the same top-1 bits | numerical-parity-repeat.jsonl |
| Missing weight / missing reference | Exit 1, names the missing file | recorded below |
| Real-model process campaign | 13 passed, 4 failed | live-e2e.jsonl |
| Complete certification entry point | Exit 1 due to those four failures | certification.log |

The four failures are unchanged: `lights-on-zh`, `lights-off-zh`, `negated-action`, and `missing-room`. No expected answer, threshold, or case was edited, and no intent word list was added.

Logits cosine on the new long prompts was 0.99999577 to 0.99999899, with matching top-1. Every layer cosine stayed above 0.999. The smallest was layer 18 at 0.999764. Layer 19's largest absolute error was 175250 on a lane mean whose dominant element is about 1.56e7 (relative error about 1.12%). After the final norm and A8 activation quant, that hidden's max absolute error was 0.0004357 and its cosine was 0.999209. Deployed logits on `the` matched `quant=True` with cosine 1.0. The float path differed by at most 0.273 and kept the same top-1. Confidence-pool cosine was 0.9999085. Confidence-head absolute error was 0.00013725.

A missing model path and a missing reference directory each exited 1 before any comparison.

## Peak memory beside the retained sample

Cold `logits` process, macOS `/usr/bin/time -l`, same pinned artifact. The request completed; the state budget did not reject it. There is no successful unbounded result.

| Request | Peak RSS | Wall time | Cosine | Top-1 |
| --- | --- | --- | --- | --- |
| Retained 1025-token sample, older binary | 536,035,328 bytes | retained | 1.0 against that binary's predecessor | match |
| 1025-token `<think>` on this binary | 536,051,712 bytes | 13.75 s | 0.99999797 | match |
| 2048-token `user`, longest new request | 536,690,688 bytes | 25.74 s | 0.99999899 | match |

The 1025-token remeasure is 16,384 bytes above the retained sample. The 2048-token peak is 655,360 bytes above that retained sample. A second run kept the same top-1 bits and the same cosine values; its peaks were 535,789,568 bytes (1025) and 536,952,832 bytes (2048). KV state remains windowed, so this is not a measurement of an 8192-token context.

## Reproduction

```sh
.cache/oracle/bin/python tools/parity.py .cache/needle3.cact .cache/needle target/release/a3s-akis
python3 tools/live_e2e.py .cache/needle3.cact
tools/certify.sh .cache/needle3.cact
```

## Not claimed

Max context (8192), a second trained depth, native behavioral compatibility, cross-platform execution, calibration, host or Apofasi adaptation, packed kernels, and soak remain open. The optional `--boundaries` set of 255/256/257 and 1023/1024/1025 is unchanged and was not re-executed as six extra rows in this run; the 1025-token length was remeasured here. No full gate is declared complete, and this is not a Needle replacement.
