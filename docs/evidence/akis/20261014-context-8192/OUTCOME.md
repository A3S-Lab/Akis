# One 8192-token logits request

Date: 2026-10-14 Asia/Shanghai. Parent: `845f6f6`.
Status: the request completed inside the current state budget. Max context, a second depth, and a Needle replacement are **not** certified.

## What changed

No inference code changed. `InferenceLimits` stayed at its default, including `max_state_bytes` of 4GiB. The release binary is the kron-axpy binary.

The prompt is `<think>` repeated so the body is 8191 tokens, and the CLI adds BOS. The request is 8192 tokens. The geometry still caps each layer's cached attention at `kv_window` 256, including the global layers.

## Result

One cold sample, macOS `/usr/bin/time -l`, pinned Needle 3 artifact.

| Item | Value |
| --- | --- |
| Tokens | 8192, ids matched `[BOS] + encode("<think>" * 8191)` |
| Cold RSS | 272,875,520 bytes |
| Wall time | 49.39 seconds |
| Cosine versus the JAX reference | 0.999999940 |
| Max absolute error | 0.000061035 |
| Top-1 | 38 and 38 |

`check_request` accepted the length. The state budget was not raised.

## Not claimed

This is one repeated-piece prompt on one machine. It does not certify max context, a second depth, native behavioral compatibility, or another platform. No full gate is declared complete, and this is not a Needle replacement.
