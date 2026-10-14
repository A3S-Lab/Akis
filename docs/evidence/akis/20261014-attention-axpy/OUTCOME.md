# Attention values use one checked axpy

Date: 2026-10-14 Asia/Shanghai. Parent: `ce31570`.
Status: the same cold logits samples got faster, and the outputs matched the previous binary. Full G4 and a Needle replacement are **not** certified.

## What changed

The value mix indexed the KV deque and checked the row width on every output element. It now checks the cached row once and adds `scale * value` across that row. Each element is still one multiply and one add. ARM64 evaluates groups of four lanes with that same rounding. The dot product, the GEMV layout, and the cosine threshold were not changed.

## Executed evidence

| Check | Result | File |
| --- | --- | --- |
| fmt / clippy all targets, warnings denied | Passed | certification.log |
| Rust tests | 39 library passed, 2 CLI signal passed, 0 failed | certification.log |
| Independent parity | 31 passed; every vector cosine above 0.999; same top-1 | numerical-parity.jsonl |
| Confidence head absolute error | 0.000696, below 0.01 | numerical-parity.jsonl |
| Logits versus the previous binary | `hello` and the 256-token `<think>` sample, max absolute error 0, same top-1 | this file |
| Real-model process campaign | 13 passed, 4 failed | certification.log |
| Complete certification entry point | Exit 1 due to those four failures | certification.log |

The 17 campaign rows match the previous binary: same status, same tool calls, same token counts, and the same head scores. The four failures remain `lights-on-zh`, `lights-off-zh`, `negated-action`, and `missing-room`. `missing-room` still emits a call whose room string is `living room`. That case was not repaired.

The lowest per-layer cosine on `hello` is layer 18 at 0.999795. Layer 19 still has a large pre-norm absolute error and the same top-1. No per-layer exception was added.

## Cold process time and RSS

One sample each, macOS `/usr/bin/time -l`, pinned Needle 3 artifact, from `tools/parity.py`.

| Request | Previous binary | This binary |
| --- | --- | --- |
| 1025-token `<think>` | 272,236,544 bytes, 9.513 seconds | 272,285,696 bytes, 6.934 seconds |
| 2048-token `user` | 273,121,280 bytes, 18.793 seconds | 273,170,432 bytes, 13.406 seconds |

RSS stays beside the previous sample. These are single cold samples, not percentiles, and not an 8192-token measurement. A separate 256-token `<think>` process on this binary took 1.98, 1.99, and 2.00 seconds wall time.

## Reproduction

```sh
.cache/oracle/bin/python tools/parity.py .cache/needle3.cact .cache/needle target/release/a3s-akis
tools/certify.sh .cache/needle3.cact
```

## Not claimed

Packed CQ GEMM, a full allocation audit, soak, other platforms, max context, a second depth, and native behavioral compatibility remain open. No full gate is declared complete, and this is not a Needle replacement.
