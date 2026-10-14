# Kronecker factor rows use the existing axpy

Date: 2026-10-14 Asia/Shanghai. Parent: `11c96b2`.
Status: the outputs matched the previous binary. The 2048-token cold sample moved by 0.206 seconds, inside one cold-sample variation, so the factor product stops here. Full G4 and a Needle replacement are **not** certified.

## What changed

Both inner rows of the two square-factor loops were scalar triple loops. Each row is now one call to the axpy already used for attention values. Each element is still one multiply and one add, in the same loop order. The 32-wide factors are not a special case, and the factor flats are not cached.

A contract test compares that routine with the scalar triple loop on small deterministic matrices, including row lengths that are not multiples of four. The test does not read a prompt or the pinned archive.

## Executed evidence

| Check | Result | File |
| --- | --- | --- |
| fmt / clippy all targets, warnings denied | Passed | certification.log |
| Rust tests | 40 library passed, 2 CLI signal passed, 0 failed | certification.log |
| Kronecker contract | Scalar triple loop, every lane, several factor sizes | certification.log |
| Independent parity | 31 passed; every vector cosine above 0.999; same top-1 | numerical-parity.jsonl |
| Confidence head absolute error | 0.000696, below 0.01 | numerical-parity.jsonl |
| Logits versus the previous binary | `hello` and the 256-token `<think>` sample, max absolute error 0, same top-1 | this file |
| Real-model process campaign | 13 passed, 4 failed | certification.log |
| Complete certification entry point | Exit 1 due to those four failures | certification.log |

The 17 campaign rows match the previous binary: same status, same tool calls, same token counts, and the same head scores. The four failures remain `lights-on-zh`, `lights-off-zh`, `negated-action`, and `missing-room`. `missing-room` still emits a call whose room string is `living room`. That case was not repaired.

The lowest per-layer cosine on `hello` is layer 18 at 0.999795. Layer 19 still has a large pre-norm absolute error and the same top-1. No per-layer exception was added. The JAX error stats match the previous binary, which agrees with the bit-identical logits above.

## Cold process time and RSS

One sample each, macOS `/usr/bin/time -l`, pinned Needle 3 artifact, from `tools/parity.py`.

| Request | Previous binary | This binary |
| --- | --- | --- |
| 1025-token `<think>` | 272,285,696 bytes, 6.934 seconds | 271,810,560 bytes, 6.762 seconds |
| 2048-token `user` | 273,170,432 bytes, 13.406 seconds | 273,006,592 bytes, 13.200 seconds |

The 2048-token wall time moved by 0.206 seconds. That sits inside the variation of one cold sample, so this slice records the numbers and does not keep tuning the factor product. RSS stays beside the previous sample. These are single cold samples, not percentiles, and not an 8192-token measurement.

## Reproduction

```sh
.cache/oracle/bin/python tools/parity.py .cache/needle3.cact .cache/needle target/release/a3s-akis
tools/certify.sh .cache/needle3.cact
```

## Not claimed

Packed CQ GEMM, a full allocation audit, soak, other platforms, max context, a second depth, and native behavioral compatibility remain open. No full gate is declared complete, and this is not a Needle replacement.
