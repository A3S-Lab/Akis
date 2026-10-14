# Complete projections use a contiguous GEMV

Date: 2026-10-14 Asia/Shanghai. Parent: `1912a70`.
Status: the same cold logits samples got faster, and the numerical gate still passed. Full G4 and a Needle replacement are **not** certified.

## What changed

Engram tables stay packed. Complete projections — the token table, attention and engram projections, and probe output projections — are transposed once at load into contiguous `[in, out]`. A step multiplies that layout directly. Candle's CPU GEMV uses its contiguous kernel for that stride. Sliced mixes stay in archive `[out, in]` order, because a step narrows output rows. The transpose is in place, so load does not keep a second copy of the matrix.

The values are a permutation of the archive matrix. The GEMV accumulation order is not the strided kernel's order, so logits are not bit-identical to the previous binary. The existing cosine gate and top-1 agreement still hold. The threshold was not changed.

## Executed evidence

| Check | Result | File |
| --- | --- | --- |
| fmt / clippy all targets, warnings denied | Passed | certification.log |
| Rust tests | 38 library passed, 2 CLI signal passed, 0 failed | certification.log |
| Independent parity | 31 passed; every vector cosine above 0.999; same top-1 | numerical-parity.jsonl |
| Confidence head absolute error | 0.000696, below 0.01 | numerical-parity.jsonl |
| Real-model process campaign | 13 passed, 4 failed | certification.log |
| Complete certification entry point | Exit 1 due to those four failures | certification.log |

The four failures are unchanged: `lights-on-zh`, `lights-off-zh`, `negated-action`, and `missing-room`. Passing cases kept the same tool calls and token counts. Head scores moved by a few thousandths. `missing-room` still emits a call when the case expects none; the incorrect room string is now `living room` rather than empty. That case was not repaired.

The lowest per-layer cosine on `hello` is layer 18 at 0.999795. Layer 19 still has a large pre-norm absolute error and the same top-1. No per-layer exception was added.

## Cold process time and RSS

One sample each, macOS `/usr/bin/time -l`, pinned Needle 3 artifact, from `tools/parity.py`.

| Request | Previous binary | This binary |
| --- | --- | --- |
| 1025-token `<think>` | 271,843,328 bytes, 12.818 seconds | 272,236,544 bytes, 9.513 seconds |
| 2048-token `user` | 272,646,144 bytes, 25.622 seconds | 273,121,280 bytes, 18.793 seconds |

The retained older 1025-token sample remains 536,035,328 bytes. RSS on this binary is beside the gather-slice sample, not a further reduction. These are single cold samples, not percentiles, and not an 8192-token measurement.

## Reproduction

```sh
.cache/oracle/bin/python tools/parity.py .cache/needle3.cact .cache/needle target/release/a3s-akis
tools/certify.sh .cache/needle3.cact
```

## Not claimed

Packed CQ GEMM, a full allocation audit, soak, other platforms, max context, a second depth, and native behavioral compatibility remain open. No full gate is declared complete, and this is not a Needle replacement.
