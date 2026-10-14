# Long prompts project each layer matrix once

Date: 2026-10-14 Asia/Shanghai. Parent: `c408935`.
Status: hello matches the previous binary. The 256-token and 2048-token logits keep the same top-1 and cosine above 0.999. One cold 2048-token sample is faster. Full G4 and a Needle replacement are **not** certified.

## What changed

A prompt of at least 128 tokens projects Q, K, V, the attention gates, the attention output, and engram key/value as one GEMM per matrix. The token loop still owns convolution, attention scores, the MLP, and the last-token unembed. Shorter prompts and narrow graphs stay on that loop. One row of a projection still uses the existing GEMV. A library test checks that a multi-row product keeps the same top-1 as those row products, with max absolute error under 1e-5 on the fixture matrix.

A 59-token prompt on an earlier cutoff moved the lights-off confidence head from 0.79703397 to 0.7850553. That cutoff is not in this binary. Prompts below 128 tokens stay on the token loop.

## Cold process time and RSS

One sample each, macOS `/usr/bin/time -l`, pinned Needle 3 artifact. Release binary `226a74b0d520d0a0bb245763fc4f25b3afd90447409379fc2d5f56144c729e09`. Previous binary `22d77e3c13857310d301723e4e6c654b8bde1c954aac68e751c73e664b49372c`.

| Request | Previous binary | This binary |
| --- | --- | --- |
| `hello` | max absolute error 0, top-1 8110 | same logits, token loop |
| 256-token `<think>` | 272,138,240 bytes, 1.87 seconds, top-1 38 | 298,074,112 bytes, 1.23 seconds, cosine 0.999997200, max absolute error 0.60337067 |
| 2048-token `user` | 273,072,128 bytes, 11.77 seconds, top-1 410 | 397,377,536 bytes, 6.18 seconds, cosine 0.999998915, max absolute error 0.2066555 |

The retained 2048-token samples on earlier binaries are 13.200 seconds and 11.40 seconds. This sample is 5.59 seconds under the same-machine previous binary. Hello wall time is load noise and is not a speed claim.

One kitchen-light `complete` matched the previous call, the 19 token ids, and head score 0.8660065.

## Executed evidence

`tools/certify.sh .cache/needle3.cact` ran fmt, clippy, `cargo test --locked`, the release build, the terminal smoke, and the 17-case campaign. 44 library tests passed. 4 CLI process tests passed. The campaign is 13 passed and 4 failed, exit 1. The failures remain `lights-on-zh`, `lights-off-zh`, `negated-action`, and `missing-room`. `missing-room` still emits room `living room`. The passing head scores match the previous binary: 0.8660065, 0.79703397, 0.9052133, and 0.6966743.

These timings are single cold samples, not percentiles. The higher RSS is the measured high-water mark, not a pass/fail gate.

## Not claimed

Packed CQ GEMM, soak, other platforms, max context, a second depth, and native behavioral compatibility remain open. No full gate is declared complete, and this is not a Needle replacement.
