# Long prompts mix each attention chunk as one product

Date: 2026-10-14 Asia/Shanghai. Parent: `f85e84c`.
Status: hello matches the previous binary. The 256-token and 2048-token logits keep the same top-1 and cosine above 0.999. One cold 2048-token sample is faster. Full G4 and a Needle replacement are **not** certified.

## What changed

A prompt of at least 128 tokens mixes each kv head's softmax weights with that head's values as one matrix product per 256-token chunk. Score products, convolution, the KV cache, and the token loop for shorter prompts stay in place. One library test checks that the product stays within 1e-4 of a per-key scaled add and keeps the same top-1 on an 8-wide fixture. Columns outside a query window stay zero.

## Cold process time and RSS

One sample each, macOS `/usr/bin/time -l`, pinned Needle 3 artifact. Release binary `63c51272828d18bf0c5e35b5258eab6aa917a2e2c5cd8c284d3f685173ad12f4`. Previous binary `69565a9dc90f03c27838dca9785e5e9e8b5b33c35e0d6d9b82b71f6005b6fb4a`.

| Request | Previous binary | This binary |
| --- | --- | --- |
| `hello` | max absolute error 0, top-1 8110 | same logits, token loop |
| 256-token `<think>` | 301,023,232 bytes, 1.17 seconds, top-1 38 | 306,675,712 bytes, 1.09 seconds, cosine 0.999998857, max absolute error 0.20584106 |
| 2048-token `user` | 389,611,520 bytes, 5.04 seconds, top-1 410 | 416,235,520 bytes, 4.49 seconds, cosine 0.999998940, max absolute error 0.22872925 |

The retained 2048-token sample on the previous binary is 5.11 seconds and 390,643,712 bytes. This sample is 0.55 seconds under the same-machine previous binary and 0.62 seconds under that retained sample. A second run of this binary on the same prompt took 4.33 seconds and 415,580,160 bytes. The paired previous process recorded 101,784 involuntary context switches. This binary recorded 178,263 and 187,416 on the two runs. Those counts include the extra matrix-product threads. Hello wall time is load noise and is not a speed claim. The 256-token gap is inside one cold-sample variation.

One kitchen-light `complete` matched the previous call, the 19 token ids, and head score 0.8660065.

Repeated 2048-token logits on this binary matched each other with max absolute error 0.

## Executed evidence

`tools/certify.sh .cache/needle3.cact` ran fmt, clippy, `cargo test --locked`, the release build, the terminal smoke, and the 17-case campaign. 47 library tests passed. 4 CLI process tests passed. The campaign is 13 passed and 4 failed, exit 1. The failures remain `lights-on-zh`, `lights-off-zh`, `negated-action`, and `missing-room`. `missing-room` still emits room `living room`. The passing head scores match the previous binary: 0.8660065, 0.79703397, 0.9052133, and 0.6966743.

These timings are single cold samples, not percentiles. RSS is the measured high-water mark, not a pass/fail gate.

## Not claimed

Packed CQ GEMM, soak, other platforms, max context, a second depth, and native behavioral compatibility remain open. No full gate is declared complete, and this is not a Needle replacement.
