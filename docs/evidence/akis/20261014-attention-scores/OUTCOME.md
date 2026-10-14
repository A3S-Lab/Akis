# Long prompts score each kv head as one matrix product

Date: 2026-10-14 Asia/Shanghai. Parent: `1d44458`.
Status: hello matches the previous binary. The 256-token and 2048-token logits keep the same top-1 and cosine above 0.999. One cold 2048-token sample is faster. Full G4 and a Needle replacement are **not** certified.

## What changed

A prompt of at least 128 tokens scores each kv head as one matrix product per 256-token chunk. Convolution, the KV cache, and the token loop for shorter prompts stay in place. One library test checks that the product keeps the same top-1 as the row dots, with max absolute error under 1e-4 on a 48-wide fixture. Another checks that the recorded key spans follow the cache deque.

## Cold process time and RSS

One sample each, macOS `/usr/bin/time -l`, pinned Needle 3 artifact. Release binary `69565a9dc90f03c27838dca9785e5e9e8b5b33c35e0d6d9b82b71f6005b6fb4a`. Previous binary `226a74b0d520d0a0bb245763fc4f25b3afd90447409379fc2d5f56144c729e09`.

| Request | Previous binary | This binary |
| --- | --- | --- |
| `hello` | max absolute error 0, top-1 8110 | same logits, token loop |
| 256-token `<think>` | 298,188,800 bytes, 1.24 seconds, top-1 38 | 300,122,112 bytes, 1.17 seconds, cosine 0.999997899, max absolute error 0.3241272 |
| 2048-token `user` | 402,833,408 bytes, 6.07 seconds, top-1 410 | 390,643,712 bytes, 5.11 seconds, cosine 0.999998738, max absolute error 0.16705513 |

The retained 2048-token sample on the previous binary is 6.18 seconds and 397,377,536 bytes. This sample is 0.96 seconds under the same-machine previous binary and 1.07 seconds under that retained sample. Two other runs of this binary on the same prompt took 5.82 seconds and 5.59 seconds. Hello wall time is load noise and is not a speed claim. The 256-token gap is inside one cold-sample variation.

One kitchen-light `complete` matched the previous call, the 19 token ids, and head score 0.8660065.

Repeated 2048-token logits on this binary matched each other with max absolute error 0.

## Executed evidence

`tools/certify.sh .cache/needle3.cact` ran fmt, clippy, `cargo test --locked`, the release build, the terminal smoke, and the 17-case campaign. 46 library tests passed. 4 CLI process tests passed. The campaign is 13 passed and 4 failed, exit 1. The failures remain `lights-on-zh`, `lights-off-zh`, `negated-action`, and `missing-room`. `missing-room` still emits room `living room`. The passing head scores match the previous binary: 0.8660065, 0.79703397, 0.9052133, and 0.6966743.

These timings are single cold samples, not percentiles. RSS is the measured high-water mark, not a pass/fail gate.

## Not claimed

Packed CQ GEMM, soak, other platforms, max context, a second depth, and native behavioral compatibility remain open. No full gate is declared complete, and this is not a Needle replacement.
