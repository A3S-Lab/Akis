# Prefix tokens skip the output projection

Date: 2026-10-14 Asia/Shanghai. Parent: `0b3a6d3`.
Status: the last-token logits match the previous binary. One cold 2048-token sample is faster. Full G4 and a Needle replacement are **not** certified.

## What changed

`logits` keeps only the last token's distribution. `generate` still scores every sampled token, and it still runs one extra cache update after a finished sequence, without scoring that pass. `embed` and `layer_readout` never read those logits, so they skip the output projection. The attention cache, convolution history, and engram history are unchanged. A library test checks that an omitted prefix leaves the same cache and the same last distribution as scoring every token.

## Cold process time and RSS

One sample each, macOS `/usr/bin/time -l`, pinned Needle 3 artifact. Release binary `22d77e3c13857310d301723e4e6c654b8bde1c954aac68e751c73e664b49372c`. Previous binary `d36d4ca35cc44f6db7bbc32fa3f6096848f4bc08910c0d0940ea60ffdab12c5b`.

| Request | Previous binary | This binary |
| --- | --- | --- |
| `hello` | max absolute error 0, top-1 8110 | same logits |
| 256-token `<think>` | 272,171,008 bytes, 1.90 seconds | 272,171,008 bytes, 1.82 seconds |
| 2048-token `user` | 273,121,280 bytes, 12.27 seconds | 272,924,672 bytes, 11.40 seconds |

The retained 2048-token sample on the previous binary is 273,006,592 bytes and 13.200 seconds. This sample is 1.80 seconds under that record and 0.87 seconds under a same-machine rerun. Hello wall time is load noise and is not a speed claim. Max absolute error is 0 on all three lengths. Top-1 is 8110, 38, and 410.

One kitchen-light `complete` matched the previous call, the 19 token ids, and head score 0.8660065 with absolute error 0.

## Executed evidence

`tools/certify.sh .cache/needle3.cact` ran fmt, clippy, `cargo test --locked`, the release build, the terminal smoke, and the 17-case campaign. 42 library tests passed. 4 CLI process tests passed. The campaign is 13 passed and 4 failed, exit 1. The failures remain `lights-on-zh`, `lights-off-zh`, `negated-action`, and `missing-room`. `missing-room` still emits room `living room`.

These timings are single cold samples, not percentiles.

## Not claimed

Packed CQ GEMM, soak, other platforms, max context, a second depth, and native behavioral compatibility remain open. No full gate is declared complete, and this is not a Needle replacement.
