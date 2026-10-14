# Long prompts finish each layer's tokens together

Date: 2026-10-14 Asia/Shanghai. Parent: `88fe81e`.
Status: hello, the 256-token logits, and the 2048-token logits match the previous binary with max absolute error 0. One cold 2048-token sample is faster. Full G4 and a Needle replacement are **not** certified.

## What changed

A prompt of at least 128 tokens on a wide graph finishes each layer's tokens together when probe cells are off. Each token still runs the same attention residual, MLP, and lane mix. Layer readout and probe cells stay on the caller thread. One library test checks that the grouped run matches the one-token run on four fixture tokens.

## Cold process time and RSS

One sample each, macOS `/usr/bin/time -l`, pinned Needle 3 artifact. Release binary `502e9bc80a2788c8dfea15d52300906c10bdf8d3b14a7c16951dfac69dc1f1b2`. Previous binary `63c51272828d18bf0c5e35b5258eab6aa917a2e2c5cd8c284d3f685173ad12f4`.

| Request | Previous binary | This binary |
| --- | --- | --- |
| `hello` | max absolute error 0, top-1 8110 | same logits, token loop |
| 256-token `<think>` | 304,594,944 bytes, 1.08 seconds, top-1 38 | 305,381,376 bytes, 0.88 seconds, max absolute error 0 |
| 2048-token `user` | 416,055,296 bytes, 4.23 seconds, top-1 410 | 418,889,728 bytes, 2.61 seconds, max absolute error 0 |

The retained 2048-token sample on the previous binary is 4.49 seconds and 416,235,520 bytes. This sample is 1.62 seconds under the same-machine previous binary and 1.88 seconds under that retained sample. A second run of this binary on the same prompt took 2.62 seconds and 420,642,816 bytes. The paired previous process recorded 173,000 involuntary context switches. This binary recorded 179,935 and 178,771 on the two runs. Hello wall time is load noise and is not a speed claim.

One kitchen-light `complete` matched the previous call, the 19 token ids, and head score 0.8660065.

Repeated 2048-token logits on this binary matched each other with max absolute error 0.

## Executed evidence

`tools/certify.sh .cache/needle3.cact` ran fmt, clippy, `cargo test --locked`, the release build, the terminal smoke, and the 17-case campaign. 48 library tests passed. 4 CLI process tests passed. The campaign is 13 passed and 4 failed, exit 1. The failures remain `lights-on-zh`, `lights-off-zh`, `negated-action`, and `missing-room`. `missing-room` still emits room `living room`. The passing head scores match the previous binary: 0.8660065, 0.79703397, 0.9052133, and 0.6966743.

These timings are single cold samples, not percentiles. RSS is the measured high-water mark, not a pass/fail gate.

## Not claimed

Packed CQ GEMM, soak, other platforms, max context, a second depth, and native behavioral compatibility remain open. No full gate is declared complete, and this is not a Needle replacement.
