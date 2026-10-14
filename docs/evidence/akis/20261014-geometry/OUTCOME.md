# Geometry edges

Date: 2026-10-14 Asia/Shanghai. Parent: `fd7c3ee`.
Status: archive geometry accepts the stated limit and rejects one past it. G0 stays Partial. The quality campaign remains 14 passed and 3 failed. A Needle replacement is **not** certified.

## What changed

No inference code changed. The release binary is `9cc7f01369941ec446f78659e2b4335aab0f3abfce509c9a3369fb643ca6cfcb`.

- A zero vocab, width, head count, kv head count, layer count, qk dimension, value dimension, context, or lane count is rejected with `invalid model geometry`.
- Layer count 64 opens. Layer count 65 is rejected.
- Head, lane, convolution-tap, and engram table, tap, dilation, and seed counts open at 256 and reject 257. Kv head count 256 opens only when the head count is a multiple of it.
- Vocab, width, qk, value, context, Hadamard, engram slots, and engram dim open at 1048576 and reject one past that cap.
- An odd qk dimension, a Hadamard value below the model width or not a power of two, a non-positive or non-finite rope theta, and an output vocab above the vocab are rejected.
- An engram site at the layer count, or two sites out of order, is rejected. The last legal site opens.
- Engram order count 4 and site count 16 open. Order count 5 and site count 17 are rejected with `invalid engram counts`.
- A site with a zero order, a zero slot count, or a table count that is not a multiple of the order count is rejected with `invalid engram geometry`. An order of 0 with no sites still opens.

## Executed evidence

`tools/certify.sh .cache/needle3.cact` ran fmt, clippy, `cargo test --locked`, the release build, the terminal smoke, the 8-case envelope, the sandboxed surface, and the 17-case campaign. 58 library tests passed. 7 CLI process tests passed (5 boundary, 2 signal). The envelope is 8 passed and 0 failed. The surface passed. The campaign is 14 passed and 3 failed, exit 1. The failures remain `lights-on-zh`, `lights-off-zh`, and `negated-action`. The log is `certification.log`.

## Not claimed

These edges are header fields on the synthetic fixture. They are not a second archive and not the needle3 allocation graph. Coverage-guided fuzz and the rest of the boundary matrix remain open. No full gate is declared complete. This is not a v0.1.0 release or a Needle replacement.
