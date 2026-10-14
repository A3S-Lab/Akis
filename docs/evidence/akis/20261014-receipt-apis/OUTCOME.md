# Receipt APIs

Date: 2026-10-14 Asia/Shanghai. Parent: `8f814a0`.
Status: a token-id receipt is returned only by a finished generate or complete. G3 stays Partial. The quality campaign remains 14 passed and 3 failed. A Needle replacement is **not** certified.

## What changed

No inference code changed. The release binary is `9cc7f01369941ec446f78659e2b4335aab0f3abfce509c9a3369fb643ca6cfcb`.

- `logits`, layer readout, `embed`, `tokenize`, and `inspect` carry no receipt.
- A finished `generate` or `complete` carries schema `a3s.power.embedded-execution-receipt.v1`. Its keys are `schema`, `model`, `runtime`, `input`, and `output`.
- The input digest is BOS plus the prompt token ids. The output digest is the returned token ids. One changed token does not match.
- The receipt JSON has no signature, attestation, accelerator, microbatch, shape profile, or timestamp.
- A cancelled `complete` writes empty stdout. Its stderr does not contain the receipt schema.
- A zero generation bound, a cancelled `generate`, and an `embed` with no head fail without that schema.

## Executed evidence

`tools/certify.sh .cache/needle3.cact` ran fmt, clippy, `cargo test --locked`, the release build, the terminal smoke, the 8-case envelope, the sandboxed surface, and the 17-case campaign. 55 library tests passed. 7 CLI process tests passed (5 boundary, 2 signal). The envelope is 8 passed and 0 failed. The surface passed. The campaign is 14 passed and 3 failed, exit 1. The failures remain `lights-on-zh`, `lights-off-zh`, and `negated-action`. The log is `certification.log`.

## Not claimed

This does not add a receipt to logits or embed. Per-stage cancel latency, non-Unix signals, and host gates remain open. No full gate is declared complete. This is not a v0.1.0 release or a Needle replacement.
