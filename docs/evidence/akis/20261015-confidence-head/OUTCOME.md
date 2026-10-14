# Confidence head

Date: 2026-10-15 Asia/Shanghai. Parent: `c23822f`.
Status: a finite code-2 probe returns one finite confidence score. A finite but overflowing projection is rejected at generate time. G0 stays Partial. The quality campaign remains 14 passed and 3 failed. A Needle replacement is **not** certified.

## What changed

No inference code changed. The release binary is `9cc7f01369941ec446f78659e2b4335aab0f3abfce509c9a3369fb643ca6cfcb`.

- A code-2 probe with finite weights loads, and `generate` of `hi` with one token returns a finite head score.
- The same probe with a finite embedding of 1, a finite gain of 1, and a projection of the maximum finite value still loads.
- `generate` then rejects that model with `invalid request: non-finite confidence head`.
- The error does not contain `non-finite probe output`, `non-finite model output`, or `non-finite tensor`.

## Executed evidence

`tools/certify.sh .cache/needle3.cact` ran fmt, clippy, `cargo test --locked`, the release build, the terminal smoke, the 8-case envelope, the sandboxed surface, and the 17-case campaign. 65 library tests passed. 7 CLI process tests passed (5 boundary, 2 signal). The envelope is 8 passed and 0 failed. The surface passed. The campaign is 14 passed and 3 failed, exit 1. The failures remain `lights-on-zh`, `lights-off-zh`, and `negated-action`. The kitchen complete stays 19 tokens with head score 0.8660065. The log is `certification.log`.

## Not claimed

This is one synthetic confidence head on the fixture. It is not a needle3 confidence forward. Section 6 of the end-to-end plan is not met: there is no user-supplied frozen bilingual set, no second non-macOS-ARM64 machine, and no percentile or soak evidence. Coverage-guided fuzz and the rest of the boundary matrix remain open. No full gate is declared complete. This is not a v0.1.0 release or a Needle replacement.
