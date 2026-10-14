# Probe output

Date: 2026-10-14 Asia/Shanghai. Parent: `4bb5e2c`.
Status: a finite code-1 probe embeds one finite value. A finite but overflowing direction is rejected at embed time. G0 stays Partial. The quality campaign remains 14 passed and 3 failed. A Needle replacement is **not** certified.

## What changed

No inference code changed. The release binary is `9cc7f01369941ec446f78659e2b4335aab0f3abfce509c9a3369fb643ca6cfcb`.

- A code-1 probe with finite weights loads, and `embed` of `hi` returns one finite value.
- The same probe with a finite embedding of 1 and a direction of the maximum finite value still loads.
- `embed` then rejects that model with `invalid request: non-finite probe output`.
- The error does not contain `non-finite embedding head` or `non-finite tensor`.

## Executed evidence

`tools/certify.sh .cache/needle3.cact` ran fmt, clippy, `cargo test --locked`, the release build, the terminal smoke, the 8-case envelope, the sandboxed surface, and the 17-case campaign. 64 library tests passed. 7 CLI process tests passed (5 boundary, 2 signal). The envelope is 8 passed and 0 failed. The surface passed. The campaign is 14 passed and 3 failed, exit 1. The failures remain `lights-on-zh`, `lights-off-zh`, and `negated-action`. The log is `certification.log`.

## Not claimed

This is one synthetic probe on the fixture. It is not a needle3 probe forward. Section 6 of the end-to-end plan is not met: there is no user-supplied frozen bilingual set, no second non-macOS-ARM64 machine, and no percentile or soak evidence. Coverage-guided fuzz and the rest of the boundary matrix remain open. No full gate is declared complete. This is not a v0.1.0 release or a Needle replacement.
