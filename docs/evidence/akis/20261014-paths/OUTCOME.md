# Path identity and empty command results

Date: 2026-10-14 Asia/Shanghai. Parent: `854b55f`.
Status: the path check and the empty-input surface pass. The quality campaign remains 14 passed and 3 failed. Full G0 and a Needle replacement are **not** certified.

## What changed

No inference code changed. The release binary is `9cc7f01369941ec446f78659e2b4335aab0f3abfce509c9a3369fb643ca6cfcb`. `inspect` of the synthetic fixture accepts a path that contains a space and a Han filename. Both sha256 fields match the plain path, and stderr has no panic. Empty-string `logits`, `layers`, and `embed` on the pinned archive return finite results. Logits and layers use ids `[2]`. Logits length is 8192. There are 20 layers of width 768 and a quantized hidden of width 768. The embedding length is 3072. A second process repeats each result.

## Executed evidence

`tools/certify.sh .cache/needle3.cact` ran fmt, clippy, `cargo test --locked`, the release build, the terminal smoke, the 8-case envelope, the surface, and the 17-case campaign. 49 library tests passed. 5 CLI process tests passed (3 boundary, 2 signal). The envelope is 8 passed and 0 failed. The surface passed. The campaign is 14 passed and 3 failed, exit 1. The failures remain `lights-on-zh`, `lights-off-zh`, and `negated-action`. The log is `certification.log`.

## Not claimed

Coverage-guided fuzz and the rest of the boundary matrix remain open. Empty input is accepted. This slice does not add a rejection. No full gate is declared complete, and this is not a v0.1.0 release or a Needle replacement.
