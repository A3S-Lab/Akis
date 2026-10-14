# Kitchen receipt fields

Date: 2026-10-14 Asia/Shanghai. Parent: `ab02b28`.
Status: two kitchen complete processes carry the same receipt fields. G3 stays Partial. The quality campaign remains 14 passed and 3 failed. A Needle replacement is **not** certified.

## What changed

No inference code changed. The release binary is `9cc7f01369941ec446f78659e2b4335aab0f3abfce509c9a3369fb643ca6cfcb`. `tools/surface_e2e.py` reads the receipt that `complete` already returns. Both kitchen processes must carry schema `a3s.power.embedded-execution-receipt.v1`, device `cpu`, weights sha256 `c9d915eca282ed42d1a09b143b592adb4cc6744ffe2d294adf5cfc5548170c38`, and identical input and output sha256 values. A missing field fails the script. The kitchen call, 19 token ids, and head score 0.8660065 stay in place. This slice does not recompute an attestation digest.

## Executed evidence

`tools/certify.sh .cache/needle3.cact` ran fmt, clippy, `cargo test --locked`, the release build, the terminal smoke, the 8-case envelope, the surface, and the 17-case campaign. 49 library tests passed. 5 CLI process tests passed (3 boundary, 2 signal). The envelope is 8 passed and 0 failed. The surface passed. The campaign is 14 passed and 3 failed, exit 1. The failures remain `lights-on-zh`, `lights-off-zh`, and `negated-action`. The log is `certification.log`.

## Not claimed

Host gates, non-Unix signals, and per-stage cancel latency remain open. No full gate is declared complete. This is not a v0.1.0 release or a Needle replacement.
