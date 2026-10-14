# Offline inference

Date: 2026-10-14 Asia/Shanghai. Parent: `08911b1`.
Status: the sandboxed surface passes. G3 stays Partial. The quality campaign remains 14 passed and 3 failed. A Needle replacement is **not** certified.

## What changed

No inference code changed. The release binary is `9cc7f01369941ec446f78659e2b4335aab0f3abfce509c9a3369fb643ca6cfcb`. `sandbox-exec` denies network and allows the rest. A synthetic-fixture `inspect` still returns the independent file digest. A missing archive exits non-zero, stdout is empty, stderr contains `No such file`, and there is no panic. The seven-command surface, including the kitchen completion, runs under that same deny. The kitchen call, 19 token ids, and head score 0.8660065 stay in place.

The quality campaign itself is not inside the sandbox. It still runs after the surface.

## Executed evidence

`tools/certify.sh .cache/needle3.cact` ran fmt, clippy, `cargo test --locked`, the release build, the terminal smoke, the 8-case envelope, the sandboxed surface, and the 17-case campaign. 49 library tests passed. 7 CLI process tests passed (5 boundary, 2 signal). The envelope is 8 passed and 0 failed. The surface passed. The campaign is 14 passed and 3 failed, exit 1. The failures remain `lights-on-zh`, `lights-off-zh`, and `negated-action`. The log is `certification.log`.

## Not claimed

This is one macOS ARM64 sandbox profile, not a packet capture and not a non-Unix proof. Host gates, non-Unix signals, and per-stage cancel latency remain open. No full gate is declared complete. This is not a v0.1.0 release or a Needle replacement.
