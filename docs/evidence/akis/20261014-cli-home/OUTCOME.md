# Explicit weights without home or a writable directory

Date: 2026-10-14 Asia/Shanghai. Parent: `7d045a2`.
Status: the process check passes. The quality campaign remains 14 passed and 3 failed. Full G0 and a Needle replacement are **not** certified.

## What changed

No inference code changed. The release binary is `9cc7f01369941ec446f78659e2b4335aab0f3abfce509c9a3369fb643ca6cfcb`. `inspect` of an explicit synthetic fixture succeeds when `HOME` points at an empty read-only directory and the working directory is read-only. Neither directory gains a file. The reported sha256 equals an independent digest of the file bytes. A directory path is rejected with `model must be a regular file`. A one-byte truncation is rejected with `invalid Needle archive`. Stderr has no panic, and stdout is empty on both rejections.

## Executed evidence

`tools/certify.sh .cache/needle3.cact` ran fmt, clippy, `cargo test --locked`, the release build, the terminal smoke, the 8-case envelope, the surface, and the 17-case campaign. 49 library tests passed. 6 CLI process tests passed (4 boundary, 2 signal). The envelope is 8 passed and 0 failed. The surface passed. The campaign is 14 passed and 3 failed, exit 1. The failures remain `lights-on-zh`, `lights-off-zh`, and `negated-action`. The log is `certification.log`.

## Not claimed

Coverage-guided fuzz and the rest of the boundary matrix remain open. The three quality failures are the same model outputs as the pinned native library. Expected answers were not edited. No full gate is declared complete. This is not a v0.1.0 release or a Needle replacement.
