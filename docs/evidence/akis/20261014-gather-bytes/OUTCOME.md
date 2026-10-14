# Packed gather residency

Date: 2026-10-14 Asia/Shanghai. Parent: `c22ec2b`.
Status: one packed gather table is counted by its stored bytes. G0 stays Partial. The quality campaign remains 14 passed and 3 failed. A Needle replacement is **not** certified.

## What changed

No inference code changed. Production `InferenceLimits` are unchanged. The release binary is `9cc7f01369941ec446f78659e2b4335aab0f3abfce509c9a3369fb643ca6cfcb`.

The synthetic fixture still has no gather table. A second fixture adds one engram site. Its gather table is a 1×1 CQ matrix with group 128 and 1 bit. Group padding stores 18 bytes. The expanded F32 matrix is 4 bytes.

- The resident sum that uses the stored 18 bytes loads.
- One byte less fails with `resident weights exceed`.
- The sum that expands that table to 4 bytes also fails with `resident weights exceed`.

## Executed evidence

`tools/certify.sh .cache/needle3.cact` ran fmt, clippy, `cargo test --locked`, the release build, the terminal smoke, the 8-case envelope, the sandboxed surface, and the 17-case campaign. 54 library tests passed. 7 CLI process tests passed (5 boundary, 2 signal). The envelope is 8 passed and 0 failed. The surface passed. The campaign is 14 passed and 3 failed, exit 1. The failures remain `lights-on-zh`, `lights-off-zh`, and `negated-action`. The log is `certification.log`.

## Not claimed

This is one synthetic gather table, not the needle3 allocation graph. Coverage-guided fuzz and the rest of the boundary matrix remain. No full gate is declared complete. This is not a v0.1.0 release or a Needle replacement.
