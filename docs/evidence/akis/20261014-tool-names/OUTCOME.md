# Tool names

Date: 2026-10-14 Asia/Shanghai. Parent: `0ec27d3`.
Status: tool count and tool name bounds reject a call before generation. G0 stays Partial. The quality campaign remains 14 passed and 3 failed. A Needle replacement is **not** certified.

## What changed

No inference code changed. The release binary is `9cc7f01369941ec446f78659e2b4335aab0f3abfce509c9a3369fb643ca6cfcb`.

- A tool list of length 0 or 65 is rejected with `tool count must be 1..64`.
- Length 64 passes that gate and then fails with `token or context limit exceeded`.
- An empty name, a 129-byte name, a control character, and a duplicate name are rejected with `invalid or duplicate tool name`.
- A 128-byte name, a name containing a space, and two distinct names pass that gate and then fail on the context limit.
- Parameters whose schema is a string are rejected with `tool parameters must be an object`.
- These rejections carry no execution receipt, and the admission permit returns to zero.

## Executed evidence

`tools/certify.sh .cache/needle3.cact` ran fmt, clippy, `cargo test --locked`, the release build, the terminal smoke, the 8-case envelope, the sandboxed surface, and the 17-case campaign. 57 library tests passed. 7 CLI process tests passed (5 boundary, 2 signal). The envelope is 8 passed and 0 failed. The surface passed. The campaign is 14 passed and 3 failed, exit 1. The failures remain `lights-on-zh`, `lights-off-zh`, and `negated-action`. The log is `certification.log`.

## Not claimed

This does not accept a 65th tool or a duplicate name. Coverage-guided fuzz and the rest of the boundary matrix remain open. No full gate is declared complete. This is not a v0.1.0 release or a Needle replacement.
