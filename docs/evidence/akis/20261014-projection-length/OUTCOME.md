# Projection payload length

Date: 2026-10-14 Asia/Shanghai. Parent: `984591f`.
Status: a projection whose opened payload length does not match its shape is rejected. G0 stays Partial. The quality campaign remains 14 passed and 3 failed. A Needle replacement is **not** certified.

## What changed

No inference code changed. The release binary is `9cc7f01369941ec446f78659e2b4335aab0f3abfce509c9a3369fb643ca6cfcb`.

- A stored `[2, 2]` projection of 16 bytes still loads.
- After the archive has opened, byte counts 0, 12, 15, and 20 on that record are rejected with `projection shape mismatch`.
- The `[10, 2]` embedding at 76 bytes is rejected with the same message.
- Those errors do not contain `expected shape` or `projection rank`.
- A width vector shortened to 4 bytes is not a projection and still loads.

## Executed evidence

`tools/certify.sh .cache/needle3.cact` ran fmt, clippy, `cargo test --locked`, the release build, the terminal smoke, the 8-case envelope, the sandboxed surface, and the 17-case campaign. 62 library tests passed. 7 CLI process tests passed (5 boundary, 2 signal). The envelope is 8 passed and 0 failed. The surface passed. The campaign is 14 passed and 3 failed, exit 1. The failures remain `lights-on-zh`, `lights-off-zh`, and `negated-action`. The log is `certification.log`.

## Not claimed

These edges mutate an opened synthetic fixture. A file whose byte count disagrees with its shape is still rejected when the archive opens. This is not a second archive and not a needle3 projection graph. Coverage-guided fuzz and the rest of the boundary matrix remain open. No full gate is declared complete. This is not a v0.1.0 release or a Needle replacement.
