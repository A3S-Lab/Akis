# Admission overlap

Date: 2026-10-14 Asia/Shanghai. Parent: `6dfcfec`.
Status: a second request is rejected while one permit is held, including during an in-flight forward. G3 stays Partial. The quality campaign remains 14 passed and 3 failed. A Needle replacement is **not** certified.

## What changed

No inference code changed. Production `InferenceLimits` are unchanged. The default cap stays 1. The release binary is `9cc7f01369941ec446f78659e2b4335aab0f3abfce509c9a3369fb643ca6cfcb`.

The new library test locks these comparisons on the production `begin` path. Power returns the active-request error without queueing, so waiting stays 0.

- A cancelled request takes no permit. Active stays 0 and the limit stays 1.
- While one permit is held, the next logits returns `already has 1 active`. Active stays 1.
- After that permit is dropped, an empty request fails and does not keep the permit. The same three tokens repeat the first logits.
- A local cap of 2 accepts two permits and rejects the third with `already has 2 active`. After one permit is dropped, the logits match the cap-1 result.
- A 4096-token forward that is still running rejects an overlapping logits with `already has 1 active`. Cancelling that forward returns the permit to 0, and the next logits runs.

## Executed evidence

`tools/certify.sh .cache/needle3.cact` ran fmt, clippy, `cargo test --locked`, the release build, the terminal smoke, the 8-case envelope, the sandboxed surface, and the 17-case campaign. 53 library tests passed. 7 CLI process tests passed (5 boundary, 2 signal). The envelope is 8 passed and 0 failed. The surface passed. The campaign is 14 passed and 3 failed, exit 1. The failures remain `lights-on-zh`, `lights-off-zh`, and `negated-action`. The log is `certification.log`.

## Not claimed

This is one in-process overlap on the synthetic fixture, not a host gate. Per-stage cancel latency and non-Unix signals remain open. No full gate is declared complete. This is not a v0.1.0 release or a Needle replacement.
