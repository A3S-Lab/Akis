# Independent receipt digests

Date: 2026-10-14 Asia/Shanghai. Parent: `cf5ce8a`.
Status: passing completions match an independent digest. G3 stays Partial. The quality campaign remains 14 passed and 3 failed. A Needle replacement is **not** certified.

## What changed

No inference code changed. The release binary is `9cc7f01369941ec446f78659e2b4335aab0f3abfce509c9a3369fb643ca6cfcb`. Two processes can repeat a receipt and still both be wrong, so agreement is not the check. Each passing completion now tokenizes the chat prompt again. Object keys in that prompt follow byte order, which is the order the model consumes. The input digest is the Power token-id hash of BOS plus those ids. The output digest is the same hash of the returned token ids. Lengths match too. Changing one input id or one output id makes the hash differ. A missing receipt field fails the case.

The kitchen completion stays at 19 tokens and head score 0.8660065. Its input digest is `87839ec0a525af138fa5f1dee66ac160013031b2e4f326a70a1ba231ca77cec5`.

## Executed evidence

`tools/certify.sh .cache/needle3.cact` ran fmt, clippy, `cargo test --locked`, the release build, the terminal smoke, the 8-case envelope, the surface, and the 17-case campaign. 49 library tests passed. 6 CLI process tests passed (4 boundary, 2 signal). The envelope is 8 passed and 0 failed. The surface passed. The campaign is 14 passed and 3 failed, exit 1. The failures remain `lights-on-zh`, `lights-off-zh`, and `negated-action`, still on the tool call, not on the digest. The log is `certification.log`.

## Not claimed

Host gates, non-Unix signals, and per-stage cancel latency remain open. This check does not sign the receipt or prove a remote attestation. No full gate is declared complete. This is not a v0.1.0 release or a Needle replacement.
