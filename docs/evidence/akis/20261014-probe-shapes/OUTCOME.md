# Probe manifest and tensor shapes

Date: 2026-10-14 Asia/Shanghai. Parent: `6e6cfe4`.
Status: matching probe heads load, and one bad shape or manifest value is rejected. G0 stays Partial. The quality campaign remains 14 passed and 3 failed. A Needle replacement is **not** certified.

## What changed

No inference code changed. The release binary is `9cc7f01369941ec446f78659e2b4335aab0f3abfce509c9a3369fb643ca6cfcb`.

- The stock fixture has no probe manifest and still loads.
- A width vector stored as one element is rejected with `tensor 1 expected shape [2]`.
- A rank-2 projection stored as `[4, 1]` is rejected with `tensor 2 expected shape [2, 2]`.
- Manifest values 0, 1.5, and 4, and the order 2 then 1, are rejected with `invalid probe head manifest`.
- A manifest of 1, or the order 1 then 2, without the head tensors is rejected with `probe tensor count mismatch`.
- Code 1 with output width 1, code 2 with output width 1, and code 3 with output width 3 load when the gain has one more row than the layer count.
- A gain with three rows is rejected with `invalid probe geometry`.
- Code 2 with output width 2, and code 3 with output width 1, are rejected with `invalid probe output width`.

## Executed evidence

`tools/certify.sh .cache/needle3.cact` ran fmt, clippy, `cargo test --locked`, the release build, the terminal smoke, the 8-case envelope, the sandboxed surface, and the 17-case campaign. 61 library tests passed. 7 CLI process tests passed (5 boundary, 2 signal). The envelope is 8 passed and 0 failed. The surface passed. The campaign is 14 passed and 3 failed, exit 1. The failures remain `lights-on-zh`, `lights-off-zh`, and `negated-action`. The log is `certification.log`.

## Not claimed

These edges add probe tensors to the synthetic fixture. They are not a needle3 probe head and not a forward of that head. Coverage-guided fuzz and the rest of the boundary matrix remain open. No full gate is declared complete. This is not a v0.1.0 release or a Needle replacement.
