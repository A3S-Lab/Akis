# Numeric payloads and load gates

Date: 2026-10-14 Asia/Shanghai. Parent: `2cb3ed8`.
Status: finite payloads load, and one bad numeric field or load gate is rejected. G0 stays Partial. The quality campaign remains 14 passed and 3 failed. A Needle replacement is **not** certified.

## What changed

No inference code changed. The release binary is `9cc7f01369941ec446f78659e2b4335aab0f3abfce509c9a3369fb643ca6cfcb`.

- A finite F32 payload loads. Negative zero stays negative zero. NaN, positive infinity, and negative infinity are rejected with `non-finite tensor`.
- An F16 payload whose byte length matches the element count loads as ones. An F16 NaN is rejected with `non-finite tensor`.
- Codebook values 1 and negative zero load. NaN and infinity are rejected with `non-finite codebook`.
- The trailing raw record is rejected with `RAW is not a numeric tensor`. A missing index is rejected with `missing tensor`. An F32 record is rejected with `gather table must be a CQ matrix`.
- KV bit widths 0, 7, and 9 are rejected with `only A8 KV archives are supported`. Width 8 loads. One fewer record is rejected with `missing model tensors`. A projection rank that is not 2 is rejected with `projection rank mismatch`.
- Hadamard permutation 1 then 0 loads. A duplicate, a value of 2, a negative value, and a fraction are rejected with `invalid Hadamard permutation`. NaN in that permutation is rejected with `non-finite tensor`.
- A 1 by 1 CQ row, group 128, 1 bit, decodes one finite value. Row 1 is rejected with `CQ row out of range`. A missing norm is rejected with `truncated CQ norm`. 17 bits are rejected with `unsupported CQ geometry`. Bit width 6 is rejected with `CQ codebook index out of range`. Bit width 4 decodes. Ternary crumbs 0, 1, and 3 decode. Crumb 2 is rejected with `invalid ternary crumb`. A NaN norm is rejected with `non-finite tensor`. A zero group is `CQ row out of range` on the row path and `gather table must be a CQ matrix` on the gather path.

## Executed evidence

`tools/certify.sh .cache/needle3.cact` ran fmt, clippy, `cargo test --locked`, the release build, the terminal smoke, the 8-case envelope, the sandboxed surface, and the 17-case campaign. 60 library tests passed. 7 CLI process tests passed (5 boundary, 2 signal). The envelope is 8 passed and 0 failed. The surface passed. The campaign is 14 passed and 3 failed, exit 1. The failures remain `lights-on-zh`, `lights-off-zh`, and `negated-action`. The log is `certification.log`.

## Not claimed

These edges use the synthetic fixture and one in-memory CQ row. They are not the needle3 tensor graph. Coverage-guided fuzz and the rest of the boundary matrix remain open. No full gate is declared complete. This is not a v0.1.0 release or a Needle replacement.
