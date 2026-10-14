# Tensor record edges

Date: 2026-10-14 Asia/Shanghai. Parent: `f4cc8f1`.
Status: archive records accept a matching byte length and reject one bad field. G0 stays Partial. The quality campaign remains 14 passed and 3 failed. A Needle replacement is **not** certified.

## What changed

No inference code changed. The release binary is `9cc7f01369941ec446f78659e2b4335aab0f3abfce509c9a3369fb643ca6cfcb`.

- A stored F32 record opens. Four dimensions open when their product still matches that byte length.
- An unknown dtype, five dimensions, and a nonzero reserved byte are rejected with `invalid tensor record`.
- One byte less or more than that F32 length is rejected with `tensor 0 byte length mismatch`.
- An empty shape, a zero dimension, and a dimension product past the address space are rejected before the length check.
- CQ parameters outside group 128 and 1 to 5 bits are rejected with `unsupported CQ geometry`. Supported parameters on the stored F32 length are a byte length mismatch.
- A second raw record, and a last record rewritten as a length-matching F32 tensor, are rejected with `expected one trailing tokenizer`. A raw matrix is rejected with `RAW attachment must be scalar`.
- A zero length, an unaligned offset, an offset inside the directory, a range past the file, and one byte past the trailing raw record are rejected with `invalid or overlapping tensor range`. One past the addressable end is `tensor range overflow`.
- One byte less on the trailing raw record still opens. Tokenizer parsing then fails with `malformed SentencePiece`. A directory that claims 1000 records is rejected with `truncated directory`.

## Executed evidence

`tools/certify.sh .cache/needle3.cact` ran fmt, clippy, `cargo test --locked`, the release build, the terminal smoke, the 8-case envelope, the sandboxed surface, and the 17-case campaign. 59 library tests passed. 7 CLI process tests passed (5 boundary, 2 signal). The envelope is 8 passed and 0 failed. The surface passed. The campaign is 14 passed and 3 failed, exit 1. The failures remain `lights-on-zh`, `lights-off-zh`, and `negated-action`. The log is `certification.log`.

## Not claimed

These edges are records in the synthetic fixture. They are not a second archive and not the needle3 tensor graph. Coverage-guided fuzz and the rest of the boundary matrix remain open. No full gate is declared complete. This is not a v0.1.0 release or a Needle replacement.
