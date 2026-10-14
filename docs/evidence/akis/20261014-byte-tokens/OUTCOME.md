# Byte tokens

Date: 2026-10-14 Asia/Shanghai. Parent: `ecde469`.
Status: a missing UTF-8 scalar is restored from byte tokens, and an id at the piece count is rejected. G0 stays Partial. The quality campaign remains 14 passed and 3 failed. A Needle replacement is **not** certified.

## What changed

No inference code changed. The release binary is `9cc7f01369941ec446f78659e2b4335aab0f3abfce509c9a3369fb643ca6cfcb`.

- All 256 byte tokens return that byte. The piece for byte `0x0a` is lowercase `<0x0a>`.
- A UTF-8 scalar missing from the vocabulary round-trips through those bytes when byte fallback is on, with and without the dummy prefix.
- The same scalar becomes the unknown token when fallback is off. Decode does not put the original character back.
- Adjacent special markers stay separate token ids. The longer marker is kept when a shorter prefix marker is also present.
- An id equal to the piece count is rejected with `token id outside vocabulary`. A sequence that contains that id does not return the earlier text.
- A lone `0xFF` byte token does not panic. `token_bytes` returns that byte. The decoded string is not that byte.
- Fallback with 255 byte pieces, a duplicate byte piece, `<0xGG>`, and `<0x0>` are rejected at load.

## Executed evidence

`tools/certify.sh .cache/needle3.cact` ran fmt, clippy, `cargo test --locked`, the release build, the terminal smoke, the 8-case envelope, the sandboxed surface, and the 17-case campaign. 56 library tests passed. 7 CLI process tests passed (5 boundary, 2 signal). The envelope is 8 passed and 0 failed. The surface passed. The campaign is 14 passed and 3 failed, exit 1. The failures remain `lights-on-zh`, `lights-off-zh`, and `negated-action`. The log is `certification.log`.

## Not claimed

This is one synthetic vocabulary, not the needle3 tokenizer differential. Coverage-guided fuzz and the rest of the boundary matrix remain open. No full gate is declared complete. This is not a v0.1.0 release or a Needle replacement.
