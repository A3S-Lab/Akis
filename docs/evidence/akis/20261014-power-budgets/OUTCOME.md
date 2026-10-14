# Power budget edges

Date: 2026-10-14 Asia/Shanghai. Parent: `ab51029`.
Status: the synthetic fixture accepts each named budget at the exact limit and rejects one byte or one token past it. G0 stays Partial. The quality campaign remains 14 passed and 3 failed. A Needle replacement is **not** certified.

## What changed

No inference code changed. Production `InferenceLimits` are unchanged. The release binary is `9cc7f01369941ec446f78659e2b4335aab0f3abfce509c9a3369fb643ca6cfcb`.

On the synthetic fixture the new tests lock these comparisons:

- A model file of length N opens and parses. Length N-1 fails open with `model byte limit exceeded` and fails parse with `invalid model length`. The opened sha256 matches the parsed file.
- Resident weight bytes equal the expanded F32 sum, or the packed byte length for a gather table. That sum loads. One byte less fails with `resident weights exceed`. This fixture has no packed gather table.
- The largest tensor element count loads. One element less fails with `element limit`.
- Token id `vocab - 1` returns finite logits. Token id `vocab` fails with `token or context limit exceeded`.
- A Power context cap of 4 accepts 4 tokens and rejects 5. The fixture context is wider than 4, so the Power cap is the binding limit.
- A generation cap of 4 accepts a 4-token `hi` generation and rejects 5.
- Input `hi` is accepted at 2 bytes by encode, generate, and embed. One byte less fails. Embed then stops on the fixture's missing probe, which is after the byte gate.
- The forward-state reservation, including probe cells, is accepted at the computed byte count and rejected one byte below it.
- Tokenizer state for `hi` uses `(len + 8)` times the existing per-byte product. That count encodes. One byte less fails with `tokenizer state`.
- A tool request whose serialized size fills `max_input_bytes` reaches generation. One token on this fixture does not finish a tool call, so the next error is `generation exhausted before a complete tool call`. One byte less stops with `tool request byte limit exceeded` and does not generate.

## Executed evidence

`tools/certify.sh .cache/needle3.cact` ran fmt, clippy, `cargo test --locked`, the release build, the terminal smoke, the 8-case envelope, the sandboxed surface, and the 17-case campaign. 52 library tests passed. 7 CLI process tests passed (5 boundary, 2 signal). The envelope is 8 passed and 0 failed. The surface passed. The campaign is 14 passed and 3 failed, exit 1. The failures remain `lights-on-zh`, `lights-off-zh`, and `negated-action`. The log is `certification.log`.

## Not claimed

This is the synthetic fixture, not an allocation audit of the needle3 graph. Coverage-guided fuzz and the rest of the boundary matrix remain. Host gates, non-Unix signals, and per-stage cancel latency remain open. No full gate is declared complete. This is not a v0.1.0 release or a Needle replacement.
