# String arguments must occur in the user input

Date: 2026-10-14 Asia/Shanghai. Parent: `a2000be`.
Status: the eight frozen Needle envelopes match. The quality campaign is 14 passed and 3 failed. Full G2 and a Needle replacement are **not** certified.

## What changed

After a schema-valid tool call is parsed, each string argument must occur as a contiguous, case-sensitive span of the user input. An empty string does not. Other argument types stay with the schema. A call that fails the check moves to `suppressed_calls`, and each missing path is `tool.field` in `ungrounded`. The check does not add a room list, a synonym, or case folding.

One library test covers the eight frozen inputs, a sentence that does contain `living room`, a case mismatch, and an empty string.

## Envelope

Release binary `9cc7f01369941ec446f78659e2b4335aab0f3abfce509c9a3369fb643ca6cfcb`. `tools/needle_e2e.py` exited 0. Passed 8, failed 0. `missing-room` now has empty `function_calls`, `suppressed_calls` of `{room: living room, on: true}`, and `ungrounded` of `set_lights.room`. Raw rows are `needle-e2e.jsonl`.

One kitchen-light `complete` kept the kitchen call, 19 token ids, and head score 0.8660065.

## Quality campaign

`tools/certify.sh .cache/needle3.cact` ran fmt, clippy, `cargo test --locked`, the release build, the terminal smoke, and the 17-case campaign. 49 library tests passed. 4 CLI process tests passed. The campaign is 14 passed and 3 failed, exit 1. The failures are `lights-on-zh`, `lights-off-zh`, and `negated-action`. `missing-room` passes the stored empty `function_calls` expectation. `suppressed_calls` on that case still holds `living room`. The stored expectations were not edited.

Passing head scores match the previous binary: 0.8660065, 0.79703397, 0.9052133, and 0.6966743.

## Not claimed

`validation.negation`, confidence, and reasoning text are not gates. The frozen bilingual set is still absent, so G2 stays failed. Native behavioral compatibility, max context, a second depth, and other platforms remain open. No full gate is declared complete, and this is not a Needle replacement.
