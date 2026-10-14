# Native tool-call diff on the eight fixtures

Date: 2026-10-14 Asia/Shanghai. Parent: `67300b9`.
Status: the eight calls match the 2026-10-13 table. Full G1 and a Needle replacement are **not** certified.

## What changed

No inference code changed. The release binary is the kron-axpy binary `d36d4ca35cc44f6db7bbc32fa3f6096848f4bc08910c0d0940ea60ffdab12c5b`. The pinned native library is `.cache/libneedle3.dylib`, sha256 `993d617e1136d644560c1fe59058e40810b9386645d01d2f6472af042f4c72ec`. The model sha256 is `c9d915eca282ed42d1a09b143b592adb4cc6744ffe2d294adf5cfc5548170c38`.

`python3 tools/native_baseline.py` called `needle_complete` on each row of `tests/e2e/cases/tool_calls.json` with max 128. The same eight inputs were sent to `a3s-akis complete` with the same tool and `max_tokens` 128. Compared fields are the tool name and arguments. Native `suppressed_calls` and `validation.negation` are recorded. Confidence, tokens per second, and peak RAM are not gates.

## Calls

Argument pairs match where both sides emit a call. JSON key order differs (`room` then `on` on the native side, `on` then `room` on the Akis side).

| id | Native `function_calls` | Akis `function_calls` | Native `suppressed_calls` | Native negation |
| --- | --- | --- | --- | --- |
| lights-on-en | kitchen, on | kitchen, on | none | false |
| lights-off-en | bedroom, off | bedroom, off | none | false |
| lights-on-zh | none | none | none | absent |
| lights-off-zh | none | none | none | absent |
| irrelevant | none | none | none | absent |
| negated-action | kitchen, off | kitchen, off | none | true |
| missing-room | none | living room, on | living room, on | false |
| quoted-action | none | none | none | absent |

Akis still returns `room` `"living room"` for `missing-room`. That string is unchanged. The native library leaves `function_calls` empty and puts the same pair in `suppressed_calls`. Akis does not copy that suppression.

The stored expectations still fail on `lights-on-zh`, `lights-off-zh`, `negated-action`, and `missing-room`. The campaign remains 13 passed and 4 failed. Answers, expectations, and thresholds were not edited. The other nine campaign cases were not re-run; the binary is the one already certified to that count.

## Not claimed

This is one pinned dylib on one Mac. It does not certify native ABI compatibility, max context, a second depth, or another platform. No full gate is declared complete, and this is not a Needle replacement.
