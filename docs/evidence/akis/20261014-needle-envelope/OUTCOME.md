# Frozen Needle engine envelope

Date: 2026-10-14 Asia/Shanghai. Parent: `3370ddd`.
Status: seven of eight tool inputs match the pinned native calls. `missing-room` still returns the call. Full G1 and a Needle replacement are **not** certified.

## What changed

No inference code changed. The release binary is `502e9bc80a2788c8dfea15d52300906c10bdf8d3b14a7c16951dfac69dc1f1b2`. The oracle is `tests/e2e/cases/needle_engine.json`, frozen from `needle_complete` on library `993d617e1136d644560c1fe59058e40810b9386645d01d2f6472af042f4c72ec` and model `c9d915eca282ed42d1a09b143b592adb4cc6744ffe2d294adf5cfc5548170c38`. `tools/needle_e2e.py` runs `a3s-akis complete` only. It does not load the native library or the SDK.

Compared fields are `function_calls`, `suppressed_calls`, and `ungrounded`. Argument key order is ignored. Confidence, reasoning, tokens per second, resident set size, and `validation.negation` are not gates.

## Calls

| id | Result |
| --- | --- |
| lights-on-en | passed |
| lights-off-en | passed |
| lights-on-zh | passed |
| lights-off-zh | passed |
| irrelevant | passed |
| negated-action | passed |
| missing-room | failed: Akis returns `{room: living room, on: true}` in `function_calls`; the frozen envelope puts that call in `suppressed_calls` and lists `set_lights.room` as ungrounded |
| quoted-action | passed |

The script exited 1. Passed 7, failed 1. Raw rows are `needle-e2e.jsonl`.

`tools/certify.sh` was not re-run. The binary sha256 is unchanged from the run that recorded 13 passed and 4 failed. The stored expectations in `tool_calls.json` were not edited.

## Not claimed

This campaign does not certify native behavioral compatibility, max context, a second depth, or another platform. No full gate is declared complete, and this is not a Needle replacement.
