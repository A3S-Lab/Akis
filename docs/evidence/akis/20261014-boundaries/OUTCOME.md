# Fail-closed boundary checks

Date: 2026-10-14 Asia/Shanghai. Parent: `af4f438`.
Status: the new checks lock rejections that were already in the tree. Full G0 and a Needle replacement are **not** certified.

## What changed

No inference code changed. Tests now cover a missing model path, an empty file, and a mode-000 file; short `tokenize` and `generate` arguments; a non-integer generation bound; an unknown `complete` field; an empty logits request; a request of length `geometry.context`; and a request one token past that context. A second model in the same test sets `max_generated_tokens` to 1 and rejects `max_tokens` 2 with the existing request error. The production default limits are unchanged.

On this Mac the mode-000 file was denied. The test does not force a failure when the owner can still open that file.

## 8193-token request

One process, release binary `d36d4ca35cc44f6db7bbc32fa3f6096848f4bc08910c0d0940ea60ffdab12c5b`, pinned needle3 archive. The text is `<think>` repeated 8192 times. The CLI adds BOS, so the request is 8193 tokens.

| Item | Value |
| --- | --- |
| Exit code | 1 |
| stdout | empty |
| stderr | `invalid request: token or context limit exceeded` |
| Wall time | 0.78 seconds |

`check_request` rejected the length before the token walk. The state budget was not raised.

## Certification

`tools/certify.sh .cache/needle3.cact` ended `EXIT:1`. Library tests: 41 passed. CLI process tests: 4 passed (2 new boundary tests and the 2 signal tests). The campaign is 13 passed and 4 failed. The failed ids are `lights-on-zh`, `lights-off-zh`, `negated-action`, and `missing-room`. `missing-room` still returns room `living room`. The release binary sha256 is unchanged.

## Not claimed

Coverage-guided fuzz and the rest of the boundary matrix remain open. This does not certify max context. No full gate is declared complete, and this is not a Needle replacement.
