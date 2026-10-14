# Seven CLI commands on one pinned archive

Date: 2026-10-14 Asia/Shanghai. Parent: `dded4c6`.
Status: the surface passes. The quality campaign remains 14 passed and 3 failed. Full G0 and a Needle replacement are **not** certified.

## What changed

No inference code changed. The release binary is `9cc7f01369941ec446f78659e2b4335aab0f3abfce509c9a3369fb643ca6cfcb`. `tools/surface_e2e.py` runs `inspect`, `tokenize`, `logits`, `layers`, `embed`, `generate`, and `complete`. `tools/certify.sh` runs the frozen engine envelope and this surface before the quality campaign.

`hello` generated with a bound of 4 does not emit EOS. The result keeps `finished` false and exactly 4 token ids. A second process repeats those ids and the head score. The kitchen completion keeps the kitchen call, 19 token ids, and head score 0.8660065. A second process repeats the call and the token ids. Logits and layer ids are the tokenize ids with BOS 2 in front. Logits length is 8192. There are 20 layers of width 768. The embedding length is 3072 and its norm is within 1e-5 of 1.

## Executed evidence

`tools/certify.sh .cache/needle3.cact` ran fmt, clippy, `cargo test --locked`, the release build, the terminal smoke, the 8-case envelope, the surface, and the 17-case campaign. 49 library tests passed. 4 CLI process tests passed. The envelope is 8 passed and 0 failed. The surface passed. The campaign is 14 passed and 3 failed, exit 1. The failures remain `lights-on-zh`, `lights-off-zh`, and `negated-action`. The log is `certification.log`.

## Not claimed

JAX parity, the 120-token tokenizer differential, and the cold-sample ledger are not part of this entry. Coverage-guided fuzz and the rest of the boundary matrix remain open. No full gate is declared complete, and this is not a Needle replacement.
