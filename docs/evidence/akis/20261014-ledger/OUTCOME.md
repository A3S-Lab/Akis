# Numerical differential and one cold ledger

Date: 2026-10-14 Asia/Shanghai. Parent: `b0c0fca`.
Status: 31 JAX comparisons and 120 tokenizer comparisons passed. The cold samples are records. Full G1, G4, and a Needle replacement are **not** certified.

## What changed

No inference code changed. The release binary is `9cc7f01369941ec446f78659e2b4335aab0f3abfce509c9a3369fb643ca6cfcb`. `tools/parity.py` ran under `.cache/oracle/bin/python`. `tools/tokenizer_parity.py` used the same interpreter because the reference exporter imports NumPy. Neither script was added to `tools/certify.sh`.

## Numerical results

JAX parity exited 0 with 31 comparisons. The lowest cosine is 0.999274850 on `quantization-activation`. Every compared row kept top-1 agreement. The confidence-head absolute error is 0.00069612. The known `layer-19` absolute error on the three-token `hello` readout remains 171382, with cosine above 0.999 and the same top-1. Raw rows are `numerical-parity.jsonl`.

The tokenizer differential passed 120 id comparisons. `hello` repeated 4000 times finished in 0.077 seconds, inside the script's 2-second budget. Raw rows are `tokenizer-parity.jsonl`.

## Cold ledger

One `/usr/bin/time -l` process each:

| Request | Time | Maximum resident set size |
| --- | --- | --- |
| `logits hello` | 0.62 seconds | 266,715,136 bytes |
| 256-token `<think>` logits | 0.88 seconds | 304,775,168 bytes |
| 2048-token `user` logits | 2.77 seconds | 418,955,264 bytes |
| kitchen-light `complete` | 1.06 seconds | 274,186,240 bytes |

The 2048-token request had 2048 ids and finite logits. It sits beside the retained sample of 2.61 seconds and 418,889,728 bytes. The kitchen completion kept head score 0.8660065 and 19 tokens. These are single cold samples, not percentiles, and not a pass/fail gate. No kernel changed.

## Not claimed

The quality campaign remains 14 passed and 3 failed. Packed CQ GEMM, soak, other platforms, max context, and a second depth remain open. No full gate is declared complete, and this is not a Needle replacement.
