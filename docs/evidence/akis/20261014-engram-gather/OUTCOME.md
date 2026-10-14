# Engram gather tables stay packed

Date: 2026-10-14 Asia/Shanghai. Parent: `8fc1c72`.
Status: residency dropped and the previous numerical cases matched bit for bit on cosine. Full G4 and a Needle replacement are **not** certified.

## What changed

Engram embedding tables are row gathers. Load keeps each CQ table in its packed bytes, checks every row for a finite legal decode, and materializes one row when inference indexes it. The row decoder is the same function that expands a CQ matrix, so a gathered row is the matrix row. Projection, attention, Monarch and embedding GEMMs still expand to F32. A table that is not CQ still expands.

The residency check counts packed bytes for those gather tables and expanded bytes for every other tensor. It still refuses a model that does not fit.

## Executed evidence

| Check | Result | File |
| --- | --- | --- |
| fmt / clippy all targets, warnings denied | Passed | certification.log |
| Rust tests | 36 library passed, 2 CLI signal passed, 0 failed | certification.log |
| CQ row decode versus full-matrix decode | Passed for widths 1 through 5 | certification.log |
| Independent parity versus the previous run | 31 passed; cosine delta 0 on every vector case; same top-1 | numerical-parity.jsonl |
| Real-model process campaign | 13 passed, 4 failed | certification.log |
| Complete certification entry point | Exit 1 due to those four failures | certification.log |

The four failures are unchanged: `lights-on-zh`, `lights-off-zh`, `negated-action`, and `missing-room`.

## Peak memory beside the retained sample

Cold `logits` process, macOS `/usr/bin/time -l`, pinned Needle 3 artifact. One sample.

| Request | Peak RSS |
| --- | --- |
| Retained 1025-token sample, older binary | 536,035,328 bytes |
| 1025-token `<think>` on the previous binary | 536,051,712 bytes |
| 1025-token `<think>` on this binary | 271,843,328 bytes |
| 2048-token `user` on the previous binary | 536,690,688 bytes |
| 2048-token `user` on this binary | 272,646,144 bytes |

The 1025-token drop from the previous binary is 264,208,384 bytes. Five packed engram tables account for that difference: their expanded F32 size minus the CQ bytes they now keep. This is one cold sample, not a percentile, and it is not an 8192-token measurement.

## Reproduction

```sh
.cache/oracle/bin/python tools/parity.py .cache/needle3.cact .cache/needle target/release/a3s-akis
tools/certify.sh .cache/needle3.cact
```

## Not claimed

Packed GEMM for projection matrices, a full allocation audit, soak, other platforms, max context, a second depth, and native behavioral compatibility remain open. No full gate is declared complete, and this is not a Needle replacement.
