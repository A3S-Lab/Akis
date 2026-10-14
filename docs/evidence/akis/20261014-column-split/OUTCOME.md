# Column-split GEMV was measured and not kept

Date: 2026-10-14 Asia/Shanghai. Parent: `4dc565a`.
Status: the split matched the kept binary and made the cold samples slower, so the tree stays on the single contiguous product. Full G4 and a Needle replacement are **not** certified.

## What was tried

`gemm_basic` is the top leaf of a 1025-token logits sample (1781 of 3114 logits samples). Candle asks for rayon, then the contiguous GEMV returns on the single-thread column kernel before that pool runs. The trial split outputs of width at least 512 into column pieces and ran the same `x.matmul(w)` on each piece. Narrow products stayed on one call. No tensor index was special-cased.

A contract test compared the split with one contiguous product on widths 512, 513, 600, and 768. Every lane matched. `hello` and the 256-token `<think>` logits also matched the kept binary with max absolute error 0 and the same top-1.

## Cold process time and RSS

One sample each, macOS `/usr/bin/time -l`, pinned Needle 3 artifact, from `tools/parity.py`, on the trial binary `9dfe8760f7a754480b8b38862c005cdfc47db7aefc1f02992922a4c33d0ad190`. That binary is not the committed release.

| Request | Kept binary | Trial binary |
| --- | --- | --- |
| 1025-token `<think>` | 271,810,560 bytes, 6.762 seconds | 274,464,768 bytes, 15.206 seconds |
| 2048-token `user` | 273,006,592 bytes, 13.200 seconds | 274,825,216 bytes, 32.020 seconds |

The trial is slower on both lengths. Cosine stayed above 0.999 with the same top-1, and the confidence-head absolute error stayed 0.000696. The split was removed. The committed release binary remains `d36d4ca35cc44f6db7bbc32fa3f6096848f4bc08910c0d0940ea60ffdab12c5b`.

These are single cold samples, not percentiles.

## Not claimed

Packed CQ GEMM, soak, other platforms, max context, a second depth, and native behavioral compatibility remain open. No full gate is declared complete, and this is not a Needle replacement.
