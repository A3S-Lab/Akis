# Packed CQ GEMV was measured and not kept

Date: 2026-10-14 Asia/Shanghai. Parent: `e7f438c`.
Status: the input-side Hadamard product stayed above cosine 0.999 with the same top-1, and the cold RSS fell, but the 256-token sample was slower. The tree stays on the expanded contiguous GEMV. Full G4 and a Needle replacement are **not** certified.

## What was tried

Complete CQ projections stayed packed. Each 128-wide input group was Hadamard-transformed once, then each output row dotted the stored centroids with that transformed group and scaled by the group norm. Widths 2 and 4 used a 4-wide multiply-then-add kernel. Widths 1, 3, and 5 stayed on the scalar decoder. A second build split rows across a pool sized by `hw.perflevel0.logicalcpu` (6 on this machine) once a matrix had at least 128 rows per worker. Engram gather tables still decode one row. No tensor index was special-cased.

A library test compared the product with a dot of the decoded row for CQ widths 1 through 5, including a 129-column partial group, and for a 512-row matrix. Wrong-width input is rejected.

## Cold process time and RSS

One sample each, macOS `/usr/bin/time -l`, pinned Needle 3 artifact. The kept binary is `d36d4ca35cc44f6db7bbc32fa3f6096848f4bc08910c0d0940ea60ffdab12c5b`. The serial trial binary was `e0c7bf1d6be2100ad4b49d77328f6f36bebd699664b47abe83ec8e04ed086e66`. The row-split trial binary was `d552be25de3938117a37f927f3730f5ffae54795491652cdc7de64eec2e5fd97`. Those trial binaries are not the committed release.

| Request | Kept binary | Serial trial | Row-split trial |
| --- | --- | --- | --- |
| `hello` | 266,485,760 bytes, 1.28 seconds | 88,031,232 bytes, 1.16 seconds | not run |
| 256-token `<think>` | 272,334,848 bytes, 2.35 seconds | 93,667,328 bytes, 3.49 seconds | 93,978,624 bytes, 3.16 seconds |

`hello` top-1 was 8110 on both measured binaries, cosine 0.999993753, max absolute error 0.17288923. The 256-token top-1 was 38 on all three, cosine 0.999996856, max absolute error 0.39241982. The 2048-token sample was not run. The confidence head was not remeasured. The campaign was not re-run; the kept release is unchanged, and the retained result stays 13 passed and 4 failed.

The trial is slower on the 256-token request. The kernel was removed. A fresh release build of this commit's parent tree matches `d36d4ca35cc44f6db7bbc32fa3f6096848f4bc08910c0d0940ea60ffdab12c5b`.

These are single cold samples, not percentiles.

## Not claimed

Packed CQ GEMM, soak, other platforms, max context, a second depth, and native behavioral compatibility remain open. No full gate is declared complete, and this is not a Needle replacement.
