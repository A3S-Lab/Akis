# Per-token allocation map, with no cached copy

Date: 2026-10-14 Asia/Shanghai. Parent: `792ce54`.
Status: the map is recorded and no vector was cached. The release binary is the kron-axpy binary. Full G4 and a Needle replacement are **not** certified.

## What changed

No inference code changed. This slice records which vectors `step` rebuilds for every token, and it applies the cache rule to that list.

A vector is cached only when it is a rank-1 scale copied again on every token, a repeat sample shows `flat` / `to_vec1` above about 5% of the step, and the extra copy would keep RSS beside the current sample. Large projections stay as the one contiguous `[in, out]` matrix. The factor flats stay uncached.

## Map, one token

Counts below are for needle3: 20 layers, width 768, lanes 4, hadamard 1024, three factor pairs per layer, attention window at most 256.

| Site | Rebuilt each token | Kind |
| --- | --- | --- |
| `step` | embedding row, lane repeat, `ap` / `ao` / `ar` flats, per-layer `pre` / `post` / `res`, per-layer bias rows, `u` and its clone, `next`, final `hidden` | small vectors and one embedding row |
| `step` | per-layer RMS weight `flat(b)`, attention RMS `flat(o+8)`, scalar gate `flat(o+9)`, MLP RMS `flat(o+10)`, final norm flat | rank-1 scales |
| `attention` | quantized input copy, Q/K/V projections, convolution taps, q and k scale flats, score vector of the cached window | gemv outputs plus rank-1 scales |
| `mlp` | five diagonal flats, three factor-pair flats, three hadamard workspace vectors | rank-1 diagonals plus 32×32 factors |
| `kron_factors` | `mid` and `out`, length of the input | workspace, not a weight copy |
| `gemv` | input tensor, matmul result, `to_vec1` | the contiguous `[in, out]` product |
| engram sites | one packed row, key and value projections, tapped mix | row decode plus gemv |

`logits` with layer capture off does not keep per-layer means. Head cells stay off for logits, generate, and embed.

## Sample

One 1025-token `<think>` logits process, macOS `sample` for 4 seconds, attached after launch, on `target/release/a3s-akis` sha256 `d36d4ca35cc44f6db7bbc32fa3f6096848f4bc08910c0d0940ea60ffdab12c5b`. The call graph reports 3114 samples in `logits`.

| Symbol | Samples | Share of the 3114 |
| --- | --- | --- |
| `flat` | 51 | 1.6% |
| `to_vec1` | 52 | 1.7% |
| `gemm_basic`, top of stack | 1781 | 57% |

`flat` and `to_vec1` stay under the cache bar. No rank-1 scale is copied into a second resident buffer. `gemm_basic` is the largest leaf. That fraction is the measurement the following CPU slice reads. This sample is not a percentile and not an RSS gate.

## Executed evidence

The binary hash matches the kron-axpy provenance, so `tools/certify.sh` was not run again. That run remains 40 library tests passed, 2 CLI signal tests passed, campaign 13 passed and 4 failed, exit 1. The four failures remain `lights-on-zh`, `lights-off-zh`, `negated-action`, and `missing-room`, and `missing-room` still emits room `living room`.

## Not claimed

Packed CQ GEMM, soak, other platforms, max context, a second depth, and native behavioral compatibility remain open. No full gate is declared complete, and this is not a Needle replacement.
