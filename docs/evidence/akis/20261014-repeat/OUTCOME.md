# One in-process repeat of 200 completes

Date: 2026-10-14 Asia/Shanghai. Parent: `f0991a4`.
Status: 200 measured calls matched inside one process. Full G4 and a Needle replacement are **not** certified.

## What changed

No inference code changed. The release binary is still `d36d4ca35cc44f6db7bbc32fa3f6096848f4bc08910c0d0940ea60ffdab12c5b`. The run is `examples/benchmark` on the pinned needle3 archive: one load, 1 warmup, then 200 `complete` calls for `turn on the kitchen lights` with `max_tokens` 128. The example returns an error if a call does not match the kitchen-on tool call, or if a later call cannot take the Power permit.

## Result

| Item | Value |
| --- | --- |
| Exit code | 0 |
| Measured calls | 200 |
| Input tokens | 59 on every call |
| Output tokens | 19 on every call |
| Cold load | 599.6 ms |
| Wall time | 93.88 seconds |
| Maximum resident set size | 279,674,880 bytes |
| Observed call time | 413.6 ms to 906.3 ms |

The raw millisecond list is `measurements.json`. The campaign remains 13 passed and 4 failed. `tools/certify.sh` was not re-run; the release binary sha256 is unchanged.

## Not claimed

This is one process on one Mac. It is not a percentile gate, not the 2-hour SOAK-01, and not the 24-hour candidate run. Packed CQ GEMM and other platforms remain open. No full gate is declared complete, and this is not a Needle replacement.
