# Boundary coverage and BPE performance checkpoint

Date: 2026-10-13. Platform: macOS 26.6.2 ARM64, Rust 1.97.1.
Parent: `8e8a5cb` (the published reference engine checkpoint).
Status: component checks passed; full certification still **failed**.

## Changes and proof

- Replaced repeated whole-string BPE scans/removals with a priority heap and
  stable linked symbol indices. Stale neighbor candidates are invalidated by
  generations; equal scores preserve the upstream leftmost rule, including
  positive/negative zero. UTF-8 spans are borrowed from the original string.
- 3,072 exhaustive synthetic cases compare heap results with the original
  scan-based algorithm across equal, unequal and signed-zero scores.
- 120 English, Chinese, Arabic, emoji, combining-character, whitespace and
  adjacent-marker cases match the unmodified pinned upstream RefTokenizer in
  both token IDs and decoded text. Raw per-case ID hashes are retained.
- Independent full Walsh-matrix multiplication verifies CQ1, CQ2, CQ3, CQ4 and
  ternary decoding with two rows, padding, crossing byte boundaries and zero
  norms. Invalid ternary crumbs are rejected.
- 10,000 deterministic grammar byte mutations never produce accepted invalid
  JSON; accepted objects are independently parsed and checked with serde_json.
- 2,000 deterministic archive mutations exercise importer, load and forward
  under small explicit Power limits without a panic. This bounded campaign is
  not a coverage-guided fuzzer or a proof of all resource bounds.
- Explicit tests cover scalar syntax, Unicode, optional/duplicate/unknown fields,
  arrays 64/65, properties 64/65, enums 255/256 and schema nesting 16/17.
- Archive loading rejects non-regular opened handles before reading contents;
  directories and model byte limits have regression coverage. This does not
  certify hostile filesystem races or cancellation during a blocking open.

## Results

| Check | Result |
| --- | --- |
| fmt / clippy all targets, warnings denied | Passed |
| Rust tests | 25 passed, 0 failed |
| Exact real-artifact tokenizer differential | 120 passed |
| Tokenizer large-input regression | Passed: 4,000 repetitions complete within frozen 2-second subprocess budget |
| Real-model process E2E | 13 passed, the same 4 semantic failures reproduced |
| A3S Test PTY | Passed, no cleanup error |
| Entire certification entry point | Exit 1, as required by those failures |

Before the fix, the 4,000-repeat case took 9.937049375 seconds including startup;
the pre-fix regression test passed all 120 identity checks then failed its
2-second performance budget. After the fix the same 4,000-repeat CLI case took
0.075312208 seconds (8,001 tokens). A separate 10,000-repeat observation went
from exceeding 15 seconds to 0.083707792 seconds (20,001 tokens).
These are single-machine diagnostic measurements, not inference throughput or
production percentile certification. The model's autoregressive compute is unchanged.

Files in this directory bind the before/after tokenizer reports, final full
certification log, source hashes and binary identity. Model/reference revisions
remain those in the [bootstrap provenance](../20261013-bootstrap/provenance.json).

## Reproduction and remaining work

```sh
tools/certify.sh .cache/needle3.cact
.cache/oracle/bin/python tools/tokenizer_parity.py .cache/needle3.cact .cache/needle target/release/a3s-akis
```

No missing quality cases were removed or marked expected-failure. No intent
word-list repairs or new runtime dependencies were introduced. Full G0 remains
partial; allocation accounting, all cancellation stages, long-context/head/depth
parity, host grounding, calibration and packed inference remain open in ROADMAP.
