# Resource, cancellation and window-boundary checkpoint

Date: 2026-10-14 Asia/Shanghai. Parent: `5b7363b`.
Status: component/numerical checks passed; full replacement certification remains **failed**.

## What changed

Power admission now covers embedded generation/embedding tokenization and complete
request preprocessing. A single permit remains held through generation, head
evaluation and response validation. All early errors release it through RAII.

`Tokenizer::encode_bounded` checks Power input/state limits before allocating BPE
working buffers and polls cancellation during scanning, symbol construction,
candidate construction, merging and output emission. The original pure `encode`
API is retained for trusted caller-owned input. Model and CLI text paths use the
bounded API. A deterministic test interrupts every polling checkpoint of a fixed
input and verifies that no partial token vector escapes.

KV queues retain each layer's effective attention window. Q/K/V convolution
queues retain the tap span, and dilated engram queues retain their required span.
Plain logits requests and generation without a confidence head avoid collecting
per-token head cells. Positional indices and engram token identities remain global.

State preflight includes rolling histories, optional head cells, probe pooling,
head output dimensions, reference scratch vectors and output buffers. Complete
measures serialized tool bytes before cloning schemas, rejects oversized metadata,
and reserves the schema/prompt budget across tokenization and inference. Scalar
enum validation avoids cloning the enum only to remove it. Non-finite head
arithmetic produces an error instead of a successful score or invalid embedding.

These are conservative model-owned buffer estimates. They are not an OS RSS
ceiling or a proof of Candle/allocator bookkeeping. Loading, every backend
allocation and concurrent process-wide residency still need further audit.

## Executed evidence

| Check | Result | File |
| --- | --- | --- |
| fmt / clippy all targets, warnings denied | Passed | certification.log |
| Rust unit tests | 32 passed, 0 failed | certification.log |
| Real model JAX logits differential | 9 passed, all top-1 identical | numerical-parity.jsonl |
| 255/256/257 and 1023/1024/1025 token boundaries | All passed; cosine 0.99999797–0.99999875 | numerical-parity.jsonl |
| Confidence-pool embedding / raw confidence head | Passed; cosine 0.9999085 / score error 0.00013725 | numerical-parity.jsonl |
| Upstream tokenizer differential | 120 passed | tokenizer-parity.jsonl |
| Serial tokenizer latency regression | Passed; 4000-repeat CLI request 0.08013s within frozen 2s budget | tokenizer-parity.jsonl |
| A3S Test terminal suite | Passed, no cleanup error | certification.log |
| Real-model process campaign | 13 passed, 4 failed | live-e2e.jsonl |
| Individual failure triage | Each original failure reproduced solo | triage.jsonl |
| Complete certification entry point | Exit 1 due to those four failures | certification.log |

New regression tests first reproduced unbounded histories, preprocessing memory
outside the state limit, omitted probe pooling and oversized tool metadata.
They now pass. Additional checks cover bounded encoder input/memory rejection,
deterministic cancellation during BPE, and overflowing finite head weights.

The six boundary prompts are repetitions of a known user-defined marker to
control exact token counts. They exercise actual learned weights and rolling
state but do not substitute for diverse natural-language long-context quality.
The first boundary harness assumed a dummy prefix that this artifact does not
enable; it failed before inference. The harness now derives that offset from the
fixed upstream tokenizer and asserts exact lengths before comparisons.

## Peak memory measurement

Same model, 1025 input token IDs, CPU, cold `logits` CLI process, macOS
`/usr/bin/time -l`, one sequential baseline/current pair:

- Baseline `5b7363b`: **716,816,384 bytes** maximum RSS.
- Current: **536,035,328 bytes** maximum RSS.
- Reduction: **180,781,056 bytes (25.22%)**.
- Input IDs identical, logits cosine **1.0**, top-1 identical.

Raw OS measurements and both binary SHA-256 values are in `memory-comparison.jsonl`.
This is a diagnostic sample, not percentile, throughput, leak or native-Needle
performance certification. Head-bearing requests still retain the cells required
by learned pooling; F32 expanded weights remain the principal residency cost.

The baseline was built from a detached git worktree under `.cache/baseline`.
A first attempt accidentally measured the same binary twice because the shared
Cargo target retained a stale executable. Those measurements were discarded.
The retained comparison uses distinct verified SHA-256 identities; the new
measurement script refuses identical binaries and checks identity after each run.
It also kills only its owned process group on timeout or interruption.

## Reproduction and next work

```sh
tools/certify.sh .cache/needle3.cact
.cache/oracle/bin/python tools/parity.py .cache/needle3.cact .cache/needle target/release/a3s-akis --boundaries
.cache/oracle/bin/python tools/tokenizer_parity.py .cache/needle3.cact .cache/needle target/release/a3s-akis
python3 tools/memory_compare.py .cache/needle3.cact .cache/akis-baseline target/release/a3s-akis
```

All test data is synthetic. Power and upstream source/model revisions remain
fixed by the bootstrap provenance; this checkpoint records current binary,
source, Cargo.lock, model and oracle hashes separately.

Chinese action, negation and missing-room cases still fail. No test was removed,
answer changed, threshold weakened or semantic word-list repair introduced.
Remaining work includes CLI signal cancellation, deterministic faults and latency
at every inference stage, full allocator accounting, diverse/max-context and
depth parity, calibrated task quality, host integration, packed kernels, soak
and platform certification. No full gate is declared complete.
