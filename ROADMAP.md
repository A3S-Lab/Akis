# Akis delivery gates

Target: a tested, efficient replacement for Needle. **No full replacement or
production certification has been granted.** Baseline: 2026-10-13.

| Gate | State | Retained evidence / next obligation |
| --- | --- | --- |
| G0 artifact, tokenizer, structure, CLI | Partial | 41 library tests and 4 CLI process tests pass. The new checks reject a missing path, an empty file, and a mode-000 file without a panic, reject short CLI arguments, a non-integer generation bound, and an unknown `complete` field, and reject an empty logits request plus one token past the fixture context. A one-token generation cap rejects the next token with the existing request error. One 8193-token needle3 logits request returned `token or context limit exceeded` with empty stdout. 120 upstream tokenizer comparisons, all CQ widths, 10,000 grammar and 2,000 artifact mutations still pass; coverage-guided fuzz and the rest of the boundary matrix remain |
| G1 numerical compatibility | Partial | Retained nine JAX logits cases include 255/256/257 and 1023/1024/1025 tokens. One fail-closed run now also covers 1280/1536/2048-token prompts, 20 per-layer lane means, and the activation-quantization path; confidence-pool embedding and confidence head still pass. The 2048-token logits request stayed inside the state budget (cold RSS 536,690,688 bytes beside the retained 1025-token sample of 536,035,328). One 8192-token `<think>` logits request on the same archive and the current limits completed in 49.39 seconds at 272,875,520 bytes, cosine 0.999999940, top-1 38. Each layer still attends at most the 256-row cache. A replay of the eight tool fixtures on pinned native library `993d617e` matches the 2026-10-13 call table on this binary: shared English calls, shared empty Chinese, irrelevant, and quoted proposals, a shared kitchen-off negation, and missing-room suppression only on the native side (Akis still returns room `living room`). Akis answers were not changed. The campaign remains 13 passed and 4 failed. Max context, a second depth, native behavioral compatibility, and cross-platform remain |
| G2 task quality and calibration | Failed | Real process campaign 13/17; Chinese action, negation and missing-room failures retained; frozen larger dataset/calibration pending |
| G3 host and cancellation | Partial | Unix CLI `SIGINT`/`SIGTERM` sets the process stop flag; grammar, head and pre-return cancellation discard the result; per-stage latency, non-Unix signals and actual host gates pending |
| G4 resources and efficiency | Not certified | Rolling caches and optional head cells previously cut 1025-token cold RSS from 717MB to 536MB. Engram gather tables stay CQ and decode one row (271,843,328 bytes on that binary). Complete projections are stored once as contiguous `[in, out]`, so each step uses the contiguous CPU GEMV. Attention mixes each cached value row with one contiguous axpy after a single width check. On the attention-axpy binary the cold 1025-token logits sample is 272,285,696 bytes and 6.934 seconds, and the 2048-token sample is 273,170,432 bytes and 13.406 seconds; the GEMV binary took 9.513 and 18.793 seconds. Kronecker factor rows now use that same axpy. On this binary the cold samples are 271,810,560 bytes and 6.762 seconds, and 273,006,592 bytes and 13.200 seconds. The 2048-token sample moved by 0.206 seconds, inside one cold-sample variation, so the factor product stops here. Cosine stayed above 0.999 with the same top-1. Hello and the 256-token logits matched the attention-axpy binary with max absolute error 0. A 4-second sample of a 1025-token logits run on this binary counts 3114 samples in `logits`, 51 in `flat`, and 52 in `to_vec1`. Those copies stay under the rank-1 cache bar, so the allocation map records no cache. `gemm_basic` is the top leaf at 1781 samples. A column-split of that product matched the single call bit for bit and was not kept: the cold samples moved from 6.762 and 13.200 seconds to 15.206 and 32.020 seconds. Packed CQ GEMM, platform measurement and soak remain |
| G5 delivery platforms | Not certified | Local macOS ARM64 only; independent consumers and other platforms pending |

Evidence anchor: [20261013 bootstrap](docs/evidence/akis/20261013-bootstrap/OUTCOME.md).
Latest: [boundary and tokenizer checks](docs/evidence/akis/20261013-boundary-tokenizer/OUTCOME.md).
Latest resource slice: [20261014 resource checks](docs/evidence/akis/20261014-resources/OUTCOME.md).
Latest cancellation slice: [20261014 signal checks](docs/evidence/akis/20261014-signals/OUTCOME.md).
Latest numerical slice: [20261014 parity checks](docs/evidence/akis/20261014-parity/OUTCOME.md).
Latest residency slice: [20261014 engram gather](docs/evidence/akis/20261014-engram-gather/OUTCOME.md).
Latest GEMV layout slice: [20261014 contiguous GEMV](docs/evidence/akis/20261014-contiguous-gemv/OUTCOME.md).
Latest attention slice: [20261014 attention axpy](docs/evidence/akis/20261014-attention-axpy/OUTCOME.md).
Latest factor slice: [20261014 kron axpy](docs/evidence/akis/20261014-kron-axpy/OUTCOME.md).
Latest allocation slice: [20261014 allocation map](docs/evidence/akis/20261014-allocation-map/OUTCOME.md).
Latest GEMV trial: [20261014 column split](docs/evidence/akis/20261014-column-split/OUTCOME.md).
Latest context sample: [20261014 8192-token logits](docs/evidence/akis/20261014-context-8192/OUTCOME.md).
Latest native diff: [20261014 native tool-call diff](docs/evidence/akis/20261014-native-diff/OUTCOME.md).
Latest boundary slice: [20261014 fail-closed boundaries](docs/evidence/akis/20261014-boundaries/OUTCOME.md).
Every subsequent landing adds a new retained evidence directory and updates this table.

Execution follows [the full E2E plan](docs/END_TO_END_TEST_PLAN.md#9-当前优先工作包):

1. Finish artifact/schema/tokenizer trust boundaries and executable real-model regression.
2. Finish grammar adversarial cases without semantic regex or word-list repairs.
3. Audit every allocation and cancellation stage; prove request and state isolation.
4. Extend independent numerical parity across heads, contexts, depths and quantization.
5. Freeze bilingual task data, compare the same native baseline, train/calibrate where
   model behavior is insufficient. Keep model quality separate from engine equivalence.
6. Integrate real host authorization/grounding and typed Apofasi probability semantics.
7. Optimize packed CQ and caches, benchmark the same workloads and certify platforms/soak.

Native 3.3.1 shares the Chinese and negation failures. Its missing-room suppression
is an additional behavior absent here; copying semantic word-list repairs would
violate A3S engineering rules. Resolve this with model evidence or typed host contracts,
then rerun the unchanged failure cases. No threshold is tuned against this test set.
