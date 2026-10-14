# Akis delivery gates

Target: a tested, efficient replacement for Needle. **No full replacement or
production certification has been granted.** Baseline: 2026-10-13.

| Gate | State | Retained evidence / next obligation |
| --- | --- | --- |
| G0 artifact, tokenizer, structure, CLI | Partial | 32 unit tests, 120 upstream tokenizer comparisons, all CQ widths, 10,000 grammar and 2,000 artifact mutations pass; coverage-guided fuzz and full boundary matrix remain |
| G1 numerical compatibility | Partial | Retained nine JAX logits cases include 255/256/257 and 1023/1024/1025 tokens. One fail-closed run now also covers 1280/1536/2048-token prompts, 20 per-layer lane means, and the activation-quantization path; confidence-pool embedding and confidence head still pass. The 2048-token logits request stayed inside the state budget (cold RSS 536,690,688 bytes beside the retained 1025-token sample of 536,035,328). Max context, a second depth, native behavioral compatibility, and cross-platform remain |
| G2 task quality and calibration | Failed | Real process campaign 13/17; Chinese action, negation and missing-room failures retained; frozen larger dataset/calibration pending |
| G3 host and cancellation | Partial | Unix CLI `SIGINT`/`SIGTERM` sets the process stop flag; grammar, head and pre-return cancellation discard the result; per-stage latency, non-Unix signals and actual host gates pending |
| G4 resources and efficiency | Not certified | Rolling caches and optional head cells previously cut 1025-token cold RSS from 717MB to 536MB. Engram gather tables stay CQ and decode one row (271,843,328 bytes on that binary). Complete projections are stored once as contiguous `[in, out]`, so each step uses the contiguous CPU GEMV. Attention mixes each cached value row with one contiguous axpy after a single width check. On this binary the cold 1025-token logits sample is 272,285,696 bytes and 6.934 seconds, and the 2048-token sample is 273,170,432 bytes and 13.406 seconds; the previous binary took 9.513 and 18.793 seconds. Cosine stayed above 0.999 with the same top-1. Hello and the 256-token logits matched that binary with max absolute error 0. Packed CQ GEMM, a full allocation audit, platform measurement and soak remain |
| G5 delivery platforms | Not certified | Local macOS ARM64 only; independent consumers and other platforms pending |

Evidence anchor: [20261013 bootstrap](docs/evidence/akis/20261013-bootstrap/OUTCOME.md).
Latest: [boundary and tokenizer checks](docs/evidence/akis/20261013-boundary-tokenizer/OUTCOME.md).
Latest resource slice: [20261014 resource checks](docs/evidence/akis/20261014-resources/OUTCOME.md).
Latest cancellation slice: [20261014 signal checks](docs/evidence/akis/20261014-signals/OUTCOME.md).
Latest numerical slice: [20261014 parity checks](docs/evidence/akis/20261014-parity/OUTCOME.md).
Latest residency slice: [20261014 engram gather](docs/evidence/akis/20261014-engram-gather/OUTCOME.md).
Latest GEMV layout slice: [20261014 contiguous GEMV](docs/evidence/akis/20261014-contiguous-gemv/OUTCOME.md).
Latest attention slice: [20261014 attention axpy](docs/evidence/akis/20261014-attention-axpy/OUTCOME.md).
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
