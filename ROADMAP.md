# Akis delivery gates

Target: a tested, efficient replacement for Needle. **No full replacement or
production certification has been granted.** Baseline: 2026-10-13.

| Gate | State | Retained evidence / next obligation |
| --- | --- | --- |
| G0 artifact, tokenizer, structure, CLI | Partial | 25 unit tests, 120 upstream tokenizer comparisons, all CQ widths, 10,000 grammar and 2,000 artifact mutations pass; coverage-guided fuzz and full boundary matrix remain |
| G1 numerical compatibility | Partial | Three short-prefix JAX comparisons plus confidence-pool embedding and confidence head pass; long context, depth, per-layer and native behavioral compatibility remain |
| G2 task quality and calibration | Failed | Real process campaign 13/17; Chinese action, negation and missing-room failures retained; frozen larger dataset/calibration pending |
| G3 host and cancellation | Partial | Power admission, pre-cancel and admitted-request cancellation pass; per-stage fault campaign and actual Use/Flow/Code host gates pending |
| G4 resources and efficiency | Not certified | BPE quadratic scan replaced with heap; 4,000-repeat tokenizer case 9.94s → 0.075s; F32 weights, packed kernels, bounded caches, peak allocation audit and soak pending |
| G5 delivery platforms | Not certified | Local macOS ARM64 only; independent consumers and other platforms pending |

Evidence anchor: [20261013 bootstrap](docs/evidence/akis/20261013-bootstrap/OUTCOME.md).
Latest: [boundary and tokenizer checks](docs/evidence/akis/20261013-boundary-tokenizer/OUTCOME.md).
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
