# A3S Akis

Rust implementation of the Needle 3 text model on the **A3S Power embedded runtime**.

**Experimental reference implementation. Not yet a drop-in Needle replacement.**
The [test plan](docs/END_TO_END_TEST_PLAN.md) defines acceptance; the
[roadmap](ROADMAP.md) records unfinished gates. Real-model tests currently expose
semantic failures. Do not connect proposals directly to external actions.

Akis imports the original `.cact` artifact, including its tokenizer and CQ
weights. It implements GQA, QKV convolution, engram memory, Monarch MLP, mHC,
A8 simulation, greedy decoding, schema-constrained tool proposals and probe
embeddings. Runtime inference is Rust; Python is used only for independent test
oracles. No upstream native engine is linked or bundled.

Power supplies the CPU tensor device, request admission, cancellation, resource
limits and execution receipts. Its server features are disabled and its Git
revision is pinned in `Cargo.toml`. Akis owns model topology and state.

## Build and run

```sh
rustup run 1.97.1 cargo build --release --locked
target/release/a3s-akis inspect /path/to/needle3.cact
target/release/a3s-akis tokenize /path/to/needle3.cact 'hello'
target/release/a3s-akis generate /path/to/needle3.cact 'hello' 32
target/release/a3s-akis embed /path/to/needle3.cact 'hello'
```

Obtain weights separately from the licensed upstream model repository. The
current certification artifact is
[Needle 3 at revision 93e88ede](https://huggingface.co/Cactus-Compute/needle3/blob/93e88ede867e4311c39151285e74df513e4b835b/needle3.cact),
SHA-256 `c9d915eca282ed42d1a09b143b592adb4cc6744ffe2d294adf5cfc5548170c38`.
Akis never downloads models at inference time.

`complete MODEL` reads one JSON **request** from stdin and writes one JSON result:

```sh
printf '%s' '{"input":"turn on the kitchen lights","tools":[{"name":"set_lights","description":"Turn lights on or off","parameters":{"type":"object","properties":{"room":{"type":"string"},"on":{"type":"boolean"}},"required":["room","on"]}}],"max_tokens":128}' |
  target/release/a3s-akis complete /path/to/needle3.cact
```

JSON here is request/response data, not product configuration. Test suites use ACL.
Successful results contain `function_calls` and `generation`; errors go to stderr
with nonzero exit status. `logits MODEL TEXT` exposes the last-token logits for
numerical verification. Library consumers use `Archive`, `Model`, `Tool`,
`Completion` and `CancellationToken` through the crate facade.

## Contracts and limits

- CPU only; CQ weights expand to F32. Artifact size is **not** resident RAM usage.
- Independent requests; one Power permit covers the complete request. The caller
  supplies the cancellation token. On Unix the CLI installs `SIGINT` and `SIGTERM`:
  the first signal requests cancellation, and a second signal restores the
  default termination action. Other platforms do not install a handler.
- Requests use `Tokenizer::encode_bounded`, which checks Power input/state budgets
  and polls cancellation through scanning, heap construction, merging and emission.
  `Tokenizer::encode` remains a pure utility for trusted caller-owned input.
- KV and convolution caches retain the effective attention/tap windows. Head cells
  are saved only for head-bearing requests. State preflight includes probe pooling,
  output buffers and persistent tool schemas across tokenization and inference.
  These conservative buffer budgets are not an OS RSS limit: allocator and Candle
  backend overhead still need platform-specific measurement.
- Supported schema: strings, numbers, integers, booleans, null, scalar enums,
  arrays and closed objects with required/optional fields. Object properties are
  closed by default. Unsupported constraints are rejected, never ignored.
- Limits: 64 tools, 64 properties per object, 64 elements per array, 255 enum
  choices, schema depth 16; context and generation additionally obey Power limits.
- A complete schema-valid proposal is not evidence that its meaning is correct.
  Host authorization, grounding, confirmation and execution remain host duties.
- `head_score` is an optional raw learned-head sigmoid; `mean_token_probability`
  is a separate decoding statistic. Neither is certified correctness probability.
- Embeddings use the embedding head when available; the current official artifact
  falls back to normalized confidence-probe pooling (3072 dimensions).
- Receipts bind input/output token IDs and weights to a runtime identity. They
  are hashes, not cryptographic signatures or remote attestation.
- No trained Apofasi `choice`/`score`/`noul` probability adapter, Python SDK/ABI
  compatibility layer, audio, training, GPU, streaming or session reuse yet.

## Verification

```sh
rustup run 1.97.1 cargo fmt --check
rustup run 1.97.1 cargo clippy --locked --all-targets -- -D warnings
rustup run 1.97.1 cargo test --locked
python3 tools/live_e2e.py /path/to/needle3.cact
```

`tools/ci-workflow.yml` is an **inactive** GitHub Actions template for component
checks. The current push credential lacks the `workflow` scope; no remote CI
execution is claimed. Local checks and the certification entry point are runnable.

`tools/certify.sh MODEL` runs the component and real-process campaign plus the
A3S Test terminal suite. It fails when a required case fails; it does not skip
missing weights. At this checkpoint the quality campaign is **red**: four of
seventeen cases fail. See retained [evidence](docs/evidence/akis/20261013-bootstrap/OUTCOME.md).
The [boundary/tokenizer checkpoint](docs/evidence/akis/20261013-boundary-tokenizer/OUTCOME.md)
extends importer/grammar coverage and removes quadratic BPE scanning; the same
four semantic failures remain visible.
The [resource checkpoint](docs/evidence/akis/20261014-resources/OUTCOME.md) adds
window-boundary numerical checks, cancellation and buffer-budget regressions,
and a measured 25% cold-process RSS reduction for a 1025-token logits request.
The [signal checkpoint](docs/evidence/akis/20261014-signals/OUTCOME.md) wires
Unix process signals into those checkpoints and proves grammar, head and
pre-return cancellation discard the result.
The [parity checkpoint](docs/evidence/akis/20261014-parity/OUTCOME.md) adds
1280/1536/2048-token comparisons, a per-layer lane-mean error, and the
activation-quantization path on the pinned artifact. The 2048-token request
completed inside the existing state budget. Full G1 and a Needle replacement
are not certified.
The [engram gather checkpoint](docs/evidence/akis/20261014-engram-gather/OUTCOME.md)
keeps engram tables packed and decodes one row at a time. The same 1025-token
logits request drops to 271,843,328 bytes cold RSS. Projection matrices are
still expanded F32. This is not a packed-kernel or replacement certification.
The [contiguous GEMV checkpoint](docs/evidence/akis/20261014-contiguous-gemv/OUTCOME.md)
stores those complete projections once as `[in, out]`. The 2048-token cold
logits request fell from 25.622 seconds to 18.793 seconds. Cold RSS stays
beside the 271,843,328-byte sample. Packed CQ GEMM and a Needle replacement
are not certified.
The [attention axpy checkpoint](docs/evidence/akis/20261014-attention-axpy/OUTCOME.md)
checks each cached value row once and adds it with a contiguous scale. The
2048-token cold logits request fell from 18.793 seconds to 13.406 seconds.
Cold RSS stays beside the 272,236,544-byte sample. Packed CQ GEMM and a
Needle replacement are not certified.
The [kron axpy checkpoint](docs/evidence/akis/20261014-kron-axpy/OUTCOME.md)
adds each Kronecker factor row with that same scale. The 2048-token cold
logits sample is 13.200 seconds beside the previous 13.406 seconds. That
difference sits inside one cold-sample variation, and the factor product
stops there. Cold RSS stays beside the 272,285,696-byte sample. Packed CQ
GEMM and a Needle replacement are not certified.
The [allocation map](docs/evidence/akis/20261014-allocation-map/OUTCOME.md)
lists the vectors rebuilt on each token. A 4-second sample puts `flat` and
`to_vec1` under the cache bar, so no second copy was added. `gemm_basic`
remains the top leaf. Packed CQ GEMM and a Needle replacement are not
certified.
The [column-split trial](docs/evidence/akis/20261014-column-split/OUTCOME.md)
split wide products across columns. Hello and the 256-token logits matched
the kept binary with max absolute error 0. The cold 2048-token sample moved
from 13.200 seconds to 32.020 seconds, so the split was not kept. Packed CQ
GEMM and a Needle replacement are not certified.
The [8192-token sample](docs/evidence/akis/20261014-context-8192/OUTCOME.md)
ran one `<think>` logits request inside the current state budget: 49.39
seconds, 272,875,520 bytes, cosine 0.999999940, top-1 38. Max context and a
Needle replacement are not certified.
The [native tool-call diff](docs/evidence/akis/20261014-native-diff/OUTCOME.md)
replayed the eight fixtures on the pinned native library. Call pairs match
that 2026-10-13 table, including missing-room suppression only on the native
side. Akis answers were not changed. The campaign remains 13 passed and 4
failed. Native compatibility and a Needle replacement are not certified.
The [boundary checks](docs/evidence/akis/20261014-boundaries/OUTCOME.md)
lock the existing rejections for a missing, empty, or unreadable model, for
short CLI arguments, and for one token past the fixture context. One
8193-token logits request on needle3 was rejected before the forward. The
campaign remains 13 passed and 4 failed. Full G0 and a Needle replacement
are not certified.
The [in-process repeat](docs/evidence/akis/20261014-repeat/OUTCOME.md) kept
one model loaded for 1 warmup and 200 kitchen-light completes. Every call
matched. Cold load was 599.6 ms and the process maximum resident set size
was 279,674,880 bytes. The campaign remains 13 passed and 4 failed. Soak
and a Needle replacement are not certified.
The [packed CQ GEMV trial](docs/evidence/akis/20261014-packed-gemv/OUTCOME.md)
kept complete CQ projections packed and applied each group Hadamard to the
input once. Hello and the 256-token logits stayed above cosine 0.999 with
the same top-1. The cold 256-token sample moved from 2.35 seconds to 3.49
seconds, and a 6-core row split still took 3.16 seconds, so the kernel was
not kept. Cold RSS on that sample fell from 272,334,848 bytes to 93,667,328
bytes on the trial binary only. Packed CQ GEMM and a Needle replacement are
not certified.
The [prefix unembed](docs/evidence/akis/20261014-prefix-logits/OUTCOME.md)
skips the output projection on every token whose logits are discarded.
Hello, the 256-token logits, and the 2048-token logits matched the previous
binary with max absolute error 0 and the same top-1. One cold 2048-token
sample took 11.40 seconds and 272,924,672 bytes, beside 12.27 seconds on
the previous binary. The campaign remains 13 passed and 4 failed. Packed CQ
GEMM and a Needle replacement are not certified.
The [layer GEMM](docs/evidence/akis/20261014-layer-gemm/OUTCOME.md) projects
Q, K, V, the attention gates, the attention output, and engram key/value once
per matrix when the prompt has at least 128 tokens. Hello matched with max
absolute error 0. The 256-token and 2048-token logits kept the same top-1,
with cosine 0.999997200 and 0.999998915. One cold 2048-token sample took
6.18 seconds and 397,377,536 bytes, beside 11.77 seconds on the previous
binary. Shorter prompts stay on the token loop after a 59-token head moved
by 0.012. The campaign remains 13 passed and 4 failed. Packed CQ GEMM and a
Needle replacement are not certified.

The [attention scores](docs/evidence/akis/20261014-attention-scores/OUTCOME.md)
score each kv head as one matrix product per 256-token chunk when the prompt
has at least 128 tokens. Hello matched with max absolute error 0. The
256-token and 2048-token logits kept the same top-1, with cosine 0.999997899
and 0.999998738. One cold 2048-token sample took 5.11 seconds and 390,643,712
bytes, beside 6.07 seconds on the previous binary. The campaign remains 13
passed and 4 failed. Packed CQ GEMM and a Needle replacement are not certified.

The [attention values](docs/evidence/akis/20261014-attention-values/OUTCOME.md)
mix each long-prompt chunk as one product of the softmax weights and that kv
head's values. Hello matched with max absolute error 0. The 256-token and
2048-token logits kept the same top-1, with cosine 0.999998857 and
0.999998940. One cold 2048-token sample took 4.49 seconds and 416,235,520
bytes, beside 5.04 seconds on the previous binary and the retained
5.11-second sample. The campaign remains 13 passed and 4 failed. Packed CQ
GEMM and a Needle replacement are not certified.

The [prefill tokens](docs/evidence/akis/20261014-prefill-tokens/OUTCOME.md)
finish each layer's tokens together when a prompt has at least 128 tokens and
probe cells are off. Hello, the 256-token logits, and the 2048-token logits
matched the previous binary with max absolute error 0 and the same top-1. One
cold 2048-token sample took 2.61 seconds and 418,889,728 bytes, beside 4.23
seconds on the previous binary and the retained 4.49-second sample. The
campaign remains 13 passed and 4 failed. Packed CQ GEMM and a Needle
replacement are not certified.

The [needle envelope](docs/evidence/akis/20261014-needle-envelope/OUTCOME.md)
compares the eight tool inputs with the pinned native calls. Seven match.
`missing-room` still returns the living-room call. Inference code is unchanged,
so the quality campaign remains 13 passed and 4 failed. Native behavioral
compatibility and a Needle replacement are not certified.

The [string span](docs/evidence/akis/20261014-string-span/OUTCOME.md) keeps a
call only when every string argument occurs in the user input. The eight
frozen native envelopes match, including `missing-room`. The kitchen-light
head score remains 0.8660065. The quality campaign is 14 passed and 3 failed.
Chinese action and negation remain. A Needle replacement is not certified.

The [command surface](docs/evidence/akis/20261014-surface/OUTCOME.md) runs
inspect, tokenize, logits, layers, embed, generate, and complete on the pinned
archive before the quality campaign. Hello generation stops at the four-token
bound with `finished` false. The kitchen completion stays at head score
0.8660065. The campaign remains 14 passed and 3 failed. A Needle replacement
is not certified.

The [numerical ledger](docs/evidence/akis/20261014-ledger/OUTCOME.md) records
31 JAX comparisons and 120 tokenizer comparisons on this binary. One cold
2048-token logits sample took 2.77 seconds and 418,955,264 bytes, beside the
retained 2.61-second sample. The samples are records. The campaign remains 14
passed and 3 failed. A Needle replacement is not certified.

The [path checks](docs/evidence/akis/20261014-paths/OUTCOME.md) accept an
inspect path that contains a space and a Han filename. Empty logits, layers,
and embed stay finite and repeat on a second process. The campaign remains 14
passed and 3 failed. A Needle replacement is not certified.

The [receipt checks](docs/evidence/akis/20261014-receipt/OUTCOME.md) require
both kitchen complete processes to carry schema
`a3s.power.embedded-execution-receipt.v1`, device cpu, the archive weights
sha256, and the same input and output sha256. The kitchen head score remains
0.8660065. The campaign remains 14 passed and 3 failed. A Needle replacement
is not certified.

The [CLI home checks](docs/evidence/akis/20261014-cli-home/OUTCOME.md) load an
explicit fixture when HOME and the working directory are read-only. The
reported sha256 matches an independent digest of the file bytes, and neither
directory gains a file. A directory path and a truncated file are rejected.
The campaign remains 14 passed and 3 failed. A Needle replacement is not
certified.

The [canonical receipt](docs/evidence/akis/20261014-canonical-receipt/OUTCOME.md)
checks each passing completion against a second tokenize of the same chat
prompt. The input digest covers BOS plus those ids. The output digest covers
the returned token ids. One changed token does not match. The kitchen head
score remains 0.8660065. The campaign remains 14 passed and 3 failed. A Needle
replacement is not certified.

The [offline checks](docs/evidence/akis/20261014-offline/OUTCOME.md) run the
seven commands with network denied. A missing archive fails with `No such
file` and does not panic. The kitchen head score remains 0.8660065. The
campaign remains 14 passed and 3 failed. A Needle replacement is not
certified.

Independent numerical checks use the pinned upstream `architecture.py`,
`export.py`, `quantize.py` with JAX/Flax/NumPy in a separate development environment:

```sh
python tools/parity.py MODEL REFERENCE_DIRECTORY target/release/a3s-akis
```

`tools/native_baseline.py LIBRARY MODEL` runs a trusted pinned upstream native
library solely as a test oracle, without SDK telemetry. It is not part of Akis
inference. Do not compare its warm timings to CLI cold-process timings.

Apache-2.0. See [NOTICE](NOTICE) for upstream architecture attribution.
