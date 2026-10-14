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

Independent numerical checks use the pinned upstream `architecture.py`,
`export.py`, `quantize.py` with JAX/Flax/NumPy in a separate development environment:

```sh
python tools/parity.py MODEL REFERENCE_DIRECTORY target/release/a3s-akis
```

`tools/native_baseline.py LIBRARY MODEL` runs a trusted pinned upstream native
library solely as a test oracle, without SDK telemetry. It is not part of Akis
inference. Do not compare its warm timings to CLI cold-process timings.

Apache-2.0. See [NOTICE](NOTICE) for upstream architecture attribution.
