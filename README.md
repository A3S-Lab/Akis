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
  supplies the cancellation token. CLI OS signals are not yet wired to that token.
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

Independent numerical checks use the pinned upstream `architecture.py`,
`export.py`, `quantize.py` with JAX/Flax/NumPy in a separate development environment:

```sh
python tools/parity.py MODEL REFERENCE_DIRECTORY target/release/a3s-akis
```

`tools/native_baseline.py LIBRARY MODEL` runs a trusted pinned upstream native
library solely as a test oracle, without SDK telemetry. It is not part of Akis
inference. Do not compare its warm timings to CLI cold-process timings.

Apache-2.0. See [NOTICE](NOTICE) for upstream architecture attribution.
