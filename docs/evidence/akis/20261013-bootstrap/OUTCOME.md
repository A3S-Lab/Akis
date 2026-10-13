# Needle 3 reference implementation checkpoint

Date: 2026-10-13. Platform: macOS 26.6.2 ARM64; Rust 1.97.1.
Status: **development checkpoint; complete replacement NOT certified**.

## Scope and identity

Akis owns a Rust Needle 3 CPU forward and `.cact` importer. A3S Power 0.10.0,
revision `830760cb9dc169893e0f9ba9e91ac5399c14d485`, owns embedded admission,
CPU device, cancellation and receipts; server features are disabled.

Upstream model/format/reference source:
`cactus-compute/needle@3da4261073199444f2e070800854436fbc31fb6a`.
Model repository: `Cactus-Compute/needle3@93e88ede867e4311c39151285e74df513e4b835b`.
Actual artifact: 35,335,380 bytes, 581 records,
SHA-256 `c9d915eca282ed42d1a09b143b592adb4cc6744ffe2d294adf5cfc5548170c38`.

This directory is bound to the initial source by `source-sha256.txt`; the
containing Git commit is the source snapshot. `provenance.json` additionally
pins the binary, Cargo.lock, model, native oracle and JAX source files.
Weights/native binaries are cached locally and excluded from Git.

## Executed results

| Check | Result | Evidence |
| --- | --- | --- |
| fmt, clippy all targets with warnings denied | Passed | certification.log |
| Rust tests | 18 passed, 0 failed | certification.log |
| Independent upstream JAX logits | 3 passed; all top-1 identical; cosine 0.999983–0.999999 | jax-parity.jsonl |
| Confidence pool embedding | 3072 dimensions; cosine 0.9999085 | jax-parity.jsonl |
| Raw confidence head | Absolute sigmoid difference 0.00013725 | jax-parity.jsonl |
| A3S Test ACL admission and real CLI PTY | Passed; root exit 0; no cleanup error | terminal-report.json, certification.log |
| Real-model process E2E | **13 passed, 4 failed** | live-e2e.jsonl |
| Native 3.3.1 comparison | 8 responses captured, no equivalence certification | native-baseline.jsonl |
| Warm Akis profiling | 1 warm-up + 10 measured calls, median 750.99 ms; cold load 552.38 ms | warm-benchmark.json |
| Entire certify entry point | **Failed as required**, exit 1 due to the four quality cases | certification.log |

The live process campaign covers artifact identity, Unicode round trip,
deterministic normalized embedding, two English actions, two Chinese actions,
irrelevant/negated/missing/quoted actions, malformed stdin, unknown command,
closed stdout pipe, zero token bound, incomplete output and unsupported schema.
Output receipt hashes are independently rebuilt using Power's canonical domain
prefix and little-endian token IDs. This is not a signature verification claim.

All fixtures are synthetic. No real tools, remote devices or business actions
were executed. The native oracle loads the pinned trusted dylib via ctypes;
it does not import the telemetry-enabled Python SDK.

## Failures retained and triaged

Each quality failure was rerun individually and reproduced:

| Case | Expected | Akis | Native 3.3.1 |
| --- | --- | --- | --- |
| lights-on-zh | Set 厨房 on | Empty proposal | Empty proposal |
| lights-off-zh | Set 客厅 off | Empty proposal | Empty proposal |
| negated-action | No action | Sets kitchen off | Also sets kitchen off; marks negation |
| missing-room | No action | Sets empty room on | Suppresses inferred living-room call |

The first three are also baseline model/product shortcomings. They still block
A3S quality acceptance; baseline agreement does not make a wrong action correct.
The fourth reveals missing native behavior. A3S prohibits semantic word-list
repairs; model evidence or typed host grounding must resolve it. Tests and
expected answers have not been weakened and no thresholds have been fitted.

Component regressions first failed before their fixes and were individually
rerun: numeric enum prefix ordering, irrelevant schema keywords, malformed
UTF-8 prefixes. Framing now uses token IDs, preserving literal marker text in
string values. Additional tests cover request isolation and permit release on
cancellation. The output writer handles a broken pipe as an error, without panic.

Test infrastructure defects corrected during this campaign: the first receipt
checker omitted Power's domain separator; the invalid-stdin check expected the
word JSON rather than the actual typed parser error; the first PTY launch used
the tool's default working directory; a JAX NumPy view was read-only. None
required changing product expectations. The first broad test filter matched
zero tests; actual test execution is the 18-test run retained here.

## Reproduction

```sh
tools/certify.sh .cache/needle3.cact
.cache/oracle/bin/python tools/parity.py .cache/needle3.cact .cache/needle target/release/a3s-akis
python3 tools/native_baseline.py .cache/libneedle3.dylib .cache/needle3.cact
rustup run 1.97.1 cargo run --release --example benchmark -- .cache/needle3.cact 10
python3 tools/live_e2e.py .cache/needle3.cact --case missing-room
```

Oracle development environment used Python 3.12, NumPy 2.5.4, JAX/JAXLIB 0.11.2,
Flax 0.12.10. No oracle package is a Rust runtime dependency.

## Remaining gates

No native SDK/ABI drop-in compatibility, native final-confidence equivalence,
full-language quality, deployment calibration, trained Apofasi probability
adapter, session/streaming API, training/audio, packed memory claim, performance
equivalence, soak or non-macOS platform certification. The warm profile is ten
samples for optimization, not p99/throughput certification. Native responses
use different prompt/output policies, so timing ratios would be misleading.

All six full gates remain unfinished. Continue in ROADMAP order; keep these
failures visible until their original assertions pass.

Delivery note: the first HTTPS push had no Git credential helper; using the
existing authenticated `gh` helper resolved authentication. GitHub then refused
the Actions workflow because that OAuth credential lacks `workflow` scope.
The unpushed checkpoint keeps the workflow as `tools/ci-workflow.yml`, an inactive
template. No credential scopes were changed and no remote CI result is claimed.
