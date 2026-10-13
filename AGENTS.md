# Akis engineering rules

- Use `rustup run 1.97.1 cargo ...`; keep Power's embedded-only dependency pinned.
- The root facade owns the public API. Akis owns model semantics; Power owns
  devices, admission, limits and receipts. Do not add a server dependency to infer.
- Product configuration and deterministic A3S Test suites use ACL. JSON model
  requests, schemas, test fixtures and result evidence are data, not configuration.
- Never infer intent, names, missing arguments or grounding with a regex or word
  list. Use learned model evidence or typed host contracts. Never execute tools
  inside the model crate or present raw head scores as calibrated correctness.
- Reject malformed archives, unsupported schemas and resource-limit violations.
  No production `unwrap()`/`expect()` or unbounded fallback.
- Keep tests in sibling files starting with `#![cfg(test)]`. Add a reproducing
  regression for bugs; rerun distinct full-suite failures individually.
- Run fmt, clippy with warnings denied, tests and the relevant real-model checks.
  `tools/certify.sh MODEL` is a fail-closed campaign, not a best-effort smoke script.
  Do not remove quality cases, loosen thresholds or silently skip missing weights.
- Land only owned files, stage each file explicitly, inspect the staged file list
  immediately before committing. Preserve unrelated in-flight work.
- Every checkpoint keeps `docs/evidence/akis/<date-slug>/OUTCOME.md`, hashes,
  actual passing/failing results and an updated ROADMAP gate row. Never delete
  retained evidence or claim a full gate on partial coverage.
- Never commit model weights, native oracle binaries, runtime caches or secrets.
