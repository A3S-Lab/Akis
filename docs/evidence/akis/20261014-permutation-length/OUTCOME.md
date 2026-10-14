# Permutation length

Date: 2026-10-14 Asia/Shanghai. Parent: `44c4c5e`.
Status: Hadamard 2 with a length-2 permutation loads. Hadamard 4 opens and is then rejected. G0 stays Partial. The quality campaign remains 14 passed and 3 failed. A Needle replacement is **not** certified.

## What changed

No inference code changed. The release binary is `9cc7f01369941ec446f78659e2b4335aab0f3abfce509c9a3369fb643ca6cfcb`.

- Hadamard 2 with the stored length-2 permutation loads.
- Hadamard 4 still opens. The stored permutations remain length 2.
- Load then rejects that archive with `invalid Needle archive: permutation shape mismatch`.
- The error does not contain `invalid Hadamard permutation` or `invalid model geometry`.

## Executed evidence

`tools/certify.sh .cache/needle3.cact` ran fmt, clippy, `cargo test --locked`, the release build, the terminal smoke, the 8-case envelope, the sandboxed surface, and the 17-case campaign. 63 library tests passed. 7 CLI process tests passed (5 boundary, 2 signal). The envelope is 8 passed and 0 failed. The surface passed. The campaign is 14 passed and 3 failed, exit 1. The failures remain `lights-on-zh`, `lights-off-zh`, and `negated-action`. The log is `certification.log`.

## Not claimed

This is one synthetic header field against the stored permutation length. It is not the needle3 Hadamard graph. Section 6 of the end-to-end plan is not met: there is no user-supplied frozen bilingual set, no second non-macOS-ARM64 machine, and no percentile or soak evidence. Coverage-guided fuzz and the rest of the boundary matrix remain open. No full gate is declared complete. This is not a v0.1.0 release or a Needle replacement.
