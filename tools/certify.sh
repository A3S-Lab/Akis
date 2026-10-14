#!/bin/sh
set -eu
project=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$project"
model=${1:?usage: tools/certify.sh MODEL}
test -f "$model"
command -v a3s-test >/dev/null
rustup run 1.97.1 cargo fmt --check
rustup run 1.97.1 cargo clippy --locked --all-targets -- -D warnings
rustup run 1.97.1 cargo test --locked
rustup run 1.97.1 cargo build --release --locked
a3s-test check tests/e2e/terminal-smoke.acl --json
a3s-test run tests/e2e/terminal-smoke.acl \
    --tui-working-directory "$project" \
    --tui-executable "$project/target/release/a3s-akis" \
    --tui-arg inspect --tui-arg "$model" \
    --tui-columns 120 --tui-rows 60 --infrastructure-retries 0 --json
python3 tools/needle_e2e.py "$model" "$project/target/release/a3s-akis"
python3 tools/surface_e2e.py "$model" "$project/target/release/a3s-akis"
python3 tools/live_e2e.py "$model"
