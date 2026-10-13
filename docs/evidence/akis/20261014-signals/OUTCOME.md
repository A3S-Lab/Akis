# Unix signal and stage-cancellation checkpoint

Date: 2026-10-14 Asia/Shanghai. Parent: `554d72a`.
Status: signal and stage checks passed; full replacement certification remains **failed**.

## What changed

On Unix, the CLI installs `SIGINT` and `SIGTERM` before doing command work. The
handler only stores a process-wide stop flag. The next inference checkpoint
cancels the caller token and returns the Power cancellation error. A second
occurrence of the same signal restores the default action and raises it again,
so the process can still be killed. Non-Unix builds do not install a handler.

`complete` reads stdin in bounded chunks, retries interrupted reads, and checks
the flag before and after the read. Successful JSON is not written after the
flag is set.

Grammar masking, confidence-head evaluation, and the point immediately before a
successful `Generation` is returned are named checkpoints. Tests arm one
checkpoint at a time. Each armed stage returns cancellation and releases the
Power permit. The grammar case forces the zero-logit tie token to be the
tool-call marker; it does not add an intent word list.

`A3S_AKIS_SIGNAL_READY_FILE` is a process-test barrier. When it is set, the CLI
creates that file after the handler is installed and waits until the flag is
set, then runs the normal command path. When it is unset, the CLI does not
wait. This is not product configuration.

## Executed evidence

| Check | Result | File |
| --- | --- | --- |
| fmt / clippy all targets, warnings denied | Passed | certification.log |
| Rust unit tests | 35 passed, 0 failed | certification.log |
| CLI `SIGINT` and `SIGTERM` during `complete` | Exit 1, stderr reports cancellation, no `function_calls` | certification.log |
| A3S Test terminal suite | Passed, no cleanup error | certification.log |
| Real-model process campaign | 13 passed, 4 failed | certification.log |
| Complete certification entry point | Exit 1 due to those four failures | certification.log |

The four failures are unchanged: Chinese light-on, Chinese light-off, negated
action, and missing room. No case was removed, no expected answer was edited,
and no threshold was loosened.

## Reproduction

```sh
tools/certify.sh .cache/needle3.cact
```

## Not claimed

Per-stage cancellation latency, a full allocator audit, non-Unix signals, host
authorization, calibration, packed kernels, soak, and platform certification
remain open. The signal barrier proves the handler reaches the normal failure
path; it is not a measurement of cancel latency inside a long forward pass.
No full gate is declared complete.
