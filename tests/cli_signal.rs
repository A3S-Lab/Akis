#![cfg(unix)]
#[path = "../src/synthetic_fixture.rs"]
mod synthetic_fixture;

use std::io::Write;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

fn signal_during_complete(signal: &str) {
    let root = std::env::temp_dir().join(format!("akis-signal-{}-{}", signal, std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let model = root.join("fixture.cact");
    let ready = root.join("ready");
    std::fs::write(&model, synthetic_fixture::fixture()).unwrap();
    let executable = std::env::var("CARGO_BIN_EXE_a3s_akis")
        .unwrap_or_else(|_| format!("{}/target/debug/a3s-akis", env!("CARGO_MANIFEST_DIR")));
    let mut child = Command::new(&executable)
        .args(["complete", model.to_str().unwrap()])
        .env("A3S_AKIS_SIGNAL_READY_FILE", &ready)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let pid = child.id();
    let finished = Arc::new(AtomicBool::new(false));
    let stop = finished.clone();
    let watchdog = std::thread::spawn(move || {
        for _ in 0..150 {
            if stop.load(Ordering::SeqCst) {
                return;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        let _ = Command::new("kill")
            .args(["-KILL", &pid.to_string()])
            .status();
    });
    let started = Instant::now();
    while !ready.exists() {
        if started.elapsed() > Duration::from_secs(5) {
            panic!("handler did not publish the ready file");
        }
        if let Some(status) = child.try_wait().unwrap() {
            panic!("process exited before the handler was ready: {status}");
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    let signaled = Command::new("kill")
        .args([&format!("-{signal}"), &child.id().to_string()])
        .status()
        .unwrap();
    assert!(signaled.success(), "failed to deliver {signal}");
    let mut stdin = child.stdin.take().unwrap();
    let _ = stdin.write_all(
        br#"{"input":"hi","tools":[{"name":"example","description":"d","parameters":{"type":"object","properties":{}}}],"max_tokens":4}"#,
    );
    drop(stdin);
    let output = child.wait_with_output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(
        output.status.code(),
        Some(1),
        "signal {signal} must cancel the command instead of killing it\nstdout: {stdout}\nstderr: {stderr}"
    );
    assert!(
        stderr.contains("cancelled"),
        "stderr did not report cancellation: {stderr}"
    );
    assert!(
        !stdout.contains("function_calls"),
        "cancelled command returned a tool call: {stdout}"
    );
    finished.store(true, Ordering::SeqCst);
    watchdog.join().unwrap();
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn sigint_cancels_complete_without_a_tool_call() {
    signal_during_complete("INT");
}

#[test]
fn sigterm_cancels_complete_without_a_tool_call() {
    signal_during_complete("TERM");
}
