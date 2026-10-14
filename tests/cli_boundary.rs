#[path = "../src/synthetic_fixture.rs"]
mod synthetic_fixture;

use std::io::Write;
use std::os::unix::fs::PermissionsExt;
use std::process::{Command, Output, Stdio};

fn executable() -> String {
    std::env::var("CARGO_BIN_EXE_a3s_akis")
        .unwrap_or_else(|_| format!("{}/target/debug/a3s-akis", env!("CARGO_MANIFEST_DIR")))
}

fn output(args: &[&str], stdin: Option<&[u8]>) -> Output {
    let mut child = Command::new(executable())
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    if let Some(bytes) = stdin {
        child.stdin.take().unwrap().write_all(bytes).unwrap();
    }
    child.wait_with_output().unwrap()
}

fn assert_rejected(result: &Output, message: &str) {
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert_ne!(result.status.code(), Some(0), "{stderr}");
    assert!(result.stdout.is_empty(), "{result:?}");
    assert!(!stderr.contains("panicked"), "{stderr}");
    assert!(stderr.contains(message), "{stderr}");
}

#[test]
fn missing_empty_and_unreadable_models_do_not_panic() {
    let root = std::env::temp_dir().join(format!("akis-boundary-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let missing = root.join("missing.cact");
    assert_rejected(
        &output(&["inspect", missing.to_str().unwrap()], None),
        "No such file",
    );

    let empty = root.join("empty.cact");
    std::fs::write(&empty, b"").unwrap();
    assert_rejected(
        &output(&["inspect", empty.to_str().unwrap()], None),
        "invalid model length",
    );

    let unreadable = root.join("unreadable.cact");
    std::fs::write(&unreadable, synthetic_fixture::fixture()).unwrap();
    std::fs::set_permissions(&unreadable, std::fs::Permissions::from_mode(0o000)).unwrap();
    let opened = output(&["inspect", unreadable.to_str().unwrap()], None);
    let stderr = String::from_utf8_lossy(&opened.stderr);
    assert!(!stderr.contains("panicked"), "{stderr}");
    if opened.status.success() {
        eprintln!("mode 000 fixture remained readable on this platform");
    } else {
        assert!(opened.stdout.is_empty(), "{opened:?}");
    }
    std::fs::set_permissions(&unreadable, std::fs::Permissions::from_mode(0o644)).unwrap();
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn short_arguments_illegal_integers_and_unknown_fields_are_rejected() {
    let root = std::env::temp_dir().join(format!("akis-cli-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let model = root.join("fixture.cact");
    std::fs::write(&model, synthetic_fixture::fixture()).unwrap();
    let model = model.to_str().unwrap();
    assert_rejected(&output(&[], None), "usage: a3s-akis");
    assert_rejected(
        &output(&["tokenize", model], None),
        "unknown command or invalid argument count",
    );
    assert_rejected(
        &output(&["generate", model, "hi"], None),
        "unknown command or invalid argument count",
    );
    assert_rejected(
        &output(&["generate", model, "hi", "xyz"], None),
        "max tokens must be a positive integer",
    );
    let request = br#"{"input":"hi","tools":[],"max_tokens":4,"extra":1}"#;
    assert_rejected(
        &output(&["complete", model], Some(request)),
        "unknown field",
    );
    let _ = std::fs::remove_dir_all(root);
}
