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

#[test]
fn inspect_accepts_paths_with_spaces_and_han_characters() {
    let root = std::env::temp_dir().join(format!("akis-path-{}", std::process::id()));
    let spaced = root.join("model dir");
    std::fs::create_dir_all(&spaced).unwrap();
    let bytes = synthetic_fixture::fixture();
    let plain = root.join("plain.cact");
    let spaced_file = spaced.join("fixture.cact");
    let han_file = root.join("模型.cact");
    std::fs::write(&plain, &bytes).unwrap();
    std::fs::write(&spaced_file, &bytes).unwrap();
    std::fs::write(&han_file, &bytes).unwrap();
    let plain_sha = sha256_field(&output(&["inspect", plain.to_str().unwrap()], None));
    assert_eq!(plain_sha, file_sha256(&bytes));
    for path in [spaced_file, han_file] {
        let result = output(&["inspect", path.to_str().unwrap()], None);
        let stderr = String::from_utf8_lossy(&result.stderr);
        assert!(result.status.success(), "{stderr}");
        assert!(!stderr.contains("panicked"), "{stderr}");
        assert_eq!(sha256_field(&result), plain_sha);
    }
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn explicit_weights_need_no_home_and_no_writable_directory() {
    let root = std::env::temp_dir().join(format!("akis-home-{}", std::process::id()));
    let writable = root.join("src");
    let work = root.join("work");
    let home = root.join("home");
    std::fs::create_dir_all(&writable).unwrap();
    std::fs::create_dir_all(&work).unwrap();
    std::fs::create_dir_all(&home).unwrap();
    let bytes = synthetic_fixture::fixture();
    let model = writable.join("fixture.cact");
    std::fs::write(&model, &bytes).unwrap();
    let truncated = writable.join("truncated.cact");
    std::fs::write(&truncated, &bytes[..bytes.len() - 1]).unwrap();
    std::fs::set_permissions(&work, std::fs::Permissions::from_mode(0o555)).unwrap();
    std::fs::set_permissions(&home, std::fs::Permissions::from_mode(0o555)).unwrap();

    let result = Command::new(executable())
        .args(["inspect", model.to_str().unwrap()])
        .current_dir(&work)
        .env("HOME", &home)
        .env_remove("XDG_CACHE_HOME")
        .env_remove("XDG_CONFIG_HOME")
        .env_remove("A3S_AKIS_SIGNAL_READY_FILE")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(result.status.success(), "{stderr}");
    assert!(!stderr.contains("panicked"), "{stderr}");
    assert_eq!(sha256_field(&result), file_sha256(&bytes));
    assert!(std::fs::read_dir(&work).unwrap().next().is_none());
    assert!(std::fs::read_dir(&home).unwrap().next().is_none());

    assert_rejected(
        &output(&["inspect", work.to_str().unwrap()], None),
        "model must be a regular file",
    );
    assert_rejected(
        &output(&["inspect", truncated.to_str().unwrap()], None),
        "invalid Needle archive",
    );

    std::fs::set_permissions(&work, std::fs::Permissions::from_mode(0o755)).unwrap();
    std::fs::set_permissions(&home, std::fs::Permissions::from_mode(0o755)).unwrap();
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn inference_does_not_require_network() {
    let root = std::env::temp_dir().join(format!("akis-offline-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let bytes = synthetic_fixture::fixture();
    let model = root.join("fixture.cact");
    std::fs::write(&model, &bytes).unwrap();
    let loaded = sandboxed(&["inspect", model.to_str().unwrap()]);
    let stderr = String::from_utf8_lossy(&loaded.stderr);
    assert!(loaded.status.success(), "{stderr}");
    assert!(!stderr.contains("panicked"), "{stderr}");
    assert_eq!(sha256_field(&loaded), file_sha256(&bytes));
    let missing = root.join("missing-akis.cact");
    let denied = sandboxed(&["inspect", missing.to_str().unwrap()]);
    let stderr = String::from_utf8_lossy(&denied.stderr);
    assert_ne!(denied.status.code(), Some(0), "{stderr}");
    assert!(denied.stdout.is_empty(), "{denied:?}");
    assert!(!stderr.contains("panicked"), "{stderr}");
    assert!(stderr.contains("No such file"), "{stderr}");
    let _ = std::fs::remove_dir_all(root);
}

fn sandboxed(args: &[&str]) -> Output {
    Command::new("sandbox-exec")
        .arg("-p")
        .arg("(version 1)(allow default)(deny network*)")
        .arg(executable())
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .unwrap()
}

fn file_sha256(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    format!("{:x}", Sha256::digest(bytes))
}

fn sha256_field(result: &Output) -> String {
    let text = String::from_utf8(result.stdout.clone()).unwrap();
    let marker = "\"sha256\": \"";
    let at = text.find(marker).unwrap() + marker.len();
    text[at..at + 64].to_string()
}
