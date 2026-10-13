#[cfg(unix)]
use a3s_akis::request_process_stop;
use a3s_akis::{
    process_stop_requested, Archive, CancellationToken, DevicePreference, EmbeddedRuntime, Error,
    InferenceLimits, Model, Result,
};
use std::io::{Read, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

fn output(value: &impl serde::Serialize, pretty: bool, cancel: &CancellationToken) -> Result<()> {
    crate_check(cancel)?;
    let mut out = std::io::stdout().lock();
    if pretty {
        serde_json::to_writer_pretty(&mut out, value)?;
    } else {
        serde_json::to_writer(&mut out, value)?;
    }
    out.write_all(b"\n")?;
    Ok(())
}

fn crate_check(cancel: &CancellationToken) -> Result<()> {
    if process_stop_requested() {
        cancel.cancel();
    }
    if cancel.is_cancelled() {
        Err(a3s_power::error::PowerError::InferenceCancelled.into())
    } else {
        Ok(())
    }
}

fn read_complete_request(limit: usize, cancel: &CancellationToken) -> Result<Vec<u8>> {
    let mut input = Vec::new();
    let mut buf = [0u8; 8192];
    let mut stdin = std::io::stdin().lock();
    loop {
        crate_check(cancel)?;
        match stdin.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => {
                if input.len().saturating_add(n) > limit {
                    return Err(Error::Request("request byte limit exceeded".into()));
                }
                input.extend_from_slice(&buf[..n]);
            }
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error.into()),
        }
    }
    crate_check(cancel)?;
    Ok(input)
}

fn run(cancel: &CancellationToken) -> Result<()> {
    crate_check(cancel)?;
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        return Err(Error::Request(
            "usage: a3s-akis inspect MODEL | tokenize MODEL TEXT | logits MODEL TEXT | generate MODEL TEXT MAX_TOKENS | complete MODEL < request.json | embed MODEL TEXT".into(),
        ));
    }
    let archive = Archive::open(&args[1], &InferenceLimits::default())?;
    crate_check(cancel)?;
    match args[0].as_str() {
        "inspect" if args.len() == 2 => output(
            &serde_json::json!({"sha256":archive.sha256(),"tensors":archive.tensor_count(),"geometry":archive.geometry()}),
            true,
            cancel,
        )?,
        "tokenize" if args.len() == 3 => {
            let tok = archive.tokenizer()?;
            let ids = tok.encode_bounded(&args[2], &InferenceLimits::default(), cancel)?;
            output(
                &serde_json::json!({"ids":ids,"text":tok.decode(&ids)?}),
                false,
                cancel,
            )?;
        }
        "generate" if args.len() == 4 => {
            let max = args[3]
                .parse()
                .map_err(|_| Error::Request("max tokens must be a positive integer".into()))?;
            let runtime = EmbeddedRuntime::new(DevicePreference::Cpu, InferenceLimits::default())?;
            let model = Model::load(archive, runtime)?;
            output(&model.generate(&args[2], max, cancel)?, false, cancel)?;
        }
        "logits" if args.len() == 3 => {
            let runtime = EmbeddedRuntime::new(DevicePreference::Cpu, InferenceLimits::default())?;
            let model = Model::load(archive, runtime)?;
            let mut ids = vec![model.tokenizer().bos];
            ids.extend(model.tokenizer().encode_bounded(
                &args[2],
                model.runtime().limits(),
                cancel,
            )?);
            let logits = model.logits(&ids, cancel)?;
            output(
                &serde_json::json!({"ids":ids,"logits":logits}),
                false,
                cancel,
            )?;
        }
        "complete" if args.len() == 2 => {
            #[derive(serde::Deserialize)]
            #[serde(deny_unknown_fields)]
            struct Request {
                input: String,
                tools: Vec<a3s_akis::Tool>,
                max_tokens: usize,
            }
            let limits = InferenceLimits::default();
            let input = read_complete_request(limits.max_input_bytes, cancel)?;
            let request: Request = serde_json::from_slice(&input)?;
            let model = Model::load(
                archive,
                EmbeddedRuntime::new(DevicePreference::Cpu, limits)?,
            )?;
            output(
                &model.complete(&request.input, &request.tools, request.max_tokens, cancel)?,
                false,
                cancel,
            )?;
        }
        "embed" if args.len() == 3 => {
            let model = Model::load(
                archive,
                EmbeddedRuntime::new(DevicePreference::Cpu, InferenceLimits::default())?,
            )?;
            output(&model.embed(&args[2], cancel)?, false, cancel)?;
        }
        _ => {
            return Err(Error::Request(
                "unknown command or invalid argument count".into(),
            ))
        }
    }
    Ok(())
}

static SIGNAL_COUNT: AtomicUsize = AtomicUsize::new(0);

#[cfg(unix)]
extern "C" fn handle_stop(signal: i32) {
    request_process_stop();
    if SIGNAL_COUNT.fetch_add(1, Ordering::SeqCst) > 0 {
        unsafe {
            libc::signal(signal, libc::SIG_DFL);
            libc::raise(signal);
        }
    }
}

fn install_os_signals() -> Result<()> {
    #[cfg(unix)]
    unsafe {
        let mut action: libc::sigaction = std::mem::zeroed();
        action.sa_sigaction = handle_stop as *const () as libc::sighandler_t;
        action.sa_flags = 0;
        if libc::sigemptyset(&mut action.sa_mask) != 0
            || libc::sigaddset(&mut action.sa_mask, libc::SIGINT) != 0
            || libc::sigaddset(&mut action.sa_mask, libc::SIGTERM) != 0
            || libc::sigaction(libc::SIGINT, &action, std::ptr::null_mut()) != 0
            || libc::sigaction(libc::SIGTERM, &action, std::ptr::null_mut()) != 0
        {
            return Err(Error::Io(std::io::Error::last_os_error()));
        }
    }
    Ok(())
}

/// Test-only barrier. Production leaves the variable unset and does not wait.
fn publish_signal_barrier() -> Result<()> {
    let Some(path) = std::env::var_os("A3S_AKIS_SIGNAL_READY_FILE") else {
        return Ok(());
    };
    if path.is_empty() {
        return Err(Error::Request("signal ready path is empty".into()));
    }
    std::fs::File::create_new(&path)?.write_all(b"ready\n")?;
    let deadline = Instant::now() + Duration::from_secs(10);
    while !process_stop_requested() {
        if Instant::now() >= deadline {
            return Err(Error::Request("signal barrier expired".into()));
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    Ok(())
}

fn main() -> std::process::ExitCode {
    if let Err(error) = install_os_signals().and_then(|()| publish_signal_barrier()) {
        let _ = writeln!(std::io::stderr().lock(), "{error}");
        return std::process::ExitCode::FAILURE;
    }
    let cancel = CancellationToken::new();
    if process_stop_requested() {
        cancel.cancel();
    }
    match run(&cancel) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            let _ = writeln!(std::io::stderr().lock(), "{error}");
            std::process::ExitCode::FAILURE
        }
    }
}
