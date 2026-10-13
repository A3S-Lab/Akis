use a3s_akis::{
    Archive, CancellationToken, DevicePreference, EmbeddedRuntime, Error, InferenceLimits, Model,
    Result,
};
use std::io::Write;

fn output(value: &impl serde::Serialize, pretty: bool) -> Result<()> {
    let mut out = std::io::stdout().lock();
    if pretty {
        serde_json::to_writer_pretty(&mut out, value)?;
    } else {
        serde_json::to_writer(&mut out, value)?;
    }
    out.write_all(b"\n")?;
    Ok(())
}

fn run() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        return Err(Error::Request(
            "usage: a3s-akis inspect MODEL | tokenize MODEL TEXT | logits MODEL TEXT | generate MODEL TEXT MAX_TOKENS | complete MODEL < request.json | embed MODEL TEXT".into(),
        ));
    }
    let archive = Archive::open(&args[1], &InferenceLimits::default())?;
    match args[0].as_str() {
        "inspect" if args.len() == 2 => output(
            &serde_json::json!({"sha256":archive.sha256(),"tensors":archive.tensor_count(),"geometry":archive.geometry()}),
            true,
        )?,
        "tokenize" if args.len() == 3 => {
            let tok = archive.tokenizer()?;
            let ids = tok.encode_bounded(
                &args[2],
                &InferenceLimits::default(),
                &CancellationToken::new(),
            )?;
            output(
                &serde_json::json!({"ids":ids,"text":tok.decode(&ids)?}),
                false,
            )?;
        }
        "generate" if args.len() == 4 => {
            let max = args[3]
                .parse()
                .map_err(|_| Error::Request("max tokens must be a positive integer".into()))?;
            let runtime = EmbeddedRuntime::new(DevicePreference::Cpu, InferenceLimits::default())?;
            let model = Model::load(archive, runtime)?;
            output(
                &model.generate(&args[2], max, &CancellationToken::new())?,
                false,
            )?;
        }
        "logits" if args.len() == 3 => {
            let runtime = EmbeddedRuntime::new(DevicePreference::Cpu, InferenceLimits::default())?;
            let model = Model::load(archive, runtime)?;
            let mut ids = vec![model.tokenizer().bos];
            ids.extend(model.tokenizer().encode_bounded(
                &args[2],
                model.runtime().limits(),
                &CancellationToken::new(),
            )?);
            let logits = model.logits(&ids, &CancellationToken::new())?;
            output(&serde_json::json!({"ids":ids,"logits":logits}), false)?;
        }
        "complete" if args.len() == 2 => {
            use std::io::Read;
            #[derive(serde::Deserialize)]
            #[serde(deny_unknown_fields)]
            struct Request {
                input: String,
                tools: Vec<a3s_akis::Tool>,
                max_tokens: usize,
            }
            let limits = InferenceLimits::default();
            let mut input = String::new();
            std::io::stdin()
                .take(limits.max_input_bytes as u64 + 1)
                .read_to_string(&mut input)?;
            if input.len() > limits.max_input_bytes {
                return Err(Error::Request("request byte limit exceeded".into()));
            }
            let request: Request = serde_json::from_str(&input)?;
            let model = Model::load(
                archive,
                EmbeddedRuntime::new(DevicePreference::Cpu, limits)?,
            )?;
            output(
                &model.complete(
                    &request.input,
                    &request.tools,
                    request.max_tokens,
                    &CancellationToken::new(),
                )?,
                false,
            )?;
        }
        "embed" if args.len() == 3 => {
            let model = Model::load(
                archive,
                EmbeddedRuntime::new(DevicePreference::Cpu, InferenceLimits::default())?,
            )?;
            output(&model.embed(&args[2], &CancellationToken::new())?, false)?;
        }
        _ => {
            return Err(Error::Request(
                "unknown command or invalid argument count".into(),
            ))
        }
    }
    Ok(())
}

fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            let _ = writeln!(std::io::stderr().lock(), "{e}");
            std::process::ExitCode::FAILURE
        }
    }
}
