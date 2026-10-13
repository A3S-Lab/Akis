//! Profiling baseline: same model stays resident for every measured request.
use a3s_akis::{
    Archive, CancellationToken, DevicePreference, EmbeddedRuntime, Error, InferenceLimits, Model,
    Result, Tool,
};
use std::time::Instant;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        return Err(Error::Request("usage: benchmark MODEL ITERATIONS".into()));
    }
    let iterations: usize = args[2]
        .parse()
        .map_err(|_| Error::Request("invalid iterations".into()))?;
    if !(1..=10000).contains(&iterations) {
        return Err(Error::Request("iterations must be 1..10000".into()));
    }
    let limits = InferenceLimits::default();
    let start = Instant::now();
    let model = Model::load(
        Archive::open(&args[1], &limits)?,
        EmbeddedRuntime::new(DevicePreference::Cpu, limits)?,
    )?;
    let load_ms = start.elapsed().as_secs_f64() * 1000.0;
    let tool = Tool {
        name: "set_lights".into(),
        description: "Turn lights on or off".into(),
        parameters: serde_json::json!({"type":"object","properties":{"room":{"type":"string"},"on":{"type":"boolean"}},"required":["room","on"]}),
    };
    let cancel = CancellationToken::new();
    let expected =
        serde_json::json!([{"name":"set_lights","arguments":{"room":"kitchen","on":true}}]);
    let mut measurements = Vec::new();
    for i in 0..iterations + 1 {
        let start = Instant::now();
        let result = model.complete(
            "turn on the kitchen lights",
            std::slice::from_ref(&tool),
            128,
            &cancel,
        )?;
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        if serde_json::to_value(&result.function_calls)? != expected {
            return Err(Error::Request("benchmark task output mismatch".into()));
        }
        if i > 0 {
            measurements.push(serde_json::json!({"ms":ms,"input_tokens":result.generation.input_tokens,"output_tokens":result.generation.token_ids.len()}));
        }
    }
    println!(
        "{}",
        serde_json::json!({"cold_load_ms":load_ms,"warmup_requests":1,"measurements":measurements})
    );
    Ok(())
}
