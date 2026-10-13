#![cfg(test)]
use super::*;
use crate::{DevicePreference, InferenceLimits};
fn model(limits: InferenceLimits) -> Model {
    Model::load(
        Archive::from_bytes(crate::test_support::fixture(), &limits).unwrap(),
        EmbeddedRuntime::new(DevicePreference::Cpu, limits).unwrap(),
    )
    .unwrap()
}
#[test]
fn deterministic_forward_has_isolated_state() {
    let m = model(InferenceLimits::default());
    let cancel = CancellationToken::new();
    let a = m.logits(&[2, 5, 4], &cancel).unwrap();
    m.logits(&[2, 6], &cancel).unwrap();
    let b = m.logits(&[2, 5, 4], &cancel).unwrap();
    assert_eq!(a, b);
    assert_eq!(a.len(), 10);
    assert!(a.iter().all(|v| v.is_finite()));
    assert!(a.iter().any(|v| *v > 0.0));
}
#[test]
fn cancellation_limits_and_admission_fail_closed() {
    let m = model(InferenceLimits::default());
    let cancel = CancellationToken::new();
    cancel.cancel();
    assert!(m.logits(&[2], &cancel).is_err());
    let cancel = CancellationToken::new();
    let permit = m.runtime.begin(&cancel).unwrap();
    assert!(m.logits(&[2], &cancel).is_err());
    drop(permit);
    assert!(m.logits(&[], &cancel).is_err());
    assert!(m.logits(&[100], &cancel).is_err());
    assert!(m.logits(&[2; 65], &cancel).is_err());
    assert!(m.logits(&[2], &cancel).is_ok());
    let m = model(InferenceLimits {
        max_state_bytes: 1,
        ..InferenceLimits::default()
    });
    assert!(m.logits(&[2], &cancel).is_err());
}
#[test]
fn no_head_does_not_invent_confidence() {
    let m = model(InferenceLimits::default());
    let result = m.generate("hi", 2, &CancellationToken::new()).unwrap();
    assert!(result.head_score.is_none());
    assert_eq!(result.receipt.runtime.name, "a3s-power-native");
    assert_eq!(result.receipt.runtime.device, "cpu");
    assert_eq!(m.runtime.admission_snapshot().active, 0);
}

#[test]
fn histories_drop_tokens_outside_the_effective_window() {
    let mut m = model(InferenceLimits::default());
    m.geometry.kv_window = 3;
    let cancel = CancellationToken::new();
    let mut state = m.state(false);
    for token in [2, 4, 5, 4, 6, 4, 5, 4, 6, 4, 5, 4] {
        m.step(token, &mut state, &cancel).unwrap();
        for layer in &state.layers {
            assert!(layer.keys.len() <= 3);
            assert!(layer.values.len() <= 3);
            assert!(layer.q.len() <= 1);
            assert!(layer.k.len() <= 1);
            assert!(layer.v.len() <= 1);
        }
    }
    assert!(state.cells.is_empty());
}

#[test]
fn tokenization_obeys_the_power_state_budget_before_forward() {
    let limits = InferenceLimits {
        max_state_bytes: 130_000,
        ..InferenceLimits::default()
    };
    let mut bytes = crate::test_support::fixture();
    bytes[13 * 4..14 * 4].copy_from_slice(&4096u32.to_le_bytes());
    let m = Model::load(
        Archive::from_bytes(bytes, &limits).unwrap(),
        EmbeddedRuntime::new(DevicePreference::Cpu, limits).unwrap(),
    )
    .unwrap();
    let error = m
        .generate(&"<tool_call>".repeat(1000), 1, &CancellationToken::new())
        .unwrap_err();
    assert!(error.to_string().contains("tokenizer"), "{error}");
    assert_eq!(m.runtime.admission_snapshot().active, 0);
}

#[test]
fn probe_pooling_is_included_in_the_state_budget() {
    let mut m = model(InferenceLimits {
        max_state_bytes: 16 * 1024,
        ..InferenceLimits::default()
    });
    let b = m.weights.len();
    for shape in [
        vec![8192, 2],
        vec![2, 4096],
        vec![1, 2],
        vec![1, 2, 4096],
        vec![1, 2],
        vec![1],
    ] {
        m.weights.push(
            Tensor::zeros(
                shape.as_slice(),
                candle_core::DType::F32,
                m.runtime.device().tensor_device(),
            )
            .unwrap(),
        );
    }
    m.heads.push((2, b));
    m.validate_shapes().unwrap();
    let error = m.generate("hi", 1, &CancellationToken::new()).unwrap_err();
    assert!(error.to_string().contains("state"), "{error}");
    assert_eq!(m.runtime.admission_snapshot().active, 0);
}

#[test]
fn overflowing_head_arithmetic_is_an_error_instead_of_a_success_score() {
    let mut m = model(InferenceLimits::default());
    let b = m.weights.len();
    for (shape, value) in [
        (vec![2, 2], 0.0),
        (vec![2, 1], f32::MAX),
        (vec![1, 2], f32::MAX),
        (vec![1, 2, 1], 0.0),
        (vec![1, 2], f32::MAX),
        (vec![1], 0.0),
    ] {
        m.weights.push(
            Tensor::from_vec(
                vec![value; shape.iter().product()],
                shape.as_slice(),
                m.runtime.device().tensor_device(),
            )
            .unwrap(),
        );
    }
    m.heads.push((2, b));
    m.validate_shapes().unwrap();
    let error = m.generate("hi", 1, &CancellationToken::new()).unwrap_err();
    assert!(error.to_string().contains("non-finite"), "{error}");
    assert_eq!(m.runtime.admission_snapshot().active, 0);
}

#[test]
fn cancellation_during_forward_releases_the_power_permit() {
    let limits = InferenceLimits::default();
    let mut bytes = crate::test_support::fixture();
    bytes[13 * 4..14 * 4].copy_from_slice(&4096u32.to_le_bytes());
    let m = Model::load(
        Archive::from_bytes(bytes, &limits).unwrap(),
        EmbeddedRuntime::new(DevicePreference::Cpu, limits).unwrap(),
    )
    .unwrap();
    let cancel = CancellationToken::new();
    std::thread::scope(|scope| {
        let worker = scope.spawn(|| m.logits(&[4; 4096], &cancel));
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while m.runtime.admission_snapshot().active == 0 {
            assert!(
                !worker.is_finished(),
                "forward ended before cancellation injection"
            );
            assert!(
                std::time::Instant::now() < deadline,
                "request never admitted"
            );
            std::thread::yield_now();
        }
        cancel.cancel();
        assert!(matches!(
            worker.join().unwrap(),
            Err(Error::Power(
                a3s_power::error::PowerError::InferenceCancelled
            ))
        ));
    });
    assert_eq!(m.runtime.admission_snapshot().active, 0);
    assert!(m.logits(&[4], &CancellationToken::new()).is_ok());
}

fn assert_cancelled(error: Error) {
    assert!(
        matches!(
            error,
            Error::Power(a3s_power::error::PowerError::InferenceCancelled)
        ),
        "{error}"
    );
}

fn zero_unembedding(m: &mut Model) {
    let shape = m.weights[0].dims().to_vec();
    m.weights[0] = Tensor::zeros(
        shape.as_slice(),
        candle_core::DType::F32,
        m.runtime.device().tensor_device(),
    )
    .unwrap();
}

#[test]
fn grammar_cancellation_does_not_return_a_tool_call() {
    let mut bytes = crate::test_support::fixture();
    bytes[13 * 4..14 * 4].copy_from_slice(&512u32.to_le_bytes());
    let mut m = Model::load(
        Archive::from_bytes(bytes, &InferenceLimits::default()).unwrap(),
        EmbeddedRuntime::new(DevicePreference::Cpu, InferenceLimits::default()).unwrap(),
    )
    .unwrap();
    zero_unembedding(&mut m);
    // Equal logits select token 0. Point the tool-call marker at that token
    // so the next decode step enters grammar without a semantic word list.
    m.tokenizer.alias_piece("<tool_call>", 0);
    let _arm = crate::test_support::arm("grammar");
    let tool = crate::Tool {
        name: "example".into(),
        description: "d".into(),
        parameters: serde_json::json!({"type":"object","properties":{}}),
    };
    let error = m
        .complete("hi", &[tool], 4, &CancellationToken::new())
        .unwrap_err();
    assert_cancelled(error);
    assert_eq!(m.runtime.admission_snapshot().active, 0);
}

#[test]
fn head_cancellation_does_not_return_a_score() {
    let _arm = crate::test_support::arm("head");
    let mut m = model(InferenceLimits::default());
    let b = m.weights.len();
    for shape in [
        vec![2, 2],
        vec![2, 1],
        vec![1, 2],
        vec![1, 2, 1],
        vec![1, 2],
        vec![1],
    ] {
        m.weights.push(
            Tensor::zeros(
                shape.as_slice(),
                candle_core::DType::F32,
                m.runtime.device().tensor_device(),
            )
            .unwrap(),
        );
    }
    m.heads.push((2, b));
    m.validate_shapes().unwrap();
    let error = m.generate("hi", 1, &CancellationToken::new()).unwrap_err();
    assert_cancelled(error);
    assert_eq!(m.runtime.admission_snapshot().active, 0);
}

#[test]
fn cancellation_before_return_discards_the_generation() {
    let _arm = crate::test_support::arm("before-return");
    let m = model(InferenceLimits::default());
    let error = m.generate("hi", 1, &CancellationToken::new()).unwrap_err();
    assert_cancelled(error);
    assert_eq!(m.runtime.admission_snapshot().active, 0);
}
