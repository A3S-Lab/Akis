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
