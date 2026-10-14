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
        m.step(token, &mut state, &cancel, super::StepLogits::Omit)
            .unwrap();
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
fn omitted_prefix_logits_keep_the_cache_and_the_last_distribution() {
    let m = model(InferenceLimits::default());
    let cancel = CancellationToken::new();
    let tokens = [2u32, 4, 5, 4];
    let mut written = m.state(false);
    let mut last = Vec::new();
    for &token in &tokens {
        last = m
            .step(token, &mut written, &cancel, super::StepLogits::Return)
            .unwrap();
    }
    assert_eq!(last.len(), m.geometry.out_vocab);
    let mut omitted = m.state(false);
    for (index, &token) in tokens.iter().enumerate() {
        let mode = if index + 1 == tokens.len() {
            super::StepLogits::Return
        } else {
            super::StepLogits::Omit
        };
        let logits = m.step(token, &mut omitted, &cancel, mode).unwrap();
        if mode == super::StepLogits::Omit {
            assert!(logits.is_empty());
        } else {
            assert_eq!(logits, last);
        }
    }
    assert_eq!(written.tokens, omitted.tokens);
    assert_eq!(written.engram_values, omitted.engram_values);
    for (left, right) in written.layers.iter().zip(&omitted.layers) {
        assert_eq!(left.keys, right.keys);
        assert_eq!(left.values, right.values);
        assert_eq!(left.q, right.q);
        assert_eq!(left.k, right.k);
        assert_eq!(left.v, right.v);
    }
}

#[test]
fn short_and_narrow_prompts_stay_on_the_token_loop() {
    let m = model(InferenceLimits::default());
    assert!(!m.batched_prefill(1));
    assert!(!m.batched_prefill(4096));
}

#[test]
fn key_spans_follow_the_cache_deque() {
    let phases = [
        (0usize, vec![1usize, 2, 3, 3, 3, 3]),
        (4, vec![4, 4, 4, 2, 2, 2]),
        (0, vec![0, 0, 0]),
        (10, vec![7, 7, 7, 7, 256]),
    ];
    for (past, windows) in phases {
        let spans = super::record_key_spans(past, &windows).unwrap();
        let mut deque = std::collections::VecDeque::new();
        for id in 0..past {
            deque.push_back(id);
        }
        for (token, &window) in windows.iter().enumerate() {
            if deque.len() == window {
                deque.pop_front();
            }
            deque.push_back(past + token);
            let (start, end) = spans[token];
            let got: Vec<usize> = deque.iter().copied().collect();
            let expect: Vec<usize> = (start..end).collect();
            assert_eq!(got, expect, "{past} {token} window {window}");
        }
    }
}

#[test]
fn batched_attention_scores_match_each_query_dot() {
    let m = model(InferenceLimits::default());
    let width = 48;
    let queries_n = 6;
    let keys_n = 17;
    let rows: Vec<f32> = (0..queries_n * width)
        .map(|i| (i % 19) as f32 * 0.03 - 0.2)
        .collect();
    let keys: Vec<f32> = (0..keys_n * width)
        .map(|i| (i % 23) as f32 * 0.02 - 0.15)
        .collect();
    let batched = m.dot_matrix(&rows, &keys, width).unwrap();
    assert_eq!(batched.len(), queries_n * keys_n);
    let top = |values: &[f32]| {
        values
            .iter()
            .enumerate()
            .max_by(|(left_i, left), (right_i, right)| {
                left.total_cmp(right).then_with(|| right_i.cmp(left_i))
            })
            .map(|(index, _)| index)
    };
    for query in 0..queries_n {
        let q = &rows[query * width..(query + 1) * width];
        let mut one = Vec::with_capacity(keys_n);
        for key in 0..keys_n {
            one.push(crate::math::dot(q, &keys[key * width..(key + 1) * width]));
        }
        let got = &batched[query * keys_n..(query + 1) * keys_n];
        let max = got
            .iter()
            .zip(&one)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0f32, f32::max);
        assert!(max < 1e-4, "{max}");
        let mut left = got.to_vec();
        let mut right = one;
        let scale = (width as f32).sqrt();
        for value in &mut left {
            *value /= scale;
        }
        for value in &mut right {
            *value /= scale;
        }
        crate::math::softmax(&mut left);
        crate::math::softmax(&mut right);
        assert_eq!(top(&left), top(&right));
    }
}

#[test]
fn batched_projection_matches_each_row_product() {
    let m = model(InferenceLimits::default());
    let device = m.runtime.device().tensor_device();
    let cols = 32;
    let outs = 24;
    let rows = 8;
    let values: Vec<f32> = (0..cols * outs)
        .map(|i| (i % 17) as f32 * 0.01 - 0.08)
        .collect();
    let w = Tensor::from_vec(values, (cols, outs), device).unwrap();
    let input: Vec<f32> = (0..rows * cols)
        .map(|i| (i % 13) as f32 * 0.02 - 0.11)
        .collect();
    let batched = m.project_matrix(&w, &input, cols).unwrap();
    assert_eq!(batched.len(), rows * outs);
    for row in 0..rows {
        let one = m.gemv(&w, &input[row * cols..(row + 1) * cols]).unwrap();
        let got = &batched[row * outs..(row + 1) * outs];
        let max = got
            .iter()
            .zip(&one)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0f32, f32::max);
        assert!(max < 1e-5, "{max}");
        let top = |values: &[f32]| {
            values
                .iter()
                .enumerate()
                .max_by(|(left_i, left), (right_i, right)| {
                    left.total_cmp(right).then_with(|| right_i.cmp(left_i))
                })
                .map(|(index, _)| index)
        };
        assert_eq!(top(got), top(&one));
    }
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
fn layer_readout_reports_every_layer_and_rejects_cancellation() {
    let m = model(InferenceLimits::default());
    let cancel = CancellationToken::new();
    let readout = m.layer_readout(&[2, 5, 4], &cancel).unwrap();
    assert_eq!(readout.layers.len(), m.geometry.layers);
    assert!(readout
        .layers
        .iter()
        .all(|layer| layer.len() == m.geometry.width));
    assert_eq!(readout.quantized_hidden.len(), m.geometry.width);
    assert!(readout
        .quantized_hidden
        .iter()
        .all(|value| value.is_finite()));
    let again = m.layer_readout(&[2, 5, 4], &cancel).unwrap();
    assert_eq!(readout.layers, again.layers);
    assert_eq!(readout.quantized_hidden, again.quantized_hidden);
    let cancel = CancellationToken::new();
    cancel.cancel();
    assert_cancelled(m.layer_readout(&[2, 5, 4], &cancel).unwrap_err());
    assert_eq!(m.runtime.admission_snapshot().active, 0);
}

#[test]
fn transpose_out_in_matches_the_index_permutation() {
    let mut wide = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    super::transpose_out_in_inplace(&mut wide, 2, 3).unwrap();
    assert_eq!(wide, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    let mut square = vec![1.0, 2.0, 3.0, 4.0];
    super::transpose_out_in_inplace(&mut square, 2, 2).unwrap();
    assert_eq!(square, vec![1.0, 3.0, 2.0, 4.0]);
    let mut column = vec![1.0, 2.0, 3.0];
    super::transpose_out_in_inplace(&mut column, 3, 1).unwrap();
    assert_eq!(column, vec![1.0, 2.0, 3.0]);
    let rows = 6;
    let cols = 4;
    let mut values: Vec<f32> = (0..rows * cols).map(|i| i as f32).collect();
    let original = values.clone();
    super::transpose_out_in_inplace(&mut values, rows, cols).unwrap();
    for row in 0..rows {
        for col in 0..cols {
            assert_eq!(values[col * rows + row], original[row * cols + col]);
        }
    }
}

#[test]
fn complete_projections_stay_in_matmul_order_and_keep_archive_rows() {
    let m = model(InferenceLimits::default());
    let width = m.geometry.width;
    let vocab = m.geometry.vocab;
    assert_eq!(m.weights[0].dims(), &[width, vocab]);
    assert!(m.gemm[0]);
    let o = 1 + if m.geometry.conv_taps > 0 { 3 } else { 0 };
    assert_eq!(m.weights[o + 22].dims(), &[width, 8]);
    assert!(!m.gemm[o + 22]);
    let mut input = vec![0.0; width];
    input[0] = 1.0;
    let projected = m.linear(0, &input).unwrap();
    assert_eq!(projected.len(), vocab);
    for (token, logit) in projected.iter().enumerate() {
        let row = m.row(0, token).unwrap();
        assert_eq!(row.len(), width);
        assert_eq!(*logit, row[0]);
    }
}

#[test]
fn cancellation_before_return_discards_the_generation() {
    let _arm = crate::test_support::arm("before-return");
    let m = model(InferenceLimits::default());
    let error = m.generate("hi", 1, &CancellationToken::new()).unwrap_err();
    assert_cancelled(error);
    assert_eq!(m.runtime.admission_snapshot().active, 0);
}

fn kron_scalar(x: &[f32], aw: &[f32], bw: &[f32], na: usize, nb: usize) -> Vec<f32> {
    let mut mid = vec![0.0; x.len()];
    let mut out = vec![0.0; x.len()];
    for i in 0..na {
        for j in 0..nb {
            for l in 0..nb {
                mid[i * nb + l] += x[i * nb + j] * bw[j * nb + l];
            }
        }
    }
    for i in 0..na {
        for k in 0..na {
            for l in 0..nb {
                out[k * nb + l] += aw[i * na + k] * mid[i * nb + l];
            }
        }
    }
    out
}

#[test]
fn context_length_and_generation_bound_fail_closed() {
    let m = model(InferenceLimits::default());
    let cancel = CancellationToken::new();
    let context = m.geometry().context;
    let limit = "token or context limit exceeded";
    let empty = m.logits(&[], &cancel).unwrap_err();
    assert!(empty.to_string().contains(limit), "{empty}");
    let accepted = m.logits(&vec![2; context], &cancel).unwrap();
    assert_eq!(accepted.len(), m.geometry().out_vocab);
    assert!(accepted.iter().all(|value| value.is_finite()));
    let over = m.logits(&vec![2; context + 1], &cancel).unwrap_err();
    assert!(over.to_string().contains(limit), "{over}");

    let cap = InferenceLimits::default().max_generated_tokens;
    let over_cap = m.generate("hi", cap + 1, &cancel).unwrap_err();
    assert!(over_cap.to_string().contains(limit), "{over_cap}");

    // The default generation cap is above this fixture's context, so the call
    // above rejects on context. This model keeps the prompt inside that
    // context and sets the generation cap to one token.
    let bounded = model(InferenceLimits {
        max_generated_tokens: 1,
        ..InferenceLimits::default()
    });
    assert!(bounded.geometry().context > 2);
    if let Err(error) = bounded.generate("hi", 1, &cancel) {
        assert!(
            !error.to_string().contains(limit),
            "a one-token generation was rejected as a context overflow: {error}"
        );
    }
    let step_over = bounded.generate("hi", 2, &cancel).unwrap_err();
    assert!(step_over.to_string().contains(limit), "{step_over}");
}

#[test]
fn kron_matches_the_scalar_triple_loop_on_every_lane() {
    for (na, nb) in [
        (0, 3),
        (2, 0),
        (1, 1),
        (1, 3),
        (2, 5),
        (3, 7),
        (4, 4),
        (5, 6),
    ] {
        let x: Vec<f32> = (0..na * nb).map(|i| (i as f32) * 0.017 - 0.4).collect();
        let aw: Vec<f32> = (0..na * na).map(|i| (i as f32) * 0.01 - 0.3).collect();
        let bw: Vec<f32> = (0..nb * nb).map(|i| (i as f32) * -0.013 + 0.2).collect();
        let found = super::kron_factors(&x, &aw, &bw, na, nb);
        let expect = kron_scalar(&x, &aw, &bw, na, nb);
        assert_eq!(found, expect, "na {na} nb {nb}");
    }
}
