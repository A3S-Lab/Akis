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
fn batched_value_mix_matches_each_scaled_add() {
    let m = model(InferenceLimits::default());
    let rows = 5;
    let keys = 9;
    let width = 8;
    let weights: Vec<f32> = (0..rows * keys)
        .map(|i| {
            if i % 4 == 0 {
                0.0
            } else {
                (i % 11) as f32 * 0.05 - 0.2
            }
        })
        .collect();
    let values: Vec<f32> = (0..keys * width)
        .map(|i| (i % 13) as f32 * 0.03 - 0.1)
        .collect();
    let got = m.mix_values(&weights, &values, rows, keys, width).unwrap();
    assert!(m.mix_values(&weights, &values, rows, keys, 0).is_err());
    let mut expect = vec![0.0f32; rows * width];
    for row in 0..rows {
        let scale = &weights[row * keys..(row + 1) * keys];
        let dst = &mut expect[row * width..(row + 1) * width];
        for key in 0..keys {
            crate::math::axpy(dst, scale[key], &values[key * width..(key + 1) * width]);
        }
    }
    let max = got
        .iter()
        .zip(&expect)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0f32, f32::max);
    assert!(max < 1e-4, "{max}");
    let top = |values: &[f32]| {
        values
            .iter()
            .enumerate()
            .max_by(|(left_i, left), (right_i, right)| {
                left.total_cmp(right).then_with(|| right_i.cmp(left_i))
            })
            .map(|(index, _)| index)
    };
    for row in 0..rows {
        let got_row = &got[row * width..(row + 1) * width];
        let expect_row = &expect[row * width..(row + 1) * width];
        assert_eq!(top(got_row), top(expect_row));
    }
}

#[test]
fn parallel_prefill_tail_matches_each_token() {
    let m = model(InferenceLimits::default());
    let d = m.geometry.width;
    let n = m.geometry.lanes;
    let scale = (d as f32).sqrt();
    let embedding: Vec<f32> = m
        .row(0, 2)
        .unwrap()
        .iter()
        .map(|value| value * scale)
        .collect();
    let stream = embedding.repeat(n);
    let ap = m.flat(m.mhc).unwrap();
    let ao = m.flat(m.mhc + 1).unwrap();
    let ar = m.flat(m.mhc + 2).unwrap();
    let carry = || m.prepare_carry(&stream, 0, &ap, &ao, &ar).unwrap();
    let count = 4;
    let mut serial_streams = vec![stream.clone(); count];
    let mut parallel_streams = vec![stream.clone(); count];
    let mut serial_carries = Vec::new();
    let mut parallel_carries = Vec::new();
    for _ in 0..count {
        serial_carries.push(carry());
        parallel_carries.push(carry());
    }
    let b = m.base(0);
    let o = b + if m.geometry.conv_taps > 0 { 3 } else { 0 };
    let att_norm = m.flat(o + 8).unwrap();
    let att_gate = m.flat(o + 9).unwrap();
    let gate = crate::math::sigmoid(*att_gate.first().unwrap());
    let mlp_norm = m.flat(o + 10).unwrap();
    let tail = LayerTail {
        att_norm: &att_norm,
        gate,
        mlp_norm: &mlp_norm,
        o,
        width: d,
    };
    let projected: Vec<f32> = (0..count * d)
        .map(|i| (i % 7) as f32 * 0.01 - 0.02)
        .collect();
    let cancel = CancellationToken::new();
    m.finish_prefill_range(
        &mut serial_streams,
        &mut serial_carries,
        &projected,
        &tail,
        &cancel,
    )
    .unwrap();
    m.finish_prefill_tokens(
        &mut parallel_streams,
        &mut parallel_carries,
        &projected,
        &tail,
        &cancel,
    )
    .unwrap();
    assert_eq!(serial_streams, parallel_streams);
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

fn load_fixture(limits: InferenceLimits) -> Result<Model> {
    let archive = Archive::from_bytes(crate::test_support::fixture(), &limits)?;
    Model::load(
        archive,
        EmbeddedRuntime::new(DevicePreference::Cpu, limits)?,
    )
}

fn resident_weight_bytes(archive: &Archive) -> u64 {
    let geometry = archive.geometry();
    let stride = if geometry.conv_taps > 0 { 27 } else { 24 };
    let engrams = 1 + stride * geometry.layers + 11;
    let gather: Vec<usize> = (0..geometry.engram_sites.len())
        .map(|site| engrams + 4 * site)
        .collect();
    archive.records[..archive.records.len() - 1]
        .iter()
        .enumerate()
        .map(|(index, record)| {
            if record.dtype == 3 && gather.contains(&index) {
                record.bytes as u64
            } else {
                record
                    .shape
                    .iter()
                    .try_fold(1u64, |product, &dimension| {
                        product.checked_mul(dimension as u64)
                    })
                    .unwrap()
                    .checked_mul(4)
                    .unwrap()
            }
        })
        .sum()
}

fn largest_tensor_elements(archive: &Archive) -> usize {
    archive
        .records
        .iter()
        .filter(|record| record.dtype != 4)
        .map(|record| record.shape.iter().product::<usize>())
        .max()
        .unwrap()
}

/// Same products `check_request` adds. The comparisons below fail when a term
/// or the `>` boundary moves.
fn reserved_state_bytes(
    model: &Model,
    token_count: usize,
    additional: usize,
    collect_cells: bool,
) -> u64 {
    let geometry = model.geometry();
    let total = token_count + additional;
    let vector = std::mem::size_of::<Vec<f32>>();
    let mut bytes = 0u64;
    add_factors(&mut bytes, &[2, total, 4]);
    add_factors(&mut bytes, &[2, token_count, 4]);
    add_factors(
        &mut bytes,
        &[
            2,
            geometry.layers + 4,
            std::mem::size_of::<super::LayerState>(),
        ],
    );
    for layer in 0..geometry.layers {
        let conv = total.min(geometry.conv_taps.max(1));
        let window = model.attention_window(layer, total);
        for (count, width) in [
            (conv, geometry.heads * geometry.qk_dim),
            (conv, geometry.kv_heads * geometry.qk_dim),
            (conv, geometry.kv_heads * geometry.v_dim),
            (window, geometry.kv_heads * geometry.qk_dim),
            (window, geometry.kv_heads * geometry.v_dim),
        ] {
            add_factors(&mut bytes, &[2, count, width, 4]);
            add_factors(&mut bytes, &[2, count + 4, vector]);
        }
    }
    if !geometry.engram_sites.is_empty() {
        let history = total.min(1 + (geometry.engram_taps - 1) * geometry.engram_dilation);
        add_factors(
            &mut bytes,
            &[2, geometry.engram_sites.len(), history, geometry.width, 4],
        );
        add_factors(
            &mut bytes,
            &[2, geometry.engram_sites.len(), history + 4, vector],
        );
    }
    if collect_cells {
        add_factors(
            &mut bytes,
            &[2, total, geometry.layers + 1, geometry.width, 4],
        );
        add_factors(&mut bytes, &[2, total + 4, vector]);
        let mut largest = 0u64;
        for &(_, base) in &model.heads {
            let rows = geometry.layers + 1;
            let k = model.weights[base + 1].dims()[1];
            let q = model.weights[base + 2].dims()[0];
            let mut head = 0u64;
            add_factors(&mut head, &[2, rows, k, geometry.width, 4]);
            add_factors(&mut head, &[2, rows * k + 4, vector]);
            add_factors(&mut head, &[2, rows, k, 4]);
            add_factors(&mut head, &[2, q, rows, k, 4]);
            add_factors(&mut head, &[2, q + 4, geometry.width, 4]);
            add_factors(&mut head, &[4, model.weights[base + 4].dims()[0], 4]);
            add_factors(&mut head, &[2, total + rows * k + 8, 4]);
            largest = largest.max(head);
        }
        bytes = bytes.checked_add(largest).unwrap();
    }
    for elements in [
        geometry.width * (8 * geometry.lanes + 32),
        32 * geometry.hadamard,
        8 * geometry.lanes * geometry.lanes,
        8 * geometry.heads * (geometry.qk_dim + geometry.v_dim),
        8 * geometry.kv_heads * (geometry.qk_dim + geometry.v_dim),
        4 * geometry.out_vocab,
        4 * geometry.engram_tables * geometry.engram_dim,
        3 * geometry.layers,
        geometry.conv_taps
            * ((geometry.heads + geometry.kv_heads) * geometry.qk_dim
                + geometry.kv_heads * geometry.v_dim),
        geometry.engram_taps * geometry.width,
    ] {
        add_factors(&mut bytes, &[elements, 4]);
    }
    add_factors(
        &mut bytes,
        &[16, additional, model.tokenizer.max_token_bytes()],
    );
    bytes
}

fn add_factors(bytes: &mut u64, factors: &[usize]) {
    super::add_bytes(bytes, factors).unwrap();
}

fn must_err<T>(result: Result<T>) -> Error {
    match result {
        Ok(_) => panic!("expected the budget to reject"),
        Err(error) => error,
    }
}

struct RemoveDir(std::path::PathBuf);
impl Drop for RemoveDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn power_budgets_accept_the_exact_limit_and_reject_one_past() {
    let cancel = CancellationToken::new();
    let bytes = crate::test_support::fixture();
    let defaults = InferenceLimits::default();
    let archive = Archive::from_bytes(bytes.clone(), &defaults).unwrap();
    assert!(archive.records.len() > 1);

    let file_limit = InferenceLimits {
        max_model_bytes: bytes.len() as u64,
        ..InferenceLimits::default()
    };
    let parsed = Archive::from_bytes(bytes.clone(), &file_limit).unwrap();
    let directory = std::env::temp_dir().join(format!("akis-pwr04-{}", std::process::id()));
    let _remove = RemoveDir(directory.clone());
    std::fs::create_dir_all(&directory).unwrap();
    let path = directory.join("fixture.cact");
    std::fs::write(&path, &bytes).unwrap();
    let opened = Archive::open(&path, &file_limit).unwrap();
    assert_eq!(opened.sha256(), parsed.sha256());
    let short_file = InferenceLimits {
        max_model_bytes: bytes.len() as u64 - 1,
        ..InferenceLimits::default()
    };
    let open_error = must_err(Archive::open(&path, &short_file));
    assert!(
        open_error.to_string().contains("model byte limit exceeded"),
        "{open_error}"
    );
    let parse_error = must_err(Archive::from_bytes(bytes.clone(), &short_file));
    assert!(
        parse_error.to_string().contains("invalid model length"),
        "{parse_error}"
    );

    let resident = resident_weight_bytes(&archive);
    assert!(resident > 1 && resident < defaults.max_resident_weight_bytes);
    load_fixture(InferenceLimits {
        max_resident_weight_bytes: resident,
        ..InferenceLimits::default()
    })
    .unwrap();
    let resident_error = must_err(load_fixture(InferenceLimits {
        max_resident_weight_bytes: resident - 1,
        ..InferenceLimits::default()
    }));
    assert!(
        resident_error
            .to_string()
            .contains("resident weights exceed"),
        "{resident_error}"
    );

    let largest = largest_tensor_elements(&archive);
    assert!(largest > 1 && largest < defaults.max_tensor_elements);
    load_fixture(InferenceLimits {
        max_tensor_elements: largest,
        ..InferenceLimits::default()
    })
    .unwrap();
    let tensor_error = must_err(load_fixture(InferenceLimits {
        max_tensor_elements: largest - 1,
        ..InferenceLimits::default()
    }));
    assert!(
        tensor_error.to_string().contains("element limit"),
        "{tensor_error}"
    );

    let probe = model(InferenceLimits::default());
    let vocab = probe.geometry().vocab;
    assert!(vocab > 1);
    let edge = probe.logits(&[(vocab - 1) as u32], &cancel).unwrap();
    assert_eq!(edge.len(), probe.geometry().out_vocab);
    assert!(edge.iter().all(|value| value.is_finite()));
    let id_error = probe.logits(&[vocab as u32], &cancel).unwrap_err();
    assert!(
        id_error
            .to_string()
            .contains("token or context limit exceeded"),
        "{id_error}"
    );

    let context_cap = 4;
    assert!(probe.geometry().context > context_cap + 1);
    let context_limited = model(InferenceLimits {
        max_context_tokens: context_cap,
        ..InferenceLimits::default()
    });
    let inside = context_limited
        .logits(&vec![2; context_cap], &cancel)
        .unwrap();
    assert_eq!(inside.len(), context_limited.geometry().out_vocab);
    assert!(inside.iter().all(|value| value.is_finite()));
    let context_error = context_limited
        .logits(&vec![2; context_cap + 1], &cancel)
        .unwrap_err();
    assert!(
        context_error
            .to_string()
            .contains("token or context limit exceeded"),
        "{context_error}"
    );
    assert_eq!(context_limited.runtime.admission_snapshot().active, 0);

    let generation_cap = 4;
    let generation_limited = model(InferenceLimits {
        max_generated_tokens: generation_cap,
        ..InferenceLimits::default()
    });
    let generated = generation_limited
        .generate("hi", generation_cap, &cancel)
        .unwrap();
    assert!(!generated.token_ids.is_empty());
    assert!(generated.token_ids.len() <= generation_cap);
    let generation_error = generation_limited
        .generate("hi", generation_cap + 1, &cancel)
        .unwrap_err();
    assert!(
        generation_error
            .to_string()
            .contains("token or context limit exceeded"),
        "{generation_error}"
    );
    assert_eq!(generation_limited.runtime.admission_snapshot().active, 0);

    let prompt = "hi";
    let longer = "hih";
    assert_eq!(longer.len(), prompt.len() + 1);
    let input_limited = model(InferenceLimits {
        max_input_bytes: prompt.len(),
        ..InferenceLimits::default()
    });
    assert!(input_limited.generate(prompt, 1, &cancel).is_ok());
    let prompt_error = input_limited.generate(longer, 1, &cancel).unwrap_err();
    assert!(
        prompt_error
            .to_string()
            .contains("invalid prompt or generation bound"),
        "{prompt_error}"
    );
    let embed_error = input_limited.embed(longer, &cancel).unwrap_err();
    assert!(
        embed_error
            .to_string()
            .contains("input byte limit exceeded"),
        "{embed_error}"
    );
    let embed_edge = input_limited.embed(prompt, &cancel).unwrap_err();
    assert!(
        embed_edge
            .to_string()
            .contains("no embedding or confidence probe"),
        "{embed_edge}"
    );
    assert_eq!(input_limited.runtime.admission_snapshot().active, 0);

    let tokens = vec![2u32; 4];
    let exact = reserved_state_bytes(&probe, tokens.len(), 0, false);
    let with_cells = reserved_state_bytes(&probe, tokens.len(), 0, true);
    assert!(with_cells > exact && exact < defaults.max_state_bytes);
    let admitted = load_fixture(InferenceLimits {
        max_state_bytes: exact,
        ..InferenceLimits::default()
    })
    .unwrap();
    let logits = admitted.logits(&tokens, &cancel).unwrap();
    assert_eq!(logits.len(), admitted.geometry().out_vocab);
    assert!(logits.iter().all(|value| value.is_finite()));
    assert_eq!(admitted.runtime.admission_snapshot().active, 0);
    let state_error = load_fixture(InferenceLimits {
        max_state_bytes: exact - 1,
        ..InferenceLimits::default()
    })
    .unwrap()
    .logits(&tokens, &cancel)
    .unwrap_err();
    assert!(
        state_error.to_string().contains("Needle reference state"),
        "{state_error}"
    );

    let cells = load_fixture(InferenceLimits {
        max_state_bytes: with_cells,
        ..InferenceLimits::default()
    })
    .unwrap();
    cells.check_request(&tokens, 0, true, 0).unwrap();
    let cell_error = load_fixture(InferenceLimits {
        max_state_bytes: with_cells - 1,
        ..InferenceLimits::default()
    })
    .unwrap()
    .check_request(&tokens, 0, true, 0)
    .unwrap_err();
    assert!(
        cell_error.to_string().contains("Needle reference state"),
        "{cell_error}"
    );
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

fn assert_admission(runtime: &EmbeddedRuntime, active: usize, limit: usize) {
    let snapshot = runtime.admission_snapshot();
    assert_eq!(snapshot.active, active);
    assert_eq!(snapshot.waiting, 0);
    assert_eq!(snapshot.active_limit, Some(limit));
}

#[test]
fn one_extra_request_is_rejected_while_a_permit_is_held() {
    let cancel = CancellationToken::new();
    let tokens = [2u32, 5, 4];
    let m = model(InferenceLimits::default());
    let stopped = CancellationToken::new();
    stopped.cancel();
    assert_cancelled(m.logits(&[2], &stopped).unwrap_err());
    assert_admission(&m.runtime, 0, 1);

    let first = m.logits(&tokens, &cancel).unwrap();
    assert_admission(&m.runtime, 0, 1);
    let held = m.runtime.begin(&cancel).unwrap();
    assert_admission(&m.runtime, 1, 1);
    let blocked = m.logits(&tokens, &cancel).unwrap_err();
    assert!(
        blocked.to_string().contains("already has 1 active"),
        "{blocked}"
    );
    assert_admission(&m.runtime, 1, 1);
    drop(held);
    assert_admission(&m.runtime, 0, 1);
    assert!(m.logits(&[], &cancel).is_err());
    assert_admission(&m.runtime, 0, 1);
    assert_eq!(m.logits(&tokens, &cancel).unwrap(), first);

    let paired = model(InferenceLimits {
        max_concurrent_requests: 2,
        ..InferenceLimits::default()
    });
    let first_permit = paired.runtime.begin(&cancel).unwrap();
    let second_permit = paired.runtime.begin(&cancel).unwrap();
    assert_admission(&paired.runtime, 2, 2);
    let third = paired.logits(&tokens, &cancel).unwrap_err();
    assert!(
        third.to_string().contains("already has 2 active"),
        "{third}"
    );
    assert_admission(&paired.runtime, 2, 2);
    drop(first_permit);
    assert_admission(&paired.runtime, 1, 2);
    assert_eq!(paired.logits(&tokens, &cancel).unwrap(), first);
    assert_admission(&paired.runtime, 1, 2);
    drop(second_permit);
    assert_admission(&paired.runtime, 0, 2);

    let mut bytes = crate::test_support::fixture();
    bytes[13 * 4..14 * 4].copy_from_slice(&4096u32.to_le_bytes());
    let limits = InferenceLimits::default();
    let running = Model::load(
        Archive::from_bytes(bytes, &limits).unwrap(),
        EmbeddedRuntime::new(DevicePreference::Cpu, limits).unwrap(),
    )
    .unwrap();
    let cancel_running = CancellationToken::new();
    std::thread::scope(|scope| {
        let worker = scope.spawn(|| running.logits(&[4; 4096], &cancel_running));
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while running.runtime.admission_snapshot().active == 0 {
            assert!(
                !worker.is_finished(),
                "forward ended before the second request"
            );
            assert!(
                std::time::Instant::now() < deadline,
                "request never admitted"
            );
            std::thread::yield_now();
        }
        let overlapped = running.logits(&[2], &CancellationToken::new()).unwrap_err();
        assert!(
            overlapped.to_string().contains("already has 1 active"),
            "{overlapped}"
        );
        assert_admission(&running.runtime, 1, 1);
        cancel_running.cancel();
        assert_cancelled(worker.join().unwrap().unwrap_err());
    });
    assert_admission(&running.runtime, 0, 1);
    assert!(running.logits(&[4], &CancellationToken::new()).is_ok());
}
