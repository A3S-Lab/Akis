#![cfg(test)]
use super::*;

// Deliberately retain the upstream exhaustive algorithm as the heap's oracle.
fn exhaustive(tok: &Tokenizer, text: &str) -> Vec<u32> {
    let mut symbols: Vec<String> = text.chars().map(|c| c.to_string()).collect();
    while symbols.len() > 1 {
        let mut best = None;
        for i in 0..symbols.len() - 1 {
            if let Some(&id) = tok.lookup.get(&(symbols[i].clone() + &symbols[i + 1])) {
                let score = tok.scores[id as usize];
                if best.is_none_or(|(_, s)| score > s) {
                    best = Some((i, score));
                }
            }
        }
        let Some((i, _)) = best else {
            break;
        };
        let next = symbols.remove(i + 1);
        symbols[i].push_str(&next);
    }
    symbols
        .into_iter()
        .map(|s| tok.lookup.get(&s).copied().unwrap_or(tok.unk))
        .collect()
}

#[test]
fn heap_matches_exhaustive_merges_for_ties_and_stale_neighbors() {
    let extras = ["ih", "hh", "ii", "hih", "ihi", "hhi", "hiih"];
    let mut bytes = crate::test_support::tokenizer_blob();
    bytes[..4].copy_from_slice(&(10 + extras.len() as u32).to_le_bytes());
    for piece in extras {
        bytes.extend(1f32.to_le_bytes());
        bytes.push(0);
        bytes.extend((piece.len() as u16).to_le_bytes());
        bytes.extend(piece.as_bytes());
    }
    let mut tok = Tokenizer::from_bytes(&bytes, 17).unwrap();
    for scores in [
        vec![1.0; 17],
        (0..17).map(|i| (i % 3) as f32).collect(),
        (0..17)
            .map(|i| if i % 2 == 0 { 0.0 } else { -0.0 })
            .collect(),
    ] {
        tok.scores = scores;
        for n in 0..1024 {
            let text: String = (0..10)
                .map(|bit| if n & (1 << bit) == 0 { 'h' } else { 'i' })
                .collect();
            let mut actual = Vec::new();
            tok.bpe(&text, &mut actual);
            assert_eq!(actual, exhaustive(&tok, &text), "{text}");
        }
    }
}

#[test]
fn bounded_encoder_rejects_bytes_and_memory_without_partial_output() {
    let tok = Tokenizer::from_bytes(&crate::test_support::tokenizer_blob(), 10).unwrap();
    let cancel = CancellationToken::new();
    let limits = InferenceLimits {
        max_input_bytes: 2,
        ..InferenceLimits::default()
    };
    assert!(tok.encode_bounded("hih", &limits, &cancel).is_err());
    let limits = InferenceLimits {
        max_state_bytes: 1,
        ..InferenceLimits::default()
    };
    assert!(tok.encode_bounded("hi", &limits, &cancel).is_err());
    assert_eq!(
        tok.encode_bounded("hi 你<tool_call>", &InferenceLimits::default(), &cancel)
            .unwrap(),
        tok.encode("hi 你<tool_call>")
    );
    cancel.cancel();
    assert!(matches!(
        tok.encode_bounded("hi", &limits, &cancel),
        Err(Error::Power(
            a3s_power::error::PowerError::InferenceCancelled
        ))
    ));
}

#[test]
fn cancellation_interrupts_symbol_scan_heap_merge_and_output() {
    let tok = Tokenizer::from_bytes(&crate::test_support::tokenizer_blob(), 10).unwrap();
    let text = "hihi".repeat(1024);
    let mut checks = 0usize;
    tok.encode_controlled(&text, &mut || {
        checks += 1;
        Ok::<_, Infallible>(())
    })
    .unwrap();
    assert!(checks > 40);
    for stop in 1..=checks {
        let mut calls = 0usize;
        let result = tok.encode_controlled(&text, &mut || {
            calls += 1;
            if calls == stop {
                Err(())
            } else {
                Ok(())
            }
        });
        assert!(result.is_err(), "cancellation checkpoint {stop}");
        assert_eq!(calls, stop);
    }
}
