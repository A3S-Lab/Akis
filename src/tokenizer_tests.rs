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

#[test]
fn tokenizer_bytes_accept_the_exact_limit() {
    let tok = Tokenizer::from_bytes(&crate::test_support::tokenizer_blob(), 10).unwrap();
    let cancel = CancellationToken::new();
    let text = "hi";
    assert_eq!(text.len(), 2);
    let input_ok = InferenceLimits {
        max_input_bytes: text.len(),
        ..InferenceLimits::default()
    };
    assert_eq!(
        tok.encode_bounded(text, &input_ok, &cancel).unwrap(),
        tok.encode(text)
    );
    let input_over = InferenceLimits {
        max_input_bytes: text.len() - 1,
        ..InferenceLimits::default()
    };
    let input_error = tok.encode_bounded(text, &input_over, &cancel).unwrap_err();
    assert!(
        input_error
            .to_string()
            .contains("input byte limit exceeded"),
        "{input_error}"
    );

    let symbols = text.len() as u64 + 8;
    let per_byte =
        2 * std::mem::size_of::<super::Symbol>() + 6 * std::mem::size_of::<super::Merge>() + 40;
    let bytes = symbols * per_byte as u64;
    assert!(bytes > 1 && bytes < InferenceLimits::default().max_state_bytes);
    let state_ok = InferenceLimits {
        max_state_bytes: bytes,
        ..InferenceLimits::default()
    };
    assert_eq!(
        tok.encode_bounded(text, &state_ok, &cancel).unwrap(),
        tok.encode(text)
    );
    let state_over = InferenceLimits {
        max_state_bytes: bytes - 1,
        ..InferenceLimits::default()
    };
    let state_error = tok.encode_bounded(text, &state_over, &cancel).unwrap_err();
    assert!(
        state_error.to_string().contains("tokenizer state"),
        "{state_error}"
    );
}

fn byte_piece(byte: u8) -> String {
    if byte == 0x0a {
        "<0x0a>".to_string()
    } else {
        format!("<0x{byte:02X}>")
    }
}

fn control_pieces(short_marker: bool) -> Vec<(String, u8)> {
    let mut pieces = vec![
        ("<pad>".to_string(), 2),
        ("</s>".to_string(), 2),
        ("<s>".to_string(), 2),
        ("<unk>".to_string(), 1),
        ("▁".to_string(), 0),
        ("你".to_string(), 0),
        ("<tool_call>".to_string(), 3),
        ("</tool_call>".to_string(), 3),
    ];
    if short_marker {
        pieces.push(("<tool_".to_string(), 3));
    }
    pieces
}

fn add_byte_pieces(pieces: &mut Vec<(String, u8)>, count: usize) {
    for byte in 0..count {
        pieces.push((byte_piece(byte as u8), 4));
    }
}

fn piece_blob(pieces: &[(String, u8)], dummy: u8, fallback: u8) -> Vec<u8> {
    let mut bytes = Vec::new();
    for value in [pieces.len() as u32, 0, 1, 2, 3] {
        bytes.extend(value.to_le_bytes());
    }
    bytes.extend([dummy, fallback, 0, 0]);
    for (index, (piece, kind)) in pieces.iter().enumerate() {
        bytes.extend((index as f32).to_le_bytes());
        bytes.push(*kind);
        bytes.extend((piece.len() as u16).to_le_bytes());
        bytes.extend(piece.as_bytes());
    }
    bytes
}

fn load(pieces: &[(String, u8)], dummy: u8, fallback: u8) -> Result<Tokenizer> {
    Tokenizer::from_bytes(&piece_blob(pieces, dummy, fallback), pieces.len())
}

fn must_err(result: Result<Tokenizer>) -> Error {
    match result {
        Ok(_) => panic!("tokenizer load returned a value"),
        Err(error) => error,
    }
}

fn raw_bytes(tok: &Tokenizer, ids: &[u32]) -> Vec<u8> {
    let mut raw = Vec::new();
    for id in ids {
        raw.extend(tok.token_bytes(*id).unwrap());
    }
    raw
}

fn assert_roundtrip(tok: &Tokenizer, text: &str) {
    let ids = tok.encode(text);
    assert_eq!(tok.decode(&ids).unwrap(), text, "{text:?} {ids:?}");
}

#[test]
fn byte_tokens_restore_split_utf8_and_reject_unknown_ids() {
    let mut pieces = control_pieces(true);
    add_byte_pieces(&mut pieces, 256);
    let tok = load(&pieces, 0, 1).unwrap();
    let mut joined = Vec::new();
    for byte in 0..=255u8 {
        let id = tok.token_id(&byte_piece(byte)).unwrap();
        let raw = tok.token_bytes(id).unwrap();
        assert_eq!(raw, vec![byte], "{byte}");
        joined.extend(raw);
    }
    assert_eq!(joined, (0..=255).collect::<Vec<u8>>());

    for text in [
        "🦀",
        "Aé€",
        "\0",
        "\n",
        "你",
        "你🦀",
        "<tool_call></tool_call>",
        "<tool_call>🦀</tool_call>",
        "</tool_call><tool_call><tool_",
    ] {
        assert_roundtrip(&tok, text);
        assert_eq!(raw_bytes(&tok, &tok.encode(text)), text.as_bytes());
    }
    assert_eq!(tok.encode("你"), vec![tok.token_id("你").unwrap()]);
    let crab = tok.encode("🦀");
    assert_eq!(crab.len(), "🦀".len());
    assert_eq!(raw_bytes(&tok, &crab), "🦀".as_bytes());
    assert_eq!(
        tok.encode("<tool_call>"),
        vec![tok.token_id("<tool_call>").unwrap()]
    );
    assert!(!tok
        .encode("<tool_call>")
        .contains(&tok.token_id("<tool_").unwrap()));
    assert_eq!(
        tok.encode("</tool_call><tool_call><tool_"),
        vec![
            tok.token_id("</tool_call>").unwrap(),
            tok.token_id("<tool_call>").unwrap(),
            tok.token_id("<tool_").unwrap(),
        ]
    );

    let dummy = load(&pieces, 1, 1).unwrap();
    for text in ["🦀", "Aé€", "\0", "\n", "你🦀", "<tool_call>🦀</tool_call>"] {
        assert_roundtrip(&dummy, text);
    }
    let marked = dummy.encode("<tool_call></tool_call>");
    assert_eq!(
        &marked[1..],
        &[
            dummy.token_id("<tool_call>").unwrap(),
            dummy.token_id("</tool_call>").unwrap(),
        ]
    );
    assert_eq!(dummy.decode(&marked).unwrap(), "<tool_call></tool_call>");

    let plain = load(&control_pieces(false), 0, 0).unwrap();
    assert_eq!(plain.unk, plain.token_id("<unk>").unwrap());
    assert_eq!(plain.encode("🦀"), vec![plain.unk]);
    assert_eq!(plain.encode("Aé"), vec![plain.unk, plain.unk]);
    assert_eq!(plain.decode(&plain.encode("🦀")).unwrap(), "");
    let mixed = plain.encode("你🦀");
    assert_eq!(mixed, vec![plain.token_id("你").unwrap(), plain.unk]);
    assert_eq!(plain.decode(&mixed).unwrap(), "你");
    assert_eq!(
        plain.encode("<tool_call></tool_call>"),
        vec![
            plain.token_id("<tool_call>").unwrap(),
            plain.token_id("</tool_call>").unwrap(),
        ]
    );
    assert_eq!(
        plain
            .decode(&plain.encode("<tool_call></tool_call>"))
            .unwrap(),
        "<tool_call></tool_call>"
    );

    let width = tok.vocab_size() as u32;
    let outside = tok.decode(&[width]).unwrap_err();
    assert!(
        outside.to_string().contains("token id outside vocabulary"),
        "{outside}"
    );
    let middle = tok
        .decode(&[tok.token_id("你").unwrap(), width])
        .unwrap_err();
    assert!(
        middle.to_string().contains("token id outside vocabulary"),
        "{middle}"
    );
    assert_eq!(tok.token_bytes(width - 1).unwrap(), vec![0xFF]);
    assert!(tok.token_bytes(width).is_err());
    assert!(tok.token_bytes(u32::MAX).is_err());
    let lone = tok
        .decode(&[tok.token_id(&byte_piece(0xFF)).unwrap()])
        .unwrap();
    assert_ne!(lone.as_bytes(), &[0xFF]);

    let mut short = control_pieces(false);
    add_byte_pieces(&mut short, 255);
    let short_error = must_err(load(&short, 0, 1));
    assert!(
        short_error.to_string().contains("malformed SentencePiece"),
        "{short_error}"
    );
    let mut duplicate = control_pieces(false);
    add_byte_pieces(&mut duplicate, 256);
    duplicate.push(("<0x00>".to_string(), 4));
    let duplicate_error = must_err(load(&duplicate, 0, 0));
    assert!(
        duplicate_error
            .to_string()
            .contains("malformed SentencePiece"),
        "{duplicate_error}"
    );
    let mut malformed = control_pieces(false);
    malformed.push(("<0xGG>".to_string(), 4));
    let malformed_error = must_err(load(&malformed, 0, 0));
    assert!(
        malformed_error
            .to_string()
            .contains("malformed SentencePiece"),
        "{malformed_error}"
    );
    let mut truncated = control_pieces(false);
    truncated.push(("<0x0>".to_string(), 4));
    let truncated_error = must_err(load(&truncated, 0, 0));
    assert!(
        truncated_error
            .to_string()
            .contains("malformed SentencePiece"),
        "{truncated_error}"
    );
    let missing = must_err(load(&control_pieces(false), 0, 1));
    assert!(
        missing.to_string().contains("malformed SentencePiece"),
        "{missing}"
    );
}
