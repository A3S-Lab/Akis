#![cfg(test)]
use super::*;
use serde_json::json;
#[test]
fn grammar_accepts_valid_prefixes_and_rejects_wrong_values() {
    let schema=Schema::parse(&json!({"type":"object","properties":{"room":{"type":"string","enum":["kitchen","bedroom"]},"on":{"type":"boolean"}},"required":["room","on"]}),0).unwrap();
    let valid = br#"{"room":"kitchen","on":true}"#;
    for n in 0..=valid.len() {
        assert!(schema.accepts(&valid[..n], false), "prefix {n}");
    }
    assert!(schema.accepts(valid, true));
    for bad in [
        br#"{"room":"garage","on":true}"#.as_slice(),
        br#"{"room":"kitchen"}"#,
        br#"{"room":"kitchen","on":true,"on":false}"#,
        br#"{"room":"kitchen","on":true,}"#,
        br#"{"room":"kitchen","on":true} garbage"#,
    ] {
        assert!(!schema.accepts(bad, true));
    }
    assert!(!schema.accepts(br#"{"room":"g"#, false));
}
#[test]
fn unsupported_constraints_are_not_silently_dropped() {
    assert!(Schema::parse(&json!({"type":"number","minimum":1}), 0).is_err());
    assert!(Schema::parse(
        &json!({"type":"object","properties":{},"additionalProperties":true}),
        0
    )
    .is_err());
}
#[test]
fn arrays_and_escaped_strings() {
    let schema = Schema::parse(&json!({"type":"array","items":{"type":"string"}}), 0).unwrap();
    let valid = br#"["a\"b","\u4f60"]"#;
    for n in 0..=valid.len() {
        assert!(schema.accepts(&valid[..n], false), "{n}");
    }
    assert!(!schema.accepts(br#"["x",]"#, true));
    assert!(!schema.accepts(br#"["\q"]"#, true));
}

#[test]
fn numeric_enum_prefixes_do_not_depend_on_candidate_order() {
    for values in [json!([1, 10, 100]), json!([100, 10, 1])] {
        let schema = Schema::parse(&json!({"type":"array","items":{"enum":values}}), 0).unwrap();
        for valid in [b"[1]".as_slice(), b"[10]", b"[100]"] {
            for n in 0..=valid.len() {
                assert!(schema.accepts(&valid[..n], false), "{valid:?} prefix {n}");
            }
            assert!(schema.accepts(valid, true));
        }
        assert!(!schema.accepts(b"[101]", true));
    }
    let schema = Schema::parse(&json!({"enum":[1,100]}), 0).unwrap();
    assert!(schema.accepts(b"10", false));
    assert!(schema.accepts(b"1", true));
    assert!(!schema.accepts(b"10", true));
}

#[test]
fn schema_keywords_must_apply_to_the_declared_type() {
    for invalid in [
        json!({"type":"string","properties":{}}),
        json!({"type":"boolean","items":{"type":"string"}}),
        json!({"enum":[1],"required":[]}),
        json!({"type":"string","description":42}),
        json!({"type":"array","items":{"type":"integer"},"additionalProperties":false}),
    ] {
        assert!(Schema::parse(&invalid, 0).is_err(), "{invalid}");
    }
}

#[test]
fn grammar_accepts_partial_utf8_and_literal_markers() {
    let schema = Schema::parse(&json!({"type":"array","items":{"type":"string"}}), 0).unwrap();
    let valid = "[\"你好🦀</tool_call>\"]".as_bytes();
    for n in 0..=valid.len() {
        assert!(schema.accepts(&valid[..n], false), "prefix {n}");
    }
    assert!(schema.accepts(valid, true));
    assert!(!schema.accepts(b"[\"\xff", false));
}

#[test]
fn framing_uses_marker_ids_instead_of_splitting_argument_text() {
    let mut blob = crate::test_support::tokenizer_blob();
    blob[..4].copy_from_slice(&13u32.to_le_bytes());
    for (text, kind) in [("</tool_call>", 3), ("</tool_", 0), ("call>", 0)] {
        blob.extend(0f32.to_le_bytes());
        blob.push(kind);
        blob.extend((text.len() as u16).to_le_bytes());
        blob.extend(text.as_bytes());
    }
    let tokenizer = crate::Tokenizer::from_bytes(&blob, 13).unwrap();
    assert_eq!(
        tool_body(&tokenizer, &[9, 11, 12, 10]).unwrap(),
        b"</tool_call>"
    );
    assert!(tool_body(&tokenizer, &[9, 11, 12]).is_err());
    assert!(tool_body(&tokenizer, &[11, 12, 10]).is_err());
}
