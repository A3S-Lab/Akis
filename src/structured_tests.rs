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

#[test]
fn every_supported_scalar_has_only_valid_complete_json() {
    let schemas = [
        (
            json!({"type":"string"}),
            vec![json!(""), json!("你🦀\n\"\\")],
        ),
        (
            json!({"type":"integer"}),
            vec![json!(-1), json!(0), json!(u64::MAX)],
        ),
        (
            json!({"type":"number"}),
            vec![json!(-1.25), json!(1e30), json!(0)],
        ),
        (json!({"type":"boolean"}), vec![json!(true), json!(false)]),
        (json!({"type":"null"}), vec![Value::Null]),
    ];
    let invalid = [
        "",
        "\"",
        "truee",
        "nul",
        "01",
        "--1",
        "1e",
        "NaN",
        "Infinity",
        "\"\\uD800\"",
        "\"\n\"",
        "1.2.3",
    ];
    for (definition, valid) in schemas {
        let schema = Schema::parse(&definition, 0).unwrap();
        for value in valid {
            let bytes = serde_json::to_vec(&value).unwrap();
            for n in 0..=bytes.len() {
                assert!(
                    schema.accepts(&bytes[..n], false),
                    "{definition}: {bytes:?} prefix {n}"
                );
            }
            assert!(schema.accepts(&bytes, true));
        }
        for text in invalid {
            assert!(
                !schema.accepts(text.as_bytes(), true),
                "{definition}: {text:?}"
            );
        }
    }
}

#[test]
fn structure_boundaries_and_optional_fields() {
    let object = Schema::parse(
        &json!({"type":"object","properties":{"a":{"type":"null"}}}),
        0,
    )
    .unwrap();
    for valid in [b"{}".as_slice(), b"{\"a\":null}", b" \n{ \"a\" : null }\t"] {
        assert!(object.accepts(valid, true));
    }
    for invalid in [
        b"{\"a\":null,".as_slice(),
        b"{\"z\":null}",
        b"{\"a\":null,\"a\":null}",
    ] {
        assert!(!object.accepts(invalid, false));
    }
    let array = Schema::parse(&json!({"type":"array","items":{"type":"integer"}}), 0).unwrap();
    assert!(array.accepts(&serde_json::to_vec(&vec![1; 64]).unwrap(), true));
    assert!(!array.accepts(&serde_json::to_vec(&vec![1; 65]).unwrap(), true));
    for count in [64, 65] {
        let properties: serde_json::Map<String, Value> = (0..count)
            .map(|i| (format!("f{i}"), json!({"type":"null"})))
            .collect();
        assert_eq!(
            Schema::parse(&json!({"type":"object","properties":properties}), 0).is_ok(),
            count == 64
        );
    }
    for count in [255, 256] {
        assert_eq!(
            Schema::parse(&json!({"enum":(0..count).collect::<Vec<_>>()}), 0).is_ok(),
            count == 255
        );
    }
    let mut nested = json!({"type":"null"});
    for depth in 1..=17 {
        nested = json!({"type":"array","items":nested});
        assert_eq!(Schema::parse(&nested, 0).is_ok(), depth <= 16);
    }
}

#[test]
fn bounded_mutation_campaign_never_accepts_invalid_json() {
    let schema = Schema::parse(&json!({"type":"object","properties":{"value":{"type":"string"},"n":{"type":"number"}},"required":["value","n"]}),0).unwrap();
    let seed = br#"{"value":"hello","n":-1.25e2}"#;
    let mut random = 0x61a9d394u32;
    for _ in 0..10000 {
        let mut bytes = seed.to_vec();
        random = random.wrapping_mul(1664525).wrapping_add(1013904223);
        let at = random as usize % bytes.len();
        random = random.wrapping_mul(1664525).wrapping_add(1013904223);
        bytes[at] = (random >> 24) as u8;
        if schema.accepts(&bytes, true) {
            let value: Value = serde_json::from_slice(&bytes).unwrap();
            let fields = value.as_object().unwrap();
            assert_eq!(fields.len(), 2);
            assert!(fields["value"].is_string());
            assert!(fields["n"].is_number());
        }
    }
}
