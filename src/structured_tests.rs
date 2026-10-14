#![cfg(test)]
use super::*;
use serde_json::{json, Value};
use std::collections::BTreeMap;
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

#[test]
fn tool_metadata_is_bounded_before_schema_cloning() {
    let limits = crate::InferenceLimits {
        max_input_bytes: 256,
        ..crate::InferenceLimits::default()
    };
    let model = Model::load(
        crate::Archive::from_bytes(crate::test_support::fixture(), &limits).unwrap(),
        crate::EmbeddedRuntime::new(crate::DevicePreference::Cpu, limits).unwrap(),
    )
    .unwrap();
    let tool = Tool {
        name: "example".into(),
        description: "x".repeat(1024),
        parameters: json!({"type":"object","properties":{}}),
    };
    let error = model
        .complete("hi", &[tool], 1, &CancellationToken::new())
        .unwrap_err();
    assert!(
        error.to_string().contains("tool request byte limit"),
        "{error}"
    );
    assert_eq!(model.runtime().admission_snapshot().active, 0);
}

#[test]
fn tool_request_accepts_the_exact_byte_limit() {
    let input = "hi";
    let tool = Tool {
        name: "example".into(),
        description: "d".into(),
        parameters: json!({"type":"object","properties":{}}),
    };
    let tools = serde_json::to_vec(std::slice::from_ref(&tool)).unwrap();
    let overhead = "<|im_start|>user\n<tools></tools>\n<|im_end|>\n<|im_start|>assistant\n".len();
    let exact = input.len() + overhead + tools.len();
    assert!(exact > input.len());
    // The stock fixture context is shorter than this chat wrapper. Widen only
    // that field so the exact byte budget reaches generation. One token does
    // not finish a tool call on this fixture.
    let mut fixture = crate::test_support::fixture();
    fixture[13 * 4..14 * 4].copy_from_slice(&512u32.to_le_bytes());
    let limits = crate::InferenceLimits {
        max_input_bytes: exact,
        ..crate::InferenceLimits::default()
    };
    let model = Model::load(
        crate::Archive::from_bytes(fixture.clone(), &limits).unwrap(),
        crate::EmbeddedRuntime::new(crate::DevicePreference::Cpu, limits).unwrap(),
    )
    .unwrap();
    let cancel = CancellationToken::new();
    let admitted = model
        .complete(input, std::slice::from_ref(&tool), 1, &cancel)
        .unwrap_err();
    assert!(
        admitted
            .to_string()
            .contains("generation exhausted before a complete tool call"),
        "{admitted}"
    );
    assert_eq!(model.runtime().admission_snapshot().active, 0);
    let tight = crate::InferenceLimits {
        max_input_bytes: exact - 1,
        ..crate::InferenceLimits::default()
    };
    let rejected = Model::load(
        crate::Archive::from_bytes(fixture, &tight).unwrap(),
        crate::EmbeddedRuntime::new(crate::DevicePreference::Cpu, tight).unwrap(),
    )
    .unwrap();
    let error = rejected.complete(input, &[tool], 1, &cancel).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("tool request byte limit exceeded"),
        "{error}"
    );
    assert_eq!(rejected.runtime().admission_snapshot().active, 0);
}

fn lights(room: &str, on: bool) -> ToolCall {
    ToolCall {
        name: "set_lights".into(),
        arguments: BTreeMap::from([
            ("on".into(), Value::Bool(on)),
            ("room".into(), Value::String(room.into())),
        ]),
    }
}

#[test]
fn string_arguments_must_occur_in_the_user_input() {
    let rows = [
        ("turn on the kitchen lights", Some(("kitchen", true)), false),
        ("turn off the bedroom lights", Some(("bedroom", false)), false),
        ("打开厨房的灯。", None, false),
        ("关闭客厅的灯。", None, false),
        ("What is the capital of France?", None, false),
        ("Do not turn on the kitchen lights.", Some(("kitchen", false)), false),
        ("Turn on the lights.", Some(("living room", true)), true),
        (
            "Translate the sentence 'turn on the kitchen lights' into French. Do not operate any lights.",
            None,
            false,
        ),
    ];
    for (input, call, suppress) in rows {
        let calls = call
            .map(|(room, on)| vec![lights(room, on)])
            .unwrap_or_default();
        let (kept, suppressed, ungrounded) = ground_string_arguments(input, calls);
        if suppress {
            assert!(kept.is_empty(), "{input}");
            assert_eq!(suppressed.len(), 1, "{input}");
            assert_eq!(ungrounded, vec!["set_lights.room".to_string()], "{input}");
        } else {
            assert!(suppressed.is_empty(), "{input}");
            assert!(ungrounded.is_empty(), "{input}");
            assert_eq!(kept.len(), if call.is_some() { 1 } else { 0 }, "{input}");
        }
    }
    let (kept, suppressed, ungrounded) = ground_string_arguments(
        "Turn on the living room lights.",
        vec![lights("living room", true)],
    );
    assert_eq!(kept.len(), 1);
    assert!(suppressed.is_empty());
    assert!(ungrounded.is_empty());
    let (kept, suppressed, ungrounded) =
        ground_string_arguments("turn on the kitchen lights", vec![lights("Kitchen", true)]);
    assert!(kept.is_empty());
    assert_eq!(ungrounded, vec!["set_lights.room".to_string()]);
    assert_eq!(suppressed.len(), 1);
    let (kept, _, ungrounded) =
        ground_string_arguments("turn on the kitchen lights", vec![lights("", true)]);
    assert!(kept.is_empty());
    assert_eq!(ungrounded, vec!["set_lights.room".to_string()]);
}

fn object_tool(name: &str) -> Tool {
    Tool {
        name: name.to_string(),
        description: "d".to_string(),
        parameters: json!({"type":"object","properties":{}}),
    }
}

#[test]
fn tool_names_and_counts_stop_before_a_call() {
    let model = Model::load(
        crate::Archive::from_bytes(
            crate::test_support::fixture(),
            &crate::InferenceLimits::default(),
        )
        .unwrap(),
        crate::EmbeddedRuntime::new(
            crate::DevicePreference::Cpu,
            crate::InferenceLimits::default(),
        )
        .unwrap(),
    )
    .unwrap();
    let cancel = CancellationToken::new();
    let context = "token or context limit exceeded";
    let count = "tool count must be 1..64";
    let name = "invalid or duplicate tool name";

    let none = model.complete("hi", &[], 1, &cancel).unwrap_err();
    assert!(none.to_string().contains(count), "{none}");
    assert!(!none.to_string().contains("embedded-execution-receipt"));
    assert_eq!(model.runtime().admission_snapshot().active, 0);

    let mut many: Vec<Tool> = (0..65).map(|i| object_tool(&format!("t{i}"))).collect();
    let extra = model.complete("hi", &many, 1, &cancel).unwrap_err();
    assert!(extra.to_string().contains(count), "{extra}");
    assert_eq!(model.runtime().admission_snapshot().active, 0);
    many.pop();
    assert_eq!(many.len(), 64);
    let full = model.complete("hi", &many, 1, &cancel).unwrap_err();
    assert!(
        !full.to_string().contains(count) && full.to_string().contains(context),
        "{full}"
    );
    assert_eq!(model.runtime().admission_snapshot().active, 0);

    let rejected = [
        String::new(),
        "a".repeat(129),
        "a\nb".to_string(),
        "\u{0001}tool".to_string(),
    ];
    for bad in &rejected {
        let error = model
            .complete("hi", &[object_tool(bad)], 1, &cancel)
            .unwrap_err();
        assert!(error.to_string().contains(name), "{bad:?} {error}");
        assert!(!error.to_string().contains("embedded-execution-receipt"));
        assert_eq!(model.runtime().admission_snapshot().active, 0);
    }
    let duplicate = [object_tool("example"), object_tool("example")];
    let duplicate_error = model.complete("hi", &duplicate, 1, &cancel).unwrap_err();
    assert!(
        duplicate_error.to_string().contains(name),
        "{duplicate_error}"
    );
    assert_eq!(model.runtime().admission_snapshot().active, 0);

    for accepted in ["a".repeat(128), "my tool".to_string()] {
        let error = model
            .complete("hi", &[object_tool(&accepted)], 1, &cancel)
            .unwrap_err();
        assert!(
            !error.to_string().contains(name) && error.to_string().contains(context),
            "{accepted:?} {error}"
        );
        assert_eq!(model.runtime().admission_snapshot().active, 0);
    }
    let distinct = [object_tool("alpha"), object_tool("beta")];
    let distinct_error = model.complete("hi", &distinct, 1, &cancel).unwrap_err();
    assert!(
        !distinct_error.to_string().contains(name) && distinct_error.to_string().contains(context),
        "{distinct_error}"
    );
    assert_eq!(model.runtime().admission_snapshot().active, 0);

    let scalar = Tool {
        name: "example".to_string(),
        description: "d".to_string(),
        parameters: json!({"type":"string"}),
    };
    let shape = model.complete("hi", &[scalar], 1, &cancel).unwrap_err();
    assert!(
        shape
            .to_string()
            .contains("tool parameters must be an object"),
        "{shape}"
    );
    assert!(!shape.to_string().contains("embedded-execution-receipt"));
    assert_eq!(model.runtime().admission_snapshot().active, 0);
}
