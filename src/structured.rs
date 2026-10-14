use crate::{CancellationToken, Error, Generation, Model, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::io::{self, Write};

struct ToolSize<'a> {
    bytes: usize,
    limit: usize,
    cancel: &'a CancellationToken,
}
impl Write for ToolSize<'_> {
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        if self.cancel.is_cancelled() {
            return Err(io::Error::new(
                io::ErrorKind::Interrupted,
                "request cancelled",
            ));
        }
        self.bytes = self
            .bytes
            .checked_add(data.len())
            .filter(|&n| n <= self.limit)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "tool request byte limit exceeded",
                )
            })?;
        Ok(data.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Tool {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ToolCall {
    pub name: String,
    pub arguments: BTreeMap<String, Value>,
}

#[derive(Debug, Serialize)]
pub struct Completion {
    pub function_calls: Vec<ToolCall>,
    pub suppressed_calls: Vec<ToolCall>,
    /// `tool.field` paths whose string values are absent from the user input.
    pub ungrounded: Vec<String>,
    pub generation: Generation,
}

/// Keep a call only when every string argument occurs as a contiguous span of
/// `input`. Other argument types are left to the schema. An empty string is
/// absent. This check does not invent a value or fold case.
pub(crate) fn ground_string_arguments(
    input: &str,
    calls: Vec<ToolCall>,
) -> (Vec<ToolCall>, Vec<ToolCall>, Vec<String>) {
    let mut kept = Vec::new();
    let mut suppressed = Vec::new();
    let mut ungrounded = Vec::new();
    for call in calls {
        let mut missing = Vec::new();
        for (field, value) in &call.arguments {
            if let Value::String(text) = value {
                if text.is_empty() || !input.contains(text.as_str()) {
                    missing.push(format!("{}.{}", call.name, field));
                }
            }
        }
        if missing.is_empty() {
            kept.push(call);
        } else {
            ungrounded.extend(missing);
            suppressed.push(call);
        }
    }
    (kept, suppressed, ungrounded)
}

#[derive(Clone, Debug)]
pub(crate) enum Schema {
    String,
    Number(bool),
    Bool,
    Null,
    Enum(Vec<Value>),
    Array(Box<Schema>),
    Object(BTreeMap<String, Schema>, BTreeSet<String>),
    Any(Vec<Schema>),
}

impl Schema {
    fn parse(value: &Value, depth: usize) -> Result<Self> {
        let bad = || {
            Error::Request(
                "unsupported tool schema; use closed objects, arrays, scalar types and enums"
                    .into(),
            )
        };
        if depth > 16 {
            return Err(bad());
        }
        let obj = value.as_object().ok_or_else(bad)?;
        let keys = [
            "type",
            "description",
            "properties",
            "required",
            "additionalProperties",
            "items",
            "enum",
        ];
        if obj.keys().any(|k| !keys.contains(&k.as_str())) {
            return Err(bad());
        }
        let kind = obj.get("type").and_then(Value::as_str);
        if obj.get("description").is_some_and(|v| !v.is_string())
            || obj.keys().any(|k| match k.as_str() {
                "properties" | "required" | "additionalProperties" => kind != Some("object"),
                "items" => kind != Some("array"),
                _ => false,
            })
        {
            return Err(bad());
        }
        if let Some(v) = obj.get("enum") {
            let vs = v.as_array().ok_or_else(bad)?;
            if vs.is_empty() || vs.len() > 255 || vs.iter().any(|v| v.is_object() || v.is_array()) {
                return Err(bad());
            }
            if matches!(kind, Some("object" | "array")) {
                return Err(bad());
            }
            if obj.contains_key("type") {
                let without = Value::Object(
                    obj.iter()
                        .filter(|(key, _)| key.as_str() != "enum")
                        .map(|(key, value)| (key.clone(), value.clone()))
                        .collect(),
                );
                let shape = Self::parse(&without, depth + 1)?;
                for v in vs {
                    if !shape.accepts(&serde_json::to_vec(v)?, true) {
                        return Err(bad());
                    }
                }
            }
            return Ok(Self::Enum(vs.clone()));
        }
        Ok(
            match obj.get("type").and_then(Value::as_str).ok_or_else(bad)? {
                "string" => Self::String,
                "integer" => Self::Number(true),
                "number" => Self::Number(false),
                "boolean" => Self::Bool,
                "null" => Self::Null,
                "array" => Self::Array(Box::new(Self::parse(
                    obj.get("items").ok_or_else(bad)?,
                    depth + 1,
                )?)),
                "object" => {
                    if obj.get("additionalProperties").is_some_and(|v| v != false) {
                        return Err(bad());
                    }
                    let props = obj
                        .get("properties")
                        .and_then(Value::as_object)
                        .ok_or_else(bad)?;
                    if props.len() > 64 {
                        return Err(bad());
                    }
                    let mut fields = BTreeMap::new();
                    for (key, value) in props {
                        fields.insert(key.clone(), Self::parse(value, depth + 1)?);
                    }
                    let mut required = BTreeSet::new();
                    if let Some(v) = obj.get("required") {
                        for v in v.as_array().ok_or_else(bad)? {
                            let k = v.as_str().ok_or_else(bad)?;
                            if !fields.contains_key(k) || !required.insert(k.to_string()) {
                                return Err(bad());
                            }
                        }
                    }
                    Self::Object(fields, required)
                }
                _ => return Err(bad()),
            },
        )
    }

    pub(crate) fn accepts(&self, bytes: &[u8], complete: bool) -> bool {
        match self.scan(bytes, 0) {
            Scan::More => !complete,
            Scan::End(at) => skip(bytes, at) == bytes.len(),
            Scan::Bad => false,
        }
    }

    fn scan(&self, b: &[u8], at: usize) -> Scan {
        let at = skip(b, at);
        if at == b.len() {
            return Scan::More;
        }
        match self {
            Self::Any(schemas) => {
                let mut more = false;
                for schema in schemas {
                    match schema.scan(b, at) {
                        Scan::More => more = true,
                        Scan::End(i) => return Scan::End(i),
                        Scan::Bad => {}
                    }
                }
                if more {
                    Scan::More
                } else {
                    Scan::Bad
                }
            }
            Self::Enum(values) => {
                let mut more = false;
                let mut end = None;
                for v in values {
                    if let Ok(encoded) = serde_json::to_vec(v) {
                        let tail = &b[at..];
                        if tail.starts_with(&encoded) {
                            end = Some(end.unwrap_or(0).max(at + encoded.len()));
                        }
                        if encoded.starts_with(tail) {
                            more = true;
                        }
                    }
                }
                if let Some(end) = end.filter(|&end| !more || end == b.len()) {
                    Scan::End(end)
                } else if more {
                    Scan::More
                } else {
                    Scan::Bad
                }
            }
            Self::Bool => Self::Enum(vec![Value::Bool(true), Value::Bool(false)]).scan(b, at),
            Self::Null => Self::Enum(vec![Value::Null]).scan(b, at),
            Self::String => string_end(b, at),
            Self::Number(integer) => {
                let end = (at..b.len())
                    .find(|&i| b[i].is_ascii_whitespace() || b",]}".contains(&b[i]))
                    .unwrap_or(b.len());
                let s = &b[at..end];
                if *integer && s.iter().any(|v| !v.is_ascii_digit() && *v != b'-') {
                    return Scan::Bad;
                }
                match serde_json::from_slice::<serde_json::Number>(s) {
                    Ok(n) if !integer || n.is_i64() || n.is_u64() => Scan::End(end),
                    Err(e) if e.is_eof() && end == b.len() => Scan::More,
                    _ => Scan::Bad,
                }
            }
            Self::Array(item) => {
                if b[at] != b'[' {
                    return Scan::Bad;
                }
                let mut i = skip(b, at + 1);
                if i == b.len() {
                    return Scan::More;
                }
                if b[i] == b']' {
                    return Scan::End(i + 1);
                }
                for _ in 0..64 {
                    i = match item.scan(b, i) {
                        Scan::End(i) => skip(b, i),
                        s => return s,
                    };
                    if i == b.len() {
                        return Scan::More;
                    }
                    match b[i] {
                        b']' => return Scan::End(i + 1),
                        b',' => i = skip(b, i + 1),
                        _ => return Scan::Bad,
                    }
                }
                Scan::Bad
            }
            Self::Object(fields, required) => {
                if b[at] != b'{' {
                    return Scan::Bad;
                }
                let mut i = skip(b, at + 1);
                let mut seen = BTreeSet::new();
                let mut after_comma = false;
                loop {
                    if i == b.len() {
                        return Scan::More;
                    }
                    if b[i] == b'}' {
                        return if required.is_subset(&seen) && !after_comma {
                            Scan::End(i + 1)
                        } else {
                            Scan::Bad
                        };
                    }
                    let choices = Self::Enum(
                        fields
                            .keys()
                            .filter(|k| !seen.contains(*k))
                            .cloned()
                            .map(Value::String)
                            .collect(),
                    );
                    let end = match choices.scan(b, i) {
                        Scan::End(i) => i,
                        s => return s,
                    };
                    let key = match serde_json::from_slice::<String>(&b[i..end]) {
                        Ok(k) => k,
                        Err(_) => return Scan::Bad,
                    };
                    let Some(schema) = fields.get(&key) else {
                        return Scan::Bad;
                    };
                    seen.insert(key);
                    i = skip(b, end);
                    if i == b.len() {
                        return Scan::More;
                    }
                    if b[i] != b':' {
                        return Scan::Bad;
                    }
                    i = match schema.scan(b, i + 1) {
                        Scan::End(i) => skip(b, i),
                        s => return s,
                    };
                    if i == b.len() {
                        return Scan::More;
                    }
                    match b[i] {
                        b'}' => {
                            return if required.is_subset(&seen) {
                                Scan::End(i + 1)
                            } else {
                                Scan::Bad
                            }
                        }
                        b',' => {
                            if seen.len() == fields.len() {
                                return Scan::Bad;
                            }
                            i = skip(b, i + 1);
                            after_comma = true;
                        }
                        _ => return Scan::Bad,
                    }
                }
            }
        }
    }
}

enum Scan {
    End(usize),
    More,
    Bad,
}
fn skip(b: &[u8], mut i: usize) -> usize {
    while i < b.len() && b" \n\r\t".contains(&b[i]) {
        i += 1;
    }
    i
}
fn string_end(b: &[u8], at: usize) -> Scan {
    if b[at] != b'"' {
        return Scan::Bad;
    }
    if std::str::from_utf8(&b[at..]).is_err_and(|e| e.error_len().is_some()) {
        return Scan::Bad;
    }
    let mut escape = false;
    for i in at + 1..b.len() {
        if !escape && b[i] == b'"' {
            return if serde_json::from_slice::<String>(&b[at..=i]).is_ok() {
                Scan::End(i + 1)
            } else {
                Scan::Bad
            };
        }
        escape = !escape && b[i] == b'\\';
    }
    match serde_json::from_slice::<String>(&b[at..]) {
        Err(e) if e.is_eof() => Scan::More,
        _ => Scan::Bad,
    }
}

impl Model {
    /// Constrained generation returns proposals only. The host owns execution.
    pub fn complete(
        &self,
        input: &str,
        tools: &[Tool],
        max_tokens: usize,
        cancel: &CancellationToken,
    ) -> Result<Completion> {
        let _permit = self.runtime().begin(cancel)?;
        if tools.is_empty() || tools.len() > 64 {
            return Err(Error::Request("tool count must be 1..64".into()));
        }
        let limits = self.runtime().limits();
        let overhead =
            "<|im_start|>user\n<tools></tools>\n<|im_end|>\n<|im_start|>assistant\n".len();
        let limit = limits
            .max_input_bytes
            .checked_sub(input.len())
            .and_then(|n| n.checked_sub(overhead))
            .ok_or_else(|| Error::Request("tool request byte limit exceeded".into()))?;
        let mut size = ToolSize {
            bytes: 0,
            limit,
            cancel,
        };
        let measured = serde_json::to_writer(&mut size, tools);
        crate::model::check_cancel(cancel)?;
        measured?;
        // Bound schema nodes/enum clones, serialized tools and the owned prompt.
        let bytes = (size.bytes as u64)
            .checked_mul(64)
            .and_then(|n| n.checked_add(input.len() as u64 + overhead as u64))
            .ok_or_else(|| Error::Request("tool schema state size overflow".into()))?;
        limits.checked_state_bytes(bytes, "tool schema state")?;
        let mut names = BTreeSet::new();
        let mut branches = Vec::new();
        for tool in tools {
            crate::model::check_cancel(cancel)?;
            if tool.name.is_empty()
                || tool.name.len() > 128
                || !names.insert(&tool.name)
                || tool.name.chars().any(char::is_control)
            {
                return Err(Error::Request("invalid or duplicate tool name".into()));
            }
            let arguments = Schema::parse(&tool.parameters, 0)?;
            if !matches!(arguments, Schema::Object(..)) {
                return Err(Error::Request("tool parameters must be an object".into()));
            }
            branches.push(Schema::Object(
                BTreeMap::from([
                    (
                        "name".into(),
                        Schema::Enum(vec![Value::String(tool.name.clone())]),
                    ),
                    ("arguments".into(), arguments),
                ]),
                BTreeSet::from(["name".into(), "arguments".into()]),
            ));
        }
        let schema = Schema::Array(Box::new(Schema::Any(branches)));
        let tools = serde_json::to_string(tools)?;
        crate::model::check_cancel(cancel)?;
        let prompt = format!(
            "<|im_start|>user\n<tools>{tools}</tools>\n{input}<|im_end|>\n<|im_start|>assistant\n"
        );
        let generation =
            self.generate_admitted(&prompt, max_tokens, cancel, Some(&schema), bytes)?;
        if !generation.finished {
            return Err(Error::Request(
                "generation exhausted before a complete tool call".into(),
            ));
        }
        let body = tool_body(&self.tokenizer, &generation.token_ids)?;
        crate::model::check_cancel(cancel)?;
        if !schema.accepts(&body, true) {
            return Err(Error::Request(
                "tool call violates the declared schema".into(),
            ));
        }
        let parsed = serde_json::from_slice(&body)?;
        crate::model::check_cancel(cancel)?;
        let (function_calls, suppressed_calls, ungrounded) = ground_string_arguments(input, parsed);
        Ok(Completion {
            function_calls,
            suppressed_calls,
            ungrounded,
            generation,
        })
    }
}

fn tool_body(tokenizer: &crate::Tokenizer, ids: &[u32]) -> Result<Vec<u8>> {
    let missing = || Error::Request("model did not return a complete tool call".into());
    let open = tokenizer.token_id("<tool_call>").ok_or_else(missing)?;
    let close = tokenizer.token_id("</tool_call>").ok_or_else(missing)?;
    let start = ids.iter().position(|&id| id == open).ok_or_else(missing)?;
    if ids.last() != Some(&close) || start + 1 >= ids.len() {
        return Err(missing());
    }
    let mut bytes = Vec::new();
    for &id in &ids[start + 1..ids.len() - 1] {
        bytes.extend(tokenizer.token_bytes(id)?);
    }
    Ok(bytes)
}

#[cfg(test)]
#[path = "structured_tests.rs"]
mod tests;
