//! Chat template handling for GGUF models
use crate::{
    completion::{ChatMessage, SenderType},
    tools::{ToolCall, ToolSpec},
};
use anyhow::{Context, Result};
use minijinja::{Environment, Error, State, context, value::Value};
use once_cell::sync::Lazy;
use regex::{Regex, escape};
use serde_json;
use std::collections::HashMap;
use tracing::{debug, error};

const DEFAULT_TEMPLATE: &str = include_str!("../../../data/default_template.jinja");

/// Applies chat template using the model's built-in template
pub fn apply_chat_template(
    template_str: &str,
    messages: &[ChatMessage],
    tools: Option<&[ToolSpec]>,
) -> Result<String> {
    debug!(
        template_from_model = !template_str.is_empty(),
        messages_count = messages.len(),
        tools_available = tools.is_some(),
        "Apply chat template"
    );
    let mut env = Environment::new();

    // We include minijinja with builtins for filters like tojson
    // Allow python compatibility methods like startswith to be available
    env.set_unknown_method_callback(unknown_method_callback);
    env.add_function("strftime_now", |format: String| {
        chrono::Local::now().format(&format).to_string()
    });

    let template = if template_str.is_empty() {
        debug!("Using default template");
        DEFAULT_TEMPLATE
    } else {
        template_str
    };

    let tmpl = env
        .template_from_str(template)
        .context("Invalid template format")?;

    // Convert ChatMessages into Jinja-compatible values
    let context_messages: Vec<_> = messages
        .iter()
        .map(|msg| {
            serde_json::json!({
                "role": match msg.sender {
                    SenderType::System => "system",
                    SenderType::User => "user",
                    SenderType::Assistant => "assistant",
                    SenderType::Tool => "tool",
                },
                "content": &msg.text,
                "tool_calls": &msg.tools,
            })
        })
        .collect();

    let tool_list = tools.unwrap_or_default();
    let context_tools: Vec<serde_json::Value> = tool_list
        .iter()
        .map(serde_json::to_value)
        .collect::<Result<_, _>>()
        .context("Failed to serialize tool spec")?;

    tmpl.render(context! {
    messages => &context_messages,
    tools => &context_tools,
    add_generation_prompt => true, // append <im_start|>assistant
    /* enable_thinking => false */
    })
    .inspect_err(|e| {
        error!(
            ?template,
            ?context_messages,
            tools = ?context_tools,
            error = ?e,
            "Failed to render chat template"
        );

        // render causes as well
        let mut err = &e as &dyn std::error::Error;
        while let Some(next_err) = err.source() {
            error!("caused by: {next_err:#}");
            err = next_err;
        }
    })
    .context("Template rendering failed")
}

fn unknown_method_callback(
    state: &State,
    value: &Value,
    method: &str,
    args: &[Value],
) -> std::result::Result<Value, Error> {
    debug!("Unknown method callback: {method}");
    minijinja_contrib::pycompat::unknown_method_callback(state, value, method, args)
}

const DEFAULT_TOOL_CALL_START_TAG: &str = "<tool_call>";
const DEFAULT_TOOL_CALL_END_TAG: &str = "</tool_call>";

static MODEL_SPECIFIC_TAGS: Lazy<HashMap<&'static str, (&'static str, &'static str)>> =
    Lazy::new(|| {
        let mut m: HashMap<&'static str, (&'static str, &'static str)> = HashMap::new();
        m.insert("granite", ("<|tool_call|>", ""));
        m
    });

pub fn get_tool_call_regexes(model_name: &str) -> (&'static str, &'static str) {
    MODEL_SPECIFIC_TAGS
        .iter()
        .find(|(k, _)| model_name.to_lowercase().contains(*k))
        .map(|(_, v)| *v)
        .unwrap_or((DEFAULT_TOOL_CALL_START_TAG, DEFAULT_TOOL_CALL_END_TAG))
}

#[derive(Debug)]
pub struct ToolCallParser {
    buffer: String,
    next_id: usize,
    complete_call_re: Regex,
    start_tag_re: Regex,
    expects_array: bool,
}

#[derive(serde::Deserialize)]
struct RawToolCall {
    name: String,
    arguments: serde_json::Value,
}

impl ToolCallParser {
    pub fn new(start_tag_pattern: &str, end_tag_pattern: &str) -> Result<Self> {
        let complete_call_pattern = if end_tag_pattern.is_empty() {
            // For models with no end tag, we'll handle parsing with a different logic.
            // This regex is a dummy that won't match anything, so the main loop is skipped.
            "$^".to_string()
        } else {
            format!(
                r"{}(?s)(.*?){}",
                escape(start_tag_pattern),
                escape(end_tag_pattern)
            )
        };
        Ok(Self {
            buffer: String::new(),
            next_id: 0,
            complete_call_re: Regex::new(&complete_call_pattern)
                .context("Invalid regex for complete tool call")?,
            start_tag_re: Regex::new(&escape(start_tag_pattern))
                .context("Invalid regex for tool call start tag")?,
            // Heuristic: Granite uses a different start tag and its content is a JSON array of tool calls.
            expects_array: start_tag_pattern == "<|tool_call|>",
        })
    }

    pub fn parse(&mut self, text: &str) -> (String, Vec<ToolCall>) {
        self.buffer.push_str(text);
        let mut tool_calls = Vec::new();
        let mut plain_text = String::new();
        let mut last_end = 0;

        // Special handling for models without a tool call end tag (e.g., Granite).
        if self.complete_call_re.as_str() == "$^" {
            if let Some(mat) = self.start_tag_re.find(&self.buffer) {
                plain_text.push_str(&self.buffer[..mat.start()]);
                let tool_call_part = self.buffer[mat.start()..].to_string();
                let content = &tool_call_part[(mat.end() - mat.start())..];

                if self.expects_array {
                    match serde_json::from_str::<Vec<RawToolCall>>(content) {
                        Ok(parsed) => {
                            for raw_tool_call in parsed {
                                tool_calls.push(ToolCall {
                                    id: format!("call_{}", self.next_id),
                                    name: raw_tool_call.name,
                                    arguments: raw_tool_call.arguments.to_string(),
                                });
                                self.next_id += 1;
                            }
                            self.buffer.clear();
                        }
                        Err(e) if e.is_eof() => {
                            // Incomplete JSON, so we buffer the tool call part.
                            self.buffer = tool_call_part;
                        }
                        Err(_) => {
                            // Invalid JSON, treat the whole tool call part as plain text.
                            plain_text.push_str(&tool_call_part);
                            self.buffer.clear();
                        }
                    }
                } else {
                    // Not expecting an array, and no end tag. This is an unsupported configuration.
                    // Treat as plain text to avoid getting stuck.
                    plain_text.push_str(&tool_call_part);
                    self.buffer.clear();
                }
            } else {
                plain_text.push_str(&self.buffer);
                self.buffer.clear();
            }
            return (plain_text, tool_calls);
        }

        for captures in self.complete_call_re.captures_iter(&self.buffer) {
            // There will be exactly one capture group in a successful match
            if let (Some(outer_match), Some(inner_match)) = (captures.get(0), captures.get(1)) {
                plain_text.push_str(&self.buffer[last_end..outer_match.start()]);

                let tool_content = inner_match.as_str().trim();
                let parsed_calls = if self.expects_array {
                    parse_tool_call_array(tool_content)
                } else {
                    parse_single_tool_call(tool_content).map(|c| vec![c])
                };

                match parsed_calls {
                    Some(raw_tool_calls) => {
                        for raw_tool_call in raw_tool_calls {
                            tool_calls.push(ToolCall {
                                id: format!("call_{}", self.next_id),
                                name: raw_tool_call.name,
                                arguments: raw_tool_call.arguments.to_string(),
                            });
                            self.next_id += 1;
                        }
                    }
                    None => {
                        error!("Failed to parse tool call JSON. Content: '{tool_content}'");
                        plain_text.push_str(outer_match.as_str());
                    }
                }
                last_end = outer_match.end();
            }
        }

        let remainder = &self.buffer[last_end..];

        if let Some(mat) = self.start_tag_re.find_iter(remainder).last() {
            let split_point = mat.start();
            plain_text.push_str(&remainder[..split_point]);
            self.buffer.drain(..last_end + split_point);
        } else {
            plain_text.push_str(remainder);
            self.buffer.clear();
        }

        (plain_text, tool_calls)
    }

    pub fn flush(&mut self) -> String {
        std::mem::take(&mut self.buffer)
    }
}

fn parse_single_tool_call(content: &str) -> Option<RawToolCall> {
    // First try parsing as direct RawToolCall
    if let Ok(raw) = serde_json::from_str::<RawToolCall>(content) {
        return Some(raw);
    }

    // Then try wrapped format: { "call": { ... RawToolCall ... } }
    if let Ok(wrapped) = serde_json::from_str::<serde_json::Value>(content)
        && let Some(call_obj) = wrapped.get("call")
        && let Ok(raw) = serde_json::from_value(call_obj.clone())
    {
        return Some(raw);
    }

    None
}

fn parse_tool_call_array(content: &str) -> Option<Vec<RawToolCall>> {
    serde_json::from_str(content).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::completion::{ChatMessage, SenderType};
    use crate::tools::ToolCall;

    use super::DEFAULT_TEMPLATE;

    #[test]
    fn test_user_message() {
        let messages = vec![ChatMessage {
            sender: SenderType::User,
            text: "Hello".into(),
            tools: vec![],
        }];

        let result = apply_chat_template(DEFAULT_TEMPLATE, &messages, None);
        assert!(result.is_ok());
        let output = result.unwrap();
        assert!(output.contains("<|im_start|>user\nHello<|im_end|>"));
    }

    #[test]
    fn test_assistant_message_with_tool_call() {
        let messages = vec![
            ChatMessage {
                sender: SenderType::User,
                text: "What's the weather in Boston?".into(),
                tools: vec![],
            },
            ChatMessage {
                sender: SenderType::Assistant,
                text: "".into(),
                tools: vec![ToolCall {
                    id: "call_123".to_string(),
                    name: "get_weather".to_string(),
                    arguments: serde_json::to_string("{\"location\": \"Boston\"}").unwrap(),
                }],
            },
        ];

        let result = apply_chat_template(DEFAULT_TEMPLATE, &messages, None);
        assert!(result.is_ok(), "Result: {result:?}");
        let output = result.unwrap();
        assert!(output.contains("<tool_call>"));
        assert!(output.contains(r#""name": "get_weather""#));
        assert!(output.contains(r#""arguments": "{\"location\": \"Boston\"}""#));
        assert!(output.contains("</tool_call>"));
    }

    #[test]
    fn test_tool_response_message() {
        let messages = vec![
            ChatMessage {
                sender: SenderType::User,
                text: "What's the weather in Boston?".into(),
                tools: vec![],
            },
            ChatMessage {
                sender: SenderType::Assistant,
                text: "".into(),
                tools: vec![ToolCall {
                    id: "call_123".to_string(),
                    name: "get_weather".to_string(),
                    arguments: "{\"location\": \"Boston\"}".to_string(),
                }],
            },
            ChatMessage {
                sender: SenderType::Tool,
                text: r#"{"temperature": 22, "unit": "celsius"}"#.into(),
                tools: vec![],
            },
        ];

        let result = apply_chat_template(DEFAULT_TEMPLATE, &messages, None);
        assert!(result.is_ok(), "Result: {result:?}");
        let output = result.unwrap();
        assert!(output.contains("<tool_response>"), "Output: {output}");
        assert!(
            output.contains(r#"{"temperature": 22, "unit": "celsius"}"#),
            "Output: {output}"
        );
        assert!(output.contains("</tool_response>"), "Output: {output}");
    }

    #[test]
    fn test_user_message_with_empty_template() {
        let messages = vec![ChatMessage {
            sender: SenderType::User,
            text: "Hello".into(),
            tools: vec![],
        }];

        let result = apply_chat_template("", &messages, None);
        assert!(result.is_ok());
        let output = result.unwrap();
        assert!(output.contains("<|im_start|>user\nHello<|im_end|>"));
    }

    #[test]
    fn test_assistant_message_with_tool_call_with_empty_template() {
        let messages = vec![
            ChatMessage {
                sender: SenderType::User,
                text: "What's the weather in Boston?".into(),
                tools: vec![],
            },
            ChatMessage {
                sender: SenderType::Assistant,
                text: "".into(),
                tools: vec![ToolCall {
                    id: "call_123".to_string(),
                    name: "get_weather".to_string(),
                    arguments: serde_json::to_string("{\"location\": \"Boston\"}").unwrap(),
                }],
            },
        ];

        let result = apply_chat_template("", &messages, None);
        assert!(result.is_ok(), "Result: {result:?}");
        let output = result.unwrap();
        assert!(output.contains("<tool_call>"));
        assert!(output.contains(r#""name": "get_weather""#));
        assert!(output.contains(r#""arguments": "{\"location\": \"Boston\"}""#));
        assert!(output.contains("</tool_call>"));
    }

    #[test]
    fn test_get_tool_call_regexes() {
        // Test default case
        let (default_start, default_end) = get_tool_call_regexes("any-other-model");
        assert_eq!(default_start, DEFAULT_TOOL_CALL_START_TAG);
        assert_eq!(default_end, DEFAULT_TOOL_CALL_END_TAG);

        // Test model-specific case
        let (specific_start, specific_end) = get_tool_call_regexes("Granite-3.3-8b-Instruct");
        assert_eq!(specific_start, "<|tool_call|>");
        assert_eq!(specific_end, "");
    }

    #[test]
    fn test_strftime_now_function() {
        // This test verifies the strftime_now filter produces the expected format
        let messages = vec![ChatMessage {
            sender: SenderType::User,
            text: "Test".into(),
            tools: vec![],
        }];

        let template_with_format = r#"{{ strftime_now("%Y-%m-%d %H:%M:%S") }}"#;
        let result = apply_chat_template(template_with_format, &messages, None);
        assert!(
            result.is_ok(),
            "Template rendering failed. Template: {}, Error: {}",
            template_with_format,
            result.err().unwrap()
        );

        let output = result.unwrap();
        println!("Formatted time: {}", output);

        // Use regex to match the formatted time pattern: YYYY-MM-DD HH:MM:SS
        let re = regex::Regex::new(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$").unwrap();
        assert!(
            re.is_match(&output),
            "Output '{}' doesn't match expected time format",
            output
        );

        // Test with another format
        let template_with_month_day = r#"{{ strftime_now("%B %d") }}"#;
        let result = apply_chat_template(template_with_month_day, &messages, None);
        assert!(
            result.is_ok(),
            "Template rendering failed. Template: {}, Error: {}",
            template_with_format,
            result.err().unwrap()
        );

        let output = result.unwrap();
        println!("Formatted month/day: {}", output);

        // Should produce something like "August 12"
        let re = regex::Regex::new(r"^\w+ \d{1,2}$").unwrap();
        assert!(
            re.is_match(&output),
            "Output '{}' doesn't match month/day format",
            output
        );
    }

    #[test]
    fn test_tool_call_parser_split_across_chunks() {
        let mut parser = ToolCallParser::new("<tool_call>", "</tool_call>").unwrap();
        let chunk1 = "Thinking...<tool_call>{\"name\": \"search\",";
        let (text, calls) = parser.parse(chunk1);
        assert_eq!(text, "Thinking...");
        assert!(calls.is_empty());

        let chunk2 = "\"arguments\": {\"query\": \"weather\"}}</tool_call>Done.";
        let (text, calls) = parser.parse(chunk2);
        assert_eq!(text, "Done.");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call_0");
        assert_eq!(calls[0].name, "search");
        assert_eq!(calls[0].arguments, r#"{"query":"weather"}"#);

        let final_text = parser.flush();
        assert_eq!(final_text, "");
    }

    #[test]
    fn test_tool_call_parser_plain_text_only() {
        let mut parser = ToolCallParser::new("<tool_call>", "</tool_call>").unwrap();
        let (text, calls) = parser.parse("This is some thinking text. ");
        assert_eq!(text, "This is some thinking text. ");
        assert!(calls.is_empty());

        let (text, calls) = parser.parse("And some more.");
        assert_eq!(text, "And some more.");
        assert!(calls.is_empty());

        let final_text = parser.flush();
        assert!(final_text.is_empty());
    }

    #[test]
    fn test_tool_call_parser_multiple_calls() {
        let mut parser = ToolCallParser::new("<tool_call>", "</tool_call>").unwrap();
        let chunk = "<tool_call>{\"name\":\"search\",\"arguments\":{\"query\":\"rust\"}}</tool_call>Then<tool_call>{\"name\":\"weather\",\"arguments\":{\"location\":\"moon\"}}</tool_call>";
        let (text, calls) = parser.parse(chunk);

        assert_eq!(text, "Then");
        assert_eq!(calls.len(), 2);

        assert_eq!(calls[0].name, "search");
        assert_eq!(calls[0].arguments, "{\"query\":\"rust\"}");
        assert_eq!(calls[1].name, "weather");
        assert_eq!(calls[1].arguments, "{\"location\":\"moon\"}");

        let final_text = parser.flush();
        assert!(final_text.is_empty());
    }

    #[test]
    fn test_tool_call_parser_mixed_content() {
        let mut parser = ToolCallParser::new("<tool_call>", "</tool_call>").unwrap();
        let chunk1 = "<tool_call>{\"name\":\"search\",\"arguments\":{\"query\":\"rust\"}}</tool_call>Now for the next one <tool_call>{\"name\":";
        let (text, calls) = parser.parse(chunk1);

        assert_eq!(text, "Now for the next one ");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "search");

        let chunk2 = "\"weather\",\"arguments\":{\"location\":\"moon\"}}</tool_call>All done.";
        let (text, calls) = parser.parse(chunk2);

        assert_eq!(text, "All done.");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "weather");

        let final_text = parser.flush();
        assert!(final_text.is_empty());
    }

    #[test]
    fn test_tool_call_parser_custom_tags() {
        let mut parser = ToolCallParser::new("<|tool_code|>", "<|/tool_code|>").unwrap();
        let chunk =
            "Thinking...<|tool_code|>{\"name\":\"search\",\"arguments\":{}}<|/tool_code|>Done.";
        let (text, calls) = parser.parse(chunk);

        assert_eq!(text, "Thinking...Done.");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "search");

        let final_text = parser.flush();
        assert!(final_text.is_empty());
    }

    #[test]
    fn test_tool_call_parser_wrapped_format() {
        let mut parser = ToolCallParser::new("<tool_call>", "</tool_call>").unwrap();
        let chunk = r#"<tool_call>{"call": {"name":"get_weather","arguments":{"location":"Paris"}}}</tool_call>"#;
        let (_, calls) = parser.parse(chunk);

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].arguments, r#"{"location":"Paris"}"#);
    }

    #[test]
    fn test_tool_call_parser_mixed_formats() {
        let mut parser = ToolCallParser::new("<tool_call>", "</tool_call>").unwrap();
        let chunk = r#"
            <tool_call>{"name":"direct_search","arguments":{"query":"Rust"}}</tool_call>
            <tool_call>{"call": {"name":"wrapped_weather","arguments":{"location":"London"}}}</tool_call>
        "#;
        let (_, calls) = parser.parse(chunk);

        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "direct_search");
        assert_eq!(calls[0].arguments, r#"{"query":"Rust"}"#);
        assert_eq!(calls[1].name, "wrapped_weather");
        assert_eq!(calls[1].arguments, r#"{"location":"London"}"#);
    }

    #[test]
    fn test_tool_call_parser_ignores_invalid() {
        let mut parser = ToolCallParser::new("<tool_call>", "</tool_call>").unwrap();
        let chunk = r#"<tool_call>This is not JSON</tool_call>
            <tool_call>{"call": {"invalid_key": "value"}}</tool_call>
            <tool_call>{"call": {"name": "missing_arguments"}}</tool_call>
            <tool_call>{"wrong_top_key": {"name":"weather"}}</tool_call>"#;
        let (text, calls) = parser.parse(chunk);

        // All tool call tags should be present in output since parsing fails
        assert_eq!(calls.len(), 0);
        assert!(text.contains("<tool_call>"));
        assert!(text.contains("</tool_call>"));
    }

    #[test]
    fn test_tool_call_parser_granite_array_of_calls() {
        let mut parser = ToolCallParser::new("<|tool_call|>", "").unwrap();
        let chunk = r#"<|tool_call|>[{"name":"search","arguments":{"query":"rust"}},{"name":"weather","arguments":{"location":"moon"}}]"#;
        let (text, calls) = parser.parse(chunk);

        assert!(text.is_empty());
        assert_eq!(calls.len(), 2);

        assert_eq!(calls[0].name, "search");
        assert_eq!(calls[0].arguments, "{\"query\":\"rust\"}");
        assert_eq!(calls[1].name, "weather");
        assert_eq!(calls[1].arguments, "{\"location\":\"moon\"}");

        let final_text = parser.flush();
        assert!(final_text.is_empty());
    }

    #[test]
    fn test_tool_call_parser_granite_array_streaming() {
        let mut parser = ToolCallParser::new("<|tool_call|>", "").unwrap();
        let chunk1 = r#"<|tool_call|>[{"name": "search", "arguments": {"query": "rust"}}, "#;
        let (text, calls) = parser.parse(chunk1);
        assert_eq!(text, "");
        assert_eq!(calls.len(), 0);

        let chunk2 = r#"{"name": "weather", "arguments": {"location": "moon"}}]"#;
        let (text, calls) = parser.parse(chunk2);
        assert_eq!(text, "");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "search");
        assert_eq!(calls[1].name, "weather");
    }
}
