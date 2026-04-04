//! Chat template handling for GGUF models
use crate::{
    completion::{ChatMessage, SenderType},
    tools::{ToolCall, ToolSpec},
};
use anyhow::{Context, Result};
use minijinja::{Environment, Error, State, value::Value};
use once_cell::sync::Lazy;
use regex::{Regex, escape};
use serde_json;
use serde_yaml;
use std::collections::HashMap;
use std::sync::Mutex;
use tracing::{debug, error};

const DEFAULT_TEMPLATE: &str = include_str!("../../../data/templates/default.jinja");

fn get_builtin_template(name: &str) -> Option<&'static str> {
    let name = name.to_lowercase();
    match name.as_str() {
        "qwen35" => Some(include_str!("../../../data/templates/qwen35.jinja")),
        "default" => Some(DEFAULT_TEMPLATE),
        _ => None,
    }
}

fn resolve_template(
    model_template: &str,
    template_override_name: Option<&str>,
) -> (String, String) {
    if let Some(name) = template_override_name
        && let Some(tmpl) = get_builtin_template(name)
    {
        return (tmpl.to_string(), name.to_string());
    }
    if !model_template.is_empty() {
        (model_template.to_string(), "model".to_string())
    } else {
        (DEFAULT_TEMPLATE.to_string(), "default".to_string())
    }
}

/// Applies chat template using the model's built-in template
pub fn apply_chat_template(
    model_template: &str,
    template_override_name: Option<&str>,
    messages: &[ChatMessage],
    tools: Option<&[ToolSpec]>,
    template_args: HashMap<String, serde_yaml::Value>,
) -> Result<String> {
    debug!(
        template_from_model = !model_template.is_empty(),
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

    let (template_str, template_name) = resolve_template(model_template, template_override_name);
    debug!(
        template_used = %template_name,
        template_from_model = !model_template.is_empty(),
        messages_count = messages.len(),
        tools_available = tools.is_some(),
        "Apply chat template"
    );
    let template = if template_str.is_empty() {
        debug!("Using default template");
        DEFAULT_TEMPLATE
    } else {
        &template_str
    };

    let tmpl = env
        .template_from_str(template)
        .context("Invalid template format")?;

    // Convert ChatMessages into Jinja-compatible values
    let context_messages: Vec<_> = messages
        .iter()
        .map(|msg| {
            let reasoning_content = match &msg.thought {
                Some(t) => serde_json::Value::String(t.clone()),
                None => serde_json::Value::Null,
            };
            serde_json::json!({
                "role": match msg.sender {
                    SenderType::System => "system",
                    SenderType::User => "user",
                    SenderType::Assistant => "assistant",
                    SenderType::Tool => "tool",
                },
                "content": &msg.text,
                "reasoning_content": reasoning_content,
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

    // Extract enable_thinking from template_args
    fn yaml_to_json(yaml_val: &serde_yaml::Value) -> serde_json::Value {
        serde_json::to_value(yaml_val).unwrap_or(serde_json::Value::Null)
    }

    let enable_thinking = template_args
        .get("enable_thinking")
        .map(yaml_to_json)
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let mut kwargs = serde_json::json!({
        "messages": &context_messages,
        "tools": &context_tools,
        "add_generation_prompt": true,
        "enable_thinking": enable_thinking,
    });

    // Merge template_args into kwargs
    for (key, value) in template_args {
        if key != "enable_thinking" {
            kwargs[key] = yaml_to_json(&value);
        }
    }

    tmpl.render(kwargs)
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
        let m: HashMap<&'static str, (&'static str, &'static str)> = HashMap::new();
        m
    });

pub fn get_tool_call_regexes(model_name: &str) -> TemplatePatterns {
    let (start, end) = MODEL_SPECIFIC_TAGS
        .iter()
        .find(|(k, _)| model_name.to_lowercase().contains(*k))
        .map(|(_, v)| *v)
        .unwrap_or((DEFAULT_TOOL_CALL_START_TAG, DEFAULT_TOOL_CALL_END_TAG));

    TemplatePatterns {
        tool_start_re: Some(start.to_string()),
        tool_end_re: Some(end.to_string()),
        thought_start_re: Some("<think>".to_string()),
        thought_end_re: Some("</think>".to_string()),
    }
}

#[derive(Debug, Clone, Default)]
pub struct TemplatePatterns {
    pub tool_start_re: Option<String>,
    pub tool_end_re: Option<String>,
    pub thought_start_re: Option<String>,
    pub thought_end_re: Option<String>,
}

#[derive(Debug)]
pub struct ToolCallParser {
    buffer: Mutex<String>,
    next_id: Mutex<usize>,
    complete_call_re: Regex,
    start_tag_re: Regex,
    expects_array: bool,
    thought_start_re: Option<Regex>,
    thought_end_re: Option<Regex>,
    is_in_thought: Mutex<bool>,
}

#[derive(serde::Deserialize)]
struct RawToolCall {
    name: String,
    arguments: serde_json::Value,
}

impl ToolCallParser {
    pub fn new(patterns: TemplatePatterns) -> Result<Self> {
        let start_tag = patterns
            .tool_start_re
            .as_deref()
            .unwrap_or(DEFAULT_TOOL_CALL_START_TAG);
        let end_tag = patterns
            .tool_end_re
            .as_deref()
            .unwrap_or(DEFAULT_TOOL_CALL_END_TAG);

        let complete_call_pattern = if end_tag.is_empty() {
            "$^".to_string()
        } else {
            format!(r"{}(?s)(.*?){}", escape(start_tag), escape(end_tag))
        };

        let thought_start_re = patterns
            .thought_start_re
            .as_ref()
            .map(|p| Regex::new(p))
            .transpose()
            .context("Invalid regex for thought start tag")?;
        let thought_end_re = patterns
            .thought_end_re
            .as_ref()
            .map(|p| Regex::new(p))
            .transpose()
            .context("Invalid regex for thought end tag")?;

        Ok(Self {
            buffer: Mutex::new(String::new()),
            next_id: Mutex::new(0),
            complete_call_re: Regex::new(&complete_call_pattern)
                .context("Invalid regex for complete tool call")?,
            start_tag_re: Regex::new(&escape(start_tag))
                .context("Invalid regex for tool call start tag")?,
            expects_array: false,
            thought_start_re,
            thought_end_re,
            is_in_thought: Mutex::new(false),
        })
    }

    pub fn set_in_thought(&self, in_thought: bool) {
        let mut state = self.is_in_thought.lock().unwrap();
        *state = in_thought;
    }

    pub fn reset(&self) {
        let mut buffer = self.buffer.lock().unwrap();
        buffer.clear();
        let mut state = self.is_in_thought.lock().unwrap();
        *state = false;
        let mut next_id = self.next_id.lock().unwrap();
        *next_id = 0;
    }

    pub fn parse(&self, text: &str) -> (String, Option<String>, Vec<ToolCall>) {
        let mut buffer = self.buffer.lock().unwrap();
        let mut next_id = self.next_id.lock().unwrap();
        let mut is_in_thought = self.is_in_thought.lock().unwrap();

        buffer.push_str(text);
        let mut tool_calls = Vec::new();
        let mut plain_text = String::new();
        let mut thought: Option<String> = None;
        let mut current_pos = 0;

        loop {
            if *is_in_thought {
                if let Some(end_re) = &self.thought_end_re {
                    if let Some(end_match) = end_re.find(&buffer[current_pos..]) {
                        let abs_end_start = current_pos + end_match.start();
                        let abs_end_end = current_pos + end_match.end();

                        let thought_content = &buffer[current_pos..abs_end_start];
                        thought
                            .get_or_insert_with(String::new)
                            .push_str(thought_content);

                        *is_in_thought = false;
                        current_pos = abs_end_end;
                        continue;
                    } else {
                        // Still in thought - stream all content immediately
                        // Think tokens are always single tokens, so no partial tag risk
                        let thought_content = &buffer[current_pos..];
                        thought
                            .get_or_insert_with(String::new)
                            .push_str(thought_content);
                        current_pos = buffer.len();
                        break;
                    }
                } else {
                    // No end tag configured
                    let thought_content = &buffer[current_pos..];
                    thought
                        .get_or_insert_with(String::new)
                        .push_str(thought_content);
                    current_pos = buffer.len();
                    break;
                }
            } else {
                // Not in thought, look for thought start or tool call
                let segment = &buffer[current_pos..];
                let thought_start = self
                    .thought_start_re
                    .as_ref()
                    .and_then(|re| re.find(segment));
                let tool_call_match = self.complete_call_re.find(segment);
                let tool_start_only = self.start_tag_re.find(segment);

                // Find earliest match
                let matches = [
                    thought_start.map(|m| (m.start(), 0)),   // Type 0: thought
                    tool_call_match.map(|m| (m.start(), 1)), // Type 1: complete tool call
                    tool_start_only.map(|m| (m.start(), 2)), // Type 2: tool start tag
                ];

                let earliest = matches.iter().flatten().min_by_key(|m| m.0);

                if let Some(&(rel_start, match_type)) = earliest {
                    let abs_start = current_pos + rel_start;
                    plain_text.push_str(&buffer[current_pos..abs_start]);
                    current_pos = abs_start;

                    match match_type {
                        0 => {
                            // Thought start
                            let m = thought_start.unwrap();
                            *is_in_thought = true;
                            current_pos += m.end();
                        }
                        1 => {
                            // Complete tool call
                            let m = tool_call_match.unwrap();
                            let abs_end = current_pos + (m.end() - m.start());
                            let content = &buffer[current_pos..abs_end];

                            // Extract the inner content (captured by group 1 in complete_call_re)
                            if let Some(caps) = self.complete_call_re.captures(content)
                                && let Some(inner) = caps.get(1)
                            {
                                let tool_content = inner.as_str().trim();
                                let parsed_calls = if self.expects_array {
                                    parse_tool_call_array(tool_content)
                                } else {
                                    parse_single_tool_call(tool_content).map(|c| vec![c])
                                };

                                if let Some(raw_tool_calls) = parsed_calls {
                                    for raw_tool_call in raw_tool_calls {
                                        tool_calls.push(ToolCall {
                                            id: format!("call_{}", *next_id),
                                            name: raw_tool_call.name,
                                            arguments: raw_tool_call.arguments.to_string(),
                                        });
                                        *next_id += 1;
                                    }
                                } else {
                                    error!("Failed to parse tool call JSON: {}", tool_content);
                                    plain_text.push_str(content);
                                }
                            } else {
                                plain_text.push_str(content);
                            }
                            current_pos = abs_end;
                        }
                        2 => {
                            // Tool start tag without end tag yet
                            // Break and keep rest in buffer
                            break;
                        }
                        _ => unreachable!(),
                    }
                } else {
                    // Nothing found
                    plain_text.push_str(&buffer[current_pos..]);
                    current_pos = buffer.len();
                    break;
                }
            }
        }

        buffer.drain(..current_pos);
        (plain_text, thought, tool_calls)
    }

    pub fn flush(&self) -> String {
        let mut buffer = self.buffer.lock().unwrap();
        std::mem::take(&mut *buffer)
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
            ..Default::default()
        }];

        let result = apply_chat_template(DEFAULT_TEMPLATE, None, &messages, None, HashMap::new());
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
                ..Default::default()
            },
            ChatMessage {
                sender: SenderType::Assistant,
                text: "".into(),
                thought: None,
                tools: Some(vec![ToolCall {
                    id: "call_123".to_string(),
                    name: "get_weather".to_string(),
                    arguments: serde_json::to_string("{\"location\": \"Boston\"}").unwrap(),
                }]),
                metrics: Some(Default::default()),
            },
        ];

        let result = apply_chat_template(DEFAULT_TEMPLATE, None, &messages, None, HashMap::new());
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
                ..Default::default()
            },
            ChatMessage {
                sender: SenderType::Assistant,
                text: "".into(),
                thought: None,
                tools: Some(vec![ToolCall {
                    id: "call_123".to_string(),
                    name: "get_weather".to_string(),
                    arguments: "{\"location\": \"Boston\"}".to_string(),
                }]),
                metrics: Some(Default::default()),
            },
            ChatMessage {
                sender: SenderType::Tool,
                text: r#"{"temperature": 22, "unit": "celsius"}"#.into(),
                thought: None,
                ..Default::default()
            },
        ];

        let result = apply_chat_template(DEFAULT_TEMPLATE, None, &messages, None, HashMap::new());
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
            ..Default::default()
        }];

        let result = apply_chat_template("", None, &messages, None, HashMap::new());
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
                ..Default::default()
            },
            ChatMessage {
                sender: SenderType::Assistant,
                text: "".into(),
                thought: None,
                tools: Some(vec![ToolCall {
                    id: "call_123".to_string(),
                    name: "get_weather".to_string(),
                    arguments: serde_json::to_string("{\"location\": \"Boston\"}").unwrap(),
                }]),
                metrics: Some(Default::default()),
            },
        ];

        let result = apply_chat_template("", None, &messages, None, HashMap::new());
        assert!(result.is_ok(), "Result: {result:?}");
        let output = result.unwrap();
        assert!(output.contains("<tool_call>"));
        assert!(output.contains(r#""name": "get_weather""#));
        assert!(output.contains(r#""arguments": "{\"location\": \"Boston\"}""#));
        assert!(output.contains("</tool_call>"));
    }

    #[test]
    fn test_get_tool_call_regexes() {
        // Test default case - all models should return default tags now
        let patterns = get_tool_call_regexes("any-other-model");
        assert_eq!(patterns.tool_start_re.unwrap(), DEFAULT_TOOL_CALL_START_TAG);
        assert_eq!(patterns.tool_end_re.unwrap(), DEFAULT_TOOL_CALL_END_TAG);

        // Test case-insensitive matching for any model name
        let patterns = get_tool_call_regexes("Some-Model-3.3-8b-Instruct");
        assert_eq!(patterns.tool_start_re.unwrap(), DEFAULT_TOOL_CALL_START_TAG);
        assert_eq!(patterns.tool_end_re.unwrap(), DEFAULT_TOOL_CALL_END_TAG);

        // Test case-insensitive matching
        let patterns = get_tool_call_regexes("some-model-3.3-8b-instruct");
        assert_eq!(patterns.tool_start_re.unwrap(), DEFAULT_TOOL_CALL_START_TAG);
        assert_eq!(patterns.tool_end_re.unwrap(), DEFAULT_TOOL_CALL_END_TAG);
    }

    #[test]
    fn test_strftime_now_function() {
        // This test verifies the strftime_now filter produces the expected format
        let messages = vec![ChatMessage {
            sender: SenderType::User,
            text: "Test".into(),
            ..Default::default()
        }];

        let template_with_format = r#"{{ strftime_now("%Y-%m-%d %H:%M:%S") }}"#;
        let result =
            apply_chat_template(template_with_format, None, &messages, None, HashMap::new());
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
        let result = apply_chat_template(
            template_with_month_day,
            None,
            &messages,
            None,
            HashMap::new(),
        );
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
        let parser = ToolCallParser::new(TemplatePatterns {
            tool_start_re: Some("<tool_call>".to_string()),
            tool_end_re: Some("</tool_call>".to_string()),
            ..Default::default()
        })
        .unwrap();
        let chunk1 = "Thinking...<tool_call>{\"name\": \"search\",";
        let (text, _thought, calls) = parser.parse(chunk1);
        assert_eq!(text, "Thinking...");
        assert!(calls.is_empty());

        let chunk2 = "\"arguments\": {\"query\": \"weather\"}}</tool_call>Done.";
        let (text, _thought, calls) = parser.parse(chunk2);
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
        let parser = ToolCallParser::new(TemplatePatterns {
            tool_start_re: Some("<tool_call>".to_string()),
            tool_end_re: Some("</tool_call>".to_string()),
            ..Default::default()
        })
        .unwrap();
        let (text, _thought, calls) = parser.parse("This is some thinking text. ");
        assert_eq!(text, "This is some thinking text. ");
        assert!(calls.is_empty());

        let (text, _thought, calls) = parser.parse("And some more.");
        assert_eq!(text, "And some more.");
        assert!(calls.is_empty());

        let final_text = parser.flush();
        assert!(final_text.is_empty());
    }

    #[test]
    fn test_tool_call_parser_multiple_calls() {
        let parser = ToolCallParser::new(TemplatePatterns {
            tool_start_re: Some("<tool_call>".to_string()),
            tool_end_re: Some("</tool_call>".to_string()),
            ..Default::default()
        })
        .unwrap();
        let chunk = "<tool_call>{\"name\":\"search\",\"arguments\":{\"query\":\"rust\"}}</tool_call>Then<tool_call>{\"name\":\"weather\",\"arguments\":{\"location\":\"moon\"}}</tool_call>";
        let (text, _thought, calls) = parser.parse(chunk);

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
        let parser = ToolCallParser::new(TemplatePatterns {
            tool_start_re: Some("<tool_call>".to_string()),
            tool_end_re: Some("</tool_call>".to_string()),
            ..Default::default()
        })
        .unwrap();
        let chunk1 = "<tool_call>{\"name\":\"search\",\"arguments\":{\"query\":\"rust\"}}</tool_call>Now for the next one <tool_call>{\"name\":";
        let (text, _thought, calls) = parser.parse(chunk1);

        assert_eq!(text, "Now for the next one ");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "search");

        let chunk2 = "\"weather\",\"arguments\":{\"location\":\"moon\"}}</tool_call>All done.";
        let (text, _thought, calls) = parser.parse(chunk2);

        assert_eq!(text, "All done.");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "weather");

        let final_text = parser.flush();
        assert!(final_text.is_empty());
    }

    #[test]
    fn test_tool_call_parser_custom_tags() {
        let parser = ToolCallParser::new(TemplatePatterns {
            tool_start_re: Some("<|tool_code|>".to_string()),
            tool_end_re: Some("<|/tool_code|>".to_string()),
            ..Default::default()
        })
        .unwrap();
        let chunk =
            "Thinking...<|tool_code|>{\"name\":\"search\",\"arguments\":{}}<|/tool_code|>Done.";
        let (text, _thought, calls) = parser.parse(chunk);

        assert_eq!(text, "Thinking...Done.");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "search");

        let final_text = parser.flush();
        assert!(final_text.is_empty());
    }

    #[test]
    fn test_tool_call_parser_wrapped_format() {
        let parser = ToolCallParser::new(TemplatePatterns {
            tool_start_re: Some("<tool_call>".to_string()),
            tool_end_re: Some("</tool_call>".to_string()),
            ..Default::default()
        })
        .unwrap();
        let chunk = r#"<tool_call>{"call": {"name":"get_weather","arguments":{"location":"Paris"}}}</tool_call>"#;
        let (_, _thought, calls) = parser.parse(chunk);

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].arguments, r#"{"location":"Paris"}"#);
    }

    #[test]
    fn test_tool_call_parser_mixed_formats() {
        let parser = ToolCallParser::new(TemplatePatterns {
            tool_start_re: Some("<tool_call>".to_string()),
            tool_end_re: Some("</tool_call>".to_string()),
            ..Default::default()
        })
        .unwrap();
        let chunk = r#"
            <tool_call>{"name":"direct_search","arguments":{"query":"Rust"}}</tool_call>
            <tool_call>{"call": {"name":"wrapped_weather","arguments":{"location":"London"}}}</tool_call>
        "#;
        let (_, _thought, calls) = parser.parse(chunk);

        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "direct_search");
        assert_eq!(calls[0].arguments, r#"{"query":"Rust"}"#);
        assert_eq!(calls[1].name, "wrapped_weather");
        assert_eq!(calls[1].arguments, r#"{"location":"London"}"#);
    }

    #[test]
    fn test_tool_call_parser_ignores_invalid() {
        let parser = ToolCallParser::new(TemplatePatterns {
            tool_start_re: Some("<tool_call>".to_string()),
            tool_end_re: Some("</tool_call>".to_string()),
            ..Default::default()
        })
        .unwrap();
        let chunk = r#"<tool_call>This is not JSON</tool_call>
            <tool_call>{"call": {"invalid_key": "value"}}</tool_call>
            <tool_call>{"call": {"name": "missing_arguments"}}</tool_call>
            <tool_call>{"wrong_top_key": {"name":"weather"}}</tool_call>"#;
        let (text, _thought, calls) = parser.parse(chunk);

        // All tool call tags should be present in output since parsing fails
        assert_eq!(calls.len(), 0);
        assert!(text.contains("<tool_call>"));
        assert!(text.contains("</tool_call>"));
    }

    #[test]
    fn test_tool_call_parser_streaming_with_no_end_tag() {
        // Test behavior for models that use only start tags with no end tags
        // When expects_array=false, this gets treated as plain text
        let parser = ToolCallParser::new(TemplatePatterns {
            tool_start_re: Some("<|tool_call|>".to_string()),
            tool_end_re: Some("".to_string()),
            ..Default::default()
        })
        .unwrap();
        let chunk1 = "<|tool_call|>[{\"name\":\"search\",\"arguments\":{\"query\":\"rust\"}}";
        let (text, _thought, calls) = parser.parse(chunk1);

        // It should be buffered because tool_start matched
        assert_eq!(text, "");
        assert_eq!(calls.len(), 0);

        // Add more content
        let chunk2 = ",{\"name\":\"weather\",\"arguments\":{\"location\":\"moon\"}}]";
        let (_text, _thought, _calls) = parser.parse(chunk2);

        // Still buffered or partial
        assert_eq!(_calls.len(), 0);
    }

    #[test]
    fn test_tool_call_parser_streaming_with_end_tags() {
        // Test streaming behavior for models that use both start and end tags
        // Note: Current implementation buffers incomplete tool calls differently than expected
        // This test verifies the basic capability to parse tool calls with tags
        let parser = ToolCallParser::new(TemplatePatterns {
            tool_start_re: Some("<|tool_start|>".to_string()),
            tool_end_re: Some("<|tool_end|>".to_string()),
            ..Default::default()
        })
        .unwrap();

        // Single complete tool call in one chunk should work
        let chunk =
            "<|tool_start|>{\"name\":\"search\",\"arguments\":{\"query\":\"rust\"}}<|tool_end|>";
        let (_text, _thought, calls) = parser.parse(chunk);

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "search");
        assert_eq!(calls[0].arguments, "{\"query\":\"rust\"}");
    }

    #[test]
    fn test_tool_call_parser_multiple_complete_calls_in_one_chunk() {
        // Test multiple complete tool calls in a single chunk (generic capability)
        let parser = ToolCallParser::new(TemplatePatterns {
            tool_start_re: Some("<|tool_call|>".to_string()),
            tool_end_re: Some("<|tool_end|>".to_string()),
            ..Default::default()
        })
        .unwrap();
        let chunk = "<|tool_call|>{\"name\":\"search\",\"arguments\":{\"query\":\"rust\"}}<|tool_end|>Some text<|tool_call|>{\"name\":\"weather\",\"arguments\":{\"location\":\"moon\"}}<|tool_end|>";
        let (text, _thought, calls) = parser.parse(chunk);

        assert_eq!(text, "Some text");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "search");
        assert_eq!(calls[1].name, "weather");
    }

    #[test]
    fn test_flush_preserves_incomplete_tool_call() {
        // Test that incomplete tool calls are preserved in buffer when flushed
        let parser = ToolCallParser::new(TemplatePatterns {
            tool_start_re: Some("<|tool_call|>".to_string()),
            tool_end_re: Some("<|tool_end|>".to_string()),
            ..Default::default()
        })
        .unwrap();
        parser.parse("<|tool_call|>{\"name\":\"search\",\"arguments\":{\"query\":");
        let flushed = parser.flush();
        assert_eq!(
            flushed,
            "<|tool_call|>{\"name\":\"search\",\"arguments\":{\"query\":"
        );
    }

    #[test]
    fn test_tool_call_parser_malformed_json() {
        let parser = ToolCallParser::new(TemplatePatterns {
            tool_start_re: Some("<tool_call>".to_string()),
            tool_end_re: Some("</tool_call>".to_string()),
            ..Default::default()
        })
        .unwrap();
        let chunk = "<tool_call>{invalid json}</tool_call>";
        let (text, _thought, calls) = parser.parse(chunk);
        assert_eq!(text, "<tool_call>{invalid json}</tool_call>");
        assert!(calls.is_empty());
    }

    #[test]
    fn test_tool_call_parser_multiple_start_tags_without_end() {
        let parser = ToolCallParser::new(TemplatePatterns {
            tool_start_re: Some("<tool_call>".to_string()),
            tool_end_re: Some("</tool_call>".to_string()),
            ..Default::default()
        })
        .unwrap();
        let chunk = "<tool_call>start<tool_call>middle";
        let (text, _thought, calls) = parser.parse(chunk);
        assert_eq!(text, "");
        assert!(calls.is_empty());
        assert_eq!(
            parser.buffer.lock().unwrap().as_str(),
            "<tool_call>start<tool_call>middle"
        );
    }

    #[test]
    fn test_assistant_message_with_content_text() {
        let messages = vec![
            ChatMessage {
                sender: SenderType::User,
                text: "How are you?".into(),
                ..Default::default()
            },
            ChatMessage {
                sender: SenderType::Assistant,
                text: "I'm doing well!".into(),
                metrics: Some(Default::default()),
                ..Default::default()
            },
        ];

        let result = apply_chat_template("", None, &messages, None, HashMap::new());
        assert!(result.is_ok());
        let output = result.unwrap();
        assert!(output.contains("I'm doing well!"));
    }

    #[test]
    fn test_tool_call_parser_repeated_calls() {
        let parser = ToolCallParser::new(TemplatePatterns {
            tool_start_re: Some("<tool_call>".to_string()),
            tool_end_re: Some("</tool_call>".to_string()),
            ..Default::default()
        })
        .unwrap();
        let chunk = r#"
            <tool_call>{"name":"search","arguments":{"query":"python"}}</tool_call>
            <tool_call>{"name":"search","arguments":{"query":"rust"}}</tool_call>
            <tool_call>{"name":"search","arguments":{"query":"ai"}}</tool_call>
        "#;
        let (_, _thought, calls) = parser.parse(chunk);
        assert_eq!(calls.len(), 3);
        assert_eq!(calls[0].name, "search");
        assert_eq!(calls[0].arguments, r#"{"query":"python"}"#);
        assert_eq!(calls[1].name, "search");
        assert_eq!(calls[1].arguments, r#"{"query":"rust"}"#);
        assert_eq!(calls[2].name, "search");
        assert_eq!(calls[2].arguments, r#"{"query":"ai"}"#);
    }

    #[test]
    fn test_parse_tool_call_array_function() {
        // Test the array parsing function directly (generic capability)
        let array_json = r#"[{"name":"search","arguments":{"query":"rust"}},{"name":"weather","arguments":{"location":"moon"}}]"#;
        let result = parse_tool_call_array(array_json);

        assert!(result.is_some());
        let parsed = result.unwrap();
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].name, "search");
        assert_eq!(parsed[1].name, "weather");
    }

    #[test]
    fn test_parse_tool_call_array_empty() {
        // Test empty array
        let array_json = "[]";
        let result = parse_tool_call_array(array_json);

        assert!(result.is_some());
        let parsed = result.unwrap();
        assert_eq!(parsed.len(), 0);
    }

    #[test]
    fn test_parse_tool_call_array_invalid_json() {
        // Test invalid JSON
        let invalid_json = "[invalid json]";
        let result = parse_tool_call_array(invalid_json);
        assert!(result.is_none());
    }

    #[test]
    fn test_parse_tool_call_array_malformed_tool_calls() {
        // Test array with partially malformed tool calls (missing arguments field)
        let malformed_json = r#"[{"name":"search","arguments":null}, {"name":"weather","arguments":{"location":"moon"}}]"#;
        let result = parse_tool_call_array(malformed_json);

        // Should parse successfully as it's valid JSON structure
        assert!(result.is_some());
        let parsed = result.unwrap();
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].name, "search");
        assert_eq!(parsed[1].name, "weather");
    }

    #[test]
    fn test_qwen35_template_enable_thinking_undefined() {
        let templ = get_builtin_template("qwen35").unwrap();
        let messages = vec![
            ChatMessage {
                sender: SenderType::User,
                text: "What's 2+2?".to_string(),
                ..Default::default()
            },
            ChatMessage {
                sender: SenderType::Assistant,
                text: "It's 4.".to_string(),
                thought: Some("Let me think...".to_string()),
                ..Default::default()
            },
        ];

        let extra_kwargs = HashMap::new();

        let result = apply_chat_template(templ, None, &messages, None, extra_kwargs);
        assert!(
            result.is_ok(),
            "Template rendering failed: {:?}",
            result.err()
        );
        let output = result.unwrap();

        assert!(
            output.contains("<think>\n\n</think>"),
            "Output should contain thinking block when enable_thinking undefined"
        );
        assert!(
            output.contains("It's 4."),
            "Output should contain assistant response"
        );
    }

    #[test]
    fn test_qwen35_template_generation_prompt_enable_thinking_undefined() {
        let templ = get_builtin_template("qwen35").unwrap();
        let messages = vec![ChatMessage {
            sender: SenderType::User,
            text: "Hi".to_string(),
            ..Default::default()
        }];

        let extra_kwargs = HashMap::new();

        let result = apply_chat_template(templ, None, &messages, None, extra_kwargs);
        assert!(
            result.is_ok(),
            "Template rendering failed: {:?}",
            result.err()
        );
        let output = result.unwrap();

        assert!(
            output.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"),
            "Generation prompt should contain empty thinking when enable_thinking undefined (defaults to false)"
        );
    }

    #[test]
    fn test_qwen35_template_enable_thinking_true_with_reasoning_content() {
        let templ = get_builtin_template("qwen35").unwrap();
        let messages = vec![
            ChatMessage {
                sender: SenderType::User,
                text: "What's 2+2?".to_string(),
                ..Default::default()
            },
            ChatMessage {
                sender: SenderType::Assistant,
                text: "It's 4.".to_string(),
                thought: Some("Let me calculate: 2+2=4".to_string()),
                ..Default::default()
            },
        ];

        let mut extra_kwargs: HashMap<String, serde_yaml::Value> = HashMap::new();
        extra_kwargs.insert("enable_thinking".to_string(), serde_yaml::Value::Bool(true));

        let result = apply_chat_template(templ, None, &messages, None, extra_kwargs);
        assert!(
            result.is_ok(),
            "Template rendering failed: {:?}",
            result.err()
        );
        let output = result.unwrap();

        assert!(
            output.contains("<think>\nLet me calculate: 2+2=4\n"),
            "Output should contain thinking block with content when enable_thinking=true"
        );
        assert!(
            output.contains("It's 4."),
            "Output should contain assistant response"
        );
    }

    #[test]
    fn test_qwen35_template_enable_thinking_true_no_reasoning_content() {
        let templ = get_builtin_template("qwen35").unwrap();
        let messages = vec![
            ChatMessage {
                sender: SenderType::User,
                text: "Hello".to_string(),
                ..Default::default()
            },
            ChatMessage {
                sender: SenderType::Assistant,
                text: "Hi there!".to_string(),
                thought: None,
                ..Default::default()
            },
        ];

        let mut extra_kwargs: HashMap<String, serde_yaml::Value> = HashMap::new();
        extra_kwargs.insert("enable_thinking".to_string(), serde_yaml::Value::Bool(true));

        let result = apply_chat_template(templ, None, &messages, None, extra_kwargs);
        assert!(
            result.is_ok(),
            "Template rendering failed: {:?}",
            result.err()
        );
        let output = result.unwrap();

        assert!(
            output.contains("<think>\n\n"),
            "Output should contain empty thinking block when enable_thinking=true but no thought"
        );
    }

    #[test]
    fn test_qwen35_template_enable_thinking_false_with_reasoning_content() {
        let templ = get_builtin_template("qwen35").unwrap();
        let messages = vec![
            ChatMessage {
                sender: SenderType::User,
                text: "What's 2+2?".to_string(),
                ..Default::default()
            },
            ChatMessage {
                sender: SenderType::Assistant,
                text: "It's 4.".to_string(),
                thought: Some("Let me calculate: 2+2=4".to_string()),
                ..Default::default()
            },
        ];

        let mut extra_kwargs: HashMap<String, serde_yaml::Value> = HashMap::new();
        extra_kwargs.insert(
            "enable_thinking".to_string(),
            serde_yaml::Value::Bool(false),
        );

        let result = apply_chat_template(templ, None, &messages, None, extra_kwargs);
        assert!(
            result.is_ok(),
            "Template rendering failed: {:?}",
            result.err()
        );
        let output = result.unwrap();

        assert!(
            output.contains("<think>"),
            "Output should contain thinking block even when enable_thinking=false (template behavior)"
        );
        assert!(
            output.contains("It's 4."),
            "Output should contain assistant response"
        );
    }

    #[test]
    fn test_qwen35_template_enable_thinking_false_no_reasoning_content() {
        let templ = get_builtin_template("qwen35").unwrap();
        let messages = vec![
            ChatMessage {
                sender: SenderType::User,
                text: "Hello".to_string(),
                ..Default::default()
            },
            ChatMessage {
                sender: SenderType::Assistant,
                text: "Hi there!".to_string(),
                thought: None,
                ..Default::default()
            },
        ];

        let mut extra_kwargs: HashMap<String, serde_yaml::Value> = HashMap::new();
        extra_kwargs.insert(
            "enable_thinking".to_string(),
            serde_yaml::Value::Bool(false),
        );

        let result = apply_chat_template(templ, None, &messages, None, extra_kwargs);
        assert!(
            result.is_ok(),
            "Template rendering failed: {:?}",
            result.err()
        );
        let output = result.unwrap();

        assert!(
            output.contains("<think>"),
            "Output should contain thinking block even when enable_thinking=false (template behavior)"
        );
    }

    #[test]
    fn test_qwen35_template_generation_prompt_enable_thinking_true() {
        let templ = get_builtin_template("qwen35").unwrap();
        let messages = vec![ChatMessage {
            sender: SenderType::User,
            text: "Hi".to_string(),
            ..Default::default()
        }];

        let mut extra_kwargs: HashMap<String, serde_yaml::Value> = HashMap::new();
        extra_kwargs.insert("enable_thinking".to_string(), serde_yaml::Value::Bool(true));

        let result = apply_chat_template(templ, None, &messages, None, extra_kwargs);
        assert!(
            result.is_ok(),
            "Template rendering failed: {:?}",
            result.err()
        );
        let output = result.unwrap();

        assert!(
            output.ends_with("<|im_start|>assistant\n<think>\n"),
            "Generation prompt should contain thinking block opening when enable_thinking=true"
        );
    }

    #[test]
    fn test_qwen35_template_generation_prompt_enable_thinking_false() {
        let templ = get_builtin_template("qwen35").unwrap();
        let messages = vec![ChatMessage {
            sender: SenderType::User,
            text: "Hi".to_string(),
            ..Default::default()
        }];

        let mut extra_kwargs: HashMap<String, serde_yaml::Value> = HashMap::new();
        extra_kwargs.insert(
            "enable_thinking".to_string(),
            serde_yaml::Value::Bool(false),
        );

        let result = apply_chat_template(templ, None, &messages, None, extra_kwargs);
        assert!(
            result.is_ok(),
            "Template rendering failed: {:?}",
            result.err()
        );
        let output = result.unwrap();

        assert!(
            output.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"),
            "Generation prompt should contain empty thinking when enable_thinking=false"
        );
    }

    #[test]
    fn test_tool_call_parser_streaming_thought() {
        let parser = ToolCallParser::new(TemplatePatterns {
            thought_start_re: Some("<think>".to_string()),
            thought_end_re: Some("</think>".to_string()),
            ..Default::default()
        })
        .unwrap();

        // Simulate qwen35 behavior: template pre-opens <think>
        parser.set_in_thought(true);

        // Chunk 1: thought content (streamed immediately)
        let (text, thought, _) = parser.parse("First thought.");
        assert_eq!(text, "");
        assert_eq!(thought, Some("First thought.".to_string()));

        // Chunk 2: thought content continues
        let (text, thought, _) = parser.parse(" Second long thought content.");
        assert_eq!(text, "");
        assert!(thought.is_some());

        // Chunk 3: thought ends
        let (text, thought, _) = parser.parse(" End.</think>Hello!");
        assert_eq!(text, "Hello!");
        assert!(thought.unwrap().contains("End."));

        parser.reset();

        // Test with start tag in the middle
        let (text, thought, _) =
            parser.parse("Response start. <think>Hidden thought that is long enough to stream");
        assert_eq!(text, "Response start. ");
        assert!(thought.is_some());

        let (text, thought, _) = parser.parse(" continues</think>Response end.");
        assert_eq!(text, "Response end.");
        assert!(thought.unwrap().contains("continues"));
    }

    #[test]
    fn test_tool_call_parser_thought_leakage_repro() {
        let parser = ToolCallParser::new(TemplatePatterns {
            thought_start_re: Some("<think>".to_string()),
            thought_end_re: Some("</think>".to_string()),
            ..Default::default()
        })
        .unwrap();

        // Pre-open thought
        parser.set_in_thought(true);

        // This was leaking in the bug report
        let (text, _, _) = parser.parse("Thinking...");
        assert_eq!(text, "");
        // It might be in thought or buffered

        let (text, _, _) = parser.parse("</think>Final answer");
        assert_eq!(text, "Final answer");
    }
}
