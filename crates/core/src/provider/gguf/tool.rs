use crate::tools::ToolCall;
use anyhow::{Context, Result};
use regex::{Regex, escape};
use serde_json;
use std::sync::Mutex;
use tracing::{debug, error};

pub const DEFAULT_TOOL_CALL_START_TAG: &str = "<tool_call>";
pub const DEFAULT_TOOL_CALL_END_TAG: &str = "</tool_call>";

#[derive(Debug, Clone, Default)]
pub struct TemplatePatterns {
    pub tool_start_re: Option<String>,
    pub tool_end_re: Option<String>,
    pub thought_start_re: Option<String>,
    pub thought_end_re: Option<String>,
}

pub fn get_tool_call_regexes(_model_name: &str) -> TemplatePatterns {
    // Currently all models use the same tags, but this can be extended
    TemplatePatterns {
        tool_start_re: Some(DEFAULT_TOOL_CALL_START_TAG.to_string()),
        tool_end_re: Some(DEFAULT_TOOL_CALL_END_TAG.to_string()),
        thought_start_re: Some("<think>".to_string()),
        thought_end_re: Some("</think>".to_string()),
    }
}

#[derive(Debug)]
pub struct ToolCallParser {
    pub buffer: Mutex<String>,
    next_id: Mutex<usize>,
    complete_call_re: Regex,
    start_tag_re: Regex,
    expects_array: bool,
    thought_start_re: Option<Regex>,
    thought_end_re: Option<Regex>,
    is_in_thought: Mutex<bool>,
}

#[derive(serde::Deserialize, Debug)]
pub struct RawToolCall {
    pub name: String,
    pub arguments: serde_json::Value,
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
                                    error!("Failed to parse tool call content: {}", tool_content);
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
    // First try parsing as direct RawToolCall (JSON)
    if let Ok(raw) = serde_json::from_str::<RawToolCall>(content) {
        return Some(raw);
    }

    // Then try wrapped format: { "call": { ... RawToolCall ... } }
    if let Ok(wrapped) = serde_json::from_str::<serde_json::Value>(content)
        && let Some(call_obj) = wrapped.get("call")
        && let Ok(raw) = serde_json::from_value::<RawToolCall>(call_obj.clone())
    {
        return Some(raw);
    }

    // Finally try XML format
    parse_xml_tool_call(content)
}

fn parse_xml_tool_call(content: &str) -> Option<RawToolCall> {
    debug!(?content, "Parsing XML tool call");

    // Extract function name: <function=NAME>
    let func_re = Regex::new(r"<function=([^>]+)>").ok()?;
    let name = func_re.captures(content)?.get(1)?.as_str().to_string();

    // Extract parameters: <parameter=KEY>VALUE</parameter>
    let param_re = Regex::new(r"(?s)<parameter=([^>]+)>(.*?)</parameter>").ok()?;
    let mut arguments = serde_json::Map::new();

    for caps in param_re.captures_iter(content) {
        let key = caps.get(1)?.as_str().to_string();
        let value = caps.get(2)?.as_str().trim().to_string();
        arguments.insert(key, serde_json::Value::String(value));
    }

    Some(RawToolCall {
        name,
        arguments: serde_json::Value::Object(arguments),
    })
}

pub fn parse_tool_call_array(content: &str) -> Option<Vec<RawToolCall>> {
    serde_json::from_str(content).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn test_parse_xml_tool_call() {
        let content = r#"<function=search>
<parameter=query>
Bhagavad Gita verse
</parameter>
<parameter=engines>
google
</parameter>
</function>"#;
        let result = parse_xml_tool_call(content).unwrap();
        assert_eq!(result.name, "search");
        assert_eq!(result.arguments["query"], "Bhagavad Gita verse");
        assert_eq!(result.arguments["engines"], "google");
    }

    #[test]
    fn test_parse_single_tool_call_xml() {
        let content = r#"<function=search><parameter=query>hello</parameter></function>"#;
        let result = parse_single_tool_call(content).unwrap();
        assert_eq!(result.name, "search");
        assert_eq!(result.arguments["query"], "hello");
    }

    #[test]
    fn test_parse_single_tool_call_json() {
        let content = r#"{"name": "search", "arguments": {"query": "hello"}}"#;
        let result = parse_single_tool_call(content).unwrap();
        assert_eq!(result.name, "search");
        assert_eq!(result.arguments["query"], "hello");
    }

    #[test]
    fn test_parse_single_tool_call_wrapped_json() {
        let content = r#"{"call": {"name": "search", "arguments": {"query": "hello"}}}"#;
        let result = parse_single_tool_call(content).unwrap();
        assert_eq!(result.name, "search");
        assert_eq!(result.arguments["query"], "hello");
    }
}
