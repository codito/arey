//! Chat template handling for GGUF models
use crate::{
    completion::{ChatMessage, SenderType},
    tools::ToolSpec,
};
use anyhow::{Context, Result};
use minijinja::{Environment, Error, State, value::Value};
use serde_json;
use serde_yaml;
use std::collections::HashMap;
use tracing::{debug, error};

pub use crate::provider::gguf::tool::{ToolCallParser, get_tool_call_regexes};

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

            let tool_calls = msg.tools.as_ref().map(|tools| {
                tools
                    .iter()
                    .map(|tc| {
                        let mut tc_val =
                            serde_json::to_value(tc).unwrap_or(serde_json::Value::Null);
                        if let Some(args_str) = tc_val.get("arguments").and_then(|a| a.as_str())
                            && let Ok(args_obj) =
                                serde_json::from_str::<serde_json::Value>(args_str)
                        {
                            tc_val["arguments"] = args_obj;
                        }
                        tc_val
                    })
                    .collect::<Vec<_>>()
            });

            serde_json::json!({
                "role": match msg.sender {
                    SenderType::System => "system",
                    SenderType::User => "user",
                    SenderType::Assistant => "assistant",
                    SenderType::Tool => "tool",
                },
                "content": &msg.text,
                "reasoning_content": reasoning_content,
                "tool_calls": tool_calls,
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
        assert!(output.contains(r#""arguments": {"location": "Boston"}"#));
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
        assert!(output.contains(r#""arguments": {"location": "Boston"}"#));
        assert!(output.contains("</tool_call>"));
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
    fn test_qwen35_template_tool_call() {
        let templ = get_builtin_template("qwen35").unwrap();
        let messages = vec![
            ChatMessage {
                sender: SenderType::User,
                text: "What's the weather in Boston?".into(),
                ..Default::default()
            },
            ChatMessage {
                sender: SenderType::Assistant,
                text: "Let me check.".into(),
                thought: Some("I need to check the weather.".into()),
                tools: Some(vec![ToolCall {
                    id: "call_123".to_string(),
                    name: "get_weather".to_string(),
                    arguments: "{\"location\": \"Boston\", \"unit\": \"celsius\"}".to_string(),
                }]),
                metrics: Some(Default::default()),
            },
        ];

        let result = apply_chat_template(templ, None, &messages, None, HashMap::new());
        assert!(result.is_ok(), "Result: {result:?}");
        let output = result.unwrap();

        assert!(output.contains("<tool_call>"));
        assert!(output.contains("<function=get_weather>"));
        assert!(output.contains("<parameter=location>\nBoston\n</parameter>"));
        assert!(output.contains("<parameter=unit>\ncelsius\n</parameter>"));
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
}
