use crate::{
    completion::{ChatMessage, SenderType},
    tools::ToolSpec,
};
use anyhow::{Context, Result};
use minijinja::{Environment, Error, ErrorKind, context, value::Value};
use minijinja_contrib::pycompat::unknown_method_callback;
use serde_json;
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

    // Allow python compatibility methods like startswith to be available
    env.set_unknown_method_callback(unknown_method_callback);
    env.add_filter("tojson", |v: Value| -> Result<String, Error> {
        serde_json::to_string(&v).map_err(|e| {
            Error::new(ErrorKind::InvalidOperation, "failed to convert to JSON").with_source(e)
        })
    });

    let template = if template_str.is_empty() {
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

    tmpl.render(context! { messages => &context_messages, tools => &context_tools })
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
        assert!(output.contains(
            "<|im_system|>system<|im_middle|>You are Kimi, an AI assistant created by Moonshot AI.<|im_end|>"
        ));
        assert!(output.contains("<|im_user|>user<|im_middle|>Hello<|im_end|>"));
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
                    arguments: "{\"location\": \"Boston\"}".to_string(),
                }],
            },
        ];

        let result = apply_chat_template(DEFAULT_TEMPLATE, &messages, None);
        assert!(result.is_ok(), "Result: {result:?}");
        let output = result.unwrap();
        assert!(output.contains(
            "<|tool_call_begin|>call_123<|tool_call_argument_begin|>{\"location\": \"Boston\"}<|tool_call_end|>"
        ));
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
        assert!(
            output.contains(
                "<|im_system|>tool<|im_middle|>{\"temperature\": 22, \"unit\": \"celsius\"}"
            ),
            "Output: {output}"
        );
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
        assert!(output.contains(
            "<|im_system|>system<|im_middle|>You are Kimi, an AI assistant created by Moonshot AI.<|im_end|>"
        ));
        assert!(output.contains("<|im_user|>user<|im_middle|>Hello<|im_end|>"));
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
                    arguments: "{\"location\": \"Boston\"}".to_string(),
                }],
            },
        ];

        let result = apply_chat_template("", &messages, None);
        assert!(result.is_ok(), "Result: {result:?}");
        let output = result.unwrap();
        assert!(output.contains(
            "<|tool_call_begin|>call_123<|tool_call_argument_begin|>{\"location\": \"Boston\"}<|tool_call_end|>"
        ));
    }
}
