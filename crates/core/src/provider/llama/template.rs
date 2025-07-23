use crate::completion::{ChatMessage, SenderType};
use anyhow::{Result, anyhow};
use llama_cpp_2::model::{LlamaChatMessage, LlamaModel};

/// Converts ChatMessage to llama_cpp style messages
pub fn to_llama_messages(messages: &[ChatMessage]) -> Result<Vec<LlamaChatMessage>> {
    messages
        .iter()
        .map(|m| {
            let role = match m.sender {
                SenderType::System => "system",
                SenderType::User => "user",
                SenderType::Assistant => "assistant",
                SenderType::Tool => "tool",
            };
            LlamaChatMessage::new(role.to_string(), m.text.clone())
                .map_err(|e| anyhow!("Failed to create LlamaChatMessage for {}: {}", role, e))
        })
        .collect()
}

/// Applies chat template using the model's built-in template
pub fn apply_chat_template(model: &LlamaModel, messages: &[ChatMessage]) -> Result<String> {
    let llama_messages = to_llama_messages(messages)?;
    let tmpl = model
        .chat_template(None)
        .map_err(|e| anyhow!("Failed to retrieve default chat template: {}", e))?;
    model
        .apply_chat_template(&tmpl, &llama_messages, true)
        .map_err(|e| anyhow!("Failed to apply chat template to messages: {}", e))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::completion::{ChatMessage, SenderType};

    #[test]
    fn test_message_conversion() {
        let messages = vec![
            ChatMessage {
                sender: SenderType::System,
                text: "You are Kimi".into(),
                tools: vec![],
            },
            ChatMessage {
                sender: SenderType::User,
                text: "What's the weather?".into(),
                tools: vec![],
            },
            ChatMessage {
                sender: SenderType::Assistant,
                text: "I'll check".into(),
                tools: vec![],
            },
            ChatMessage {
                sender: SenderType::Tool,
                text: "Weather data".into(),
                tools: vec![],
            },
        ];

        let llama_msgs = to_llama_messages(&messages).unwrap();
        assert_eq!(llama_msgs.len(), 4);

        // Since LlamaChatMessage fields are private, we can only test the conversion succeeded
        // The actual content verification would require integration tests
    }

    // Remove the test that requires mocking LlamaModel since we can't properly mock it
    // The functionality is tested implicitly through integration tests
}
