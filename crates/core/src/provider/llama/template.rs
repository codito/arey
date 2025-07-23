use crate::completion::{ChatMessage, SenderType};
use anyhow::{Result, anyhow};
use llama_cpp_2::model::{LlamaChatMessage, LlamaModel};

/// Applies chat template using the model's built-in template
pub fn apply_chat_template(model: &LlamaModel, messages: &[ChatMessage]) -> Result<String> {
    let llama_messages: Vec<LlamaChatMessage> = messages
        .iter()
        .map(|m| {
            let role = match m.sender {
                SenderType::System => "system",
                SenderType::User => "user",
                SenderType::Assistant => "assistant",
                SenderType::Tool => "tool",
            };
            // Convert role to String and use the message text
            LlamaChatMessage::new(role.to_string(), m.text.clone())
                .map_err(|e| anyhow!("Failed to create LlamaChatMessage for {}: {}", role, e))
        })
        .collect::<Result<Vec<_>>>()?;

    // Get the model's default chat template
    let tmpl = model
        .chat_template(None)
        .map_err(|e| anyhow!("Failed to retrieve default chat template: {}", e))?;

    // Apply the template
    model
        .apply_chat_template(&tmpl, &llama_messages, true)
        .map_err(|e| anyhow!("Failed to apply chat template to messages: {}", e))
}
