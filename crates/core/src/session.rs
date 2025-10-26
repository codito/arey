//! A session is a shared context between a human and the AI assistant.
//! Context includes the conversation, shared artifacts, and tools.

use crate::{
    completion::{CancellationToken, ChatMessage, Completion, CompletionModel, SenderType},
    get_completion_llm,
    model::{ModelConfig, ModelMetrics, ModelProvider},
    provider::test_provider::TestProviderModel,
    tools::{Tool, ToolCall, ToolResult},
};
use anyhow::{Context, Result};
use futures::stream::BoxStream;
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, error};

/// A session with shared context between Human and AI model.
pub struct Session {
    model: Box<dyn CompletionModel>,
    model_key: String,
    context_size: usize,
    system_prompt: String,
    messages: Vec<(ChatMessage, Option<usize>)>,
    tools: Vec<Arc<dyn Tool>>,
    settings: HashMap<String, String>,
    metrics: Option<ModelMetrics>,
}

impl Session {
    /// Create a new session with the given model configuration
    pub fn new(model_config: ModelConfig, system_prompt: &str) -> Result<Self> {
        let model_key = model_config.key.clone();
        let model = get_completion_llm(model_config.clone())
            .context("Failed to initialize session model")?;

        let context_size = model_config.get_setting::<usize>("n_ctx").unwrap_or(4096);
        let metrics = Some(model.metrics());

        Ok(Self {
            model,
            model_key,
            context_size,
            system_prompt: system_prompt.to_string(),
            messages: Vec::new(),
            tools: Vec::new(),
            settings: HashMap::new(),
            metrics,
        })
    }

    /// Set a new system prompt for the session.
    pub fn set_system_prompt(&mut self, prompt: &str) -> Result<()> {
        self.system_prompt = prompt.to_string();
        Ok(())
    }

    /// Get model key for retrieving details from config
    pub fn model_key(&self) -> String {
        self.model_key.clone()
    }

    /// Get system prompt
    pub fn system_prompt(&self) -> String {
        self.system_prompt.clone()
    }

    /// Transfer existing message history to session
    pub fn set_messages(&mut self, messages: Vec<ChatMessage>) -> Result<()> {
        self.messages = messages
            .into_iter()
            .map(|msg| (msg, None)) // Reset token counts for new model
            .collect();
        Ok(())
    }

    /// Add a new message to the conversation history
    pub fn add_message(&mut self, message: ChatMessage) -> Result<()> {
        if message.sender != SenderType::Assistant {
            self.messages.push((message, None));
            return Ok(());
        }

        // Add assistant message. Assume assistant message is fully received.
        // Back-fill token counts for messages sent in the last prompt and adds the
        // assistant's response with its token count.
        let messages = &mut self.messages;

        let mut known_prompt_tokens: usize = 0;
        let mut unknown_message_indices = Vec::new();
        let mut total_unknown_chars = 0;

        // Identify messages that were part of the prompt and don't have a token count yet.
        for (i, (msg, count)) in messages.iter().enumerate() {
            if let Some(count) = *count {
                known_prompt_tokens += count;
            } else {
                unknown_message_indices.push(i);
                // Use character length as a heuristic for distribution.
                total_unknown_chars += msg.text.len();
            }
        }

        // Calculate new tokens added by the assistant's response.
        let metrics = message.clone().metrics.unwrap();
        let new_tokens = (metrics.prompt_tokens as usize).saturating_sub(known_prompt_tokens);

        if !unknown_message_indices.is_empty() {
            if total_unknown_chars > 0 {
                let mut distributed_tokens: usize = 0;
                let last_index = unknown_message_indices.len() - 1;

                // Distribute tokens proportionally, giving remainder to the last message.
                for (i, msg_idx) in unknown_message_indices.iter().enumerate() {
                    let msg_chars = messages[*msg_idx].0.text.len();
                    let msg_tokens = if i == last_index {
                        new_tokens.saturating_sub(distributed_tokens)
                    } else {
                        (new_tokens * msg_chars) / total_unknown_chars
                    };
                    messages[*msg_idx].1 = Some(msg_tokens);
                    distributed_tokens += msg_tokens;
                }
            } else if unknown_message_indices.len() == 1 {
                // Handle case where there's one message with no text.
                messages[unknown_message_indices[0]].1 = Some(new_tokens);
            }
        }

        messages.push((message, Some(metrics.completion_tokens as usize)));
        Ok(())
    }

    /// Set the tools available for this session
    pub fn set_tools(&mut self, tools: Vec<Arc<dyn Tool>>) -> Result<()> {
        self.tools = tools;
        Ok(())
    }

    /// Returns a clone of the tools available in the session.
    pub fn tools(&self) -> Vec<Arc<dyn Tool>> {
        self.tools.clone()
    }

    /// Replace the current model with a new one, ensuring proper cleanup
    pub fn set_model(&mut self, model_config: ModelConfig) -> Result<()> {
        let model_key = model_config.key.clone();

        // Create a temporary placeholder model to replace the old one
        let temp_model = Box::new(TestProviderModel::new(ModelConfig {
            key: "temp".to_string(),
            name: "temp".to_string(),
            provider: ModelProvider::Test,
            settings: HashMap::new(),
        })?) as Box<dyn CompletionModel>;

        // Replace and drop the old model to free GPU memory
        let old_model = std::mem::replace(&mut self.model, temp_model);
        drop(old_model);

        // Now create the new model with the old model's memory freed
        self.model = get_completion_llm(model_config)?;
        self.model_key = model_key;

        Ok(())
    }

    /// Update agent configuration (prompt and tools) atomically
    pub fn update_agent(&mut self, prompt: String, tools: Vec<Arc<dyn Tool>>) -> Result<()> {
        self.system_prompt = prompt;
        self.tools = tools;
        Ok(())
    }

    /// Update generation settings
    pub fn update_settings(&mut self, new_settings: HashMap<String, String>) -> Result<()> {
        self.settings = new_settings;
        Ok(())
    }

    /// Generate a response stream for the current conversation
    pub async fn generate(
        &self,
        settings: HashMap<String, String>,
        cancel_token: CancellationToken,
    ) -> Result<BoxStream<'static, Result<Completion>>> {
        let tool_slice = if self.tools.is_empty() {
            None
        } else {
            Some(self.tools.as_slice())
        };

        let max_tokens = settings
            .get("max_tokens")
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(1024);

        let messages_to_send =
            self.get_trimmed_messages(&self.messages, max_tokens, &self.system_prompt);

        Ok(self
            .model
            .complete(&messages_to_send, tool_slice, &settings, cancel_token)
            .await)
    }

    /// Clear the conversation history
    pub fn clear_history(&mut self) -> Result<()> {
        self.messages.clear();
        Ok(())
    }

    /// Get model metrics if available
    pub fn metrics(&self) -> Option<&ModelMetrics> {
        self.metrics.as_ref()
    }

    /// Get all messages in the conversation
    pub fn all_messages(&self) -> Vec<ChatMessage> {
        self.messages.iter().map(|(msg, _)| msg.clone()).collect()
    }

    /// Get the last message from the assistant
    pub fn last_assistant_message(&self) -> Option<ChatMessage> {
        self.messages
            .iter()
            .rev()
            .find(|(m, _)| m.sender == SenderType::Assistant)
            .map(|(m, _)| m.clone())
    }

    /// Trims a single, oversized block of messages to fit within the available token budget.
    ///
    /// This function works backwards from the end of the block, prioritizing newer messages.
    /// It aggressively truncates `Tool` messages if necessary to make the block fit, ensuring
    /// the returned list of messages strictly adheres to the token limit.
    fn trim_oversized_block(
        &self,
        block_messages: &[(ChatMessage, Option<usize>)],
        available_tokens: usize,
        estimate_tokens_fn: &dyn Fn(&ChatMessage) -> usize,
    ) -> Vec<ChatMessage> {
        let mut final_messages = Vec::new();
        let mut used_tokens = 0;

        // Iterate forwards through the block to preserve conversational order.
        for (msg, count) in block_messages.iter() {
            let mut msg_to_add = msg.clone();
            let mut tokens = count.unwrap_or_else(|| estimate_tokens_fn(&msg_to_add));

            if used_tokens + tokens > available_tokens {
                // If the message doesn't fit, check if we can truncate it.
                // We only truncate tool messages as they are the most likely to be verbose.
                if msg_to_add.sender == SenderType::Tool {
                    let budget_for_this_msg = available_tokens.saturating_sub(used_tokens);
                    // Use a 4 chars/token heuristic to determine how much text to keep.
                    let chars_to_keep = budget_for_this_msg * 4;
                    if msg_to_add.text.len() > chars_to_keep {
                        msg_to_add.text.truncate(chars_to_keep);
                        msg_to_add.text.push_str("\n... [truncated]");
                        // After truncation, the token cost is now the budget we had.
                        tokens = budget_for_this_msg;
                    } else {
                        // It's a tool message but can't be truncated enough. Stop.
                        break;
                    }
                } else {
                    // It's not a tool message and it doesn't fit. Stop.
                    break;
                }
            }

            // Add the message only if it fits (either originally or after truncation).
            if used_tokens + tokens <= available_tokens {
                used_tokens += tokens;
                final_messages.push(msg_to_add);
            } else {
                // This can happen if budget was 0 and truncation wasn't possible. Stop.
                break;
            }
        }

        debug!(
            "Trimmed block of {} messages to {} messages using {} tokens",
            block_messages.len(),
            final_messages.len(),
            used_tokens
        );

        final_messages
    }

    /// Returns a list of messages that fits within the model's context window.
    /// It trims older messages, attempting to keep conversational turns intact.
    fn get_trimmed_messages(
        &self,
        messages: &[(ChatMessage, Option<usize>)],
        max_output_tokens: usize,
        system_prompt: &str,
    ) -> Vec<ChatMessage> {
        // Leave room for the response. TODO: use the max_tokens parameter from completion model.
        let max_tokens = self.context_size - max_output_tokens;

        // Heuristic for messages without a known token count.
        let estimate_tokens = |msg: &ChatMessage| -> usize {
            let mut tokens = msg.text.len() / 4;
            if let Some(tools) = &msg.tools {
                for tool_call in tools {
                    tokens += tool_call.name.len() / 4;
                    if let Ok(input_str) = serde_json::to_string(&tool_call.arguments) {
                        tokens += input_str.len() / 4;
                    }
                }
            }
            tokens
        };

        let get_token_count = |(msg, count): &(ChatMessage, Option<usize>)| {
            count.unwrap_or_else(|| estimate_tokens(msg))
        };

        let system_prompt_tokens = if system_prompt.is_empty() {
            0
        } else {
            system_prompt.len() / 4
        };
        let available_tokens: usize = max_tokens.saturating_sub(system_prompt_tokens);
        debug!(
            "Token trimming: available tokens: {}, System prompt tokens: {}",
            available_tokens, system_prompt_tokens
        );

        let mut start_index = messages.len();
        let mut current_tokens = 0;
        let mut current_block_tokens = 0;

        for i in (0..messages.len()).rev() {
            let msg = &messages[i];
            current_block_tokens += get_token_count(msg);

            // A "block" is a User message followed by non-User messages.
            if msg.0.sender == SenderType::User || i == 0 {
                if current_tokens + current_block_tokens <= available_tokens {
                    current_tokens += current_block_tokens;
                    start_index = i;
                    current_block_tokens = 0;
                } else {
                    // This block is too big to fit with the others.
                    if start_index == messages.len() {
                        // This is the *most recent* block, and it's oversized.
                        // We cannot simply discard it, as that would mean sending an empty prompt.
                        // Instead, we must truncate it to fit the context window.
                        debug!(
                            "Token trimming: Most recent block is oversized ({} tokens > {} available). Truncating it.",
                            current_block_tokens, available_tokens
                        );
                        let oversized_block = &messages[i..];
                        return self.trim_oversized_block(
                            oversized_block,
                            available_tokens,
                            &estimate_tokens,
                        );
                    } else {
                        // This is an older block that doesn't fit. We stop here and
                        // only include the newer blocks that have already been approved.
                        break;
                    }
                }
            }
        }
        debug!("Token trimming: start index: {}", start_index);

        let mut final_messages: Vec<ChatMessage> = messages[start_index..]
            .iter()
            .map(|(m, _)| m.clone())
            .collect();

        // Truncate all but the last tool message to a reasonable limit.
        let last_tool_idx = final_messages
            .iter()
            .rposition(|m| m.sender == SenderType::Tool);

        if let Some(last_idx) = last_tool_idx {
            const MAX_TOOL_RESPONSE_CHARS: usize = 512;
            const TRUNCATION_MARKER: &str = "... [truncated]";

            for (i, msg) in final_messages.iter_mut().enumerate() {
                if i < last_idx
                    && msg.sender == SenderType::Tool
                    && msg.text.len() > MAX_TOOL_RESPONSE_CHARS
                {
                    // Deserialize or handle errors gracefully
                    let mut tool_output = match serde_json::from_str::<ToolResult>(
                        msg.text.as_str(),
                    ) {
                        Ok(t) => t,
                        Err(e) => {
                            error!(
                                "Token trimming: Failed to deserialize ToolResult from message text: {}. Error: {}",
                                msg.text, e
                            );
                            ToolResult {
                                call: ToolCall {
                                    id: "invalid".to_string(),
                                    name: "invalid".to_string(),
                                    arguments: "{}".to_string(),
                                },
                                output: serde_json::Value::String(msg.text.clone()),
                            }
                        }
                    };

                    // Serialize output field only
                    let mut content = match serde_json::to_string(&tool_output.output) {
                        Ok(c) => c,
                        Err(e) => {
                            error!(
                                "Token trimming: Failed to serialize tool output: {:?}. Error: {}",
                                tool_output.output, e
                            );
                            "[Tool output not serializable]".to_string()
                        }
                    };
                    if content.len() > MAX_TOOL_RESPONSE_CHARS {
                        let mut new_len = MAX_TOOL_RESPONSE_CHARS;

                        // Ensure we don't cut off in the middle of a UTF-8 character.
                        while !content.is_char_boundary(new_len) {
                            new_len -= 1;
                        }
                        content.truncate(new_len);
                        content.push_str(TRUNCATION_MARKER);
                    }

                    // Update output field and serialize full ToolResult
                    tool_output.output = serde_json::Value::String(content);
                    msg.text = match serde_json::to_string(&tool_output) {
                        Ok(s) => s,
                        Err(e) => {
                            error!(
                                "Token trimming: Failed to serialize ToolResult: {:?}. Error: {}",
                                tool_output, e
                            );
                            "[Tool output not available]".to_string()
                        }
                    };
                }
            }
        }

        // Add system prompt at beginning of the messages
        if !system_prompt.is_empty() {
            final_messages.insert(
                0,
                ChatMessage {
                    sender: SenderType::System,
                    text: system_prompt.to_string(),
                    ..Default::default()
                },
            );
        }
        final_messages
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        completion::{Completion, CompletionMetrics},
        model::ModelProvider,
        tools::{ToolCall, ToolResult},
    };
    use async_trait::async_trait;
    use serde_json::Value;
    use std::collections::HashMap;

    // A mock CompletionModel for testing purposes.
    struct MockModel {}

    impl MockModel {
        fn new() -> Self {
            Self {}
        }
    }

    #[async_trait]
    impl CompletionModel for MockModel {
        fn metrics(&self) -> ModelMetrics {
            ModelMetrics {
                init_latency_ms: 0.0,
            }
        }
        async fn complete(
            &self,
            _messages: &[ChatMessage],
            _tools: Option<&[Arc<dyn Tool>]>,
            _settings: &HashMap<String, String>,
            _cancel_token: CancellationToken,
        ) -> BoxStream<'static, Result<Completion>> {
            let stream = futures::stream::iter(Vec::<Result<Completion>>::new());
            Box::pin(stream)
        }
    }

    fn new_chat_msg(sender: SenderType, text: &str) -> ChatMessage {
        ChatMessage {
            sender,
            text: text.to_string(),
            ..Default::default()
        }
    }

    fn new_assistant_msg(
        text: &str,
        metrics: CompletionMetrics,
        tools: Vec<ToolCall>,
    ) -> ChatMessage {
        ChatMessage {
            sender: SenderType::Assistant,
            text: text.to_string(),
            tools: Some(tools),
            metrics: Some(metrics),
        }
    }

    fn new_session(context_size: usize) -> Session {
        Session {
            model: Box::new(MockModel::new()),
            model_key: "test-model".to_string(),
            context_size,
            system_prompt: "System".to_string(), // 6 chars -> 1 token est.
            messages: Vec::new(),
            tools: Vec::new(),
            settings: HashMap::new(),
            metrics: Some(ModelMetrics {
                init_latency_ms: 0.0,
            }),
        }
    }

    fn new_tool_call(name: &str) -> ToolCall {
        ToolCall {
            id: "{name}_id".to_string(),
            name: name.to_string(),
            arguments: "{}".to_string(),
        }
    }

    #[test]
    fn test_add_message() {
        let mut session = new_session(100);

        let _ = session.add_message(new_chat_msg(SenderType::User, "Hello"));

        assert_eq!(session.messages.len(), 1);
        assert_eq!(session.messages[0].0.text, "Hello");
        assert_eq!(session.messages[0].1, None);
    }

    #[tokio::test]
    async fn test_add_message_assistant_simple() {
        let mut session = new_session(100);
        let _ = session.add_message(new_chat_msg(SenderType::User, "User 1"));

        let metrics = CompletionMetrics {
            prompt_tokens: 10,
            completion_tokens: 20,
            ..Default::default()
        };
        let _ = session.add_message(new_assistant_msg("Response 1", metrics, Vec::new()));

        assert_eq!(session.messages.len(), 2);
        assert_eq!(session.messages[0].1, Some(10));
        assert_eq!(session.messages[1].1, Some(20));
    }

    #[tokio::test]
    async fn test_add_message_assistant_multiple_unknown() {
        let mut session = new_session(100);
        let _ = session.add_message(new_chat_msg(
            SenderType::User,
            "This is a slightly longer message.",
        )); // 34 chars
        let _ = session.add_message(new_chat_msg(SenderType::User, "Short one.")); // 10 chars

        let metrics = CompletionMetrics {
            prompt_tokens: 100,
            completion_tokens: 20,
            ..Default::default()
        };
        let _ = session.add_message(new_assistant_msg("Response 1", metrics, Vec::new()));
        assert_eq!(session.messages.len(), 3);
        // 100 * (34 / (34+10)) = 77.27 -> 77
        assert_eq!(session.messages[0].1, Some(77));
        // Remainder: 100 - 77 = 23
        assert_eq!(session.messages[1].1, Some(23));
        assert_eq!(session.messages[2].1, Some(20));
    }

    #[test]
    fn test_get_trimmed_messages_no_trimming() {
        let session = new_session(100);
        let messages = vec![
            (new_chat_msg(SenderType::User, "U1"), Some(10)),
            (new_chat_msg(SenderType::Assistant, "A1"), Some(10)),
        ];
        let system_prompt = session.system_prompt.clone();
        let trimmed = session.get_trimmed_messages(&messages, 0, &system_prompt);

        assert_eq!(trimmed.len(), 3);
    }

    #[test]
    fn test_get_trimmed_messages_simple_trim() {
        let session = new_session(30); // small context
        let messages = vec![
            (new_chat_msg(SenderType::User, "U1"), Some(10)),
            (new_chat_msg(SenderType::Assistant, "A1"), Some(10)),
            (new_chat_msg(SenderType::User, "U2"), Some(10)),
        ];
        let system_prompt = session.system_prompt.clone();
        // available_tokens = 30-1=29.
        // U2 block (10) fits. current_tokens=10.
        // U1 block (20) doesn't fit with U2 (10+20 > 29). Trim U1 block.
        let trimmed = session.get_trimmed_messages(&messages, 0, &system_prompt);

        assert_eq!(trimmed.len(), 2);
        assert_eq!(trimmed[1].text, "U2");
    }

    #[test]
    fn test_get_trimmed_messages_keeps_blocks() {
        let session = new_session(25); // Very small context to force trimming
        let messages = vec![
            (new_chat_msg(SenderType::User, "U1"), Some(10)),
            (new_chat_msg(SenderType::Assistant, "A1"), Some(10)),
            (new_chat_msg(SenderType::User, "U2"), Some(10)),
        ];
        let system_prompt = session.system_prompt.clone();
        // available_tokens = 25-1=24.
        // U2 block (10) fits. U1+A1 block (20) doesn't fit with U2 (10+20 > 24).
        // Should keep only U2 block (most recent).
        let trimmed = session.get_trimmed_messages(&messages, 0, &system_prompt);

        assert_eq!(trimmed.len(), 2);
        assert_eq!(trimmed[1].text, "U2");
    }

    #[test]
    fn test_get_trimmed_messages_with_system_prompt() {
        let session = new_session(30); // smaller context to force trimming
        let system_prompt = "System prompt with much more text to consume tokens";
        let messages = vec![
            (new_chat_msg(SenderType::User, "U1"), Some(10)),
            (new_chat_msg(SenderType::Assistant, "A1"), Some(10)),
            (new_chat_msg(SenderType::User, "U2"), Some(10)),
        ];
        // available_tokens = 30 - (system_prompt.len() / 4) = 30 - (48/4) = 30 - 12 = 18.
        // U2 block (10) fits. U1+A1 block (20) doesn't fit with U2 (10+20 > 18).
        // Should keep only U2 block.
        let trimmed = session.get_trimmed_messages(&messages, 0, system_prompt);

        assert_eq!(trimmed.len(), 2);
        assert_eq!(trimmed[1].text, "U2");
    }

    #[test]
    fn test_get_trimmed_messages_exact_fit() {
        let session = new_session(21); // exact fit for U2
        let messages = vec![
            (new_chat_msg(SenderType::User, "U1"), Some(20)),
            (new_chat_msg(SenderType::Assistant, "A1"), Some(20)),
            (new_chat_msg(SenderType::User, "U2"), Some(10)),
        ];
        let system_prompt = session.system_prompt.clone();
        // available_tokens = 21-1=20.
        // U2 block (10) fits. current_tokens=10.
        // U1 block (20) doesn't fit with U2 (10+20 > 20). Trim U1 block.
        let trimmed = session.get_trimmed_messages(&messages, 0, &system_prompt);

        assert_eq!(trimmed.len(), 2);
        assert_eq!(trimmed[1].text, "U2");
    }

    #[test]
    fn test_get_trimmed_messages_empty_input() {
        let session = new_session(30);
        let messages = Vec::new();
        let system_prompt = session.system_prompt.clone();

        let trimmed = session.get_trimmed_messages(&messages, 0, &system_prompt);

        // Only system prompt
        assert_eq!(trimmed.len(), 1);
    }

    #[test]
    fn test_get_trimmed_messages_all_fit() {
        let session = new_session(100); // large context
        let messages = vec![
            (new_chat_msg(SenderType::User, "U1"), Some(10)),
            (new_chat_msg(SenderType::Assistant, "A1"), Some(10)),
            (new_chat_msg(SenderType::User, "U2"), Some(10)),
        ];
        let system_prompt = session.system_prompt.clone();
        // available_tokens = 100-1=99.
        // All messages fit easily.
        let trimmed = session.get_trimmed_messages(&messages, 0, &system_prompt);
        assert_eq!(trimmed.len(), 4);
        assert_eq!(trimmed[1].text, "U1");
        assert_eq!(trimmed[2].text, "A1");
        assert_eq!(trimmed[3].text, "U2");
    }

    #[test]
    fn test_full_flow() {
        let mut session = new_session(35); // Reduced from 50 to force trimming

        // Turn 1
        let _ = session.add_message(new_chat_msg(
            SenderType::User,
            "User message 1, quite long to ensure it costs something",
        ));
        let metrics1 = CompletionMetrics {
            prompt_tokens: 20,
            completion_tokens: 10,
            ..Default::default()
        };
        let _ = session.add_message(new_assistant_msg(
            "Assistant response 1",
            metrics1,
            Vec::new(),
        ));
        assert_eq!(session.messages[0].1, Some(20));
        assert_eq!(session.messages[1].1, Some(10));

        // Turn 2
        let _ = session.add_message(new_chat_msg(SenderType::User, "User message 2")); // est. 3 tokens
        let messages_to_send: Vec<(ChatMessage, Option<usize>)> = session
            .messages
            .iter()
            .map(|(msg, tokens)| (msg.clone(), *tokens))
            .collect();
        let system_prompt = session.system_prompt.clone();
        let trimmed = session.get_trimmed_messages(&messages_to_send, 0, &system_prompt);
        // Sys(1)+U1_block(30)+U2_est(3) = 34 > 35. All still fit.
        assert_eq!(trimmed.len(), 4);

        let metrics2 = CompletionMetrics {
            prompt_tokens: 35, // 20+10 known, so U2 cost is 5
            completion_tokens: 5,
            ..Default::default()
        };
        let _ = session.add_message(new_assistant_msg(
            "Assistant response 2",
            metrics2,
            Vec::new(),
        ));
        assert_eq!(session.messages[2].1, Some(5));
        assert_eq!(session.messages[3].1, Some(5));

        // Turn 3: force trimming - with context of 35, system (1) + available = 34 tokens
        // Messages: U1(20) + A1(10) + U2(5) + A2(5) + U3(est.7) = 47 tokens total
        // Expected to trim U1+A1 block (30 tokens), keeping U2+A2+U3 (17 tokens)
        let _ = session.add_message(new_chat_msg(
            SenderType::User,
            "User message 3 is also quite long",
        )); // est. 7 tokens
        let messages_to_send: Vec<(ChatMessage, Option<usize>)> = session
            .messages
            .iter()
            .map(|(msg, tokens)| (msg.clone(), *tokens))
            .collect();
        let system_prompt = session.system_prompt.clone();
        let trimmed = session.get_trimmed_messages(&messages_to_send, 0, &system_prompt);

        // Should have trimmed the oldest block (U1+A1) to fit within context
        // Keeping: U2(5) + A2(5) + U3(est.7) = 17 tokens + system(1) = 18 tokens total
        assert_eq!(trimmed.len(), 4); // Sys prompt, U2, A2, U3 - U1+A1 was trimmed
        assert_eq!(trimmed[1].text, "User message 2"); // First message should be U2
        assert_eq!(trimmed[2].text, "Assistant response 2"); // Second message should be A2
        assert_eq!(trimmed[3].text, "User message 3 is also quite long"); // Third message should be U3
    }

    #[test]
    fn test_trim_messages_with_tool_calls() {
        let mut session = new_session(200); // Larger context to fit all messages
        let _ = session.add_message(new_chat_msg(SenderType::User, "Use the search tool"));
        let _ = session.add_message(new_assistant_msg(
            "I'll search for that information.",
            CompletionMetrics::default(),
            Vec::new(),
        ));

        // Simulate tool calls in the assistant message
        session.messages[1].0.tools =
            Some(vec![new_tool_call("search"), new_tool_call("calculate")]);

        let _ = session.add_message(new_chat_msg(SenderType::Tool, "Search result"));
        let _ = session.add_message(new_chat_msg(SenderType::Tool, "Calculation result"));

        // Get trimmed messages with limited context
        let messages_to_send: Vec<(ChatMessage, Option<usize>)> = session
            .messages
            .iter()
            .map(|(msg, tokens)| (msg.clone(), *tokens))
            .collect();
        let system_prompt = session.system_prompt.clone();
        let trimmed = session.get_trimmed_messages(&messages_to_send, 0, &system_prompt);
        // Should include all messages since they fit in context
        assert_eq!(trimmed.len(), 5);
    }

    #[test]
    fn test_trim_messages_oversized_tool_call() {
        let mut session = new_session(50); // Very small context to force trimming
        let _ = session.add_message(new_chat_msg(SenderType::User, "Use the search tool"));
        let _ = session.add_message(new_assistant_msg(
            "I'll search for that information.",
            CompletionMetrics::default(),
            Vec::new(),
        ));

        // Simulate a large tool call that exceeds context
        session.messages[1].0.tools = Some(vec![new_tool_call("search")]);

        let _ = session.add_message(new_chat_msg(SenderType::Tool, "Search result"));

        // Add an extra message that should be trimmed
        let _ = session.add_message(new_chat_msg(SenderType::User, "Follow up question"));

        // Get trimmed messages with limited context
        let messages_to_send: Vec<(ChatMessage, Option<usize>)> = session
            .messages
            .iter()
            .map(|(msg, tokens)| (msg.clone(), *tokens))
            .collect();
        let system_prompt = session.system_prompt.clone();
        let trimmed = session.get_trimmed_messages(&messages_to_send, 0, &system_prompt);
        // Should trim the oldest user message to fit the tool call
        assert!(trimmed.len() >= 3); // At least the assistant, tool, and latest user message
    }

    #[test]
    fn test_trim_messages_with_invalid_tool_result() {
        let mut session = new_session(100);
        let _ = session.add_message(new_chat_msg(SenderType::User, "Use the search tool"));
        let _ = session.add_message(new_assistant_msg(
            "I'll search for that information.",
            CompletionMetrics::default(),
            Vec::new(),
        ));

        // Simulate tool calls in the assistant message
        session.messages[1].0.tools = Some(vec![new_tool_call("search")]);

        // Add a tool result with invalid JSON (make it long enough to trigger truncation)
        let invalid_json = format!(
            r#"{{"call":{{"id":"search_id","name":"search","arguments":"{{}}"}},"output":""{}""#,
            "invalid json ".repeat(100)
        );
        let _ = session.add_message(new_chat_msg(SenderType::Tool, &invalid_json));

        // Add another tool result (this will be the last one, forcing the first to be truncated)
        let _ = session.add_message(new_chat_msg(SenderType::Tool, "Second tool result"));

        // Get trimmed messages
        let messages_to_send: Vec<(ChatMessage, Option<usize>)> = session
            .messages
            .iter()
            .map(|(msg, tokens)| (msg.clone(), *tokens))
            .collect();
        let system_prompt = session.system_prompt.clone();
        let trimmed = session.get_trimmed_messages(&messages_to_send, 0, &system_prompt);
        // Should handle invalid JSON gracefully
        assert_eq!(trimmed.len(), 4);

        // Check if the invalid tool result was truncated
        assert!(
            trimmed[2].text.contains("... [truncated]"),
            "Expected tool result to be truncated"
        );
    }

    #[test]
    fn test_record_completion_with_tools() {
        let mut session = new_session(100);
        let _ = session.add_message(new_chat_msg(SenderType::User, "Use the search tool"));

        let metrics = CompletionMetrics {
            prompt_tokens: 10,
            completion_tokens: 20,
            ..Default::default()
        };
        let assistant_message = new_assistant_msg(
            "I'll search for that information.",
            metrics,
            vec![new_tool_call("search")],
        );

        let _ = session.add_message(assistant_message);

        assert_eq!(session.messages.len(), 2);
        assert_eq!(session.messages[1].1, Some(20));
        assert_eq!(session.messages[1].0.tools.as_ref().unwrap().len(), 1);
    }

    #[test]
    fn test_trim_messages_extremely_large_tool_result() {
        let mut session = new_session(300); // Very small context to force truncation
        let _ = session.add_message(new_chat_msg(SenderType::User, "Use the search tool"));

        let assistant_message = new_assistant_msg(
            "I'll search for that information.",
            CompletionMetrics::default(),
            vec![new_tool_call("search")],
        );

        let _ = session.add_message(assistant_message);

        // Create a very large tool result
        let large_tool_result =
            r#"{"call":{"id":"search_id","name":"search","arguments":"{}"},"output":""#.to_string()
                + &"x".repeat(1000)
                + r#""#;
        let tool_message = new_chat_msg(SenderType::Tool, &large_tool_result);
        let _ = session.add_message(tool_message);

        // Create another tool result (this will be the last one and won't be truncated)
        let _ = session.add_message(new_chat_msg(SenderType::Tool, "Small result"));

        // Create another message
        // let _ = session.add_message(new_chat_msg(SenderType::User, "Follow up"));

        // Get trimmed messages
        let messages_to_send: Vec<(ChatMessage, Option<usize>)> = session
            .messages
            .iter()
            .map(|(msg, tokens)| (msg.clone(), *tokens))
            .collect();
        let system_prompt = session.system_prompt.clone();
        let trimmed = session.get_trimmed_messages(&messages_to_send, 0, &system_prompt);
        // Check that it didn't panic and the tool message was truncated correctly.
        // The tool result is very large (1070 chars) and should be truncated to 512 chars
        assert!(
            trimmed[3].text.contains("... [truncated]"),
            "Expected tool result to be truncated. Text: {}...",
            &trimmed[3].text[..100.min(trimmed[2].text.len())]
        );
        assert!(!trimmed[4].text.contains("... [truncated]")); // Follow-up message should not be truncated
    }

    #[test]
    fn test_session_new() {
        let model_config = ModelConfig {
            key: "test_key".to_string(),
            name: "test_model".to_string(),
            provider: ModelProvider::Test,
            settings: HashMap::new(),
        };
        let session = Session::new(model_config, "test_system_prompt").unwrap();
        assert_eq!(session.model_key, "test_key");
        assert_eq!(session.context_size, 4096); // default value
        assert_eq!(session.system_prompt, "test_system_prompt");
        assert!(session.messages.is_empty());
        assert!(session.tools.is_empty());
        assert!(session.settings.is_empty());
    }

    #[test]
    fn test_clear_history() {
        let mut session = new_session(100);
        let _ = session.add_message(new_chat_msg(SenderType::User, "Hello"));
        let _ = session.add_message(new_assistant_msg(
            "Hi there",
            CompletionMetrics::default(),
            Vec::new(),
        ));
        assert_eq!(session.messages.len(), 2);

        session.clear_history().unwrap();
        assert_eq!(session.messages.len(), 0);
    }

    #[test]
    fn test_settings() {
        let mut session = new_session(100);
        session
            .settings
            .insert("temperature".to_string(), "0.7".to_string());
        session
            .settings
            .insert("top_p".to_string(), "0.9".to_string());

        assert_eq!(
            session.settings.get("temperature"),
            Some(&"0.7".to_string())
        );
        assert_eq!(session.settings.get("top_p"), Some(&"0.9".to_string()));
    }

    #[test]
    fn test_tool_response_truncation_with_multibyte_chars() {
        // This test ensures that truncating a tool response with multi-byte characters
        // does not panic by cutting a character in half.
        let mut session = new_session(4096);

        // Construct a string where the truncation boundary (512) falls within a multi-byte character.
        // `content` will be `"` + `long_string` + `"`.
        // The euro sign '€' is 3 bytes and starts at byte index 511 of `content`.
        let long_string = "a".to_string().repeat(510) + "€"; // 513 bytes as a string slice.

        let tool_outputs = vec![
            ToolResult {
                call: new_tool_call("tool_1"),
                output: Value::String(long_string),
            },
            ToolResult {
                call: new_tool_call("tool_2"),
                output: Value::String("b".to_string().repeat(10)), // A dummy second tool call
            },
        ];

        let serialized_outputs: Vec<String> = tool_outputs
            .iter()
            .map(|out| serde_json::to_string(out).unwrap())
            .collect();

        let _ = session.add_message(new_chat_msg(SenderType::User, "U1"));
        let _ = session.add_message(new_chat_msg(SenderType::Tool, &serialized_outputs[0]));
        let _ = session.add_message(new_chat_msg(SenderType::User, "U2"));
        let _ = session.add_message(new_chat_msg(SenderType::Tool, &serialized_outputs[1]));

        // Get trimmed messages using the same pattern as other tests
        let messages_to_send: Vec<(ChatMessage, Option<usize>)> = session
            .messages
            .iter()
            .map(|(msg, tokens)| (msg.clone(), *tokens))
            .collect();
        let system_prompt = session.system_prompt.clone();
        let trimmed = session.get_trimmed_messages(&messages_to_send, 0, &system_prompt);

        // Check that it didn't panic and the first message was truncated correctly.
        assert!(trimmed[2].text.contains("... [truncated]"));
        assert!(!trimmed[4].text.contains("... [truncated]"));
    }

    #[test]
    fn test_set_model() -> Result<()> {
        let mut session = new_session(100);
        let original_model_key = session.model_key();
        assert_eq!(original_model_key, "test-model");

        // Create a new model config with different key
        let new_model_config = ModelConfig {
            key: "new-test-model".to_string(),
            name: "new-test-model".to_string(),
            provider: ModelProvider::Test,
            settings: HashMap::new(),
        };

        // Set new model
        session.set_model(new_model_config)?;

        // Verify model was updated
        assert_eq!(session.model_key(), "new-test-model");

        Ok(())
    }

    #[test]
    fn test_set_model_preserves_session_data() -> Result<()> {
        let mut session = new_session(100);

        // Add some data to the session
        session.add_message(new_chat_msg(SenderType::User, "Hello"))?;
        session.add_message(new_assistant_msg(
            "Hi there",
            CompletionMetrics::default(),
            Vec::new(),
        ))?;
        let original_prompt = session.system_prompt();

        // Create a new model config
        let new_model_config = ModelConfig {
            key: "new-test-model".to_string(),
            name: "new-test-model".to_string(),
            provider: ModelProvider::Test,
            settings: HashMap::new(),
        };

        // Set new model
        session.set_model(new_model_config)?;

        // Verify session data is preserved
        assert_eq!(session.model_key(), "new-test-model");
        assert_eq!(session.system_prompt(), original_prompt);
        assert_eq!(session.all_messages().len(), 2);
        assert_eq!(session.all_messages()[0].text, "Hello");
        assert_eq!(session.all_messages()[1].text, "Hi there");

        Ok(())
    }

    #[test]
    fn test_set_model_with_gguf_config() -> Result<()> {
        let mut session = new_session(100);

        // Create a GGUF model config (even if path doesn't exist, we test the flow)
        let mut gguf_settings = HashMap::new();
        gguf_settings.insert("path".to_string(), "/nonexistent/model.gguf".into());

        let gguf_config = ModelConfig {
            key: "gguf-test-model".to_string(),
            name: "gguf-test-model".to_string(),
            provider: ModelProvider::Gguf,
            settings: gguf_settings,
        };

        // This should fail gracefully due to missing file, but the logic should work
        let result = session.set_model(gguf_config);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Model loading failed")
        );

        // Original model should still be intact
        assert_eq!(session.model_key(), "test-model");

        Ok(())
    }
}
