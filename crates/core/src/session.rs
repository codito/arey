//! A session is a shared context between a human and the AI assistant.
//! Context includes the conversation, shared artifacts, and tools.
use crate::model::ModelConfig;

use crate::tools::{ToolCall, ToolResult};
use crate::{
    completion::{
        CancellationToken, ChatMessage, Completion, CompletionMetrics, CompletionModel, SenderType,
    },
    model::ModelMetrics,
    tools::Tool,
};
use anyhow::{Context, Result};
use futures::stream::BoxStream;
use std::{collections::HashMap, sync::Arc};
use tracing::{debug, error};

/// A session with shared context between Human and AI model.
pub struct Session {
    model: Box<dyn CompletionModel + Send + Sync>,
    // Model key is used to identify the model in config.
    model_key: String,
    context_size: usize,
    system_prompt: String,
    // Store messages with their optional token count.
    messages: Vec<(ChatMessage, Option<usize>)>,
    tools: Vec<Arc<dyn Tool>>,
    metrics: Option<ModelMetrics>,
}

impl Session {
    /// Create a new session with the given model configuration
    pub async fn new(model_config: ModelConfig, system_prompt: &str) -> Result<Self> {
        let model_key = model_config.key.clone();
        let mut model = crate::get_completion_llm(model_config.clone())
            .context("Failed to initialize session model")?;

        model
            .load(system_prompt)
            .await
            .context("Failed to load model with system prompt")?;

        let context_size = model_config.get_setting::<usize>("n_ctx").unwrap_or(4096);

        let metrics = Some(model.metrics());

        Ok(Self {
            model,
            model_key,
            context_size,
            system_prompt: system_prompt.to_string(),
            messages: Vec::new(),
            tools: Vec::new(),
            metrics,
        })
    }

    /// Set a new system prompt for the session.
    /// This reloads the model and clears the message history.
    pub async fn set_system_prompt(&mut self, prompt: &str) -> Result<()> {
        self.system_prompt = prompt.to_string();
        self.model
            .load(prompt)
            .await
            .context("Failed to load model with new system prompt")?;
        self.messages.clear();
        Ok(())
    }

    /// Get model key for retrieving details from config
    pub fn model_key(&self) -> &str {
        &self.model_key
    }

    /// Get system prompt
    pub fn system_prompt(&self) -> &str {
        &self.system_prompt
    }

    /// Transfer existing message history to session
    pub fn set_messages(&mut self, messages: Vec<ChatMessage>) {
        self.messages = messages
            .into_iter()
            .map(|msg| (msg, None)) // Reset token counts for new model
            .collect();
    }

    /// Add a new message to the conversation history
    pub fn add_message(&mut self, sender: SenderType, text: &str) {
        self.messages.push((
            ChatMessage {
                sender,
                text: text.to_string(),
                tools: Vec::new(),
            },
            None,
        ));
    }

    /// Set the tools available for this session
    pub fn set_tools(&mut self, tools: Vec<Arc<dyn Tool>>) {
        self.tools = tools;
    }

    /// Returns a clone of the tools available in the session.
    pub fn tools(&self) -> Vec<Arc<dyn Tool>> {
        self.tools.clone()
    }

    /// Generate a response stream for the current conversation
    pub async fn generate(
        &mut self,
        settings: HashMap<String, String>,
        cancel_token: CancellationToken,
    ) -> Result<BoxStream<'_, Result<Completion>>> {
        let tool_slice = if self.tools.is_empty() {
            None
        } else {
            Some(self.tools.as_slice())
        };
        let max_tokens = settings
            .get("max_tokens")
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(1024);

        let messages_to_send = self.get_trimmed_messages(max_tokens);

        Ok(self
            .model
            .complete(&messages_to_send, tool_slice, &settings, cancel_token)
            .await)
    }

    /// Clear the conversation history
    pub fn clear_history(&mut self) {
        self.messages.clear();
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
    pub fn last_assistant_message(&self) -> Option<&ChatMessage> {
        self.messages
            .iter()
            .rev()
            .find(|(m, _)| m.sender == SenderType::Assistant)
            .map(|(m, _)| m)
    }

    /// Back-fills token counts for messages sent in the last prompt and adds the
    /// assistant's response with its token count. This should be called after a
    /// response has been fully received.
    pub fn record_completion(
        &mut self,
        assistant_message: ChatMessage,
        metrics: &CompletionMetrics,
    ) {
        let mut known_prompt_tokens: usize = 0;
        let mut unknown_message_indices = Vec::new();
        let mut total_unknown_chars = 0;

        // Identify messages that were part of the prompt and don't have a token count yet.
        for (i, (msg, count)) in self.messages.iter().enumerate() {
            if let Some(count) = *count {
                known_prompt_tokens += count;
            } else {
                unknown_message_indices.push(i);
                // Use character length as a heuristic for distribution.
                total_unknown_chars += msg.text.len();
            }
        }

        let new_tokens = (metrics.prompt_tokens as usize).saturating_sub(known_prompt_tokens);

        if !unknown_message_indices.is_empty() {
            if total_unknown_chars > 0 {
                let mut distributed_tokens: usize = 0;
                let last_index = unknown_message_indices.len() - 1;

                // Distribute tokens proportionally, giving remainder to the last message.
                for (i, msg_idx) in unknown_message_indices.iter().enumerate() {
                    let msg_chars = self.messages[*msg_idx].0.text.len();
                    let msg_tokens = if i == last_index {
                        new_tokens.saturating_sub(distributed_tokens)
                    } else {
                        (new_tokens * msg_chars) / total_unknown_chars
                    };
                    self.messages[*msg_idx].1 = Some(msg_tokens);
                    distributed_tokens += msg_tokens;
                }
            } else if unknown_message_indices.len() == 1 {
                // Handle case where there's one message with no text.
                self.messages[unknown_message_indices[0]].1 = Some(new_tokens);
            }
        }

        self.messages
            .push((assistant_message, Some(metrics.completion_tokens as usize)));
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
    fn get_trimmed_messages(&self, max_output_tokens: usize) -> Vec<ChatMessage> {
        // Leave room for the response. TODO: use the max_tokens parameter from completion model.
        let max_tokens = self.context_size - max_output_tokens;

        // Heuristic for messages without a known token count.
        let estimate_tokens = |msg: &ChatMessage| -> usize {
            let mut tokens = msg.text.len() / 4;
            for tool_call in &msg.tools {
                tokens += tool_call.name.len() / 4;
                if let Ok(input_str) = serde_json::to_string(&tool_call.arguments) {
                    tokens += input_str.len() / 4;
                }
            }
            tokens
        };

        let get_token_count = |(msg, count): &(ChatMessage, Option<usize>)| {
            count.unwrap_or_else(|| estimate_tokens(msg))
        };

        let system_prompt_tokens = if self.system_prompt.is_empty() {
            0
        } else {
            self.system_prompt.len() / 4
        };
        let available_tokens: usize = max_tokens.saturating_sub(system_prompt_tokens);
        debug!(
            "Token trimming: available tokens: {}, System prompt tokens: {}",
            available_tokens, system_prompt_tokens
        );

        let mut start_index = self.messages.len();
        let mut current_tokens = 0;
        let mut current_block_tokens = 0;

        for i in (0..self.messages.len()).rev() {
            let msg = &self.messages[i];
            current_block_tokens += get_token_count(msg);

            // A "block" is a User message followed by non-User messages.
            if msg.0.sender == SenderType::User || i == 0 {
                if current_tokens + current_block_tokens <= available_tokens {
                    current_tokens += current_block_tokens;
                    start_index = i;
                    current_block_tokens = 0;
                } else {
                    // This block is too big to fit with the others.
                    if start_index == self.messages.len() {
                        // This is the *most recent* block, and it's oversized.
                        // We cannot simply discard it, as that would mean sending an empty prompt.
                        // Instead, we must truncate it to fit the context window.
                        debug!(
                            "Token trimming: Most recent block is oversized ({} tokens > {} available). Truncating it.",
                            current_block_tokens, available_tokens
                        );
                        let oversized_block = &self.messages[i..];
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

        let mut final_messages: Vec<ChatMessage> = self.messages[start_index..]
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
                    let mut content = serde_json::to_string(&tool_output.output).unwrap();
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

        final_messages
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{completion::Completion, tools::ToolCall};
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
        async fn load(&mut self, _text: &str) -> Result<()> {
            Ok(())
        }
        async fn complete(
            &mut self,
            _messages: &[ChatMessage],
            _tools: Option<&[Arc<dyn Tool>]>,
            _settings: &HashMap<String, String>,
            _cancel_token: CancellationToken,
        ) -> BoxStream<'_, Result<Completion>> {
            let stream = futures::stream::iter(Vec::<Result<Completion>>::new());
            Box::pin(stream)
        }
    }

    fn new_chat_msg(sender: SenderType, text: &str) -> ChatMessage {
        ChatMessage {
            sender,
            text: text.to_string(),
            tools: Vec::new(),
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
        session.add_message(SenderType::User, "Hello");
        assert_eq!(session.messages.len(), 1);
        assert_eq!(session.messages[0].0.text, "Hello");
        assert_eq!(session.messages[0].1, None);
    }

    #[test]
    fn test_record_completion_simple() {
        let mut session = new_session(100);
        session.add_message(SenderType::User, "User 1");

        let metrics = CompletionMetrics {
            prompt_tokens: 10,
            completion_tokens: 20,
            ..Default::default()
        };
        session.record_completion(new_chat_msg(SenderType::Assistant, "Response 1"), &metrics);

        assert_eq!(session.messages.len(), 2);
        assert_eq!(session.messages[0].1, Some(10));
        assert_eq!(session.messages[1].1, Some(20));
    }

    #[test]
    fn test_record_completion_multiple_unknown() {
        let mut session = new_session(100);
        session.add_message(SenderType::User, "This is a slightly longer message."); // 34 chars
        session.add_message(SenderType::User, "Short one."); // 10 chars

        let metrics = CompletionMetrics {
            prompt_tokens: 100,
            completion_tokens: 20,
            ..Default::default()
        };
        session.record_completion(new_chat_msg(SenderType::Assistant, "Response 1"), &metrics);
        assert_eq!(session.messages.len(), 3);
        // 100 * (34 / (34+10)) = 77.27 -> 77
        assert_eq!(session.messages[0].1, Some(77));
        // Remainder: 100 - 77 = 23
        assert_eq!(session.messages[1].1, Some(23));
        assert_eq!(session.messages[2].1, Some(20));
    }

    #[test]
    fn test_get_trimmed_messages_no_trimming() {
        let mut session = new_session(100);
        session
            .messages
            .push((new_chat_msg(SenderType::User, "U1"), Some(10)));
        session
            .messages
            .push((new_chat_msg(SenderType::Assistant, "A1"), Some(10)));
        session.add_message(SenderType::User, "U2, estimated"); // 14 chars -> 3 tokens est.
        // System(1) + U1(10) + A1(10) + U2_est(3) = 24 < 100. No trimming.
        let trimmed = session.get_trimmed_messages(0);
        assert_eq!(trimmed.len(), 3);
    }

    #[test]
    fn test_get_trimmed_messages_simple_trim() {
        let mut session = new_session(30); // small context
        session
            .messages
            .push((new_chat_msg(SenderType::User, "U1"), Some(10)));
        session
            .messages
            .push((new_chat_msg(SenderType::Assistant, "A1"), Some(10)));
        session
            .messages
            .push((new_chat_msg(SenderType::User, "U2"), Some(10)));
        // available_tokens = 30-1=29.
        // U2 block (10) fits. current_tokens=10.
        // U1 block (20) doesn't fit with U2 (10+20 > 29). Trim U1 block.
        let trimmed = session.get_trimmed_messages(0);
        assert_eq!(trimmed.len(), 1);
        assert_eq!(trimmed[0].text, "U2");
    }

    #[test]
    fn test_get_trimmed_messages_keeps_blocks() {
        let mut session = new_session(35); // small context
        session
            .messages
            .push((new_chat_msg(SenderType::User, "U1"), Some(10)));
        session
            .messages
            .push((new_chat_msg(SenderType::Assistant, "A1"), Some(10)));
        session
            .messages
            .push((new_chat_msg(SenderType::User, "U2"), Some(15)));
        // available=34. U2 block (15) fits. current=15.
        // U1 block (20) doesn't fit with U2 (15+20 > 34). Trim U1 block.
        let trimmed = session.get_trimmed_messages(0);
        assert_eq!(trimmed.len(), 1);
        assert_eq!(trimmed[0].text, "U2");
    }

    #[test]
    fn test_full_flow() {
        let mut session = new_session(50);

        // Turn 1
        session.add_message(
            SenderType::User,
            "User message 1, quite long to ensure it costs something",
        );
        let metrics1 = CompletionMetrics {
            prompt_tokens: 20,
            completion_tokens: 10,
            ..Default::default()
        };
        session.record_completion(
            new_chat_msg(SenderType::Assistant, "Assistant response 1"),
            &metrics1,
        );
        assert_eq!(session.messages[0].1, Some(20));
        assert_eq!(session.messages[1].1, Some(10));

        // Turn 2
        session.add_message(SenderType::User, "User message 2"); // est. 3 tokens
        let messages_to_send = session.get_trimmed_messages(0);
        // Sys(1)+U1_block(30)+U2_est(3) = 34 < 50. All sent.
        assert_eq!(messages_to_send.len(), 3);

        let metrics2 = CompletionMetrics {
            prompt_tokens: 35, // 20+10 known, so U2 cost is 5
            completion_tokens: 5,
            ..Default::default()
        };
        session.record_completion(
            new_chat_msg(SenderType::Assistant, "Assistant response 2"),
            &metrics2,
        );
        assert_eq!(session.messages[2].1, Some(5));
        assert_eq!(session.messages[3].1, Some(5));

        // Turn 3: force trimming
        session.add_message(SenderType::User, "User message 3 is also quite long"); // est. 8 tokens
        session
            .messages
            .push((new_chat_msg(SenderType::User, "another one"), Some(10)));
        let messages_to_send_2 = session.get_trimmed_messages(0);
        // available=49. "another one" block(10) fits. current=10.
        // "U3" block(est 8) fits. current=18.
        // "U2" block(10) fits. current=28.
        // "U1" block(30) does not fit (28+30 > 49). Trim.
        // Should have U2, A2, U3, "another one". Total 4 messages.
        assert_eq!(messages_to_send_2.len(), 4);
        assert_eq!(messages_to_send_2[0].text, "User message 2");
    }

    #[test]
    fn test_get_trimmed_messages_oversized_block_with_tool_message() {
        // This test simulates a single conversational block that is too large for the context.
        // The block contains a large valid tool message that should be truncated.
        let mut session = new_session(40); // small context
        session.add_message(SenderType::User, "Check weather"); // 3 tokens est (12 chars)

        // Create valid tool message with 200 char output
        let serialized_tool_output = {
            let tool_result = ToolResult {
                call: new_tool_call("weather"),
                output: Value::String("a".repeat(200)),
            };
            serde_json::to_string(&tool_result).unwrap()
        };
        session.add_message(SenderType::Tool, &serialized_tool_output);

        // Available tokens: 40-1=39
        // Block tokens estimated: User=3, Tool=50+
        // The block must be truncated
        let trimmed = session.get_trimmed_messages(0);
        assert_eq!(trimmed.len(), 2, "Both messages should be present");
        assert_eq!(trimmed[0].text, "Check weather");
        assert!(
            trimmed[1].text.contains("... [truncated]"),
            "Tool message should be truncated"
        );
    }

    #[test]
    fn test_get_trimmed_messages_with_max_output_tokens() {
        let mut session = new_session(100);
        session
            .messages
            .push((new_chat_msg(SenderType::User, "U1"), Some(25)));
        session
            .messages
            .push((new_chat_msg(SenderType::Assistant, "A1"), Some(25)));

        // available for prompt: 100 (context) - 40 (output) - 1 (system) = 59
        // messages are 25+25=50. 50 < 59, so no trimming.
        let trimmed_spacious = session.get_trimmed_messages(40);
        assert_eq!(trimmed_spacious.len(), 2);

        // available for prompt: 100 (context) - 51 (output) - 1 (system) = 48
        // messages are 50. 50 > 48, so it must trim the oversized block.
        let trimmed_tight = session.get_trimmed_messages(51);
        assert_eq!(trimmed_tight.len(), 1, "Should trim the assistant message");
        assert_eq!(trimmed_tight[0].text, "U1");
    }

    #[test]
    fn test_tool_response_truncation() {
        // Test that intermediate tool responses are truncated, but the last one is not.
        let mut session = new_session(4096); // Large context to avoid other trimming

        // Create 3 valid tool messages with different content
        let tool_outputs = vec![
            ToolResult {
                call: new_tool_call("tool_1"),
                output: Value::String("a".repeat(600)),
            },
            ToolResult {
                call: new_tool_call("tool_2"),
                output: Value::String("b".repeat(600)),
            },
            ToolResult {
                call: new_tool_call("tool_3"),
                output: Value::String("c".repeat(600)),
            },
        ];

        let serialized_outputs: Vec<String> = tool_outputs
            .iter()
            .map(|out| serde_json::to_string(out).unwrap())
            .collect();

        session.add_message(SenderType::User, "U1");
        session.add_message(SenderType::Tool, &serialized_outputs[0]);
        session.add_message(SenderType::User, "U2");
        session.add_message(SenderType::Tool, &serialized_outputs[1]);
        session.add_message(SenderType::User, "U3");
        session.add_message(SenderType::Tool, &serialized_outputs[2]);

        let trimmed = session.get_trimmed_messages(0);
        assert_eq!(trimmed.len(), 6);

        // First two tool messages should be truncated (only need to check last 5 chars)
        assert!(trimmed[1].text.contains("... [truncated]"));
        assert!(trimmed[3].text.contains("... [truncated]"));

        // Last tool message should contain full content
        assert!(!trimmed[5].text.contains("... [truncated]"));

        // Test with single tool response
        let mut session_single = new_session(4096);
        session_single.add_message(SenderType::User, "U1");
        session_single.add_message(SenderType::Tool, &serialized_outputs[0]);
        let trimmed_single = session_single.get_trimmed_messages(0);
        assert_eq!(trimmed_single.len(), 2);
        assert!(!trimmed_single[1].text.contains("... [truncated]"));
    }

    #[test]
    fn test_get_trimmed_messages_invalid_tool_result() {
        let mut session = new_session(4096);
        session.add_message(SenderType::User, "U1");
        session.add_message(
            SenderType::Tool,
            "invalid json".to_string().repeat(100).as_str(),
        );
        session.add_message(SenderType::User, "U2");
        session.add_message(
            SenderType::Tool,
            "invalid json for last message is not truncated",
        );

        let trimmed = session.get_trimmed_messages(0);
        assert_eq!(trimmed.len(), 4);
        assert_eq!(trimmed[0].text, "U1");

        // When deserializing invalid JSON fails, it creates a ToolResult with the text as output
        // and then serializes it back, which escapes the string content.
        // Use escaped quotes for string "invalid json" as it gets serialized.
        let expected_start = r#"{"call":{"id":"invalid","name":"invalid","arguments":"{}"},"output":"\"invalid json"#;
        assert!(
            trimmed[1].text.starts_with(expected_start),
            "Actual text: {}\nExpected: {}",
            trimmed[1].text,
            expected_start
        );
        assert!(
            trimmed[1].text.ends_with(r#"... [truncated]"}"#),
            "Actual text: {}",
            trimmed[1].text
        );
        assert_eq!(trimmed[2].text, "U2");
        assert_eq!(
            trimmed[3].text,
            "invalid json for last message is not truncated"
        );
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

        session.add_message(SenderType::User, "U1");
        session.add_message(SenderType::Tool, &serialized_outputs[0]);
        session.add_message(SenderType::User, "U2");
        session.add_message(SenderType::Tool, &serialized_outputs[1]);

        // This call would panic without the fix.
        let trimmed = session.get_trimmed_messages(0);

        // Check that it didn't panic and the first message was truncated correctly.
        assert!(trimmed[1].text.contains("... [truncated]"));
        assert!(!trimmed[3].text.contains("... [truncated]"));
    }
}
