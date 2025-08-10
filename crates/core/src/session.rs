//! A session is a shared context between a human and the AI assistant.
//! Context includes the conversation, shared artifacts, and tools.
use crate::{
    completion::{
        CancellationToken, ChatMessage, Completion, CompletionMetrics, CompletionModel, SenderType,
    },
    model::{ModelConfig, ModelMetrics},
    tools::Tool,
};
use anyhow::{Context, Result};
use futures::stream::BoxStream;
use std::{collections::HashMap, sync::Arc};
use tracing::debug;

/// A session with shared context between Human and AI model.
pub struct Session {
    model: Box<dyn CompletionModel + Send + Sync>,
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
        let mut model = crate::get_completion_llm(model_config.clone())
            .context("Failed to initialize session model")?;

        model
            .load(system_prompt)
            .await
            .context("Failed to load model with system prompt")?;

        let context_size = model_config
            .settings
            .get("n_ctx")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(4096);

        let metrics = Some(model.metrics());

        Ok(Self {
            model,
            context_size,
            system_prompt: system_prompt.to_string(),
            messages: Vec::new(),
            tools: Vec::new(),
            metrics,
        })
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

        let messages_to_send = self.get_trimmed_messages();

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
        let mut trimmed_messages = std::collections::VecDeque::new();
        let mut used_tokens = 0;

        // Iterate backwards over the block, adding newest messages first.
        for (msg, count) in block_messages.iter().rev() {
            let mut msg_to_add = msg.clone();
            let mut tokens = count.unwrap_or_else(|| estimate_tokens_fn(&msg_to_add));

            if used_tokens + tokens > available_tokens {
                // If the message makes the block too large, try to shrink it.
                // We only shrink tool responses as they are the most likely to be verbose.
                if msg_to_add.sender == SenderType::Tool {
                    let budget_for_this_msg = available_tokens.saturating_sub(used_tokens);
                    // Use a 4 chars/token heuristic to determine how much text to keep.
                    let chars_to_keep = budget_for_this_msg * 4;
                    if msg_to_add.text.len() > chars_to_keep {
                        msg_to_add.text.truncate(chars_to_keep);
                        msg_to_add.text.push_str("\n... [truncated]");
                        // After truncation, the token cost is now the budget we had.
                        tokens = budget_for_this_msg;
                    }
                }
            }

            // Add the message only if it fits (either originally or after truncation).
            if used_tokens + tokens <= available_tokens {
                used_tokens += tokens;
                trimmed_messages.push_front(msg_to_add);
            } else {
                // This message (even after potential truncation) is too large.
                // Since we are iterating from newest to oldest, we can't skip this
                // and take an older one, so we stop here.
                debug!(
                    "Token trimming: Skipping message to fit budget. Used tokens: {}, available: {}",
                    used_tokens, available_tokens
                );
                break;
            }
        }
        trimmed_messages.into_iter().collect()
    }

    /// Returns a list of messages that fits within the model's context window.
    /// It trims older messages, attempting to keep conversational turns intact.
    fn get_trimmed_messages(&self) -> Vec<ChatMessage> {
        let max_tokens = self.context_size;

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

        self.messages[start_index..]
            .iter()
            .map(|(m, _)| m.clone())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::completion::Completion;
    use async_trait::async_trait;
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
            context_size,
            system_prompt: "System".to_string(), // 6 chars -> 1 token est.
            messages: Vec::new(),
            tools: Vec::new(),
            metrics: Some(ModelMetrics {
                init_latency_ms: 0.0,
            }),
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
        let trimmed = session.get_trimmed_messages();
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
        let trimmed = session.get_trimmed_messages();
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
        let trimmed = session.get_trimmed_messages();
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
        let messages_to_send = session.get_trimmed_messages();
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
        let messages_to_send_2 = session.get_trimmed_messages();
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
        // The block contains a large tool message that should be truncated.
        let mut session = new_session(40); // small context
        session.add_message(SenderType::User, "Check the weather for me."); // 6 tokens est
        let large_tool_output = "a".repeat(200); // 50 tokens est
        session.add_message(SenderType::Tool, &large_tool_output); // This is part of the same block

        // available_tokens = 40 (context) - 1 (system prompt) = 39
        // block_tokens_est = 6 (user) + 50 (tool) = 56. This is > 39.
        // The block must be truncated.
        let trimmed = session.get_trimmed_messages();
        assert_eq!(trimmed.len(), 2, "Both messages should be present");

        assert_eq!(
            trimmed[0].text, "Check the weather for me.",
            "User message should be untouched"
        );
        assert!(
            trimmed[1].text.ends_with("\n... [truncated]"),
            "Tool message should be truncated"
        );

        // Calculate expected truncated size
        // available_tokens = 39. User msg cost = 6. Remaining budget for tool msg = 33.
        // Expected chars = 33 * 4 = 132.
        let expected_truncated_text_len = 132 + "\n... [truncated]".len();
        assert_eq!(
            trimmed[1].text.len(),
            expected_truncated_text_len,
            "Tool message should be truncated to fit the budget"
        );
    }
}
