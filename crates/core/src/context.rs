use crate::completion::{ChatMessage, SenderType};
use anyhow::Result;

const DEFAULT_MAX_CONTEXT_TOKENS: usize = 4096;
const DEFAULT_CHUNK_SIZE_TOKENS: usize = 1024;
const DEFAULT_PRESERVE_RECENT_TURNS: usize = 1;
const COMPACTION_THRESHOLD: f32 = 0.80;

#[derive(Debug, Clone)]
pub struct CompactionConfig {
    pub max_context_tokens: usize,
    pub chunk_size_tokens: usize,
    pub preserve_recent_turns: usize,
    pub threshold: f32,
}

impl Default for CompactionConfig {
    fn default() -> Self {
        Self {
            max_context_tokens: DEFAULT_MAX_CONTEXT_TOKENS,
            chunk_size_tokens: DEFAULT_CHUNK_SIZE_TOKENS,
            preserve_recent_turns: DEFAULT_PRESERVE_RECENT_TURNS,
            threshold: COMPACTION_THRESHOLD,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CompactionResult {
    pub original_messages: usize,
    pub compacted_messages: usize,
    pub summary_tokens: usize,
}

#[derive(Debug, Clone)]
pub struct MessageChunk {
    pub messages: Vec<ChatMessage>,
    pub _token_estimate: usize,
}

#[derive(Clone)]
pub struct Context {
    all_messages: Vec<(ChatMessage, Option<usize>)>,
    messages: Vec<(ChatMessage, Option<usize>)>,
    context_size: usize,
    system_prompt: String,
    config: CompactionConfig,
}

impl Default for Context {
    fn default() -> Self {
        Self {
            all_messages: Vec::new(),
            messages: Vec::new(),
            context_size: DEFAULT_MAX_CONTEXT_TOKENS,
            system_prompt: String::new(),
            config: CompactionConfig::default(),
        }
    }
}

impl Context {
    pub fn new(context_size: usize) -> Self {
        Self {
            all_messages: Vec::new(),
            messages: Vec::new(),
            context_size,
            system_prompt: String::new(),
            config: CompactionConfig::default(),
        }
    }

    pub fn with_system(mut self, prompt: String) -> Self {
        self.system_prompt = prompt;
        self
    }

    pub fn with_messages(mut self, messages: Vec<(ChatMessage, Option<usize>)>) -> Self {
        self.all_messages = messages.clone();
        self.messages = messages;
        self
    }

    // ----- Messages management -----

    pub fn add_message(&mut self, message: ChatMessage) -> Result<()> {
        // Update both all_messages (archive) and messages (working copy)
        self.all_messages.push((message.clone(), None));

        if message.sender != SenderType::Assistant {
            self.messages.push((message, None));
            return Ok(());
        }

        let messages = &mut self.messages;

        let mut known_prompt_tokens: usize = 0;
        let mut unknown_message_indices = Vec::new();
        let mut total_unknown_chars = 0;

        for (i, (msg, count)) in messages.iter().enumerate() {
            if let Some(count) = *count {
                known_prompt_tokens += count;
            } else {
                unknown_message_indices.push(i);
                total_unknown_chars += msg.text.len();
            }
        }

        let metrics = if let Some(m) = message.metrics.clone() {
            m
        } else {
            self.messages.push((message, None));
            return Ok(());
        };

        let new_tokens = (metrics.prompt_tokens as usize).saturating_sub(known_prompt_tokens);

        if !unknown_message_indices.is_empty() {
            if total_unknown_chars > 0 {
                let mut distributed_tokens: usize = 0;
                let last_index = unknown_message_indices.len() - 1;

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
                messages[unknown_message_indices[0]].1 = Some(new_tokens);
            }
        }

        messages.push((message, Some(metrics.completion_tokens as usize)));
        Ok(())
    }

    pub fn messages_tuple(&self) -> &[(ChatMessage, Option<usize>)] {
        &self.messages
    }

    pub fn clear(&mut self) {
        self.all_messages.clear();
        self.messages.clear();
    }

    pub fn trim(&mut self, force: bool) -> Result<bool> {
        // Force trim: OOM recovery - keep last 50%
        if force {
            let keep_count = self.messages.len().div_ceil(2);
            if keep_count < self.messages.len() {
                let original_len = self.messages.len();
                let kept: Vec<_> = self.messages.iter().skip(keep_count).cloned().collect();
                self.messages = kept;
                tracing::debug!(
                    "Force trim: removed {} messages, kept {}",
                    original_len - self.messages.len(),
                    self.messages.len()
                );
                return Ok(true);
            }
            return Ok(false);
        }

        // Normal trim: fit to budget
        let safety_margin = Self::safety_margin(self.context_size);
        let system_prompt_tokens = self.system_prompt.len() / 4 + 50;
        let available_for_messages = self
            .context_size
            .saturating_sub(safety_margin)
            .saturating_sub(system_prompt_tokens);

        let total_tokens: usize = self
            .messages
            .iter()
            .map(|(msg, count)| count.unwrap_or_else(|| self.estimate_tokens(msg)))
            .sum();

        if total_tokens <= available_for_messages {
            return Ok(false);
        }

        if let Some((last_msg, count)) = self.messages.last() {
            let last_tokens = count.unwrap_or_else(|| self.estimate_tokens(last_msg));
            if last_tokens > available_for_messages {
                return Err(anyhow::anyhow!(
                    "Last message ({} tokens, {} chars) exceeds available context budget ({} tokens)",
                    last_tokens,
                    last_msg.text.len(),
                    available_for_messages
                ));
            }
        }

        let mut trimmed = Vec::new();
        let mut used_tokens = 0;
        let mut i = self.messages.len();

        while i > 0 {
            let block_start = self.find_block_start(&self.messages, i);
            let block_tokens: usize = self.messages[block_start..i]
                .iter()
                .map(|(msg, count)| count.unwrap_or_else(|| self.estimate_tokens(msg)))
                .sum();

            if used_tokens + block_tokens <= available_for_messages {
                used_tokens += block_tokens;
                trimmed.splice(
                    0..0,
                    self.messages[block_start..i]
                        .iter()
                        .map(|(msg, _)| msg.clone()),
                );
                i = block_start;
            } else {
                let trimmed_block = self.trim_oversized_block(
                    &self.messages[block_start..i],
                    available_for_messages.saturating_sub(used_tokens),
                )?;
                if !trimmed_block.is_empty() {
                    trimmed.splice(0..0, trimmed_block);
                }
                break;
            }
        }

        let original_count = self.messages.len();
        self.messages = trimmed.into_iter().map(|m| (m, None)).collect();
        let did_trim = self.messages.len() < original_count;

        Ok(did_trim)
    }

    fn find_block_start(&self, messages: &[(ChatMessage, Option<usize>)], end_idx: usize) -> usize {
        if end_idx == 0 {
            return 0;
        }

        let mut idx = end_idx - 1;
        while idx > 0 {
            if messages[idx].0.sender == SenderType::User {
                return idx;
            }
            idx -= 1;
        }
        0
    }

    fn trim_oversized_block(
        &self,
        block_messages: &[(ChatMessage, Option<usize>)],
        available_tokens: usize,
    ) -> Result<Vec<ChatMessage>> {
        let mut final_messages = Vec::new();
        let mut used_tokens = 0;

        for (msg, count) in block_messages.iter() {
            let mut msg_to_add = msg.clone();
            let mut tokens = count.unwrap_or_else(|| self.estimate_tokens(&msg_to_add));

            if used_tokens + tokens > available_tokens {
                if msg_to_add.sender == SenderType::Tool {
                    let budget_for_this_msg = available_tokens.saturating_sub(used_tokens);
                    let chars_to_keep = budget_for_this_msg * 4;
                    if msg_to_add.text.chars().count() > chars_to_keep {
                        let truncated: String =
                            msg_to_add.text.chars().take(chars_to_keep).collect();
                        msg_to_add.text = truncated;
                        msg_to_add.text.push_str("\n... [truncated]");
                        tokens = budget_for_this_msg;
                    } else {
                        used_tokens += tokens;
                        final_messages.push(msg_to_add);
                        continue;
                    }
                } else {
                    let preview: String = msg_to_add.text.chars().take(50).collect();
                    return Err(anyhow::anyhow!(
                        "Message (type: {:?}, preview: \"{}...\", {} chars) exceeds available \
                         budget ({} tokens). Only Tool messages can be truncated.",
                        msg_to_add.sender,
                        preview,
                        msg_to_add.text.len(),
                        available_tokens
                    ));
                }
            }

            used_tokens += tokens;
            final_messages.push(msg_to_add);
        }

        Ok(final_messages)
    }

    fn safety_margin(context_size: usize) -> usize {
        if context_size <= 640 {
            context_size / 20
        } else {
            (context_size / 10).min(512)
        }
    }

    fn estimate_tokens(&self, message: &ChatMessage) -> usize {
        let base = message.text.len() / 4;
        let overhead = 50;

        let tool_overhead = message.tools.as_ref().map(|t| t.len() * 100).unwrap_or(0);

        base + overhead + tool_overhead
    }

    pub fn needs_compaction(&self, messages: &[(ChatMessage, Option<usize>)]) -> bool {
        let total: usize = {
            messages
                .iter()
                .map(|(msg, count)| count.unwrap_or_else(|| self.estimate_tokens(msg)))
                .sum()
        };
        let current_threshold = total as f32 / self.config.max_context_tokens as f32;
        current_threshold > self.config.threshold
    }

    fn chunk_messages(&self, messages: &[ChatMessage], target_size: usize) -> Vec<MessageChunk> {
        if messages.is_empty() {
            return vec![];
        }

        let mut chunks = Vec::new();
        let mut current_chunk: Vec<ChatMessage> = Vec::new();
        let mut current_tokens = 0;

        for msg in messages {
            let msg_tokens = self.estimate_tokens(msg);

            if !current_chunk.is_empty() && current_tokens + msg_tokens > target_size {
                chunks.push(MessageChunk {
                    messages: std::mem::take(&mut current_chunk),
                    _token_estimate: current_tokens,
                });
                current_tokens = 0;
            }

            current_chunk.push(msg.clone());
            current_tokens += msg_tokens;
        }

        if !current_chunk.is_empty() {
            chunks.push(MessageChunk {
                messages: current_chunk,
                _token_estimate: current_tokens,
            });
        }

        chunks
    }

    // TODO: Implement compact() - LLM summarization
    // - Split messages into recent + older (preserve last N turns via config.preserve_recent_turns)
    // - Archive older messages to all_messages
    // - Call LLM with summarization prompt (see data/prompt/summarize.md)
    // - Replace older with summary exchange in messages working copy
    pub fn compact(&mut self) -> CompactionResult {
        let original_count = self.messages.len();

        if self.messages.is_empty() || !self.needs_compaction(&self.messages) {
            return CompactionResult {
                original_messages: original_count,
                compacted_messages: original_count,
                summary_tokens: 0,
            };
        }

        let (recent, older) = {
            let this = &self;
            let messages: &[(ChatMessage, Option<usize>)] = &self.messages;
            let preserve_count = this.config.preserve_recent_turns * 2;

            if messages.len() <= preserve_count {
                (messages.to_vec(), vec![])
            } else {
                let split_idx = messages.len() - preserve_count;
                (
                    messages[split_idx..].to_vec(),
                    messages[..split_idx].to_vec(),
                )
            }
        };

        if older.is_empty() {
            return CompactionResult {
                original_messages: original_count,
                compacted_messages: original_count,
                summary_tokens: 0,
            };
        }

        // Archive older messages to all_messages (for future retrieval)
        self.all_messages.extend(older.clone());

        // TODO: Call LLM to summarize older messages
        // For now, just keep recent messages (placeholder for LLM summarization)
        let older_messages: Vec<ChatMessage> = older.iter().map(|(m, _)| m.clone()).collect();
        let chunks = self.chunk_messages(&older_messages, self.config.chunk_size_tokens);

        let mut summaries: Vec<ChatMessage> = Vec::new();

        for chunk in chunks {
            summaries.extend(chunk.messages);
        }

        let summary_tokens = {
            let this = &self;
            let messages: &[(ChatMessage, Option<usize>)] = &summaries
                .iter()
                .map(|m| (m.clone(), None))
                .collect::<Vec<_>>();
            messages
                .iter()
                .map(|(msg, count)| count.unwrap_or_else(|| this.estimate_tokens(msg)))
                .sum()
        };

        self.messages = recent;
        self.messages
            .extend(summaries.into_iter().map(|m| (m, None)));

        let compacted_count = self.messages.len();

        CompactionResult {
            original_messages: original_count,
            compacted_messages: compacted_count,
            summary_tokens,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::completion::SenderType;
    use crate::tools::ToolCall;
    use yare::parameterized;

    pub fn new_chat_msg(sender: SenderType, text: &str) -> ChatMessage {
        ChatMessage {
            sender,
            text: text.to_string(),
            ..Default::default()
        }
    }

    pub fn msg_tuple(msg: ChatMessage) -> (ChatMessage, Option<usize>) {
        (msg, None)
    }

    fn new_tool_call(name: &str) -> ToolCall {
        ToolCall {
            id: format!("call_{}", name),
            name: name.to_string(),
            arguments: "{}".to_string(),
            ..Default::default()
        }
    }

    #[test]
    fn test_new_with_default_context() {
        let ctx = Context::new(4096);
        assert!(ctx.messages_tuple().is_empty());
    }

    #[test]
    fn test_with_messages() {
        let msgs = vec![
            msg_tuple(new_chat_msg(SenderType::User, "Hello")),
            msg_tuple(new_chat_msg(SenderType::Assistant, "Hi")),
        ];
        let ctx = Context::new(4096).with_messages(msgs);
        assert_eq!(ctx.messages_tuple().len(), 2);
    }

    #[test]
    fn test_with_system() {
        let ctx = Context::new(4096).with_system("You are helpful".into());
        assert_eq!(ctx.system_prompt, "You are helpful");
    }

    #[test]
    fn test_messages_tuple() {
        let mut ctx = Context::default();
        ctx.messages
            .push(msg_tuple(new_chat_msg(SenderType::User, "Test")));
        let tuple = ctx.messages_tuple();
        assert_eq!(tuple.len(), 1);
        assert_eq!(tuple[0].0.text, "Test");
    }

    #[test]
    fn test_clear() {
        let mut ctx = Context::default();
        ctx.messages
            .push(msg_tuple(new_chat_msg(SenderType::User, "Test")));
        ctx.clear();
        assert!(ctx.messages_tuple().is_empty());
    }

    #[parameterized(
        small_context = { 640, 32 },
        medium_context = { 4096, 409 },
        large_context = { 8192, 512 },
    )]
    fn test_safety_margin(context_size: usize, expected: usize) {
        assert_eq!(Context::safety_margin(context_size), expected);
    }

    #[test]
    fn test_add_message_with_tools() {
        let mut ctx = Context::default();

        let user_msg = ChatMessage {
            sender: SenderType::User,
            text: "Use the search tool".to_string(),
            ..Default::default()
        };
        let _ = ctx.add_message(user_msg);

        let tool_call = new_tool_call("search");
        let assistant_msg = ChatMessage {
            sender: SenderType::Assistant,
            text: "I'll search for that".to_string(),
            tools: Some(vec![tool_call]),
            metrics: Some(crate::completion::CompletionMetrics {
                prompt_tokens: 10,
                completion_tokens: 20,
                ..Default::default()
            }),
            ..Default::default()
        };
        let _ = ctx.add_message(assistant_msg);

        assert_eq!(ctx.messages_tuple().len(), 2);
        assert_eq!(ctx.messages_tuple()[1].0.tools.as_ref().unwrap().len(), 1);
    }

    #[test]
    fn test_add_message_syncs_both_fields() {
        let mut ctx = Context::new(4096);

        let msg = new_chat_msg(SenderType::User, "Test");
        ctx.add_message(msg).unwrap();

        assert_eq!(ctx.messages_tuple().len(), 1);
        assert!(!ctx.all_messages.is_empty());
    }

    #[test]
    fn test_builder_chain() {
        let msgs = vec![msg_tuple(new_chat_msg(SenderType::User, "Hello"))];
        let ctx = Context::new(4096)
            .with_system("You are helpful".into())
            .with_messages(msgs);

        assert_eq!(ctx.system_prompt, "You are helpful");
        assert_eq!(ctx.messages_tuple().len(), 1);
    }
}

#[cfg(test)]
mod trim_tests {
    use super::*;
    use tests::{msg_tuple, new_chat_msg};
    use yare::parameterized;

    #[test]
    fn test_trim_messages_extremely_large_tool_result() {
        let mut ctx = Context::new(100000);

        ctx.messages = vec![
            msg_tuple(new_chat_msg(SenderType::User, "Q1")),
            msg_tuple(new_chat_msg(SenderType::Assistant, "A1")),
            msg_tuple(new_chat_msg(SenderType::Tool, &"Y".repeat(50000))),
        ];

        ctx.trim(false).unwrap();
        assert!(!ctx.messages_tuple().is_empty());
    }

    #[test]
    fn test_trim_oversized_block_multibyte_chars() {
        let mut ctx = Context::new(5000);

        let long_string = "a".repeat(510) + "€"; // € is 3 bytes

        ctx.messages = vec![
            (new_chat_msg(SenderType::User, "Q1"), None),
            (new_chat_msg(SenderType::Tool, &long_string), None),
        ];

        ctx.trim(false).unwrap();
        // Just verify no panic on multibyte
    }

    #[test]
    fn test_trim_last_message_too_large() {
        let mut ctx = Context::new(500);

        ctx.messages = vec![(new_chat_msg(SenderType::User, &"X".repeat(2000)), None)];

        let result = ctx.trim(false);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("exceeds available context budget"));
    }

    #[parameterized(
        user_message_too_large = { 500, vec![(SenderType::User, "X".repeat(2000))], true },
        assistant_and_user_too_large = { 500, vec![(SenderType::User, "X".repeat(2000)), (SenderType::Assistant, "Response".to_string())], true },
    )]
    fn test_trim_error_cases(
        context_size: usize,
        messages: Vec<(SenderType, String)>,
        expect_error: bool,
    ) {
        let mut ctx = Context::new(context_size);
        ctx.messages = messages
            .into_iter()
            .map(|(s, t)| (new_chat_msg(s, &t), None))
            .collect();

        let result = ctx.trim(false);
        if expect_error {
            assert!(result.is_err(), "Expected error but got success");
        } else {
            assert!(
                result.is_ok(),
                "Expected success but got error: {:?}",
                result.err()
            );
        }
    }

    #[parameterized(
        empty_messages = { 4096, vec![] },
        simple_conversation = { 10000, vec![(SenderType::User, "Hi".to_string()), (SenderType::Assistant, "Hello".to_string())] },
    )]
    fn test_trim_success_cases(context_size: usize, messages: Vec<(SenderType, String)>) {
        let mut ctx = Context::new(context_size);
        ctx.messages = messages
            .into_iter()
            .map(|(s, t)| (new_chat_msg(s, &t), None))
            .collect();

        let result = ctx.trim(false);
        assert!(
            result.is_ok(),
            "Expected success but got error: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_trim_empty_prompt_and_messages() {
        let mut ctx = Context::new(4096);

        let result = ctx.trim(false);
        assert!(result.is_ok());
        assert!(!result.unwrap()); // no trim performed
    }

    #[test]
    fn test_trim_force_oom_recovery() {
        let mut ctx = Context::new(1000);

        ctx.messages = vec![
            msg_tuple(new_chat_msg(SenderType::User, "Message 1")),
            msg_tuple(new_chat_msg(SenderType::Assistant, "Response 1")),
            msg_tuple(new_chat_msg(SenderType::User, "Message 2")),
            msg_tuple(new_chat_msg(SenderType::Assistant, "Response 2")),
            msg_tuple(new_chat_msg(SenderType::User, "Message 3")),
        ];

        let original_count = ctx.messages.len();
        assert!(original_count > 2);

        let result = ctx.trim(true);
        assert!(result.is_ok());
        assert!(result.unwrap());

        let remaining = ctx.messages.len();
        assert!(remaining < original_count);
    }

    #[test]
    fn test_trim_with_system_prompt() {
        let mut ctx = Context::new(2000);
        ctx.system_prompt = "You are a helpful assistant with a very long system prompt that takes up context tokens".to_string();

        ctx.messages = vec![msg_tuple(new_chat_msg(SenderType::User, "Short question"))];

        let result = ctx.trim(false);
        assert!(result.is_ok());
        assert!(!result.unwrap());
    }

    #[test]
    fn test_trim_non_tool_message_exceeds_budget() {
        let mut ctx = Context::new(500);

        ctx.messages = vec![(new_chat_msg(SenderType::User, &"X".repeat(5000)), None)];

        let result = ctx.trim(false);
        assert!(result.is_err());
        let err = result.unwrap_err();
        println!("Error: {}", err);
        assert!(err.to_string().contains("exceeds available context budget"));
    }
}

#[cfg(test)]
mod compact_tests {
    use super::*;
    use tests::{msg_tuple, new_chat_msg};
    use yare::parameterized;

    #[parameterized(
        empty_messages = { vec![], 0.0 },
        under_threshold = { vec![msg_tuple(new_chat_msg(SenderType::User, "Hi"))], 0.3 },
        over_threshold = { vec![msg_tuple(new_chat_msg(SenderType::User, &"X".repeat(10000)))], 0.9 },
    )]
    fn test_needs_compaction(messages: Vec<(ChatMessage, Option<usize>)>, usage: f32) {
        let mut ctx = Context::new(10000);
        ctx.messages = messages.clone();
        ctx.config.threshold = 0.8;
        ctx.config.max_context_tokens = if messages.is_empty() {
            10000
        } else {
            messages.len() * 100
        };

        let result = ctx.needs_compaction(&messages);
        if usage > 0.8 {
            assert!(result, "Expected compaction needed at usage {}", usage);
        } else {
            assert!(!result, "Expected no compaction at usage {}", usage);
        }
    }

    #[test]
    fn test_needs_compaction_at_exact_threshold() {
        let mut ctx = Context::new(10000);
        ctx.config.threshold = 0.8;
        ctx.config.max_context_tokens = 1000;

        let msg_text = "X".repeat(800);
        ctx.messages = vec![msg_tuple(new_chat_msg(SenderType::User, &msg_text))];

        let result = ctx.needs_compaction(&ctx.messages);
        assert!(
            !result,
            "Should NOT need compaction at exactly 0.8 threshold"
        );
    }

    #[parameterized(
        noop_when_empty = { vec![], true },
        noop_when_under_threshold = { vec![msg_tuple(new_chat_msg(SenderType::User, "Hi"))], true },
        compact_when_over_threshold = { vec![msg_tuple(new_chat_msg(SenderType::User, &"X".repeat(10000)))], false },
    )]
    fn test_compact(messages: Vec<(ChatMessage, Option<usize>)>, expect_noop: bool) {
        let mut ctx = Context::new(10000);
        ctx.messages = messages.clone();
        ctx.config.threshold = 0.8;
        ctx.config.max_context_tokens = 10000;
        ctx.config.preserve_recent_turns = 1;
        ctx.config.chunk_size_tokens = 1024;

        let result = ctx.compact();
        if expect_noop {
            assert_eq!(result.compacted_messages, result.original_messages);
        } else {
            assert!(result.compacted_messages <= result.original_messages);
        }
    }
}
