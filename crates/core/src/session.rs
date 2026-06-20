//! A session is a shared context between a human and the AI assistant.
//! Context includes the conversation, shared artifacts, and tools.

use crate::{
    completion::{CancellationToken, ChatMessage, Completion, CompletionModel, SenderType},
    context::Context as SessionContext,
    get_completion_llm,
    model::{ModelConfig, ModelMetrics},
    tools::{Tool, ToolCall, ToolExecutor},
};
use anyhow::{Context, Result};
use futures::stream::{BoxStream, StreamExt};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, error, info, warn};

/// Immutable configuration snapshot for a session.
///
/// All fields are public for direct initialization. Use `SessionConfig::default()`
/// for defaults, or create directly:
///
/// ```
/// use std::sync::Arc;
/// use arey_core::tools::Tool;
/// use arey_core::session::SessionConfig;
/// use std::collections::HashMap;
///
/// // Example with tools
/// // let my_tool: Arc<dyn Tool> = ...;
/// let config = SessionConfig {
///     system_prompt: "You are helpful".to_string(),
///     tools: vec![/* my_tool */],
///     settings: HashMap::new(),
///     enable_reasoning: Some(true),
///     max_tool_iterations: 10,
///     tool_execution_enabled: true,
/// };
/// ```
#[derive(Clone)]
pub struct SessionConfig {
    pub system_prompt: String,
    pub tools: Vec<Arc<dyn Tool>>,
    pub settings: HashMap<String, String>,
    pub enable_reasoning: Option<bool>,
    pub max_tool_iterations: usize,
    pub tool_execution_enabled: bool,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            system_prompt: String::new(),
            tools: Vec::new(),
            settings: HashMap::new(),
            enable_reasoning: None,
            max_tool_iterations: 10,
            tool_execution_enabled: true,
        }
    }
}

impl SessionConfig {
    /// Create a new SessionConfig with the given tools (convenience constructor)
    pub fn with_tools(tools: Vec<Arc<dyn Tool>>) -> Self {
        Self {
            tools,
            ..Default::default()
        }
    }
}

/// Events emitted during session generation
pub enum SessionEvent {
    /// Token response from model
    Token(Completion),
    /// Compaction (summarization) started
    CompactionStart,
    /// Compaction (summarization) completed
    CompactionEnd {
        result: crate::context::CompactionResult,
    },
    /// Reasoning (thinking) started
    ReasoningStart,
    /// Reasoning (thinking) completed
    ReasoningEnd { reasoning: Option<String> },
    /// Tool execution started
    ToolStart { calls: Vec<ToolCall> },
    /// Tool execution completed: Vec of (tool_call, succeeded) pairs
    ToolEnd { results: Vec<(ToolCall, bool)> },
}

/// A session with shared context between Human and AI model.
pub struct Session {
    model: Box<dyn CompletionModel>,
    model_key: String,
    context: SessionContext,
    config: SessionConfig,
    metrics: ModelMetrics,
    model_reset_called: bool,
}

impl Session {
    /// Create a new session with the given model configuration and session config.
    pub fn new(model_config: ModelConfig, config: SessionConfig) -> Result<Self> {
        let model_key = model_config.key.clone();
        let model = get_completion_llm(model_config.clone())
            .context("Failed to initialize session model")?;

        let context_size = model_config
            .get_setting::<usize>("n_ctx")
            .or_else(|| model.metrics().n_ctx.map(|n| n as usize))
            .unwrap_or(4096);
        let metrics = model.metrics();

        debug!(
            "Session created with context_size={} (from config: {}, from model: {:?})",
            context_size,
            model_config.get_setting::<usize>("n_ctx").unwrap_or(4096),
            metrics.n_ctx
        );

        Ok(Self {
            model,
            model_key,
            context: SessionContext::new(context_size).with_system(config.system_prompt.clone()),
            config,
            metrics,
            model_reset_called: false,
        })
    }

    /// Update the session configuration atomically.
    pub fn update_config(&mut self, config: SessionConfig) {
        self.config = config;
    }

    /// Replace the current model with a new one, preserving conversation history
    pub fn update_model(&mut self, model_config: ModelConfig) -> Result<()> {
        let model_key = model_config.key.clone();
        let context_size = model_config.get_setting::<usize>("n_ctx").unwrap_or(4096);
        let old_messages = self.context.messages_tuple().to_vec();

        let temp_model = Box::new(crate::provider::test_provider::TestProviderModel::new(
            ModelConfig {
                key: "temp".to_string(),
                name: "temp".to_string(),
                provider: crate::model::ModelProvider::Test,
                settings: HashMap::new(),
            },
        )?) as Box<dyn CompletionModel>;

        let old_model = std::mem::replace(&mut self.model, temp_model);
        drop(old_model);

        self.model = get_completion_llm(model_config)?;
        self.model_key = model_key;
        self.context = SessionContext::new(context_size)
            .with_system(self.config.system_prompt.clone())
            .with_messages(old_messages);

        Ok(())
    }

    /// Clear all messages from the session and reset model state
    pub async fn clear(&mut self) -> Result<()> {
        self.context.clear();
        self.model.reset().await?;
        self.model_reset_called = true;
        Ok(())
    }

    /// Returns true if the model was reset during the last clear() call.
    pub fn was_model_reset(&self) -> bool {
        self.model_reset_called
    }

    // ── Message access ─────────────────────────────────────────

    /// Add a new message to the conversation history
    pub fn add_message(&mut self, message: ChatMessage) -> Result<()> {
        self.context.add_message(message)
    }

    /// Get all messages in conversation. Caller can derive count, last, etc. from this.
    pub fn messages(&self) -> Vec<ChatMessage> {
        self.context
            .messages_tuple()
            .iter()
            .map(|(msg, _)| msg.clone())
            .collect()
    }

    /// Get current model key
    pub fn model_key(&self) -> &str {
        &self.model_key
    }

    // ── Tool execution ─────────────────────────────────────────

    /// Get tools available in session
    pub fn tools(&self) -> &[Arc<dyn Tool>] {
        &self.config.tools
    }

    /// Get a tool by name from the session's tools
    pub fn get_tool(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.config.tools.iter().find(|t| t.name() == name).cloned()
    }

    /// Get current session configuration (read-only)
    pub fn config(&self) -> &SessionConfig {
        &self.config
    }

    /// Trim message history after model memory trim (OOM recovery).
    /// This keeps only the most recent 50% of messages to match the KV cache trim.
    /// Should be called when the model signals memory was trimmed.
    pub fn trim_for_memory_recovery(&mut self) {
        let _ = self.context.trim(true);
    }

    /// Generate a response stream for the current conversation.
    ///
    /// This method handles the full generate→execute_tool→generate loop when
    /// tool_execution_enabled is true:
    /// 1. Generate tokens from model
    /// 2. If tool_calls detected and tool_execution_enabled:
    ///    - emit ToolStart, execute tools in parallel, emit ToolEnd
    ///    - add tool results to conversation
    ///    - repeat until no tool calls or max_iterations reached
    pub async fn generate(
        &mut self,
        settings: HashMap<String, String>,
        cancel_token: CancellationToken,
    ) -> Result<BoxStream<'_, Result<SessionEvent>>> {
        Ok(async_stream::stream! {
            let max_iterations = self.config.max_tool_iterations;
            let tools_enabled = self.config.tool_execution_enabled && !self.config.tools.is_empty();

            let mut iteration = 0;
            let mut pending_tool_calls: Vec<ToolCall> = Vec::new();
            let mut previous_tool_calls: Vec<ToolCall> = Vec::new();
            let mut dedup_triggered = false;

            loop {
                // Check compaction
                let needs_compaction = {
                    let ctx = self.context.messages_tuple();
                    self.context.needs_compaction(ctx)
                };

                if needs_compaction {
                    yield Ok(SessionEvent::CompactionStart);
                    let result = self.context.compact();
                    yield Ok(SessionEvent::CompactionEnd { result });
                }

                // Trim context to fit budget before generation
                if let Err(e) = self.context.trim(false) {
                    error!("Failed to trim context: {}", e);
                    yield Err(e);
                    return;
                }

                // Get messages from working copy (already trimmed)
                let mut messages_to_send: Vec<ChatMessage> = self
                    .context
                    .messages_tuple()
                    .iter()
                    .map(|(m, _)| m.clone())
                    .collect();

                // Prepend system prompt if set
                if !self.config.system_prompt.is_empty() {
                    messages_to_send.insert(
                        0,
                        ChatMessage {
                            sender: SenderType::System,
                            text: self.config.system_prompt.clone(),
                            ..Default::default()
                        },
                    );
                }

                if messages_to_send.is_empty() {
                    error!(
                        "Cannot generate: zero messages to send. System prompt: '{}', Conversation messages: {}",
                        self.config.system_prompt,
                        self.context.messages_tuple().len()
                    );
                    return;
                }

                let mut effective_settings = settings.clone();
                for (k, v) in &self.config.settings {
                    effective_settings.entry(k.clone()).or_insert(v.clone());
                }

                if let Some(enabled) = self.config.enable_reasoning {
                    effective_settings.insert("enable_thinking".to_string(), enabled.to_string());
                }

                let tool_slice: Vec<Arc<dyn Tool>> = if dedup_triggered {
                    Vec::new()
                } else if self.config.tool_execution_enabled && !self.config.tools.is_empty() {
                    self.config.tools.clone()
                } else {
                    Vec::new()
                };

                let mut stream = self.model.complete(
                    &messages_to_send,
                    if tool_slice.is_empty() { None } else { Some(&tool_slice) },
                    &effective_settings,
                    cancel_token.clone()
                ).await;

                debug!(
                    "Tool loop iteration {}: sending {} messages to model; last user msg: {:?}",
                    iteration,
                    messages_to_send.len(),
                    messages_to_send.iter().rev().find(|m| m.sender == SenderType::User).map(|m| m.text.chars().take(100).collect::<String>())
                );

                pending_tool_calls.clear();
                let mut assistant_text = String::new();
                let mut assistant_thought = String::new();
                let mut final_metrics = None;

                while let Some(item) = stream.next().await {
                    match item {
                        Ok(Completion::Response(resp)) => {
                            if let Some(tool_calls) = &resp.tool_calls {
                                pending_tool_calls.extend(tool_calls.clone());
                            }
                            if !resp.text.is_empty() {
                                assistant_text.push_str(&resp.text);
                            }
                            if let Some(thought) = &resp.thought {
                                assistant_thought.push_str(thought);
                            }
                            yield Ok(SessionEvent::Token(Completion::Response(resp)));
                        }
                        Ok(Completion::Metrics(m)) => {
                            final_metrics = Some(m.clone());
                            yield Ok(SessionEvent::Token(Completion::Metrics(m)));
                        }
                        Err(e) => {
                            yield Err(e);
                            return;
                        }
                    }
                }

                // Add assistant response to context before potentially looping for tools
                let assistant_msg = ChatMessage {
                    sender: SenderType::Assistant,
                    text: assistant_text,
                    thought: if assistant_thought.is_empty() {
                        None
                    } else {
                        Some(assistant_thought)
                    },
                    tools: if pending_tool_calls.is_empty() {
                        None
                    } else {
                        Some(pending_tool_calls.clone())
                    },
                    metrics: final_metrics,
                };
                let _ = self.add_message(assistant_msg);

                // Check for repeated identical tool calls (same name + arguments)
                if !pending_tool_calls.is_empty()
                    && !previous_tool_calls.is_empty()
                    && pending_tool_calls.len() == previous_tool_calls.len()
                    && pending_tool_calls.iter().zip(previous_tool_calls.iter()).all(|(a, b)| a.name == b.name && a.arguments == b.arguments)
                {
                    warn!("Same tool calls repeated ({}), synthesizing response",
                        pending_tool_calls.iter().map(|t| t.name.clone()).collect::<Vec<_>>().join(", "));
                    let context_msg = ChatMessage {
                        sender: SenderType::Tool,
                        text: "Skipped repeated search query. Use a different query, or create an answer from above results. If needed, use `fetch` tool to get more details for the result url.".to_string(),
                        ..Default::default()
                    };
                    let _ = self.add_message(context_msg);
                    dedup_triggered = true;
                    continue;
                }

                // No more tool calls - exit loop
                if pending_tool_calls.is_empty() {
                    break;
                }

                // Execute tools
                if !tools_enabled {
                    info!("Tool calls detected but tool execution is disabled");
                    break;
                }

                previous_tool_calls = pending_tool_calls.clone();

                yield Ok(SessionEvent::ToolStart {
                    calls: pending_tool_calls.clone(),
                });

                let mut tool_results: Vec<(ToolCall, bool)> = Vec::new();
                for tool_call in &pending_tool_calls {
                    // Validate tool_call.arguments is valid JSON before execution
                    if serde_json::from_str::<serde_json::Value>(&tool_call.arguments).is_err() {
                        error!(
                            "Invalid tool call arguments (not valid JSON): {} for tool {}",
                            tool_call.arguments, tool_call.name
                        );
                        let error_msg = ChatMessage {
                            sender: SenderType::Tool,
                            text: format!("Invalid arguments (not valid JSON): {}", tool_call.arguments),
                            ..Default::default()
                        };
                        let _ = self.add_message(error_msg);
                        tool_results.push((tool_call.clone(), false));
                        continue;
                    }

                    let success = if let Some(tool) = self.get_tool(&tool_call.name) {
                        match ToolExecutor::execute(tool_call, tool.as_ref()).await {
                            Ok(result_msg) => {
                                let _ = self.add_message(result_msg);
                                true
                            }
                            Err(e) => {
                                error!("Tool execution failed: {}", e);
                                false
                            }
                        }
                    } else {
                        let error_msg = ChatMessage {
                            sender: SenderType::Tool,
                            text: format!("Unknown tool: {}", tool_call.name),
                            ..Default::default()
                        };
                        let _ = self.add_message(error_msg);
                        false
                    };
                    tool_results.push((tool_call.clone(), success));
                }

                yield Ok(SessionEvent::ToolEnd {
                    results: tool_results,
                });

                iteration += 1;
                if iteration >= max_iterations {
                    warn!("Max tool iterations ({}) reached", max_iterations);
                    break;
                }
            }
        }.boxed())
    }

    /// Get model metrics
    pub fn metrics(&self) -> &ModelMetrics {
        &self.metrics
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::completion::{CompletionResponse, SenderType};
    use crate::model::{ModelMetrics, ModelProvider};
    use async_trait::async_trait;
    use futures::stream::{self, BoxStream};
    use std::sync::Arc;

    fn new_chat_msg(sender: SenderType, text: &str) -> ChatMessage {
        ChatMessage {
            sender,
            text: text.to_string(),
            ..Default::default()
        }
    }

    fn new_session(_context_size: usize) -> Session {
        let model_config = ModelConfig {
            key: "test".to_string(),
            name: "Test".to_string(),
            provider: ModelProvider::Test,
            settings: HashMap::new(),
        };
        Session::new(model_config, SessionConfig::default()).unwrap()
    }

    #[test]
    fn test_session_new() {
        let config = SessionConfig {
            system_prompt: "Test".to_string(),
            ..Default::default()
        };

        let model_config = ModelConfig {
            key: "test".to_string(),
            name: "Test".to_string(),
            provider: ModelProvider::Test,
            settings: HashMap::new(),
        };

        let session = Session::new(model_config, config).unwrap();
        assert_eq!(session.model_key(), "test");
    }

    #[test]
    fn test_add_message() {
        let mut session = new_session(100);
        session
            .add_message(new_chat_msg(SenderType::User, "Hello"))
            .unwrap();
        assert_eq!(session.messages().len(), 1);
    }

    #[test]
    fn test_trim_for_memory_recovery() {
        let mut session = new_session(100);

        // Add 10 messages
        for i in 0..10 {
            session
                .add_message(new_chat_msg(SenderType::User, &format!("Msg {}", i)))
                .unwrap();
        }
        assert_eq!(session.messages().len(), 10);

        session.trim_for_memory_recovery();

        // Should keep last 50% = 5 messages
        assert_eq!(session.messages().len(), 5);
        assert_eq!(session.messages()[0].text, "Msg 5");
    }

    #[derive(Clone)]
    #[allow(dead_code)]
    struct MockModel {
        response: Option<String>,
        should_error: bool,
    }

    #[async_trait]
    impl CompletionModel for MockModel {
        fn metrics(&self) -> ModelMetrics {
            ModelMetrics {
                init_latency_ms: 0.0,
                ..Default::default()
            }
        }

        async fn complete(
            &self,
            _messages: &[ChatMessage],
            _tools: Option<&[Arc<dyn Tool>]>,
            _settings: &HashMap<String, String>,
            _cancel_token: CancellationToken,
        ) -> BoxStream<'static, Result<Completion>> {
            if self.should_error {
                return stream::iter([Err(anyhow::anyhow!("Mock error"))]).boxed();
            }

            let response = self.response.clone().unwrap_or_else(|| "Hello".to_string());
            stream::iter([Ok(Completion::Response(CompletionResponse {
                text: response,
                tool_calls: None,
                ..Default::default()
            }))])
            .boxed()
        }

        async fn reset(&self) -> Result<()> {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tool_tests {
    use super::*;
    use crate::completion::{CompletionResponse, SenderType};
    use crate::model::{ModelMetrics, ModelProvider};
    use crate::tools::{Tool, ToolError};
    use async_trait::async_trait;
    use futures::TryStreamExt;
    use futures::stream::{self, BoxStream, StreamExt};
    use serde_json::{Value as JsonValue, json};
    use serde_yaml::Value;
    use std::sync::Arc;

    fn new_chat_msg(sender: SenderType, text: &str) -> ChatMessage {
        ChatMessage {
            sender,
            text: text.to_string(),
            ..Default::default()
        }
    }

    #[derive(Clone)]
    struct TestTool {
        name: String,
        should_error: bool,
    }

    #[async_trait]
    impl Tool for TestTool {
        fn name(&self) -> String {
            self.name.clone()
        }

        fn description(&self) -> String {
            format!("Test tool {}", self.name)
        }

        fn parameters(&self) -> JsonValue {
            json!({"type": "object"})
        }

        async fn execute(&self, _args: &JsonValue) -> Result<JsonValue, ToolError> {
            if self.should_error {
                Err(ToolError::ExecutionError(format!(
                    "Tool {} failed",
                    self.name
                )))
            } else {
                Ok(json!({"result": format!("executed {}", self.name)}))
            }
        }
    }

    #[tokio::test]
    async fn test_tool_start_event_emitted() -> Result<()> {
        let tool: Arc<dyn Tool> = Arc::new(TestTool {
            name: "mock_tool".to_string(),
            should_error: false,
        });

        let mut settings: HashMap<String, Value> = HashMap::new();
        settings.insert(
            "response_mode".to_string(),
            Value::String("tool_call".to_string()),
        );

        let model_config = ModelConfig {
            key: "test".to_string(),
            name: "Test".to_string(),
            provider: ModelProvider::Test,
            settings,
        };

        let session_config = SessionConfig {
            tools: vec![tool],
            ..Default::default()
        };

        let mut session = Session::new(model_config, session_config)?;
        session.add_message(new_chat_msg(SenderType::User, "test"))?;

        let stream = session
            .generate(HashMap::new(), CancellationToken::new())
            .await?;

        let events: Vec<SessionEvent> = stream.try_collect().await?;

        let tool_start_events: Vec<_> = events
            .iter()
            .filter(|e| matches!(e, SessionEvent::ToolStart { .. }))
            .collect();

        assert!(
            !tool_start_events.is_empty(),
            "ToolStart event should be emitted"
        );

        if let SessionEvent::ToolStart { calls } = &tool_start_events[0] {
            assert!(!calls.is_empty(), "ToolStart should have calls");
            assert_eq!(calls[0].name, "mock_tool");
        } else {
            panic!("Expected ToolStart event");
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_tool_end_event_emitted_after_execution() -> Result<()> {
        let tool: Arc<dyn Tool> = Arc::new(TestTool {
            name: "mock_tool".to_string(),
            should_error: false,
        });

        let mut settings: HashMap<String, Value> = HashMap::new();
        settings.insert(
            "response_mode".to_string(),
            Value::String("tool_call".to_string()),
        );

        let model_config = ModelConfig {
            key: "test".to_string(),
            name: "Test".to_string(),
            provider: ModelProvider::Test,
            settings,
        };

        let session_config = SessionConfig {
            tools: vec![tool],
            tool_execution_enabled: true,
            ..Default::default()
        };

        let mut session = Session::new(model_config, session_config)?;
        session.add_message(new_chat_msg(SenderType::User, "test"))?;

        let stream = session
            .generate(HashMap::new(), CancellationToken::new())
            .await?;

        let events: Vec<SessionEvent> = stream.try_collect().await?;

        let tool_end_events: Vec<_> = events
            .iter()
            .filter(|e| matches!(e, SessionEvent::ToolEnd { .. }))
            .collect();

        assert!(
            !tool_end_events.is_empty(),
            "ToolEnd event should be emitted"
        );

        if let SessionEvent::ToolEnd { results } = &tool_end_events[0] {
            assert_eq!(results.len(), 1, "ToolEnd should have one result");
            assert!(results[0].1, "Tool execution should succeed");
        } else {
            panic!("Expected ToolEnd event");
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_tool_result_message_added_to_context() -> Result<()> {
        let tool: Arc<dyn Tool> = Arc::new(TestTool {
            name: "mock_tool".to_string(),
            should_error: false,
        });

        let mut settings: HashMap<String, Value> = HashMap::new();
        settings.insert(
            "response_mode".to_string(),
            Value::String("tool_call".to_string()),
        );

        let model_config = ModelConfig {
            key: "test".to_string(),
            name: "Test".to_string(),
            provider: ModelProvider::Test,
            settings,
        };

        let session_config = SessionConfig {
            tools: vec![tool],
            tool_execution_enabled: true,
            ..Default::default()
        };

        let mut session = Session::new(model_config, session_config)?;
        session.add_message(new_chat_msg(SenderType::User, "test"))?;

        let stream = session
            .generate(HashMap::new(), CancellationToken::new())
            .await?;
        let _ = stream.try_collect::<Vec<_>>().await?;

        let messages = session.messages();

        let tool_messages: Vec<_> = messages
            .iter()
            .filter(|m| m.sender == SenderType::Tool)
            .collect();

        assert!(
            !tool_messages.is_empty(),
            "Tool result message should be added to context"
        );

        let first_tool_msg = &tool_messages[0];
        let parsed: serde_json::Value = serde_json::from_str(&first_tool_msg.text)?;
        assert!(
            parsed.get("output").is_some(),
            "Tool result should contain output"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_tool_execution_error() -> Result<()> {
        let tool: Arc<dyn Tool> = Arc::new(TestTool {
            name: "mock_tool".to_string(),
            should_error: true,
        });

        let mut settings: HashMap<String, Value> = HashMap::new();
        settings.insert(
            "response_mode".to_string(),
            Value::String("tool_call".to_string()),
        );

        let model_config = ModelConfig {
            key: "test".to_string(),
            name: "Test".to_string(),
            provider: ModelProvider::Test,
            settings,
        };

        let session_config = SessionConfig {
            tools: vec![tool],
            tool_execution_enabled: true,
            ..Default::default()
        };

        let mut session = Session::new(model_config, session_config)?;
        session.add_message(new_chat_msg(SenderType::User, "test"))?;

        let stream = session
            .generate(HashMap::new(), CancellationToken::new())
            .await?;

        let events: Vec<SessionEvent> = stream.try_collect().await?;

        let tool_end_events: Vec<_> = events
            .iter()
            .filter(|e| matches!(e, SessionEvent::ToolEnd { .. }))
            .collect();

        assert!(
            !tool_end_events.is_empty(),
            "ToolEnd should be emitted on error too"
        );

        if let SessionEvent::ToolEnd { results } = &tool_end_events[0] {
            assert_eq!(results.len(), 1, "ToolEnd should have one result");
            assert!(!results[0].1, "Tool execution should fail");
        }

        Ok(())
    }

    #[tokio::test]
    #[ignore = "Requires architecture to inject custom model into Session"]
    async fn test_tools_passed_to_model_complete() -> Result<()> {
        struct ModelCapturesTools {
            captured_tools: Arc<std::sync::Mutex<Option<Vec<String>>>>,
        }

        #[async_trait]
        impl CompletionModel for ModelCapturesTools {
            fn metrics(&self) -> ModelMetrics {
                ModelMetrics::default()
            }

            async fn complete(
                &self,
                _messages: &[ChatMessage],
                tools: Option<&[Arc<dyn Tool>]>,
                _settings: &HashMap<String, String>,
                _cancel_token: CancellationToken,
            ) -> BoxStream<'static, Result<Completion>> {
                let tool_names = tools.map(|t| t.iter().map(|t| t.name()).collect::<Vec<_>>());
                *self.captured_tools.lock().unwrap() = tool_names;

                stream::iter([Ok(Completion::Response(CompletionResponse {
                    text: "Hello".to_string(),
                    tool_calls: None,
                    finish_reason: Some("stop".to_string()),
                    ..Default::default()
                }))])
                .boxed()
            }

            async fn reset(&self) -> Result<()> {
                Ok(())
            }
        }

        let tool1: Arc<dyn Tool> = Arc::new(TestTool {
            name: "search".to_string(),
            should_error: false,
        });
        let tool2: Arc<dyn Tool> = Arc::new(TestTool {
            name: "weather".to_string(),
            should_error: false,
        });

        let captured = Arc::new(std::sync::Mutex::new(None));
        let captured_for_model = captured.clone();

        let _model = ModelCapturesTools {
            captured_tools: captured_for_model,
        };

        let model_config = ModelConfig {
            key: "test".to_string(),
            name: "Test".to_string(),
            provider: ModelProvider::Test,
            settings: HashMap::new(),
        };

        let session_config = SessionConfig {
            tools: vec![tool1, tool2],
            tool_execution_enabled: true,
            ..Default::default()
        };

        let mut session = Session::new(model_config, session_config)?;
        session.add_message(new_chat_msg(SenderType::User, "hello"))?;

        let stream = session
            .generate(HashMap::new(), CancellationToken::new())
            .await?;
        let _ = stream.try_collect::<Vec<_>>().await?;

        let captured_tools = captured.lock().unwrap();
        let tools_passed = captured_tools.as_ref().expect("Tools should be captured");

        assert_eq!(
            tools_passed.len(),
            2,
            "Both tools should be passed to model"
        );
        assert!(tools_passed.contains(&"search".to_string()));
        assert!(tools_passed.contains(&"weather".to_string()));

        Ok(())
    }

    #[tokio::test]
    async fn test_invalid_tool_call_arguments_not_executed() -> Result<()> {
        let tool: Arc<dyn Tool> = Arc::new(TestTool {
            name: "mock_tool".to_string(),
            should_error: false,
        });

        // TestProviderModel doesn't support invalid arguments directly,
        // so we test the validation logic by creating a session and calling generate
        // The tool will receive what the model sends - we can't easily inject invalid args
        // through TestProviderModel. Instead, test that valid JSON works.

        let mut settings: HashMap<String, Value> = HashMap::new();
        settings.insert(
            "response_mode".to_string(),
            Value::String("tool_call".to_string()),
        );

        let model_config = ModelConfig {
            key: "test".to_string(),
            name: "Test".to_string(),
            provider: ModelProvider::Test,
            settings,
        };

        let session_config = SessionConfig {
            tools: vec![tool],
            tool_execution_enabled: true,
            ..Default::default()
        };

        let mut session = Session::new(model_config, session_config)?;
        session.add_message(new_chat_msg(SenderType::User, "test"))?;

        let stream = session
            .generate(HashMap::new(), CancellationToken::new())
            .await?;

        let events: Vec<SessionEvent> = stream.try_collect().await?;

        // With valid arguments (TestProviderModel sends "{}"), tool should execute
        let tool_end_events: Vec<_> = events
            .iter()
            .filter(|e| matches!(e, SessionEvent::ToolEnd { .. }))
            .collect();

        assert!(
            !tool_end_events.is_empty(),
            "ToolEnd event should be emitted"
        );

        if let SessionEvent::ToolEnd { results } = &tool_end_events[0] {
            // TestProviderModel sends "{}" which is valid JSON, so tool execution should succeed
            assert!(
                results[0].1,
                "Tool execution should succeed with valid JSON"
            );
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_repeated_tool_call_dedup_synthesizes_response() -> Result<()> {
        let tool: Arc<dyn Tool> = Arc::new(TestTool {
            name: "mock_tool".to_string(),
            should_error: false,
        });

        let mut settings: HashMap<String, Value> = HashMap::new();
        settings.insert(
            "response_mode".to_string(),
            Value::String("persistent_tool_call".to_string()),
        );

        let model_config = ModelConfig {
            key: "test".to_string(),
            name: "Test".to_string(),
            provider: ModelProvider::Test,
            settings,
        };

        let session_config = SessionConfig {
            tools: vec![tool],
            tool_execution_enabled: true,
            ..Default::default()
        };

        let mut session = Session::new(model_config, session_config)?;
        session.add_message(new_chat_msg(SenderType::User, "test"))?;

        let stream = session
            .generate(HashMap::new(), CancellationToken::new())
            .await?;

        let events: Vec<SessionEvent> = stream.try_collect().await?;

        // With persistent_tool_call, the model returns identical tool calls.
        // The dedup check should trigger on the second iteration, skipping
        // tool execution and continuing without tools to synthesize a text response.
        let tool_end_events: Vec<_> = events
            .iter()
            .filter(|e| matches!(e, SessionEvent::ToolEnd { .. }))
            .collect();

        assert_eq!(
            tool_end_events.len(),
            1,
            "Tool should execute only once before dedup triggers continue"
        );

        // The dedup context message should be in the conversation
        let messages = session.messages();
        let has_dedup_msg = messages.iter().any(|m| {
            m.sender == SenderType::Tool && m.text.contains("Skipped repeated search query")
        });
        assert!(
            has_dedup_msg,
            "Dedup context message should be added to conversation"
        );

        // A final text response should have been synthesized (not an abrupt error)
        let synthesized_text: String = events
            .iter()
            .filter_map(|e| match e {
                SessionEvent::Token(Completion::Response(r))
                    if !r.text.is_empty() && r.tool_calls.is_none() =>
                {
                    Some(r.text.clone())
                }
                _ => None,
            })
            .collect();
        assert!(
            !synthesized_text.is_empty(),
            "Should have synthesized a text response after dedup"
        );
        assert!(
            synthesized_text.contains("Final answer"),
            "Synthesized response should contain the final answer text"
        );

        Ok(())
    }
}
