//! A mock LLM provider for unit testing purposes.
use crate::completion::{
    CancellationToken, ChatMessage, Completion, CompletionMetrics, CompletionModel,
    CompletionResponse, SenderType,
};
use crate::model::{ModelConfig, ModelMetrics};
use crate::tools::{Tool, ToolCall};
use anyhow::{Result, anyhow};
use async_trait::async_trait;
use futures::stream::{self, BoxStream};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// A mock `CompletionModel` for use in unit tests.
///
/// Its behavior can be configured via settings in the `ModelConfig`.
/// The `response_mode` setting controls what kind of response it generates:
/// - `""` (default): A simple "Hello world" streaming text response.
/// - `"tool_call"`: A response containing a tool call to `mock_tool`.
/// - `"persistent_tool_call"`: Always returns a tool call on every invocation,
///   ignoring whether the last message is a tool result. Used to test tool-loop
///   dedup logic.
/// - `"error"`: An error response.
///
/// It also responds with a final answer if the last message in the chat history
/// is from a tool, simulating a complete tool-use cycle (unless overridden by
/// `persistent_tool_call`).
#[derive(Debug)]
pub struct TestProviderModel {
    config: ModelConfig,
    metrics: ModelMetrics,
    reset_called: Arc<AtomicBool>,
}

impl TestProviderModel {
    /// Creates a new `TestProviderModel`.
    pub fn new(config: ModelConfig) -> Result<Self> {
        Ok(Self {
            config,
            metrics: ModelMetrics::default(),
            reset_called: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Returns true if reset() has been called on this model.
    pub fn was_reset(&self) -> bool {
        self.reset_called.load(Ordering::SeqCst)
    }

    /// Resets the reset_called flag (for test isolation).
    #[cfg(test)]
    pub fn reset_flag(&self) {
        self.reset_called.store(false, Ordering::SeqCst);
    }
}

#[async_trait]
impl CompletionModel for TestProviderModel {
    fn metrics(&self) -> ModelMetrics {
        self.metrics.clone()
    }

    async fn complete(
        &self,
        messages: &[ChatMessage],
        _tools: Option<&[Arc<dyn Tool>]>,
        _settings: &HashMap<String, String>,
        _cancel_token: CancellationToken,
    ) -> BoxStream<'static, Result<Completion>> {
        let response_mode: String = self.config.get_setting("response_mode").unwrap_or_default();

        // When persistent_tool_call mode sees no tools were passed (session
        // stripped them after dedup triggered), return a text response so the
        // tool loop can exit naturally.
        if response_mode == "persistent_tool_call" && _tools.is_none() {
            let response = Completion::Response(CompletionResponse {
                text: "Final answer based on available information".to_string(),
                thought: None,
                tool_calls: None,
                finish_reason: Some("stop".to_string()),
                raw_chunk: None,
            });
            let metrics = Completion::Metrics(CompletionMetrics::default());
            let stream = stream::iter(vec![Ok(response), Ok(metrics)]);
            return Box::pin(stream);
        }

        // If the last message is a tool result, return the "final answer" —
        // unless persistent_tool_call mode is set (for testing tool-loop dedup).
        if response_mode != "persistent_tool_call"
            && let Some(last_msg) = messages.last()
            && last_msg.sender == SenderType::Tool
        {
            let response = Completion::Response(CompletionResponse {
                text: "Tool output is mock tool output".to_string(),
                thought: None,
                tool_calls: None,
                finish_reason: Some("stop".to_string()),
                raw_chunk: None,
            });
            let metrics = Completion::Metrics(CompletionMetrics::default());
            let stream = stream::iter(vec![Ok(response), Ok(metrics)]);
            return Box::pin(stream);
        }

        match response_mode.as_str() {
            "error" => {
                let stream = stream::once(async { Err(anyhow!("TestProviderModel error")) });
                Box::pin(stream)
            }
            "tool_call" | "persistent_tool_call" => {
                let tool_call = ToolCall {
                    id: "c1".to_string(),
                    name: "mock_tool".to_string(),
                    arguments: "{}".to_string(),
                    ..Default::default()
                };
                let response = Completion::Response(CompletionResponse {
                    text: "".to_string(),
                    thought: None,
                    tool_calls: Some(vec![tool_call]),
                    finish_reason: Some("tool_calls".to_string()),
                    raw_chunk: None,
                });
                let metrics = Completion::Metrics(CompletionMetrics::default());
                let stream = stream::iter(vec![Ok(response), Ok(metrics)]);
                Box::pin(stream)
            }
            _ => {
                // default is simple text response
                let response1 = Completion::Response(CompletionResponse {
                    text: "Hello".to_string(),
                    thought: None,
                    tool_calls: None,
                    finish_reason: None,
                    raw_chunk: None,
                });
                let response2 = Completion::Response(CompletionResponse {
                    text: " world".to_string(),
                    thought: None,
                    tool_calls: None,
                    finish_reason: Some("stop".to_string()),
                    raw_chunk: None,
                });
                let metrics = Completion::Metrics(CompletionMetrics::default());
                let stream = stream::iter(vec![Ok(response1), Ok(response2), Ok(metrics)]);
                Box::pin(stream)
            }
        }
    }

    async fn reset(&self) -> Result<()> {
        self.reset_called.store(true, Ordering::SeqCst);
        Ok(())
    }
}
