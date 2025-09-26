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

/// A mock `CompletionModel` for use in unit tests.
///
/// Its behavior can be configured via settings in the `ModelConfig`.
/// The `response_mode` setting controls what kind of response it generates:
/// - `""` (default): A simple "Hello world" streaming text response.
/// - `"tool_call"`: A response containing a tool call to `mock_tool`.
/// - `"error"`: An error response.
///
/// It also responds with a final answer if the last message in the chat history
/// is from a tool, simulating a complete tool-use cycle.
#[derive(Debug)]
pub struct TestProviderModel {
    config: ModelConfig,
    metrics: ModelMetrics,
}

impl TestProviderModel {
    /// Creates a new `TestProviderModel`.
    pub fn new(config: ModelConfig) -> Result<Self> {
        Ok(Self {
            config,
            metrics: ModelMetrics::default(),
        })
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

        // If the last message is a tool result, always return the "final answer".
        if let Some(last_msg) = messages.last()
            && last_msg.sender == SenderType::Tool
        {
            let response = Completion::Response(CompletionResponse {
                text: "Tool output is mock tool output".to_string(),
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
            "tool_call" => {
                let tool_call = ToolCall {
                    id: "c1".to_string(),
                    name: "mock_tool".to_string(),
                    arguments: "{}".to_string(),
                };
                let response = Completion::Response(CompletionResponse {
                    text: "".to_string(),
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
                    tool_calls: None,
                    finish_reason: None,
                    raw_chunk: None,
                });
                let response2 = Completion::Response(CompletionResponse {
                    text: " world".to_string(),
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
}
