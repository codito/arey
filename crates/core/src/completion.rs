use crate::model::ModelMetrics;
use crate::tools::{Tool, ToolCall};
use anyhow::Result;
use async_trait::async_trait;
use futures::stream::BoxStream;
use std::collections::HashMap;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

#[derive(Default, Debug, Clone)]
pub struct CancellationToken {
    cancelled: Arc<AtomicBool>,
}

impl CancellationToken {
    pub fn new() -> Self {
        Self {
            cancelled: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum SenderType {
    System,
    Assistant,
    User,
    Tool,
}

impl From<SenderType> for String {
    fn from(val: SenderType) -> Self {
        val.as_str().into()
    }
}

impl SenderType {
    pub fn as_str(&self) -> &'static str {
        match &self {
            SenderType::System => "system",
            SenderType::User => "user",
            SenderType::Assistant => "assistant",
            SenderType::Tool => "tool",
        }
    }
}

#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub text: String,
    pub sender: SenderType,
    pub tools: Vec<ToolCall>,
}

pub enum Completion {
    Response(CompletionResponse),
    Metrics(CompletionMetrics),
}

#[derive(Debug, Clone, Default)]
pub struct CompletionMetrics {
    pub prompt_tokens: u32,
    pub prompt_eval_latency_ms: f32,
    pub completion_tokens: u32,
    pub completion_latency_ms: f32,
    pub raw_chunk: Option<String>,
}

#[derive(Debug)]
pub struct CompletionResponse {
    pub text: String,
    pub tool_calls: Option<Vec<ToolCall>>,
    pub finish_reason: Option<String>,
    pub raw_chunk: Option<String>,
}

#[async_trait]
pub trait CompletionModel: Send + Sync {
    fn metrics(&self) -> ModelMetrics;
    async fn complete(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Arc<dyn Tool>]>,
        settings: &HashMap<String, String>,
        cancel_token: CancellationToken,
    ) -> BoxStream<'static, Result<Completion>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cancellation_token() {
        let token = CancellationToken::new();
        assert!(!token.is_cancelled());
        token.cancel();
        assert!(token.is_cancelled());

        let cloned_token = token.clone();
        assert!(cloned_token.is_cancelled()); // Cloned token reflects original state
    }

    #[test]
    fn test_cancellation_token_default() {
        let token: CancellationToken = Default::default();
        assert!(!token.is_cancelled());
    }

    #[test]
    fn test_sender_type() {
        assert_eq!(SenderType::System.as_str(), "system");
        assert_eq!(SenderType::User.as_str(), "user");
        assert_eq!(SenderType::Assistant.as_str(), "assistant");
        assert_eq!(SenderType::Tool.as_str(), "tool");

        let s: String = SenderType::User.into();
        assert_eq!(s, "user");
    }
}
