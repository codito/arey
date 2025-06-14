use crate::core::model::ModelMetrics;
use anyhow::Result;
use async_trait::async_trait;
use futures::stream::BoxStream;
use std::collections::HashMap;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

#[derive(Debug, Clone)]
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
}

impl SenderType {
    pub fn role(&self) -> &'static str {
        match self {
            SenderType::System => "system",
            SenderType::Assistant => "assistant",
            SenderType::User => "user",
        }
    }
}

#[derive(Debug)]
pub struct ChatMessage {
    pub text: String,
    pub sender: SenderType,
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
    pub completion_runs: u32,
    pub completion_latency_ms: f32,
}

#[derive(Debug)]
pub struct CompletionResponse {
    pub text: String,
    pub finish_reason: Option<String>,
    pub raw_chunk: Option<String>,
}

#[async_trait]
pub trait CompletionModel: Send + Sync {
    fn context_size(&self) -> usize;
    fn metrics(&self) -> ModelMetrics;
    async fn load(&mut self, text: &str) -> Result<()>;
    async fn complete(
        &mut self,
        messages: &[ChatMessage],
        settings: &HashMap<String, String>,
        cancel_token: CancellationToken,
    ) -> BoxStream<'_, Result<Completion>>;
    async fn free(&mut self);
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::stream;

    #[test]
    fn test_sender_type_role() {
        assert_eq!(SenderType::System.role(), "system");
        assert_eq!(SenderType::Assistant.role(), "assistant");
        assert_eq!(SenderType::User.role(), "user");
    }

    // Mock implementation for testing
    pub struct MockCompletionModel;

    #[async_trait::async_trait]
    impl CompletionModel for MockCompletionModel {
        fn context_size(&self) -> usize {
            4096
        }
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
            _settings: &HashMap<String, String>,
            _cancel_token: CancellationToken, // Added for mock
        ) -> BoxStream<'_, Result<Completion>> {
            Box::pin(stream::empty())
        }
        async fn free(&mut self) {}
    }

    #[tokio::test]
    async fn test_mock_completion_model() {
        let mut model = MockCompletionModel;
        model.load("test").await.unwrap();
        assert_eq!(model.context_size(), 4096);
    }

    #[test]
    fn test_cancellation_token() {
        let token = CancellationToken::new();
        assert!(!token.is_cancelled());
        token.cancel();
        assert!(token.is_cancelled());

        let cloned_token = token.clone();
        assert!(cloned_token.is_cancelled()); // Cloned token reflects original state
    }
}
