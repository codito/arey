use crate::core::model::ModelMetrics;
use anyhow::Result;
use async_trait::async_trait;
use futures::stream::BoxStream;
use std::collections::HashMap;

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

#[derive(Debug)]
pub struct CompletionMetrics {
    pub prompt_tokens: usize,
    pub prompt_eval_latency_ms: f32,
    pub completion_tokens: usize,
    pub completion_runs: usize,
    pub completion_latency_ms: f32,
}

#[derive(Debug)]
pub struct CompletionResponse {
    pub text: String,
    pub finish_reason: Option<String>,
    pub metrics: CompletionMetrics,
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
    ) -> BoxStream<'_, Result<CompletionResponse>>;
    async fn count_tokens(&self, text: &str) -> usize;
    async fn free(&mut self);
}

pub fn combine_metrics(usage_series: &[CompletionMetrics]) -> CompletionMetrics {
    let mut response_latency = 0.0;
    let mut response_tokens = 0;
    let prompt_tokens = usage_series.first().map(|u| u.prompt_tokens).unwrap_or(0);
    let prompt_eval_latency = usage_series
        .first()
        .map(|u| u.prompt_eval_latency_ms)
        .unwrap_or(0.0);

    for u in usage_series {
        response_latency += u.completion_latency_ms;
        response_tokens += u.completion_tokens;
    }

    CompletionMetrics {
        prompt_tokens,
        prompt_eval_latency_ms: prompt_eval_latency,
        completion_tokens: response_tokens,
        completion_runs: usage_series.len(),
        completion_latency_ms: response_latency,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::stream;

    #[test]
    fn test_combine_metrics_empty() {
        let combined = combine_metrics(&[]);
        assert_eq!(combined.prompt_tokens, 0);
        assert_eq!(combined.completion_tokens, 0);
    }

    #[test]
    fn test_combine_metrics() {
        let metrics1 = CompletionMetrics {
            prompt_tokens: 100,
            prompt_eval_latency_ms: 50.0,
            completion_tokens: 20,
            completion_runs: 1,
            completion_latency_ms: 200.0,
        };

        let metrics2 = CompletionMetrics {
            prompt_tokens: 100,
            prompt_eval_latency_ms: 0.0,
            completion_tokens: 30,
            completion_runs: 1,
            completion_latency_ms: 300.0,
        };

        let combined = combine_metrics(&[metrics1, metrics2]);

        assert_eq!(combined.prompt_tokens, 100);
        assert_eq!(combined.prompt_eval_latency_ms, 50.0);
        assert_eq!(combined.completion_tokens, 50);
        assert_eq!(combined.completion_runs, 2);
        assert_eq!(combined.completion_latency_ms, 500.0);
    }

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
        ) -> BoxStream<'_, CompletionResponse> {
            Box::pin(stream::empty())
        }
        async fn count_tokens(&self, _text: &str) -> usize {
            0
        }
        async fn free(&mut self) {}
    }

    #[tokio::test]
    async fn test_mock_completion_model() {
        let mut model = MockCompletionModel;
        model.load("test").await.unwrap();
        assert_eq!(model.context_size(), 4096);
    }
}
