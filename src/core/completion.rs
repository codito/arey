use crate::core::model::{ModelCapability, ModelConfig, ModelMetrics};
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
    ) -> BoxStream<'_, CompletionResponse>;
    async fn count_tokens(&self, text: &str) -> usize;
    async fn free(&mut self);
}

pub fn combine_metrics(usage_series: &[CompletionMetrics]) -> CompletionMetrics {
    let mut response_latency = 0.0;
    let mut response_tokens = 0;
    let prompt_tokens = usage_series.first().map(|u| u.prompt_tokens).unwrap_or(0);
    let prompt_eval_latency = usage_series.first().map(|u| u.prompt_eval_latency_ms).unwrap_or(0.0);

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
