use crate::core::completion::{
    CancellationToken, ChatMessage, CompletionMetrics, CompletionModel, CompletionResponse,
};
use crate::core::model::{ModelConfig, ModelMetrics};
use anyhow::{Result, anyhow};
use async_trait::async_trait;
use futures::stream::BoxStream;
use serde::Deserialize;
use std::collections::HashMap;

#[derive(Debug, Deserialize)]
pub struct LlamaSettings {
    n_ctx: usize,
    n_gpu_layers: i32,
    seed: u32,
}

pub struct LlamaBaseModel {
    config: ModelConfig,
    model: Option<llama_cpp_2::model::LlamaModel>,
    // context: Option<llama_cpp_2::Context>,
    metrics: ModelMetrics,
    settings: LlamaSettings,
}

impl LlamaBaseModel {
    pub fn new(config: ModelConfig) -> Result<Self> {
        let settings: LlamaSettings = serde_yaml::from_value(
            serde_yaml::to_value(&config.settings)
                .map_err(|_e| anyhow!("Invalid settings structure"))?,
        )?;

        // let mut model_builder = llama_cpp_2::Model::new();
        // model_builder.n_ctx(settings.n_ctx);

        Ok(Self {
            config,
            model: None,
            // context: None,
            metrics: ModelMetrics {
                init_latency_ms: 0.0,
            },
            settings,
        })
    }
}

#[async_trait]
impl CompletionModel for LlamaBaseModel {
    fn context_size(&self) -> usize {
        self.settings.n_ctx
    }

    fn metrics(&self) -> ModelMetrics {
        self.metrics.clone()
    }

    async fn load(&mut self, _text: &str) -> Result<()> {
        // TODO: Implement actual loading
        Ok(())
    }

    async fn complete(
        &mut self,
        _messages: &[ChatMessage],
        _settings: &HashMap<String, String>,
        cancel_token: CancellationToken, // Add parameter
    ) -> BoxStream<'_, Result<CompletionResponse>> {
        // Add cancellation token handling
        let stream = async_stream::stream! {
            // This would be replaced with actual streaming implementation
            for i in 0..10 {
                // Check cancellation token periodically
                if cancel_token.is_cancelled() {
                    yield Err(anyhow::anyhow!("Cancelled by user"));
                    break;
                }

                // Simulate work
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                yield Ok(CompletionResponse {
                    text: format!("Chunk {}\n", i),
                    finish_reason: None,
                    metrics: CompletionMetrics::default(),
                    raw_chunk: None
                });
            }
        };
        Box::pin(stream)
    }

    async fn count_tokens(&self, _text: &str) -> usize {
        // TODO: Implement token counting
        0
    }

    async fn free(&mut self) {
        // TODO: Release resources
    }
}
