use crate::core::completion::{ChatMessage, CompletionModel, CompletionResponse, CancellationToken, CompletionMetrics};
use crate::core::model::{ModelConfig, ModelMetrics};
use anyhow::{Result, anyhow};
use async_trait::async_trait;
use futures::stream::BoxStream;
use serde::Deserialize;
use std::collections::HashMap;

#[derive(Debug, Deserialize)]
pub struct OllamaSettings {
    host: String,
    model: String,
    temperature: f32,
}

pub struct OllamaBaseModel {
    config: ModelConfig,
    metrics: ModelMetrics,
    settings: OllamaSettings,
}

impl OllamaBaseModel {
    pub fn new(config: ModelConfig) -> Result<Self> {
        let settings: OllamaSettings = serde_yaml::from_value(
            serde_yaml::to_value(&config.settings)
                .map_err(|_e| anyhow!("Invalid settings structure"))?,
        )?;

        Ok(Self {
            config,
            metrics: ModelMetrics {
                init_latency_ms: 0.0,
            },
            settings,
        })
    }
}

#[async_trait]
impl CompletionModel for OllamaBaseModel {
    fn context_size(&self) -> usize {
        // TODO: Implement actual context size
        4096
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
        cancel_token: CancellationToken,  // Add parameter
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
