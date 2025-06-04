use crate::core::ChatMessage;
use crate::core::completion::{CompletionMetrics, CompletionModel, CompletionResponse};
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
                .map_err(|e| anyhow!("Invalid settings structure"))?,
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

    async fn load(&mut self, text: &str) -> Result<()> {
        // TODO: Implement actual loading
        Ok(())
    }

    async fn complete(
        &mut self,
        messages: &[ChatMessage],
        settings: &HashMap<String, String>,
    ) -> BoxStream<'_, CompletionResponse> {
        // TODO: Implement completion
        Box::pin(futures::stream::empty())
    }

    async fn count_tokens(&self, text: &str) -> usize {
        // TODO: Implement token counting
        0
    }

    async fn free(&mut self) {
        // TODO: Release resources
    }
}
