use crate::core::completion::{ChatMessage, CompletionModel, CompletionResponse};
use crate::core::model::{ModelConfig, ModelMetrics};
use anyhow::{Result, anyhow};
use async_trait::async_trait;
use futures::stream::BoxStream;
use reqwest::Client;
use serde::Deserialize;
use std::collections::HashMap;

#[derive(Debug, Deserialize)]
pub struct OpenAISettings {
    token: String,
    organization: Option<String>,
    model: String,
    temperature: f32,
}

pub struct OpenAIBaseModel {
    config: ModelConfig,
    client: Client,
    metrics: ModelMetrics,
    settings: OpenAISettings,
}

impl OpenAIBaseModel {
    pub fn new(config: ModelConfig) -> Result<Self> {
        let settings: OpenAISettings = serde_yaml::from_value(
            serde_yaml::to_value(&config.settings)
                .map_err(|_e| anyhow!("Invalid settings structure"))?,
        )?;

        let client = Client::new();
        Ok(Self {
            config,
            client,
            metrics: ModelMetrics {
                init_latency_ms: 0.0,
            },
            settings,
        })
    }
}

#[async_trait]
impl CompletionModel for OpenAIBaseModel {
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
    ) -> BoxStream<'_, CompletionResponse> {
        // TODO: Implement completion
        Box::pin(futures::stream::empty())
    }

    async fn count_tokens(&self, _text: &str) -> usize {
        // TODO: Implement token counting
        0
    }

    async fn free(&mut self) {
        // TODO: Release resources
    }
}
