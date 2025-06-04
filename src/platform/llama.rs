use crate::core::completion::{
    ChatMessage, CompletionModel, CompletionResponse,
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
                .map_err(|e| anyhow!("Invalid settings structure"))?,
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
