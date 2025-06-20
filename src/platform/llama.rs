use crate::core::completion::{
    CancellationToken, ChatMessage, Completion, CompletionModel, CompletionResponse, SenderType,
};
use crate::core::model::{ModelConfig, ModelMetrics};
use anyhow::{Result, anyhow};
use async_trait::async_trait;
use futures::stream::{BoxStream, Stream};
use tokio::sync::mpsc;
use llama_cpp_2::{
    context::params::LlamaContextParams,
    llama_backend::LlamaBackend,
    model::Special,
    llama_batch::LlamaBatch,
    model::{AddBos, LlamaModel, params::LlamaModelParams},
    sampling::LlamaSampler,
};
use std::sync::Arc;
use std::{num::NonZeroU32, path::PathBuf};
use tokio::sync::Mutex;

pub struct LlamaBaseModel {
    backend: LlamaBackend,
    model: Arc<Mutex<LlamaModel>>,
    context_params: LlamaContextParams,
    model_config: ModelConfig,
    metrics: ModelMetrics,
}

impl LlamaBaseModel {
    pub fn new(model_config: ModelConfig) -> Result<Self> {
        let path = model_config
            .settings
            .get("path")
            .and_then(|v| v.as_str())
            .map(PathBuf::from)
            .ok_or_else(|| anyhow!("'path' setting is required for llama model"))?;

        let backend = LlamaBackend::init().map_err(|e| anyhow!("Backend init failed: {e}"))?;

        let mut model_params = LlamaModelParams::default();
        if let Some(n_gpu_layers) = model_config.settings.get("n_gpu_layers") {
            if let Some(val) = n_gpu_layers.as_i64() {
                model_params = model_params.with_n_gpu_layers(val as u32);
            }
        }

        let model = LlamaModel::load_from_file(&backend, &path, &model_params)
            .map_err(|e| anyhow!("Model loading failed: {e}"))?;

        let mut context_params = LlamaContextParams::default();
        if let Some(threads) = model_config
            .settings
            .get("n_threads")
            .and_then(|v| v.as_i64())
        {
            context_params = context_params.with_n_threads(threads as i32);
        }
        if let Some(n_ctx) = model_config
            .settings
            .get("n_ctx")
            .and_then(|v| v.as_i64())
            .and_then(|v| NonZeroU32::new(v as u32))
        {
            context_params = context_params.with_n_ctx(Some(n_ctx));
        }

        Ok(Self {
            backend,
            model: Arc::new(Mutex::new(model)),
            context_params,
            model_config,
            metrics: ModelMetrics {
                init_latency_ms: 0.0,
            },
        })
    }

    fn format_prompt(messages: &[ChatMessage]) -> String {
        let mut prompt = String::new();
        for msg in messages {
            let role = match msg.sender {
                SenderType::System => "system",
                SenderType::User => "user",
                SenderType::Assistant => "assistant",
            };
            prompt.push_str(&format!("{role}: {}\n", msg.text));
        }
        prompt
    }
}

#[async_trait]
impl CompletionModel for LlamaBaseModel {
    fn metrics(&self) -> ModelMetrics {
        self.metrics.clone()
    }

    async fn load(&mut self, _text: &str) -> Result<()> {
        // No-op for GGUF models
        Ok(())
    }

    async fn complete(
        &mut self,
        messages: &[ChatMessage],
        settings: &std::collections::HashMap<String, String>,
        cancel_token: CancellationToken,
    ) -> BoxStream<'_, Result<Completion>> {
        // Create channel for results
        let (tx, rx) = mpsc::channel(32);
        let prompt = Self::format_prompt(messages).clone();
        let max_tokens = settings
            .get("max_tokens")
            .and_then(|s| s.parse().ok())
            .unwrap_or(1024);
        let model_config = self.model_config.clone();
        let context_params = self.context_params.clone();
        let backend = self.backend.clone();
        let model_ref = self.model.clone();
        let cancel_token_clone = cancel_token.clone(); // Clone for the blocking task

        // Spawn blocking task
        tokio::task::spawn_blocking(move || {
            if let Err(e) = (|| -> Result<()> {
                let model = model_ref.blocking_lock();
                let mut ctx = model
                    .new_context(&backend, context_params.clone())
                    .map_err(|e| anyhow!("Context creation failed: {e}"))?;

                let tokens = model.str_to_token(&prompt, AddBos::Never)
                    .map_err(|e| anyhow!("Tokenization failed: {e}"))?;

                let mut batch = LlamaBatch::new(512, 1);
                for (i, &token) in tokens.iter().enumerate() {
                    batch.add(token, i as i32, &[], i == tokens.len() - 1)
                        .map_err(|e| anyhow!("Batch add failed: {e}"))?;
                }

                // Process prompt
                ctx.decode(&mut batch)
                    .map_err(|e| anyhow!("Prompt decoding failed: {e}"))?;

                // Get seed from settings or default
                let seed = settings
                    .get("seed")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(1234);
                let mut sampler = LlamaSampler::chain_simple([
                    LlamaSampler::dist(seed),
                    LlamaSampler::greedy(),
                ]);

                let mut n_cur = batch.n_tokens();
                let mut token_count = 0;
                let mut decoder = encoding_rs::UTF_8.new_decoder();

                // Generation loop
                while token_count < max_tokens {
                    if cancel_token_clone.is_cancelled() {
                        break;
                    }

                    let token = sampler.sample(&ctx, batch.n_tokens() - 1);
                    sampler.accept(token);

                    // Skip special tokens
                    if model.is_eog_token(token) {
                        break;
                    }

                    // Get token bytes and decode
                    let token_bytes = model.token_to_bytes(token, Special::Tokenize)
                        .map_err(|e| anyhow!("Token conversion failed: {e}"))?;

                    let mut last_chunk = String::new();
                    decoder.decode_to_string(&token_bytes, &mut last_chunk, false);

                    let mut next_batch = LlamaBatch::new(1, 1);
                    next_batch.add(token, n_cur, &[], true)?;
                    ctx.decode(&mut next_batch)?;

                    n_cur += 1;
                    token_count += 1;

                    // Send token through channel
                    let _ = tx.blocking_send(Ok(Completion::Response(CompletionResponse {
                        text: last_chunk,
                        finish_reason: None,
                        raw_chunk: None,
                    })));
                }
                Ok(())
            })() {
                let _ = tx.blocking_send(Err(e));
            }
        });

        // Convert receiver to async stream
        let stream = async_stream::try_stream! {
            for await result in rx {
                yield result?;
            }
        };
        Box::pin(stream)
    }
}
