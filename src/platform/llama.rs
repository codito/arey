use crate::core::completion::{
    CancellationToken, ChatMessage, Completion, CompletionMetrics, CompletionModel,
    CompletionResponse,
};
use crate::core::model::{ModelConfig, ModelMetrics};
use anyhow::{Result, anyhow};
use async_trait::async_trait;
use futures::stream::BoxStream;
use llama_cpp_2::model::LlamaChatMessage;
use llama_cpp_2::{LogOptions, send_logs_to_tracing};
use llama_cpp_2::{
    context::params::LlamaContextParams,
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::Special,
    model::{AddBos, LlamaModel, params::LlamaModelParams},
    sampling::LlamaSampler,
};
use std::sync::Arc;
use std::{num::NonZeroU32, path::PathBuf};
use tokio::sync::Mutex;
use tokio::sync::mpsc;

pub struct LlamaBaseModel {
    backend: Arc<LlamaBackend>,
    model: Arc<Mutex<LlamaModel>>,
    context_params: LlamaContextParams,
    // model_config: ModelConfig,
    metrics: ModelMetrics,
}

impl LlamaBaseModel {
    pub fn new(model_config: ModelConfig) -> Result<Self> {
        // if verbose {
        //     tracing_subscriber::fmt().init();
        // }
        send_logs_to_tracing(LogOptions::default().with_logs_enabled(false));

        let path = model_config
            .settings
            .get("path")
            .ok_or_else(|| anyhow!("'path' setting is required for llama model"))?
            .as_str()
            .unwrap();
        let expand_path = shellexpand::tilde(path).into_owned();
        let model_path = PathBuf::from(expand_path);

        let backend = LlamaBackend::init().map_err(|e| anyhow!("Backend init failed: {e}"))?;

        let mut model_params = LlamaModelParams::default();
        if let Some(n_gpu_layers) = model_config.settings.get("n_gpu_layers") {
            if let Some(val) = n_gpu_layers.as_i64() {
                model_params = model_params.with_n_gpu_layers(val as u32);
            }
        }

        let model = LlamaModel::load_from_file(&backend, &model_path, &model_params)
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
            backend: Arc::new(backend),
            model: Arc::new(Mutex::new(model)),
            context_params,
            // model_config,
            metrics: ModelMetrics {
                init_latency_ms: 0.0,
            },
        })
    }
}

#[async_trait]
impl CompletionModel for LlamaBaseModel {
    fn metrics(&self) -> ModelMetrics {
        self.metrics.clone()
    }

    async fn load(&mut self, _text: &str) -> Result<()> {
        Ok(())
    }

    async fn complete(
        &mut self,
        messages: &[ChatMessage],
        settings: &std::collections::HashMap<String, String>,
        cancel_token: CancellationToken,
    ) -> BoxStream<'_, Result<Completion, anyhow::Error>> {
        // Create channel for results
        let (tx, rx) = mpsc::channel(32);
        let max_tokens = settings
            .get("max_tokens")
            .and_then(|s| s.parse().ok())
            .unwrap_or(1024);
        // Get seed from settings or default
        let seed = settings
            .get("seed")
            .and_then(|s| s.parse().ok())
            .unwrap_or(1234);

        let context_params = self.context_params.clone();
        let model_ref = self.model.clone();
        let cancel_token = cancel_token.clone();
        let shared_backend = self.backend.clone();
        let llama_messages: Vec<LlamaChatMessage> = messages
            .iter()
            .map(|m| LlamaChatMessage::new(m.sender.clone().into(), m.text.clone()).unwrap())
            .collect();

        tokio::task::spawn_blocking(move || {
            if let Err(e) = (|| -> Result<()> {
                let model = model_ref.blocking_lock();
                let mut ctx = model
                    .new_context(&shared_backend, context_params)
                    .map_err(|e| anyhow!("Context creation failed: {e}"))?;

                let prompt = model
                    .chat_template(None)
                    .map_err(|e| anyhow!("Failed to retrieve default chat template: {e}"))
                    .and_then(|tmpl| {
                        model
                            .apply_chat_template(&tmpl, &llama_messages, true)
                            .map_err(|e| anyhow!("Failed to apply chat template to messages: {e}"))
                    })?;

                let tokens = model
                    .str_to_token(&prompt, AddBos::Always)
                    .map_err(|e| anyhow!("Tokenization failed: {e}"))?;

                let mut batch = LlamaBatch::new(512, 1);
                for (i, &token) in tokens.iter().enumerate() {
                    batch
                        .add(token, i as i32, &[0], i == tokens.len() - 1)
                        .map_err(|e| anyhow!("Batch add failed: {e}"))?;
                }

                // Start timing for prompt evaluation
                let start_time = std::time::Instant::now();

                // Process prompt
                ctx.decode(&mut batch)
                    .map_err(|e| anyhow!("Prompt decoding failed: {e}"))?;

                // Capture prompt metrics
                let prompt_token_count = tokens.len() as u32;
                let prompt_eval_end = std::time::Instant::now();
                let prompt_eval_latency_ms =
                    prompt_eval_end.duration_since(start_time).as_millis() as f32;

                let mut sampler =
                    LlamaSampler::chain_simple([LlamaSampler::dist(seed), LlamaSampler::greedy()]);

                let mut n_cur = batch.n_tokens();
                let mut token_count = 0;
                let mut decoder = encoding_rs::UTF_8.new_decoder();

                // Generation loop
                while token_count < max_tokens {
                    if cancel_token.is_cancelled() {
                        break;
                    }

                    let token = sampler.sample(&ctx, batch.n_tokens() - 1);
                    sampler.accept(token);

                    // Skip special tokens: break on end of stream
                    if model.is_eog_token(token) {
                        break;
                    }

                    // Get token bytes and decode
                    let token_bytes = model
                        .token_to_bytes(token, Special::Tokenize)
                        .map_err(|e| anyhow!("Token conversion failed: {e}"))?;

                    let mut last_chunk = String::with_capacity(32);
                    let _ = decoder.decode_to_string(&token_bytes, &mut last_chunk, false);

                    batch.clear();
                    batch.add(token, n_cur, &[0], true)?;
                    ctx.decode(&mut batch)?;

                    n_cur += 1;
                    token_count += 1;

                    // Send token through channel
                    let _ = tx.blocking_send(Ok(Completion::Response(CompletionResponse {
                        text: last_chunk,
                        finish_reason: None,
                        raw_chunk: None,
                    })));
                }

                // Send finish reason
                let finish_reason = if token_count < max_tokens {
                    "Stop"
                } else {
                    "Length"
                };
                let _ = tx.blocking_send(Ok(Completion::Response(CompletionResponse {
                    text: "".to_string(),
                    finish_reason: Some(finish_reason.to_string()),
                    raw_chunk: None,
                })));

                // After generation loop, send completion metrics
                let completion_end = std::time::Instant::now();
                let completion_latency_ms =
                    completion_end.duration_since(prompt_eval_end).as_millis() as f32;

                let _ = tx.blocking_send(Ok(Completion::Metrics(CompletionMetrics {
                    prompt_tokens: prompt_token_count,
                    prompt_eval_latency_ms,
                    completion_tokens: token_count,
                    completion_latency_ms,
                })));

                Ok(())
            })() {
                let _ = tx.blocking_send(Err(e));
            }
        });

        // Use the Receiver as a Stream
        Box::pin(tokio_stream::wrappers::ReceiverStream::new(rx))
    }
}
