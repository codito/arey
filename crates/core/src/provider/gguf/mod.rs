use crate::completion::{
    CancellationToken, ChatMessage, Completion, CompletionMetrics, CompletionModel,
    CompletionResponse,
};
use crate::model::{ModelConfig, ModelMetrics};
use crate::tools::{Tool, ToolSpec};
use anyhow::{Context, Result, anyhow};
use async_trait::async_trait;
use futures::stream::BoxStream;
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
use tracing::instrument;

mod template;
use crate::provider::gguf::template::{ToolCallParser, apply_chat_template};

pub struct GgufBaseModel {
    backend: Arc<LlamaBackend>,
    model: Arc<Mutex<LlamaModel>>,
    context_params: LlamaContextParams,
    // model_config: ModelConfig,
    metrics: ModelMetrics,
    model_name: String,
}

impl GgufBaseModel {
    #[instrument(skip(model_config))]
    pub fn new(model_config: ModelConfig) -> Result<Self> {
        // if verbose {
        //     tracing_subscriber::fmt().init();
        // }
        send_logs_to_tracing(LogOptions::default().with_logs_enabled(true));

        let path = model_config
            .settings
            .get("path")
            .ok_or_else(|| anyhow!("'path' setting is required for gguf model"))?
            .as_str()
            .unwrap();
        let expand_path = shellexpand::tilde(path).into_owned();
        let model_path = PathBuf::from(expand_path);

        if !model_path.exists() {
            return Err(anyhow!(
                "Model loading failed: file not found at path: {}",
                model_path.display()
            ));
        }

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
            model_name: model_config.name,
        })
    }
}

#[async_trait]
impl CompletionModel for GgufBaseModel {
    fn metrics(&self) -> ModelMetrics {
        self.metrics.clone()
    }

    async fn load(&mut self, _text: &str) -> Result<()> {
        Ok(())
    }

    #[instrument(skip_all)]
    async fn complete(
        &mut self,
        messages: &[ChatMessage],
        tools: Option<&[Arc<dyn Tool>]>,
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

        let model_name = self.model_name.clone();
        let context_params = self.context_params.clone();
        let model_ref = self.model.clone();
        let cancel_token = cancel_token.clone();
        let shared_backend = self.backend.clone();
        // Clone messages to capture owned copies to move into the closure
        let messages: Vec<ChatMessage> = messages.to_vec();
        let tool_specs: Option<Vec<ToolSpec>> =
            tools.map(|t| t.iter().map(|tool| tool.clone().into()).collect());

        tokio::task::spawn_blocking(move || {
            if let Err(e) = (|| -> Result<()> {
                let (start_re, end_re) = template::get_tool_call_regexes(&model_name);
                let mut tool_parser = ToolCallParser::new(start_re, end_re)
                    .context("Failed to create tool call parser")?;
                let model = model_ref.blocking_lock();
                let mut ctx = model
                    .new_context(&shared_backend, context_params)
                    .map_err(|e| anyhow!("Context creation failed: {e}"))?;

                let template_str = model
                    .chat_template(None)
                    .context("Failed to retrieve default chat template")?
                    .to_string()?;
                let prompt = apply_chat_template(&template_str, &messages, tool_specs.as_deref())?;

                let tokens = model
                    .str_to_token(&prompt, AddBos::Always)
                    .map_err(|e| anyhow!("Tokenization failed: {e}"))?;

                let mut batch = LlamaBatch::new(tokens.len(), 1);
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

                    // Parse chunk for tool calls and text
                    let (plain_text, tool_calls) = tool_parser.parse(&last_chunk);

                    if !plain_text.is_empty() || !tool_calls.is_empty() {
                        // Send token through channel
                        let _ = tx.blocking_send(Ok(Completion::Response(CompletionResponse {
                            text: plain_text,
                            tool_calls: if tool_calls.is_empty() {
                                None
                            } else {
                                Some(tool_calls)
                            },
                            finish_reason: None,
                            raw_chunk: None,
                        })));
                    }
                }

                // Flush any remaining text from parser
                let remaining_text = tool_parser.flush();
                if !remaining_text.is_empty() {
                    let _ = tx.blocking_send(Ok(Completion::Response(CompletionResponse {
                        text: remaining_text,
                        tool_calls: None,
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
                    tool_calls: None,
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
                    raw_chunk: None,
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

#[cfg(test)]
mod tests {
    use crate::{model::ModelConfig, provider::gguf::GgufBaseModel};
    use std::collections::HashMap;

    #[test]
    fn test_gguf_model_new_missing_path() {
        let model_config = ModelConfig {
            name: "test-gguf".to_string(),
            provider: crate::model::ModelProvider::Gguf,
            settings: HashMap::new(),
        };
        let model = GgufBaseModel::new(model_config);
        assert!(model.is_err());
        assert!(
            model
                .err()
                .unwrap()
                .to_string()
                .contains("'path' setting is required for gguf model")
        );
    }

    #[test]
    fn test_gguf_model_new_path_not_found() {
        let mut settings = HashMap::new();
        settings.insert(
            "path".to_string(),
            "/path/to/non/existent/model.gguf".into(),
        );
        let model_config = ModelConfig {
            name: "test-gguf".to_string(),
            provider: crate::model::ModelProvider::Gguf,
            settings,
        };
        let model = GgufBaseModel::new(model_config);
        assert!(model.is_err());
        assert!(
            model
                .err()
                .unwrap()
                .to_string()
                .contains("Model loading failed")
        );
    }

    #[test]
    #[ignore = "requires a valid GGUF model file for a full integration test"]
    fn test_gguf_model_complete() {
        // This test requires a real model and is complex to set up.
        // It would involve:
        // 极速1. Pointing to a valid GGUF model file.
        // 2. Creating a GgufBaseModel.
        // 3. Calling complete with sample messages.
        // 4. Asserting on the streamed response.
    }
}
