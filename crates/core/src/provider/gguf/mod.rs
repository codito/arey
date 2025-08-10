use crate::completion::{
    CancellationToken, ChatMessage, Completion, CompletionMetrics, CompletionModel,
    CompletionResponse,
};
use crate::model::{ModelConfig, ModelMetrics};
use crate::tools::{Tool, ToolSpec};
use anyhow::{Context, Result, anyhow};
use async_trait::async_trait;
use futures::stream::BoxStream;
use llama_cpp_2::token::LlamaToken;
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
use tracing::level_filters::LevelFilter;
use tracing::{debug, instrument};

mod template;
use crate::provider::gguf::template::{ToolCallParser, apply_chat_template};

struct GgufCacheState {
    tokens: Vec<LlamaToken>,
    raw_state: Vec<u8>,
}

pub struct GgufBaseModel {
    backend: Arc<LlamaBackend>,
    model: Arc<Mutex<LlamaModel>>,
    context_params: LlamaContextParams,
    // model_config: ModelConfig,
    metrics: ModelMetrics,
    cache_state: Arc<Mutex<Option<GgufCacheState>>>,
    model_name: String,
}

impl ModelConfig {
    /// Convert the model config to model params
    pub fn to_model_params(&self) -> LlamaModelParams {
        let n_gpu_layers = self
            .get_setting::<u32>("n_gpu_layers")
            .inspect(|n| debug!("CUDA enabled: n_gpu_layers = {}", n));

        LlamaModelParams::default().with_n_gpu_layers(n_gpu_layers.unwrap_or(0))
    }

    /// Convert the model config to context params
    pub fn to_context_params(&self) -> LlamaContextParams {
        LlamaContextParams::default()
            .with_n_threads(self.get_setting::<i32>("n_threads").unwrap_or(-1))
            .with_n_batch(self.get_setting::<u32>("n_batch").unwrap_or(512))
            .with_n_ctx(self.get_setting::<u32>("n_ctx").and_then(NonZeroU32::new))
    }
}

impl GgufBaseModel {
    #[instrument(skip(model_config))]
    pub fn new(model_config: ModelConfig) -> Result<Self> {
        // Send logs to tracing if verbose is enabled
        let tracing_enabled = is_tracing_enabled();
        send_logs_to_tracing(LogOptions::default().with_logs_enabled(tracing_enabled));

        let path: String = model_config
            .get_setting("path")
            .ok_or_else(|| anyhow!("'path' setting is required for gguf model"))?;
        let expand_path = shellexpand::tilde(&path).into_owned();
        let model_path = PathBuf::from(expand_path);
        debug!("Model path: {}", model_path.display());

        if !model_path.exists() {
            return Err(anyhow!(
                "Model loading failed: file not found at path: {}",
                model_path.display()
            ));
        }

        let backend = LlamaBackend::init().map_err(|e| anyhow!("Backend init failed: {e}"))?;

        let model_params = model_config.to_model_params();
        let context_params = model_config.to_context_params();
        debug!("Model configuration: {:?}", model_config.settings);
        debug!("Model params: {:?}", model_params);
        debug!("Context params: {:?}", context_params);

        let model = LlamaModel::load_from_file(&backend, &model_path, &model_params)
            .map_err(|e| anyhow!("Model loading failed: {e}"))?;

        Ok(Self {
            backend: Arc::new(backend),
            model: Arc::new(Mutex::new(model)),
            context_params,
            // model_config,
            metrics: ModelMetrics {
                init_latency_ms: 0.0,
            },
            cache_state: Arc::new(Mutex::new(None)),
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
        let cache_state = self.cache_state.clone();
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

                let n_batch = context_params.n_batch() as usize;

                // Create a new context
                let mut ctx = model
                    .new_context(&shared_backend, context_params)
                    .map_err(|e| anyhow!("Context creation failed: {e}"))?;

                // Prepare the prompt
                let prompt_eval_start = std::time::Instant::now();
                let template_str = model
                    .chat_template(None)
                    .context("Failed to retrieve default chat template")?
                    .to_string()?;
                let prompt = apply_chat_template(&template_str, &messages, tool_specs.as_deref())?;

                // Lock the shared cache state
                let cache_guard = cache_state.blocking_lock();

                let (prompt_tokens, common_prefix_len) = {
                    let all_tokens = model
                        .str_to_token(&prompt, AddBos::Always)
                        .map_err(|e| anyhow!("Tokenization failed: {e}"))?;

                    if let Some(cached) = &*cache_guard {
                        // Deserialize previous state into the new context
                        unsafe {
                            ctx.set_state_data(&cached.raw_state);
                        }

                        // Find the common prefix between the old and new prompt
                        let common_len = all_tokens
                            .iter()
                            .zip(cached.tokens.iter())
                            .take_while(|(a, b)| a == b)
                            .count();

                        // If the prompt has changed, remove the divergent part of the KV cache
                        if common_len < cached.tokens.len() {
                            // Seq ID 0, remove from common_len to the end of the cached sequence.
                            ctx.clear_kv_cache_seq(
                                Some(0),
                                Some(common_len as u32),
                                Some(cached.tokens.len() as u32),
                            )?;
                        }
                        (all_tokens, common_len)
                    } else {
                        (all_tokens, 0)
                    }
                };

                // We are done reading the cache, so we can release the lock
                drop(cache_guard);

                // Add the prompt tokens to context in batches
                let tokens_to_process = &prompt_tokens[common_prefix_len..];
                debug!(
                    "Evaluating prompt with {} tokens in batches of {} ({} from cache)",
                    tokens_to_process.len(),
                    n_batch,
                    common_prefix_len,
                );
                let mut batch = LlamaBatch::new(n_batch, 1);
                let mut in_token_count = common_prefix_len as i32; // Number of tokens in the current context
                for (chunk_idx, chunk) in tokens_to_process.chunks(n_batch).enumerate() {
                    batch.clear();
                    for (token_idx, token) in chunk.iter().enumerate() {
                        // Token position in context memory (KV cache) must be sequential
                        // (`token_pos`).
                        // Logits are computed only for the last token in the prompt. We set
                        // `is_last` accordingly.
                        let token_pos =
                            (common_prefix_len + chunk_idx * n_batch + token_idx) as i32;
                        let is_last = token_pos as usize == prompt_tokens.len() - 1;
                        batch.add(*token, token_pos, &[0], is_last)?;
                    }
                    in_token_count += batch.n_tokens();

                    ctx.decode(&mut batch)
                        .map_err(|e| anyhow!("Prompt decoding failed: {e}"))?;
                }

                // Capture prompt metrics
                let prompt_token_count = prompt_tokens.len() as u32;
                let prompt_eval_end = std::time::Instant::now();
                let prompt_eval_latency_ms = prompt_eval_end
                    .duration_since(prompt_eval_start)
                    .as_millis() as f32;
                debug!(
                    "Prompt evaluation took {}ms for {} tokens",
                    prompt_eval_latency_ms, prompt_token_count
                );

                let mut sampler =
                    LlamaSampler::chain_simple([LlamaSampler::dist(seed), LlamaSampler::greedy()]);

                let mut out_token_count = 0;
                let mut decoder = encoding_rs::UTF_8.new_decoder();

                // Generation loop
                debug!("Starting generation loop");
                // Use the token pos to sample in the last batch
                let mut sample_pos = (in_token_count - 1) % n_batch as i32;
                while out_token_count < max_tokens {
                    if cancel_token.is_cancelled() {
                        break;
                    }

                    let token = sampler.sample(&ctx, sample_pos);
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

                    // Note we're now decoding one token at a time to get the logits
                    batch.clear();
                    batch.add(token, in_token_count, &[0], true)?;
                    ctx.decode(&mut batch)?;

                    in_token_count += 1;
                    out_token_count += 1;
                    sample_pos = batch.n_tokens() - 1;

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
                let finish_reason = if out_token_count < max_tokens {
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
                debug!(
                    "Completed text generation with finish reason: {}",
                    finish_reason
                );

                // After generation loop, send completion metrics
                let completion_end = std::time::Instant::now();
                let completion_latency_ms =
                    completion_end.duration_since(prompt_eval_end).as_millis() as f32;

                // After the loop, before the task ends, save the new state.
                let state_size = ctx.get_state_size();
                let mut new_raw_state = vec![0u8; state_size];
                unsafe {
                    ctx.copy_state_data(new_raw_state.as_mut_ptr());
                }

                // Re-acquire lock to write the new state
                let mut cache_guard = cache_state.blocking_lock();
                *cache_guard = Some(GgufCacheState {
                    tokens: prompt_tokens,
                    raw_state: new_raw_state,
                });

                let _ = tx.blocking_send(Ok(Completion::Metrics(CompletionMetrics {
                    prompt_tokens: prompt_token_count,
                    prompt_eval_latency_ms,
                    completion_tokens: out_token_count,
                    completion_latency_ms,
                    raw_chunk: None,
                })));
                debug!(
                    "Generated {} tokens in {}ms",
                    out_token_count, completion_latency_ms
                );

                Ok(())
            })() {
                let _ = tx.blocking_send(Err(e));
            }
        });

        // Use the Receiver as a Stream
        Box::pin(tokio_stream::wrappers::ReceiverStream::new(rx))
    }
}

/// Returns true if any subscriber is active and its max level is at least TRACE.
pub fn is_tracing_enabled() -> bool {
    tracing::level_filters::LevelFilter::current() != LevelFilter::OFF
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
        // 1. Pointing to a valid GGUF model file.
        // 2. Creating a GgufBaseModel.
        // 3. Calling complete with sample messages.
        // 4. Asserting on the streamed response.
    }

    #[tokio::test]
    #[ignore = "requires a valid GGUF model file for a full integration test"]
    async fn test_gguf_model_kv_cache() {
        use crate::completion::{CancellationToken, ChatMessage, CompletionModel, SenderType};
        use futures::stream::StreamExt;
        use std::env;

        let model_path = env::var("AREY_TEST_GGUF_MODEL_PATH");
        if model_path.is_err() {
            println!("Skipping test_gguf_model_kv_cache: AREY_TEST_GGUF_MODEL_PATH not set");
            return;
        }

        let mut settings = HashMap::new();
        settings.insert("path".to_string(), model_path.unwrap().into());
        let model_config = ModelConfig {
            name: "test-gguf-cache".to_string(),
            provider: crate::model::ModelProvider::Gguf,
            settings,
        };
        let mut model = GgufBaseModel::new(model_config).unwrap();

        let messages = vec![ChatMessage {
            sender: SenderType::User,
            text: "The first three letters of the alphabet are:".to_string(),
            tools: Vec::new(),
        }];

        let mut settings = HashMap::new();
        settings.insert("max_tokens".to_string(), "10".to_string());

        let mut first_latency = 0.0;
        {
            let mut stream = model
                .complete(&messages, None, &settings, CancellationToken::new())
                .await;

            while let Some(Ok(completion)) = stream.next().await {
                if let crate::completion::Completion::Metrics(metrics) = completion {
                    first_latency = metrics.prompt_eval_latency_ms;
                    break;
                }
            }
        }

        assert!(first_latency > 0.0, "First latency should be positive");

        // Second call with same prompt should be faster due to cache
        let mut second_latency = 0.0;
        {
            let mut stream = model
                .complete(&messages, None, &settings, CancellationToken::new())
                .await;

            while let Some(Ok(completion)) = stream.next().await {
                if let crate::completion::Completion::Metrics(metrics) = completion {
                    second_latency = metrics.prompt_eval_latency_ms;
                    break;
                }
            }
        }

        assert!(second_latency > 0.0, "Second latency should be positive");
        assert!(
            second_latency < first_latency,
            "Second call with cache should be faster. first: {}, second: {}",
            first_latency,
            second_latency
        );

        // Third call with different prompt should have latency, but may still be fast.
        // This part mainly ensures it doesn't crash.
        let messages2 = vec![ChatMessage {
            sender: SenderType::User,
            text: "The first three letters of the alphabet are not:".to_string(),
            tools: Vec::new(),
        }];

        let mut third_latency = 0.0;
        {
            let mut stream = model
                .complete(&messages2, None, &settings, CancellationToken::new())
                .await;

            while let Some(Ok(completion)) = stream.next().await {
                if let crate::completion::Completion::Metrics(metrics) = completion {
                    third_latency = metrics.prompt_eval_latency_ms;
                    break;
                }
            }
        }
        assert!(third_latency > 0.0, "Third latency should be positive");
    }
}
