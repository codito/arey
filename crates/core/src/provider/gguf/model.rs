//! GgufBaseModel: GGUF model with dedicated thread for persistent context.
//!
//! This module provides a GGUF model implementation that creates a dedicated OS thread
//! to own the LlamaModel and LlamaContext. This allows the context to persist across
//! requests without lifetime issues.

use crate::{
    completion::{
        CacheMetrics, CancellationToken, ChatMessage, Completion, CompletionMetrics,
        CompletionModel, CompletionResponse, SenderType,
    },
    model::ModelConfig,
    tools::Tool,
};
use anyhow::{Context, Result, anyhow};
use async_trait::async_trait;
use futures::stream::BoxStream;
use llama_cpp_2::{
    LogOptions,
    context::params::LlamaContextParams,
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{AddBos, LlamaModel, params::LlamaModelParams},
    sampling::LlamaSampler,
    send_logs_to_tracing,
    token::LlamaToken,
};
use std::{
    collections::HashMap,
    num::NonZeroU32,
    path::PathBuf,
    sync::Arc,
    thread::{self, JoinHandle},
};
use tokio::sync::{mpsc, oneshot};
use tracing::debug;

use crate::provider::gguf::checkpoint::{CheckpointManager, TransitionType};
use crate::provider::gguf::template::{ToolCallParser, apply_chat_template, get_tool_call_regexes};

static GGUF_BACKEND: once_cell::sync::Lazy<Arc<LlamaBackend>> = once_cell::sync::Lazy::new(|| {
    Arc::new(LlamaBackend::init().expect("llama backend init failed"))
});

pub struct GgufBaseModel {
    request_tx: mpsc::Sender<Request>,
    _thread_guard: Arc<ThreadGuard>,
}

/// Load status for tracking model loading progress.
#[derive(Debug, Clone)]
enum LoadStatus {
    /// Model is currently loading.
    Loading,
    /// Model loaded successfully.
    Loaded,
    /// Model loading failed with error message.
    Failed(String),
}

/// Returns a channel for sending load status updates.
type LoadStatusSender = mpsc::Sender<LoadStatus>;

struct ThreadGuard {
    _handle: Option<JoinHandle<()>>,
}

impl Drop for ThreadGuard {
    fn drop(&mut self) {
        if let Some(handle) = self._handle.take() {
            handle.join().ok();
        }
    }
}

/// Immutable model resources (set once at startup)
struct ModelResources<'a> {
    model: &'a LlamaModel,
    model_template: String,
    template_override_name: Option<String>,
    template_override_args: HashMap<String, serde_yaml::Value>,
    tool_parser: &'a ToolCallParser,
}

/// Mutable state that changes across requests
struct RequestState {
    n_batch: usize,
    previous_tokens: Vec<LlamaToken>,
    position: i32,
    checkpoint_manager: CheckpointManager,
    is_hybrid: bool,
    n_ctx: i32,
}

impl GgufBaseModel {
    /// Creates a new GGUF model instance.
    ///
    /// This is a blocking call - it waits for the model to fully load before returning.
    pub fn new(model_config: ModelConfig) -> Result<Self> {
        let model_path = model_config
            .get_setting::<String>("path")
            .context("GGUF model path not found")?;
        let model_path = PathBuf::from(shellexpand::tilde(&model_path).into_owned());

        if !model_path.exists() {
            anyhow::bail!("model file not found: {}", model_path.display());
        }

        let (request_tx, request_rx) = mpsc::channel::<Request>(4);
        let (status_tx, status_rx) = mpsc::channel::<LoadStatus>(4);

        let backend = GGUF_BACKEND.clone();
        let thread = thread::Builder::new()
            .name(format!("gguf-model-{}", model_config.key))
            .spawn(move || {
                run_event_loop(
                    backend,
                    model_path,
                    model_config,
                    request_rx,
                    Some(status_tx),
                );
            })
            .context("failed to spawn GGUF model thread")?;

        // Block until model is loaded (run in separate thread to avoid blocking async runtime)
        let mut status_rx = status_rx;
        std::thread::spawn(move || {
            loop {
                match status_rx.blocking_recv() {
                    Some(LoadStatus::Loaded) => return Ok(()),
                    Some(LoadStatus::Failed(e)) => {
                        return Err(anyhow!("failed to load model: {}", e));
                    }
                    Some(LoadStatus::Loading) => continue,
                    None => {
                        return Err(anyhow!("model loading channel closed unexpectedly"));
                    }
                }
            }
        })
        .join()
        .expect("model loading thread panicked")
        .context("failed to load model")?;

        Ok(Self {
            request_tx,
            _thread_guard: Arc::new(ThreadGuard {
                _handle: Some(thread),
            }),
        })
    }
}

#[async_trait]
impl CompletionModel for GgufBaseModel {
    fn metrics(&self) -> crate::model::ModelMetrics {
        crate::model::ModelMetrics::default()
    }

    async fn complete(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Arc<dyn Tool>]>,
        settings: &HashMap<String, String>,
        _cancel_token: CancellationToken,
    ) -> BoxStream<'static, Result<Completion>> {
        let (tx, rx) = mpsc::channel::<Result<Completion>>(100);

        let max_tokens = settings
            .get("max_tokens")
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(1024);

        tracing::debug!("GGUF complete() settings: max_tokens={}", max_tokens);
        let seed = settings
            .get("seed")
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(0);

        let messages: Vec<ChatMessage> = messages.to_vec();
        let tools: Option<Vec<_>> =
            tools.map(|t| t.iter().map(|tool| tool.clone().into()).collect());

        let request = Request {
            messages,
            tools: tools.map(|t| t.into_boxed_slice()),
            max_tokens,
            seed,
            settings: settings.clone(),
            response_tx: Some(ResponseSender::Streaming(tx)),
            reset: false,
        };

        let _ = self.request_tx.send(request).await;

        Box::pin(tokio_stream::wrappers::ReceiverStream::new(rx))
    }

    async fn reset(&self) -> Result<()> {
        let (tx, rx) = oneshot::channel();
        let request = Request {
            messages: vec![],
            tools: None,
            max_tokens: 0,
            seed: 0,
            settings: HashMap::new(),
            response_tx: Some(ResponseSender::OneShot(tx)),
            reset: true,
        };
        self.request_tx
            .send(request)
            .await
            .map_err(|e| anyhow!("failed to send reset request: {}", e))?;
        rx.await
            .map_err(|e| anyhow!("reset request failed: {}", e))?
    }
}

enum ResponseSender {
    Streaming(mpsc::Sender<Result<Completion>>),
    OneShot(oneshot::Sender<Result<()>>),
}

struct Request {
    messages: Vec<ChatMessage>,
    tools: Option<Box<[crate::tools::ToolSpec]>>,
    max_tokens: u32,
    seed: u32,
    settings: HashMap<String, String>,
    response_tx: Option<ResponseSender>,
    reset: bool,
}

fn run_event_loop(
    backend: Arc<LlamaBackend>,
    model_path: PathBuf,
    model_config: ModelConfig,
    mut request_rx: mpsc::Receiver<Request>,
    status_sender: Option<LoadStatusSender>,
) {
    send_logs_to_tracing(LogOptions::default());

    // Report loading status
    let _ = status_sender
        .as_ref()
        .map(|s| s.try_send(LoadStatus::Loading));

    let n_ctx = model_config.get_setting::<u32>("n_ctx").unwrap_or(4096);
    let n_batch = model_config.get_setting::<u32>("n_batch").unwrap_or(512);
    let n_gpu_layers = model_config.get_setting::<u32>("n_gpu_layers").unwrap_or(0);

    let model_params = LlamaModelParams::default().with_n_gpu_layers(n_gpu_layers);
    let context_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(n_ctx))
        .with_n_batch(n_batch);

    let model = match LlamaModel::load_from_file(&backend, &model_path, &model_params) {
        Ok(m) => m,
        Err(e) => {
            tracing::error!("failed to load model: {}", e);
            let _ = status_sender
                .as_ref()
                .map(|s| s.try_send(LoadStatus::Failed(e.to_string())));
            return;
        }
    };

    // Get model's built-in template from gguf metadata
    let model_template = model
        .chat_template(None)
        .ok()
        .and_then(|t| t.to_string().ok())
        .unwrap_or_default();

    // Get template override if specified in config
    let template_override = model_config.template_override();
    let (template_override_name, template_override_args) = match &template_override {
        Some(t) => (Some(t.name().to_string()), t.args()),
        None => (None, HashMap::new()),
    };

    tracing::debug!(
        model_has_template = !model_template.is_empty(),
        template_override = template_override_name.as_deref().unwrap_or("none"),
        "Template configuration"
    );

    let basename = model.meta_val_str("general.basename").unwrap_or_default();
    let is_hybrid = model.is_recurrent() || model.is_hybrid();

    let patterns = get_tool_call_regexes(&basename);
    let tool_parser = match ToolCallParser::new(patterns) {
        Ok(p) => p,
        Err(e) => {
            tracing::error!("failed to create tool parser: {}", e);
            let _ = status_sender
                .as_ref()
                .map(|s| s.try_send(LoadStatus::Failed(e.to_string())));
            return;
        }
    };

    let mut context = match model.new_context(&backend, context_params) {
        Ok(ctx) => ctx,
        Err(e) => {
            tracing::error!("failed to create context: {}", e);
            let _ = status_sender
                .as_ref()
                .map(|s| s.try_send(LoadStatus::Failed(e.to_string())));
            return;
        }
    };

    // Report loaded status
    let _ = status_sender
        .as_ref()
        .map(|s| s.try_send(LoadStatus::Loaded));

    let n_batch_ctx = context.n_batch() as usize;

    // Only allocate checkpoint storage for hybrid/recurrent models.
    // Will dynamically adjust based on actual checkpoint size (up to 16)
    let max_checkpoints = if is_hybrid { 16 } else { 0 };
    let mut state = RequestState {
        n_batch: n_batch_ctx,
        previous_tokens: Vec::new(),
        position: 0,
        checkpoint_manager: CheckpointManager::new(max_checkpoints, is_hybrid, n_ctx as usize),
        is_hybrid,
        n_ctx: n_ctx as i32,
    };

    while let Some(request) = request_rx.blocking_recv() {
        let resources = ModelResources {
            model: &model,
            model_template: model_template.clone(),
            template_override_name: template_override_name.clone(),
            template_override_args: template_override_args.clone(),
            tool_parser: &tool_parser,
        };
        handle_request(resources, &mut state, &mut context, request);
    }
}

fn determine_transition_type(messages: &[ChatMessage]) -> TransitionType {
    for msg in messages.iter().rev() {
        match msg.sender {
            SenderType::Tool => return TransitionType::ToolResponse,
            SenderType::Assistant => {
                if msg.tools.is_some() {
                    return TransitionType::ToolCall;
                }
                return TransitionType::TurnEnd;
            }
            SenderType::User => return TransitionType::TurnStart,
            SenderType::System => continue,
        }
    }
    TransitionType::TurnStart
}

#[tracing::instrument(skip_all)]
fn handle_request(
    resources: ModelResources,
    state: &mut RequestState,
    context: &mut llama_cpp_2::context::LlamaContext,
    request: Request,
) {
    let Request {
        messages,
        tools,
        max_tokens,
        seed,
        settings,
        response_tx,
        reset,
    } = request;

    if reset {
        debug!("handle_request: resetting model state");
        context.clear_kv_cache();
        state.position = 0;
        state.previous_tokens.clear();
        state.checkpoint_manager.clear();
        if let Some(ResponseSender::OneShot(tx)) = response_tx {
            let _ = tx.send(Ok(()));
        }
        return;
    }

    // For non-reset requests, we need a response channel for streaming
    let Some(ResponseSender::Streaming(response_tx)) = response_tx else {
        debug!("handle_request: no response channel for non-reset request");
        return;
    };

    debug!(
        "handle_request: messages={}, position={}",
        messages.len(),
        state.position
    );

    // Extract enable_thinking from settings and merge with template_args
    let mut template_args = resources.template_override_args.clone();
    let mut enable_thinking = template_args
        .get("enable_thinking")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    if let Some(enabled) = settings.get("enable_thinking") {
        let enabled_bool = enabled.parse::<bool>().unwrap_or(false);
        template_args.insert(
            "enable_thinking".to_string(),
            serde_yaml::Value::Bool(enabled_bool),
        );
        enable_thinking = enabled_bool;
    }

    // Save checkpoint at START of request (before processing new tokens)
    // This captures the state at conversation boundary - highest priority for matching
    if state.is_hybrid && state.position > 0 {
        state.checkpoint_manager.save_at_transition(
            context,
            vec![],
            state.position as usize,
            TransitionType::RequestStart,
        );
    }

    let prompt = match apply_chat_template(
        &resources.model_template,
        resources.template_override_name.as_deref(),
        &messages,
        tools.as_deref(),
        template_args,
    ) {
        Ok(p) => p,
        Err(e) => {
            let _ = response_tx.blocking_send(Err(anyhow!("template failed: {}", e)));
            return;
        }
    };

    let tokens = match resources.model.str_to_token(&prompt, AddBos::Always) {
        Ok(t) => t,
        Err(e) => {
            let _ = response_tx.blocking_send(Err(anyhow!("tokenization failed: {}", e)));
            return;
        }
    };

    debug!(
        "Tokenization: prompt length={}, token count={}",
        prompt.len(),
        tokens.len()
    );

    if tokens.is_empty() {
        let _ = response_tx.blocking_send(Ok(Completion::Response(CompletionResponse {
            text: String::new(),
            thought: None,
            tool_calls: None,
            finish_reason: Some("stop".to_string()),
            raw_chunk: None,
        })));
        let _ = response_tx.blocking_send(Ok(Completion::Metrics(CompletionMetrics {
            prompt_tokens: 0,
            prompt_eval_latency_ms: 0.0,
            completion_tokens: 0,
            completion_latency_ms: 0.0,
            thought: None,
            raw_chunk: None,
            cache_metrics: Some(CacheMetrics {
                cache_hit: false,
                strategy: Some(if state.is_hybrid {
                    "Hybrid".to_string()
                } else {
                    "KvCacheOnly".to_string()
                }),
                tokens_skipped: None,
                n_ctx: Some(state.n_ctx),
                checkpoint_transition: None,
            }),
        })));
        return;
    }

    // Compute cache status using checkpoint manager
    let cache_status = state.checkpoint_manager.cache_status(
        state.is_hybrid,
        state.position,
        &tokens,
        &state.previous_tokens,
    );

    // Debug logging for cache status
    debug!(
        "Cache status: checkpoint_restored={}, tokens_to_skip={}, checkpoint_count={}, previous_tokens_len={}, current_tokens_len={}",
        cache_status.checkpoint_restored,
        cache_status.tokens_to_skip,
        state.checkpoint_manager.len(),
        state.previous_tokens.len(),
        tokens.len()
    );

    // Log checkpoint divergence for hybrid models when no match found
    if state.is_hybrid && !cache_status.checkpoint_restored && !state.checkpoint_manager.is_empty()
    {
        for cp in state.checkpoint_manager.checkpoints() {
            if cp.tokens.is_empty() || cp.tokens.len() > tokens.len() {
                continue;
            }
            let diverge_pos = cp
                .tokens
                .iter()
                .zip(tokens.iter())
                .position(|(a, b)| a != b);

            if let Some(pos) = diverge_pos {
                // Decode tokens to strings with more detail
                let decode_token = |t: &LlamaToken| -> String {
                    let token_id = t.0;
                    match resources.model.token_to_piece_bytes(*t, 32, true, None) {
                        Ok(bytes) => {
                            let s = String::from_utf8_lossy(&bytes).to_string();
                            if s.chars().any(|c| c.is_control()) {
                                format!("{}:{:?}", token_id, bytes)
                            } else {
                                format!("{}:'{}'", token_id, s)
                            }
                        }
                        Err(e) => format!("{}:error={}", token_id, e),
                    }
                };

                // Format token slice centered on pos with indices
                let format_with_indices = |tokens: &[LlamaToken], pos: usize| -> String {
                    let start = pos.saturating_sub(5);
                    let end = (pos + 6).min(tokens.len());
                    tokens[start..end]
                        .iter()
                        .enumerate()
                        .map(|(i, t)| format!("[{}]{}", start + i, decode_token(t)))
                        .collect::<Vec<_>>()
                        .join(" | ")
                };

                debug!(
                    "Checkpoint diverge: cp_pos={}, diverge_at={}\n  cp:     [{}]\n  curr:   [{}]",
                    cp.position,
                    pos,
                    format_with_indices(&cp.tokens, pos),
                    format_with_indices(&tokens, pos)
                );
            }
        }
    }

    // Restore checkpoint if found (needs context)
    if cache_status.checkpoint_restored
        && let Some(cp) = state.checkpoint_manager.checkpoints().iter().find(|cp| {
            cp.tokens.len() == cache_status.tokens_to_skip
                && cp.tokens.len() <= tokens.len()
                && cp.tokens.iter().zip(tokens.iter()).all(|(a, b)| a == b)
        })
    {
        debug!(
            "Checkpoint found: restoring state at position {}",
            cp.position
        );
        if state.checkpoint_manager.restore(context, cp) {
            state.position = cp.position as i32;
            debug!(
                "Checkpoint restored: skipping {} tokens, position now {}",
                cache_status.tokens_to_skip, state.position
            );
        }
    }

    let mut tokens_to_skip = cache_status.tokens_to_skip;
    // Cap tokens_to_skip for slicing (can't skip more than available)
    // Note: position is set from checkpoint restore, not from this value
    tokens_to_skip = tokens_to_skip.min(tokens.len());
    let tokens_to_process = &tokens[tokens_to_skip..];

    if cache_status.cache_hit {
        debug!(
            "Cache hit: skipping {} tokens, processing {} new tokens",
            tokens_to_skip,
            tokens_to_process.len()
        );
    }

    let total_tokens = tokens.len() as i32;
    if total_tokens > state.n_ctx {
        let _ = response_tx.blocking_send(Err(anyhow!(
            "prompt ({}) exceeds context window ({})",
            total_tokens,
            state.n_ctx
        )));
        return;
    }

    // Check for overflow AFTER computing tokens_to_skip
    // Calculate effective position after any checkpoint restore
    let effective_position = cache_status.restored_position.unwrap_or(state.position);

    // Check if prompt exceeds context window
    if tokens.len() as i32 > state.n_ctx {
        let _ = response_tx.blocking_send(Err(anyhow!(
            "prompt ({}) exceeds context window ({})",
            tokens.len(),
            state.n_ctx
        )));
        return;
    }

    let mut had_overflow = false;
    let has_overflow = state.checkpoint_manager.is_context_overflow(
        state.n_ctx,
        effective_position,
        tokens.len(),
        tokens_to_skip,
    );
    if has_overflow {
        let new_tokens = (tokens.len() as i32 - tokens_to_skip as i32).max(0);
        debug!(
            "Context overflow: effective_pos={}, total={}, new={}, tokens_to_skip={}, n_ctx={}, available={}, clearing",
            effective_position,
            tokens.len(),
            new_tokens,
            tokens_to_skip,
            state.n_ctx,
            state.n_ctx - effective_position
        );
        context.clear_kv_cache();
        state.position = 0;
        state.previous_tokens.clear();
        state.checkpoint_manager.clear();
        had_overflow = true;
    }

    let mut batch = LlamaBatch::new(state.n_batch, 1);
    let prompt_start = std::time::Instant::now();
    let mut in_token_count = state.position;

    if !tokens_to_process.is_empty() {
        // Process tokens in chunks to handle prompts larger than n_batch
        for chunk in tokens_to_process.chunks(state.n_batch) {
            batch.clear();
            for (i, &token) in chunk.iter().enumerate() {
                let want_logits = i == chunk.len() - 1;
                if let Err(e) = batch.add(token, in_token_count, &[0], want_logits) {
                    let _ = response_tx.blocking_send(Err(anyhow!("batch add failed: {}", e)));
                    return;
                }
                in_token_count += 1;
            }
            if let Err(e) = context.decode(&mut batch) {
                let _ = response_tx.blocking_send(Err(anyhow!("decode failed: {}", e)));
                return;
            }

            // Save periodic checkpoint if we've crossed the interval
            if state.is_hybrid
                && state
                    .checkpoint_manager
                    .should_save_periodic(in_token_count as usize)
            {
                // tokens contains new tokens for this request
                // in_token_count = previous_position + tokens_processed_in_this_request
                // We need to save all tokens up to current position = previous_tokens + processed new tokens
                let tokens_processed = (in_token_count - state.position) as usize;
                let tokens_for_checkpoint = state
                    .previous_tokens
                    .iter()
                    .chain(tokens.iter().take(tokens_processed))
                    .cloned()
                    .collect::<Vec<_>>();
                state.checkpoint_manager.save_at_transition(
                    context,
                    tokens_for_checkpoint,
                    in_token_count as usize,
                    TransitionType::Periodic,
                );
            }
        }
    } else {
        debug!(
            "Cache hit: no tokens to process, using existing KV cache at position {}",
            state.position
        );
    }

    state.position = in_token_count;
    state.previous_tokens = tokens.clone();

    debug!(
        "After prompt: position={}, batch_n_tokens={}, is_hybrid={}, tokens_to_process.len()={}",
        state.position,
        batch.n_tokens(),
        state.is_hybrid,
        tokens_to_process.len()
    );

    let prompt_eval_latency_ms = prompt_start.elapsed().as_millis() as f32;
    let generation_start = std::time::Instant::now();

    let mut sampler =
        LlamaSampler::chain_simple([LlamaSampler::dist(seed), LlamaSampler::greedy()]);
    debug!("Sampler created, about to call sample()");
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let mut generated_tokens = 0u32;

    resources.tool_parser.set_in_thought(enable_thinking);

    while generated_tokens < max_tokens {
        // Use -1 to sample from last token in batch (only last token has logits).
        // Fixes "invalid logits id" error.
        let token = sampler.sample(context, -1);
        sampler.accept(token);

        if resources.model.is_eog_token(token) {
            break;
        }

        let token_bytes = match resources.model.token_to_piece_bytes(token, 32, true, None) {
            Ok(b) => b,
            Err(_) => {
                let _ = response_tx.blocking_send(Err(anyhow!("token conversion failed")));
                return;
            }
        };

        let mut chunk = String::with_capacity(32);
        let _ = decoder.decode_to_string(&token_bytes, &mut chunk, false);

        batch.clear();
        if let Err(e) = batch.add(token, in_token_count, &[0], true) {
            let _ = response_tx.blocking_send(Err(anyhow!("batch add failed: {}", e)));
            return;
        }
        if let Err(e) = context.decode(&mut batch) {
            let _ = response_tx.blocking_send(Err(anyhow!("decode failed: {}", e)));
            return;
        }

        in_token_count += 1;
        state.position = in_token_count;
        let _sample_pos = batch.n_tokens() - 1;

        let (plain_text, thought, tool_calls) = resources.tool_parser.parse(&chunk);
        if !plain_text.is_empty() || thought.is_some() || !tool_calls.is_empty() {
            let _ = response_tx.blocking_send(Ok(Completion::Response(CompletionResponse {
                text: plain_text,
                thought,
                tool_calls: if tool_calls.is_empty() {
                    None
                } else {
                    Some(tool_calls)
                },
                finish_reason: None,
                raw_chunk: None,
            })));
        }

        generated_tokens += 1;
    }

    let completion_latency_ms = generation_start.elapsed().as_millis() as f32;

    let _ = response_tx.blocking_send(Ok(Completion::Response(CompletionResponse {
        text: String::new(),
        thought: None,
        tool_calls: None,
        finish_reason: Some(
            if generated_tokens < max_tokens {
                "stop"
            } else {
                "length"
            }
            .to_string(),
        ),
        raw_chunk: None,
    })));

    let _ = response_tx.blocking_send(Ok(Completion::Metrics(CompletionMetrics {
        prompt_tokens: tokens.len() as u32,
        prompt_eval_latency_ms,
        completion_tokens: generated_tokens,
        completion_latency_ms,
        thought: None,
        raw_chunk: None,
        cache_metrics: Some(CacheMetrics {
            cache_hit: cache_status.cache_hit,
            strategy: Some(if state.is_hybrid {
                "Hybrid".to_string()
            } else {
                "KvCacheOnly".to_string()
            }),
            tokens_skipped: if cache_status.cache_hit {
                Some(cache_status.tokens_to_skip)
            } else {
                None
            },
            n_ctx: Some(state.n_ctx),
            checkpoint_transition: cache_status.restored_transition.map(|t| format!("{:?}", t)),
        }),
    })));

    if state.is_hybrid && !had_overflow {
        let current_position = state.position as usize;
        if current_position > 0 {
            let transition = determine_transition_type(&messages);
            debug!(
                "Saving checkpoint at position {} for {:?}",
                current_position, transition
            );
            state.checkpoint_manager.save_at_transition(
                context,
                tokens.to_vec(),
                current_position,
                transition,
            );
        }
    }

    resources.tool_parser.reset();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gguf_base_model2_creation_missing_file() {
        let mut settings = HashMap::new();
        settings.insert(
            "path".to_string(),
            serde_yaml::Value::String("/nonexistent/model.gguf".to_string()),
        );
        let config = ModelConfig {
            key: "test".to_string(),
            name: "test".to_string(),
            provider: crate::model::ModelProvider::Gguf,
            settings,
        };
        let result = GgufBaseModel::new(config);
        assert!(result.is_err());
        assert!(
            result
                .err()
                .unwrap()
                .to_string()
                .contains("model file not found")
        );
    }
}
