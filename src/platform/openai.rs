use crate::core::completion::{
    ChatMessage, CompletionMetrics, CompletionModel, CompletionResponse,
};
use crate::core::model::{ModelConfig, ModelMetrics};
use anyhow::{Result, anyhow};
use async_openai::config::OpenAIConfig;
use async_openai::{
    Client as OpenAIClient,
    types::{ChatCompletionRequestMessage, CreateChatCompletionRequestArgs},
};
use async_trait::async_trait;
use futures::stream::{BoxStream, StreamExt};
use std::collections::HashMap;
use std::time::Instant;

#[derive(Debug, Clone, serde::Deserialize)]
pub struct OpenAISettings {
    base_url: String,
    api_key: String,
}

pub struct OpenAIBaseModel {
    config: ModelConfig,
    client: OpenAIClient<OpenAIConfig>,
    metrics: ModelMetrics,
    settings: OpenAISettings,
}

impl OpenAIBaseModel {
    pub fn new(model_config: ModelConfig) -> Result<Self> {
        let settings: OpenAISettings = serde_yaml::from_value(
            serde_yaml::to_value(&model_config.settings)
                .map_err(|_e| anyhow!("Invalid settings structure"))?,
        )?;

        // If api_key starts with "env:", read from environment variable
        let api_key = if settings.api_key.starts_with("env:") {
            let env_key = &settings.api_key[4..].trim();
            std::env::var(env_key)
                .map_err(|_| anyhow!("Environment variable {} not found", env_key))?
        } else {
            settings.api_key.clone()
        };

        // Create OpenAI configuration
        let config = OpenAIConfig::new()
            .with_api_key(api_key.clone())
            .with_api_base(settings.base_url.clone());

        let client = OpenAIClient::with_config(config);

        // Replace the settings with the resolved key
        let mut resolved_settings = settings.clone();
        resolved_settings.api_key = api_key;

        Ok(Self {
            config: model_config,
            client,
            metrics: ModelMetrics {
                init_latency_ms: 0.0,
            },
            settings: resolved_settings,
        })
    }

    fn to_openai_message(msg: &ChatMessage) -> ChatCompletionRequestMessage {
        match msg.sender {
            crate::core::completion::SenderType::System => ChatCompletionRequestMessage::System(
                async_openai::types::ChatCompletionRequestSystemMessageArgs::default()
                    .content(msg.text.as_str()) // Use as_str to get &str
                    .build()
                    .unwrap(),
            ),
            crate::core::completion::SenderType::Assistant => {
                ChatCompletionRequestMessage::Assistant(
                    async_openai::types::ChatCompletionRequestAssistantMessageArgs::default()
                        .content(msg.text.as_str()) // Use as_str to get &str
                        .build()
                        .unwrap(),
                )
            }
            crate::core::completion::SenderType::User => ChatCompletionRequestMessage::User(
                async_openai::types::ChatCompletionRequestUserMessageArgs::default()
                    .content(msg.text.as_str()) // Use as_str to get &str
                    .build()
                    .unwrap(),
            ),
        }
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
        // No-op for remote models
        Ok(())
    }

    async fn complete(
        &mut self,
        messages: &[ChatMessage],
        settings: &HashMap<String, String>,
    ) -> BoxStream<'_, CompletionResponse> {
        let mut request = CreateChatCompletionRequestArgs::default();

        // Map messages to OpenAI message types
        let openai_messages: Vec<ChatCompletionRequestMessage> = messages
            .iter()
            .map(OpenAIBaseModel::to_openai_message)
            .collect();

        // Set model and messages
        request.model(&self.config.name).messages(openai_messages);

        // Set max_tokens and temperature if provided
        if let Some(max_tokens_str) = settings.get("max_tokens") {
            if let Ok(max_tokens) = max_tokens_str.parse::<u32>() {
                request.max_tokens(max_tokens);
            }
        }

        if let Some(temperature_str) = settings.get("temperature") {
            if let Ok(temperature) = temperature_str.parse::<f64>() {
                request.temperature(temperature as f32);
            }
        }

        // Build request
        let request = match request.build() {
            Ok(req) => req,
            Err(err) => {
                return Box::pin(futures::stream::once(async move {
                    CompletionResponse {
                        text: format!("Invalid request: {:?}", err),
                        finish_reason: Some("error".to_string()),
                        metrics: CompletionMetrics {
                            prompt_tokens: 0,
                            prompt_eval_latency_ms: 0.0,
                            completion_tokens: 0,
                            completion_runs: 0,
                            completion_latency_ms: 0.0,
                        },
                    }
                }));
            }
        };

        // Start the timer
        let start_time = Instant::now();
        let mut prev_time = start_time;
        let mut first_chunk = true;

        // Create the stream
        let outer_stream = async_stream::stream! {
            let start_time = start_time.clone();
            let mut prev_time = prev_time.clone();
            let mut first_chunk = first_chunk.clone();

            // Send the request and get back a streaming response
            match self.client.chat().create_stream(request).await {
                Ok(response) => {
                    let mut stream = response;

                    while let Some(next) = stream.next().await {
                        // Time measurement
                        let now = Instant::now();
                        let elapsed = now.duration_since(prev_time).as_millis() as f32;
                        prev_time = now;

                        match next {
                            Ok(chunk) => {
                                // Check if we have a content block in the first choice
                                if let Some(choice) = chunk.choices.first() {
                                    let text = choice.delta.content.clone().unwrap_or_else(|| "".to_string());

                                    // For the first chunk, set the prompt_eval_latency_ms
                                    let mut prompt_eval_latency = 0.0;
                                    let mut completion_latency = elapsed;
                                    if first_chunk {
                                        prompt_eval_latency = elapsed;
                                        completion_latency = 0.0;
                                        first_chunk = false;
                                    }

                                    yield CompletionResponse {
                                        text: text.to_string(),
                                        finish_reason: choice.finish_reason.as_ref().map(|x| format!("{:?}", x)),
                                        metrics: CompletionMetrics {
                                            prompt_tokens: 0, // TODO: token counting
                                            prompt_eval_latency_ms: prompt_eval_latency,
                                            completion_tokens: 0, // TODO: token counting
                                            completion_runs: 1,
                                            completion_latency_ms: completion_latency,
                                        },
                                    };
                                }
                            }
                            Err(err) => {
                                yield CompletionResponse {
                                    text: format!("OpenAI error: {}", err),
                                    finish_reason: Some("error".to_string()),
                                    metrics: CompletionMetrics {
                                        prompt_tokens: 0,
                                        prompt_eval_latency_ms: 0.0,
                                        completion_tokens: 0,
                                        completion_runs: 0,
                                        completion_latency_ms: 0.0,
                                    },
                                };
                            }
                        }
                    }
                }
                Err(err) => {
                    yield CompletionResponse {
                        text: format!("Request failed: {:?}", err),
                        finish_reason: Some("error".to_string()),
                        metrics: CompletionMetrics {
                            prompt_tokens: 0,
                            prompt_eval_latency_ms: 0.0,
                            completion_tokens: 0,
                            completion_runs: 0,
                            completion_latency_ms: 0.0,
                        },
                    };
                }
            }
        };

        Box::pin(outer_stream)
    }

    async fn count_tokens(&self, _text: &str) -> usize {
        // TODO: Implement token counting
        0
    }

    async fn free(&mut self) {
        // No resources to free
    }
}
