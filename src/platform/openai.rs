use crate::core::completion::{ChatMessage, CompletionMetrics, CompletionModel, CompletionResponse};
use crate::core::model::{ModelConfig, ModelMetrics};
use anyhow::{Result, anyhow, Context};
use async_stream::stream;
use async_trait::async_trait;
use futures::stream::BoxStream;
use futures::{pin_mut, StreamExt, TryStreamExt};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tokio::sync::OnceCell;

#[derive(Debug, Deserialize, Clone)]
pub struct OpenAISettings {
    base_url: String,
    api_key: String,
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

        // If api_key starts with "env:", read from environment variable
        let api_key = if settings.api_key.starts_with("env:") {
            let env_key = &settings.api_key[4..].trim();
            std::env::var(env_key).map_err(|_| anyhow!("Environment variable {} not found", env_key))?
        } else {
            settings.api_key.clone()
        };

        // Replace the settings with the resolved key
        let mut resolved_settings = settings.clone();
        resolved_settings.api_key = api_key;

        let client = Client::new();
        Ok(Self {
            config,
            client,
            metrics: ModelMetrics {
                init_latency_ms: 0.0,
            },
            settings: resolved_settings,
        })
    }
}

#[derive(Debug, Serialize)]
struct OpenAIMessage {
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct OpenAIRequest {
    model: String,
    messages: Vec<OpenAIMessage>,
    stream: bool,
    temperature: f64,
}

#[derive(Debug, Deserialize)]
struct OpenAIDelta {
    content: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAIChoice {
    delta: OpenAIDelta,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAISSE {
    choices: Vec<OpenAIChoice>,
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
         // Convert ChatMessage to OpenAI format
         let messages: Vec<OpenAIMessage> = messages
         .iter()
         .map(|msg| OpenAIMessage {
             role: msg.sender.role().to_string(),
             content: msg.text.clone(),
         })
         .collect();

        // Merge settings: start with defaults, then override with passed settings
        let mut request_settings = HashMap::new();
        request_settings.insert("temperature".to_string(), "0.7".to_string());
        for (k, v) in settings {
            request_settings.insert(k.clone(), v.clone());
        }

        // Extract temperature from settings
        let temperature = request_settings
            .get("temperature")
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(0.7);

        let api_url = format!("{}/chat/completions", self.settings.base_url);
        let api_key = self.settings.api_key.clone();
        let body = OpenAIRequest {
            model: self.config.name.clone(),
            messages,
            stream: true,
            temperature,
        };

        // Create and stream the response
        let s = stream! {
            let start_time = Instant::now();
            let mut response = match self.client
                .post(&api_url)
                .json(&body)
                .bearer_auth(api_key)
                .send()
                .await
            {
                Ok(res) => res,
                Err(_e) => {
                    yield Err(anyhow!("Request failed"));
                    return;
                }
            };

            if !response.status().is_success() {
                let status = response.status();
                let error_body = response.text().await.ok().unwrap_or_default();
                yield Err(anyhow!("OpenAI API error: {} - {}", status, error_body));
                return;
            }

            let mut stream = response.bytes_stream();
            let mut buffer = String::new();

            // Track whether we've received first response
            let mut first_chunk_received = false;
            let mut finish_reason: Option<String> = None;

            while let Some(chunk) = stream.try_next().await.map_err(|e| anyhow!("Stream error: {}", e)).transpose() {
                let chunk = match chunk {
                    Ok(chunk) => chunk,
                    Err(e) => {
                        yield Err(e);
                        break;
                    }
                };

                // Convert chunk to text and add to buffer
                let chunk_str = match String::from_utf8(chunk.to_vec()) {
                    Ok(s) => s,
                    Err(_) => continue,
                };
                buffer.push_str(&chunk_str);

                // Process each SSE event (data: { ... })
                while let Some(end) = buffer.find("\n\n") {
                    let event = buffer.drain(..=end).collect::<String>();
                    let event = event.trim();

                    if !event.starts_with("data:") {
                        continue;
                    }

                    // "data:" is 5 characters, so skip that
                    let json_str = &event[4..].trim();
                    if json_str == "[DONE]" {
                        finish_reason = Some("stop".to_string());
                        break;
                    }

                    let openai_event: OpenAISSE = match serde_json::from_str(json_str) {
                        Ok(event) => event,
                        Err(_) => continue,
                    };

                    if let Some(choice) = openai_event.choices.first() {
                        // First chunk in stream
                        if !first_chunk_received {
                            let prompt_eval_latency = start_time.elapsed().as_millis() as f32;
                            first_chunk_received = true;
                            
                            // Yield first chunk with prompt latency
                            let chunk_text = choice.delta.content.clone().unwrap_or_default();
                            let metrics = CompletionMetrics {
                                prompt_tokens: 0, // TODO: token counting
                                prompt_eval_latency_ms: prompt_eval_latency,
                                completion_tokens: 0, // TODO: token counting
                                completion_runs: 1,
                                completion_latency_ms: 0.0, // first chunk latency is in prompt
                            };
                            yield Ok(CompletionResponse {
                                text: chunk_text,
                                finish_reason: choice.finish_reason.clone(),
                                metrics,
                            });
                        } else {
                            // Subsequent chunks
                            let latency_ms = start_time.elapsed().as_micros() as f32 / 1000.0;
                            let chunk_text = choice.delta.content.clone().unwrap_or_default();
                            let metrics = CompletionMetrics {
                                prompt_tokens: 0,
                                prompt_eval_latency_ms: 0.0,
                                completion_tokens: 0,
                                completion_runs: 1,
                                completion_latency_ms: latency_ms,
                            };
                            yield Ok(CompletionResponse {
                                text: chunk_text,
                                finish_reason: choice.finish_reason.clone(),
                                metrics,
                            });
                        }
                        if let Some(reason) = &choice.finish_reason {
                            finish_reason = Some(reason.clone());
                        }
                    }
                }
            }
        };

        Box::pin(s)
    }

    async fn count_tokens(&self, _text: &str) -> usize {
        // TODO: Implement token counting
        0
    }

    async fn free(&mut self) {
        // No resources to free
    }
}
