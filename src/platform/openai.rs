use crate::core::completion::{
    ChatMessage, CompletionMetrics, CompletionModel, CompletionResponse,
};
use crate::core::model::{ModelConfig, ModelMetrics};
use anyhow::{Result, anyhow};
use async_stream::stream;
use async_trait::async_trait;
use futures::StreamExt;
use futures::stream::BoxStream;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

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

#[cfg(test)]
mod tests {
    use super::*;
    use mockito::Server;
    use crate::core::completion::SenderType;
    use crate::core::model::{ModelProvider, ModelCapability};
    use serde_yaml::{Mapping, Value};

    #[tokio::test]
    async fn test_openai_successful_response() {
        let mut server = mockito::Server::new();
        let mock_url = server.url();
        
        let model_config = ModelConfig {
            name: "test-model".to_string(),
            r#type: ModelProvider::Openai,
            capabilities: vec![ModelCapability::Completion],
            settings: Value::Mapping(Mapping::from_iter(vec![
                ("base_url".to_string(), mock_url.clone().into()),
                ("api_key".to_string(), "dummy_key".into()),
            ]))
        };

        let mut model = OpenAIBaseModel::new(model_config).unwrap();
        let messages = vec![ChatMessage {
            sender: SenderType::User,
            text: "Hello".to_string(),
        }];
        let settings = HashMap::new();

        let mock_response = server
            .mock("POST", "/chat/completions")
            .with_status(200)
            .with_body(
                r#"data: {"choices": [{"delta": {"content": "Test "}, "finish_reason": null}]}
data: {"choices": [{"delta": {"content": "response"}, "finish_reason": "stop"}]}
data: [DONE]"#
            )
            .create();

        let mut stream = model.complete(&messages, &settings).await;
        let mut responses = vec![];
        while let Some(response) = stream.next().await {
            responses.push(response);
        }

        assert_eq!(responses.len(), 2);
        assert_eq!(responses[0].text, "Test ");
        assert_eq!(responses[1].text, "response");
        mock_response.assert();
    }

    #[tokio::test]
    async fn test_openai_error_response() {
        let mut server = mockito::Server::new();
        let mock_url = server.url();
        
        let model_config = ModelConfig {
            name: "test-model".to_string(),
            r#type: ModelProvider::Openai,
            capabilities: vec![ModelCapability::Completion],
            settings: Value::Mapping(Mapping::from_iter(vec![
                ("base_url".to_string(), mock_url.clone().into()),
                ("api_key".to_string(), "dummy_key".into()),
            ]))
        };

        let mut model = OpenAIBaseModel::new(model_config).unwrap();
        let messages = vec![ChatMessage {
            sender: SenderType::User,
            text: "Hello".to_string(),
        }];
        let settings = HashMap::new();

        let mock_response = server
            .mock("POST", "/chat/completions")
            .with_status(400)
            .with_body("Bad Request")
            .create();

        let mut stream = model.complete(&messages, &settings).await;
        let response = stream.next().await.unwrap();

        assert_eq!(response.text, "OpenAI API error: 400 Bad Request - Bad Request");
        mock_response.assert();
    }
}

        // If api_key starts with "env:", read from environment variable
        let api_key = if settings.api_key.starts_with("env:") {
            let env_key = &settings.api_key[4..].trim();
            std::env::var(env_key)
                .map_err(|_| anyhow!("Environment variable {} not found", env_key))?
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
            let response = match self.client
                .post(&api_url)
                .json(&body)
                .bearer_auth(api_key)
                .send()
                .await
            {
                Ok(res) => res,
                Err(e) => {
                    yield CompletionResponse {
                        text: format!("Request failed: {}", e),
                        finish_reason: Some("error".to_string()),
                        metrics: CompletionMetrics {
                            prompt_tokens: 0,
                            prompt_eval_latency_ms: 0.0,
                            completion_tokens: 0,
                            completion_runs: 0,
                            completion_latency_ms: 0.0,
                        },
                    };
                    return;
                }
            };

            if !response.status().is_success() {
                let status = response.status();
                let error_body = response.text().await.unwrap_or_else(|_| "Failed to read error response body".to_string());
                yield CompletionResponse {
                    text: format!("OpenAI API error: {} - {}", status, error_body),
                    finish_reason: Some("error".to_string()),
                    metrics: CompletionMetrics {
                        prompt_tokens: 0,
                        prompt_eval_latency_ms: 0.0,
                        completion_tokens: 0,
                        completion_runs: 0,
                        completion_latency_ms: 0.0,
                    },
                };
                return;
            }

            let mut stream = response.bytes_stream();
            let mut buffer = String::new();

            // Track whether we've received first response
            let mut first_chunk_received = false;
            while let Some(item) = stream.next().await {
                let chunk = match item {
                    Ok(chunk) => chunk,
                    Err(e) => {
                        yield CompletionResponse {
                            text: format!("Stream error: {}", e),
                            finish_reason: Some("error".to_string()),
                            metrics: CompletionMetrics {
                                prompt_tokens: 0,
                                prompt_eval_latency_ms: 0.0,
                                completion_tokens: 0,
                                completion_runs: 0,
                                completion_latency_ms: 0.0,
                            },
                        };
                        return;
                    }
                };

                // Convert chunk to text and add to buffer
                match String::from_utf8(chunk.to_vec()) {
                    Ok(chunk_str) => buffer.push_str(&chunk_str),
                    Err(_) => continue,
                };

                // Process each SSE event (data: { ... })
                while let Some(end) = buffer.find("\n\n") {
                    let event = buffer.drain(..=end).collect::<String>();
                    let event = event.trim();

                    if !event.starts_with("data:") {
                        continue;
                    }

                    // "data:" is 5 characters, so skip that
                    let json_str = &event[5..].trim();
                    if *json_str == "[DONE]" {
                        break;
                    }

                    if let Ok(openai_event) = serde_json::from_str::<OpenAISSE>(json_str) {
                        if let Some(choice) = openai_event.choices.first() {
                            // First chunk in stream
                            let chunk_text = choice.delta.content.clone().unwrap_or_default();
                            if !first_chunk_received {
                                let prompt_eval_latency = start_time.elapsed().as_millis() as f32;
                                first_chunk_received = true;

                                yield CompletionResponse {
                                    text: chunk_text,
                                    finish_reason: choice.finish_reason.clone(),
                                    metrics: CompletionMetrics {
                                        prompt_tokens: 0, // TODO: token counting
                                        prompt_eval_latency_ms: prompt_eval_latency,
                                        completion_tokens: 0, // TODO: token counting
                                        completion_runs: 1,
                                        completion_latency_ms: 0.0,
                                    },
                                };
                            } else {
                                let latency_ms = start_time.elapsed().as_millis() as f32;
                                yield CompletionResponse {
                                    text: chunk_text,
                                    finish_reason: choice.finish_reason.clone(),
                                    metrics: CompletionMetrics {
                                        prompt_tokens: 0,
                                        prompt_eval_latency_ms: 0.0,
                                        completion_tokens: 0,
                                        completion_runs: 1,
                                        completion_latency_ms: latency_ms,
                                    },
                                };
                            }
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
