use crate::completion::{
    CancellationToken, ChatMessage, Completion, CompletionMetrics, CompletionModel,
    CompletionResponse, SenderType,
};
use crate::model::{ModelConfig, ModelMetrics};
use anyhow::{Result, anyhow};
use async_openai::config::OpenAIConfig;
use async_openai::{
    Client as OpenAIClient,
    types::{
        ChatCompletionRequestMessage, ChatCompletionStreamOptions, CreateChatCompletionRequestArgs,
    },
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
        })
    }

    fn to_openai_message(msg: &ChatMessage) -> ChatCompletionRequestMessage {
        match msg.sender {
            SenderType::System => ChatCompletionRequestMessage::System(
                async_openai::types::ChatCompletionRequestSystemMessageArgs::default()
                    .content(msg.text.as_str())
                    .build()
                    .unwrap(),
            ),
            SenderType::Assistant => ChatCompletionRequestMessage::Assistant(
                async_openai::types::ChatCompletionRequestAssistantMessageArgs::default()
                    .content(msg.text.as_str())
                    .build()
                    .unwrap(),
            ),
            SenderType::User => ChatCompletionRequestMessage::User(
                async_openai::types::ChatCompletionRequestUserMessageArgs::default()
                    .content(msg.text.as_str())
                    .build()
                    .unwrap(),
            ),
        }
    }
}

#[async_trait]
impl CompletionModel for OpenAIBaseModel {
    // fn context_size(&self) -> usize {
    //     // TODO: Implement actual context size
    //     4096
    // }

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
        cancel_token: CancellationToken,
    ) -> BoxStream<'_, Result<Completion>> {
        // Map messages to OpenAI message types
        let openai_messages: Vec<ChatCompletionRequestMessage> = messages
            .iter()
            .map(OpenAIBaseModel::to_openai_message)
            .collect();

        // Set max_tokens and temperature if provided
        let max_tokens = settings
            .get("max_tokens")
            .and_then(|s| s.parse::<u32>().ok())
            .unwrap_or(1024u32);
        let temperature = settings
            .get("temperature")
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(0.0);

        // Build request
        let stream_options = ChatCompletionStreamOptions {
            include_usage: true,
        };
        let request = CreateChatCompletionRequestArgs::default()
            .model(self.config.name.clone())
            .messages(openai_messages)
            .max_tokens(max_tokens)
            .temperature(temperature)
            .stream(true)
            .stream_options(stream_options)
            .build();

        let request = match request {
            Ok(req) => req,
            Err(err) => {
                return Box::pin(futures::stream::once(async move {
                    Err(anyhow!("Invalid request: {:?}", err))
                }));
            }
        };

        // Start the timer
        let start_time = Instant::now();
        let prev_time = start_time;
        let mut first_chunk = true;

        // Create the stream
        let outer_stream = async_stream::stream! {
            let _start_time = start_time;
            let mut prev_time = prev_time;
            let mut prompt_eval_latency = 0.0;
            let mut completion_latency = 0.0;

            // Send the request and get back a streaming response
            match self.client.chat().create_stream(request).await {
                Ok(response) => {
                    let mut stream = response;

                    while let Some(next) = stream.next().await {
                        // Check for cancellation *before* processing the chunk
                        if cancel_token.is_cancelled() {
                            yield Err(anyhow::anyhow!("Cancelled by user")); // Yield a cancellation error
                            break; // Exit the loop
                        }

                        // Time measurement
                        let now = Instant::now();
                        let elapsed = now.duration_since(prev_time).as_millis() as f32;
                        prev_time = now;

                        match next {
                            Ok(chunk) => {
                                let raw_json = serde_json::to_string(&chunk).unwrap_or_else(|_| String::from(""));

                                // Check if we have a content block in the first choice
                                if let Some(choice) = chunk.choices.first() {
                                    let text = choice.delta.content.clone().unwrap_or_else(|| "".to_string());
                                    // For the first chunk, set the prompt_eval_latency_ms
                                    if first_chunk {
                                        prompt_eval_latency = elapsed;
                                        first_chunk = false;
                                    }

                                    completion_latency += elapsed;

                                    yield Ok(Completion::Response(CompletionResponse {
                                        text: text.to_string(),
                                        finish_reason: choice.finish_reason.as_ref().map(|x| format!("{x:?}")),
                                        raw_chunk: Some(raw_json.clone()),
                                    }));
                                }

                                // Some openai compatible servers (Gemini) club usage with the
                                // final response, others send a separate chunk.
                                if let Some(usage) = chunk.usage {
                                    // FIXME possible duplicate logs in raw_chunk
                                    yield Ok(Completion::Metrics(CompletionMetrics{
                                        prompt_tokens: usage.prompt_tokens,
                                        prompt_eval_latency_ms: prompt_eval_latency,
                                        completion_tokens: usage.completion_tokens,
                                        completion_latency_ms: completion_latency,
                                        raw_chunk: Some(raw_json.clone())
                                    }));
                                }
                            }
                            Err(err) => {
                                println!("{err:?}");
                                yield Err(anyhow!("OpenAI stream error: {}", err));
                            }
                        }
                    }
                }
                Err(err) => {
                    yield Err(anyhow!("OpenAI request failed: {:?}", err));
                }
            }
        };

        Box::pin(outer_stream)
    }

    // async fn free(&mut self) {
    //     // No resources to free
    // }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::completion::SenderType;
    use crate::model::ModelProvider;
    use serde_json::json;
    use wiremock::{
        Mock, MockServer, ResponseTemplate,
        matchers::{method, path},
    };

    // Create a mock event stream body
    fn mock_event_stream_body() -> String {
        let events = vec![
            json!({
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "created": 1684,
                "model": "gpt-3.5-turbo",
                "choices": [{
                    "delta": {"content": "Hello"},
                    "index": 0,
                    "finish_reason": serde_json::Value::Null
                }]
            }),
            json!({
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "created": 1684,
                "model": "gpt-3.5-turbo",
                "choices": [{
                    "delta": {"content": " world"},
                    "index": 0,
                    "finish_reason": serde_json::Value::Null
                }]
            }),
            json!({
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "created": 1684,
                "model": "gpt-3.5-turbo",
                "choices": [{
                    "delta": {},
                    "index": 0,
                    "finish_reason": "stop"
                }],
            }),
            json!({
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "created": 1684,
                "model": "gpt-3.5-turbo",
                "choices": [],
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 30,
                    "total_tokens": 50,
                    "prompt_tokens_details": {},
                    "completion_tokens_details": {"reasoning_tokens": 5}
                }
            }),
        ];

        let mut mock_body = events
            .into_iter()
            .map(|event| format!("data: {}\n\n", serde_json::to_string(&event).unwrap()))
            .collect::<String>();
        mock_body.push_str("data: [DONE]\n\n");
        mock_body
    }

    // Create a test model configuration with mock server URL
    fn create_mock_model_config(server_url: &str) -> Result<ModelConfig> {
        let settings: HashMap<String, serde_yaml::Value> = HashMap::from([
            ("base_url".to_string(), server_url.into()),
            ("api_key".to_string(), "MOCK_OPENAI_API_KEY".into()),
        ]);

        let config = ModelConfig {
            name: "test-model".to_string(),
            provider: ModelProvider::Openai,
            settings,
        };

        Ok(config)
    }

    #[tokio::test]
    async fn test_openai_new_model() {
        let server = MockServer::start().await;
        // We don't need any mocks since we're not making HTTP calls in this test
        let server_url = server.uri();

        let config = create_mock_model_config(&server_url).unwrap();
        let model = OpenAIBaseModel::new(config).unwrap();

        assert_eq!(model.config.name, "test-model");
    }

    #[tokio::test]
    async fn test_openai_complete_api() {
        let server = MockServer::start().await;
        let server_url = server.uri();
        let config = create_mock_model_config(&server_url).unwrap();

        let mock_response = ResponseTemplate::new(200)
            .set_body_raw(mock_event_stream_body(), "text/event-stream")
            .insert_header("Connection", "close");

        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(mock_response)
            .mount(&server)
            .await;

        let mut model = OpenAIBaseModel::new(config).unwrap();

        let messages = vec![ChatMessage {
            text: "Hello".to_string(),
            sender: SenderType::User,
        }];

        let cancel_token = CancellationToken::new(); // Create a token for the test
        let mut stream = model
            .complete(&messages, &HashMap::new(), cancel_token)
            .await;

        // Collect and assert on responses
        let mut responses = Vec::new();
        let mut metrics = CompletionMetrics::default();
        while let Some(chunk_result) = stream.next().await {
            match chunk_result.unwrap() {
                Completion::Response(response) => {
                    println!("{response:?}");
                    responses.push(response);
                }
                Completion::Metrics(m) => metrics = m,
            }
        }

        // We expect 3 responses: two content chunks and one finish reason
        assert_eq!(responses.len(), 3);
        assert_eq!(responses[0].text, "Hello");
        assert_eq!(responses[1].text, " world");
        assert_eq!(responses[2].text, "");
        assert_eq!(responses[2].finish_reason, Some("Stop".to_string()));

        assert_eq!(metrics.prompt_tokens, 20);
        assert_eq!(metrics.completion_tokens, 30);
        assert!(metrics.completion_latency_ms != 0.0);
        assert!(metrics.prompt_eval_latency_ms != 0.0)
    }
}
