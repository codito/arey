use anyhow::{Context, Result};
use futures::stream::BoxStream;
use std::collections::HashMap;

use arey_core::completion::{
    CancellationToken, ChatMessage, Completion, CompletionModel, SenderType,
};
use arey_core::get_completion_llm;
use arey_core::model::{ModelConfig, ModelMetrics};

/// Represents a single, non-interactive instruction to be executed by a model.
pub struct Task {
    instruction: String,
    model_config: ModelConfig,
    model: Option<Box<dyn CompletionModel + Send + Sync>>,
}

impl Task {
    /// Creates a new `Task` with the given instruction and model configuration.
    pub fn new(instruction: String, model_config: ModelConfig) -> Self {
        Self {
            instruction,
            model_config,
            model: None,
        }
    }

    /// Loads the language model for the task.
    ///
    /// This method initializes the model and loads the system prompt.
    /// It should be called before `run`.
    pub async fn load_model(&mut self) -> Result<ModelMetrics> {
        let config = self.model_config.clone();
        let mut model = get_completion_llm(config).context("Failed to initialize model")?;

        // Load empty system prompt for tasks
        model
            .load("")
            .await
            .context("Failed to load model with system prompt")?;

        let metrics = model.metrics();
        self.model = Some(model);
        Ok(metrics)
    }

    /// Executes the task and returns a stream of completion results.
    ///
    /// # Panics
    ///
    /// This method will panic if `load_model` has not been called first.
    pub async fn run(&mut self) -> Result<BoxStream<'_, Result<Completion>>> {
        let message = ChatMessage {
            sender: SenderType::User,
            text: self.instruction.clone(),
            tools: Vec::new(),
        };

        let settings = HashMap::new(); // Use default settings for now
        let cancel_token = CancellationToken::new();

        let model = self
            .model
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("Model not loaded"))?;
        let stream = model
            .complete(&[message], None, &settings, cancel_token)
            .await;

        Ok(stream)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arey_core::model::ModelProvider;
    use futures::stream::StreamExt;
    use serde_json::json;
    use serde_yaml::Value;
    use wiremock::{
        Mock, MockServer, ResponseTemplate,
        matchers::{method, path},
    };

    fn create_mock_model_config(server_url: &str) -> ModelConfig {
        let settings: HashMap<String, Value> = HashMap::from([
            ("base_url".to_string(), server_url.into()),
            ("api_key".to_string(), "MOCK_OPENAI_API_KEY".into()),
        ]);

        ModelConfig {
            name: "test-model".to_string(),
            provider: ModelProvider::Openai,
            settings,
        }
    }

    fn mock_event_stream_body() -> String {
        let events = [
            json!({
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "created": 1684,
                "model": "gpt-3.5-turbo",
                "choices": [{
                    "delta": {"content": "Hello"},
                    "index": 0,
                    "finish_reason": null
                }]
            }),
            json!({
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "created": 1684,
                "model": "gpt-3.5-turbo",
                "choices": [{
                    "delta": {"content": " world!"},
                    "index": 0,
                    "finish_reason": "stop"
                }]
            }),
        ];
        let mut body: String = events.iter().map(|e| format!("data: {e}\n\n")).collect();
        body.push_str("data: [DONE]\n\n");
        body
    }

    #[tokio::test]
    async fn test_task_new() {
        let model_config = create_mock_model_config("");
        let task = Task::new("test".to_string(), model_config.clone());
        assert_eq!(task.instruction, "test");
        assert_eq!(task.model_config.name, model_config.name);
        assert!(task.model.is_none());
    }

    #[tokio::test]
    async fn test_task_logic() -> Result<()> {
        let server = MockServer::start().await;
        let response = ResponseTemplate::new(200)
            .set_body_bytes(mock_event_stream_body().as_bytes())
            .insert_header("Content-Type", "text/event-stream");

        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(response)
            .mount(&server)
            .await;

        let model_config = create_mock_model_config(&server.uri());
        let mut task = Task::new("test instruction".to_string(), model_config);

        let metrics = task.load_model().await?;
        assert!(metrics.init_latency_ms >= 0.0);
        assert!(task.model.is_some());

        let mut stream = task.run().await?;

        let mut response_text = String::new();
        let mut final_finish_reason: Option<String> = None;

        while let Some(result) = stream.next().await {
            match result? {
                Completion::Response(r) => {
                    response_text.push_str(&r.text);
                    if r.finish_reason.is_some() {
                        final_finish_reason = r.finish_reason;
                    }
                }
                Completion::Metrics(_) => {}
            }
        }

        assert_eq!(response_text, "Hello world!");
        assert_eq!(final_finish_reason, Some("Stop".to_string()));

        Ok(())
    }
}
