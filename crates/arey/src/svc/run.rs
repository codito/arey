use anyhow::{Context, Result};
use futures::stream::BoxStream;
use std::collections::HashMap;
use std::sync::Arc;

use arey_core::completion::{
    CancellationToken, ChatMessage, Completion, CompletionModel, SenderType,
};
use arey_core::config::Config;
use arey_core::get_completion_llm;
use arey_core::model::{ModelConfig, ModelMetrics};
use arey_core::tools::Tool;

use crate::ext::get_tools;

/// Represents a single, non-interactive instruction to be executed by a model.
pub struct Task<'a> {
    instruction: String,
    model_config: ModelConfig,
    config: &'a Config,
    agent_name: String,
    model: Option<Box<dyn CompletionModel + Send + Sync>>,
}

impl<'a> Task<'a> {
    /// Creates a new `Task` with the given instruction, model configuration, and config.
    pub fn new(
        instruction: String,
        model_config: ModelConfig,
        config: &'a Config,
        agent_name: String,
    ) -> Self {
        Self {
            instruction,
            model_config,
            config,
            agent_name,
            model: None,
        }
    }

    /// Loads the language model for the task.
    ///
    /// This method initializes the model and loads the system prompt from the agent if specified.
    /// It should be called before `run`.
    pub async fn load_model(&mut self) -> Result<ModelMetrics> {
        let config = self.model_config.clone();
        let mut model = get_completion_llm(config).context("Failed to initialize model")?;

        // Use agent prompt if available, otherwise empty for tasks
        let system_prompt = self
            .config
            .agents
            .get(self.agent_name.as_str())
            .map(|agent| agent.prompt.clone())
            .unwrap_or_default();

        model
            .load(&system_prompt)
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

        let mut settings = HashMap::new();
        if let Some(agent) = self.config.agents.get(self.agent_name.as_str()) {
            settings.insert(
                "temperature".to_string(),
                agent.profile.temperature.to_string(),
            );
            settings.insert(
                "repeat_penalty".to_string(),
                agent.profile.repeat_penalty.to_string(),
            );
            settings.insert("top_k".to_string(), agent.profile.top_k.to_string());
            settings.insert("top_p".to_string(), agent.profile.top_p.to_string());
        }

        let available_tools = get_tools(self.config).unwrap_or_default();
        let tools: Vec<Arc<dyn Tool>> =
            if let Some(agent) = self.config.agents.get(self.agent_name.as_str()) {
                agent
                    .tools
                    .iter()
                    .filter_map(|tool_name| available_tools.get(tool_name.as_str()).cloned())
                    .collect()
            } else {
                vec![]
            };

        let cancel_token = CancellationToken::new();

        let model = self
            .model
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("Model not loaded"))?;
        let stream = model
            .complete(&[message], Some(&tools), &settings, cancel_token)
            .await;

        Ok(stream)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arey_core::config::{ModeConfig, ProfileConfig};
    use arey_core::model::ModelProvider;
    use futures::stream::StreamExt;
    use serde_json::json;
    use serde_yaml::Value;
    use std::collections::HashMap;
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
            key: "test-key".to_string(),
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
        let mut models = HashMap::new();
        models.insert("test-model".to_string(), create_mock_model_config(""));

        let config = Config {
            models,
            profiles: HashMap::new(),
            agents: HashMap::new(),
            chat: ModeConfig {
                model: create_mock_model_config(""),
                profile: ProfileConfig::default(),
                profile_name: None,
                agent_name: "default".to_string(),
            },
            task: ModeConfig {
                model: create_mock_model_config(""),
                profile: ProfileConfig::default(),
                profile_name: None,
                agent_name: "default".to_string(),
            },
            theme: "light".to_string(),
            tools: HashMap::new(),
        };

        let model_config = config.task.model.clone();
        let task = Task::new(
            "test".to_string(),
            model_config,
            &config,
            "default".to_string(),
        );
        assert_eq!(task.instruction, "test");
        assert_eq!(task.model_config.name, "test-model");
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

        let mut models = HashMap::new();
        models.insert(
            "test-model".to_string(),
            create_mock_model_config(&server.uri()),
        );

        let config = Config {
            models,
            profiles: HashMap::new(),
            agents: HashMap::new(),
            chat: ModeConfig {
                model: create_mock_model_config(&server.uri()),
                profile: ProfileConfig::default(),
                profile_name: None,
                agent_name: "default".to_string(),
            },
            task: ModeConfig {
                model: create_mock_model_config(&server.uri()),
                profile: ProfileConfig::default(),
                profile_name: None,
                agent_name: "default".to_string(),
            },
            theme: "light".to_string(),
            tools: HashMap::new(),
        };

        let model_config = config.task.model.clone();
        let mut task = Task::new(
            "test instruction".to_string(),
            model_config,
            &config,
            "default".to_string(),
        );

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
                    if let Some(fr) = r.finish_reason {
                        final_finish_reason = Some(fr);
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
