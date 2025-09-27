use anyhow::{Context, Result};
use futures::stream::BoxStream;
use std::collections::HashMap;
use std::sync::Arc;

use arey_core::agent::Agent;
use arey_core::completion::{CancellationToken, Completion, SenderType};
use arey_core::config::Config;
use arey_core::model::ModelConfig;
use arey_core::session::Session;
use arey_core::tools::Tool;

use crate::ext::get_tools;

/// Represents a single, non-interactive instruction to be executed by a model.
pub struct Task<'a> {
    instruction: String,
    config: &'a Config,
    current_agent: Agent,
    session: Option<Session>,
}

impl<'a> Task<'a> {
    /// Creates a new `Task` with the given instruction, model configuration, and config.
    pub fn new(
        instruction: String,
        _model_config: ModelConfig,
        config: &'a Config,
        agent_name: String,
    ) -> Result<Self> {
        let current_agent = config
            .agents
            .get(&agent_name)
            .cloned()
            .context(format!("Agent '{agent_name}' not found in config."))?;

        // Mark this agent as active
        let mut agent = current_agent;
        agent.set_active(true);

        Ok(Self {
            instruction,
            config,
            current_agent: agent,
            session: None,
        })
    }

    /// Loads the session for the task.
    ///
    /// This method initializes the session with the model and agent configuration.
    /// It should be called before `run`.
    pub async fn load_session(&mut self) -> Result<()> {
        let model_config = self.config.task.model.clone();
        let system_prompt = &self.current_agent.prompt;

        let mut session =
            Session::new(model_config, system_prompt).context("Failed to create task session")?;

        // Get tools for the agent
        let available_tools = get_tools(self.config).unwrap_or_default();
        let tools: Vec<Arc<dyn Tool>> = self
            .current_agent
            .effective_tools()
            .iter()
            .filter_map(|tool_name| available_tools.get(tool_name.as_str()).cloned())
            .collect();

        session.set_tools(tools)?;

        self.session = Some(session);
        Ok(())
    }

    /// Executes the task and returns a stream of completion results.
    ///
    /// # Panics
    ///
    /// This method will panic if `load_session` has not been called first.
    /// For backwards compatibility, also provide load_model method
    /// that returns model metrics
    pub async fn load_model(&mut self) -> Result<arey_core::model::ModelMetrics> {
        self.load_session().await?;

        // Return metrics from the session
        self.session
            .as_ref()
            .and_then(|s| s.metrics().cloned())
            .ok_or_else(|| anyhow::anyhow!("No metrics available"))
    }

    pub async fn run(&mut self) -> Result<BoxStream<'_, Result<Completion>>> {
        let session = self
            .session
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("Session not loaded"))?;

        // Add the user instruction to the session
        session
            .add_message(SenderType::User, &self.instruction)
            .context("Failed to add instruction message")?;

        let mut settings = HashMap::new();
        let profile = self.current_agent.effective_profile();
        settings.insert("temperature".to_string(), profile.temperature.to_string());
        settings.insert(
            "repeat_penalty".to_string(),
            profile.repeat_penalty.to_string(),
        );
        settings.insert("top_k".to_string(), profile.top_k.to_string());
        settings.insert("top_p".to_string(), profile.top_p.to_string());

        let cancel_token = CancellationToken::new();

        let stream = session
            .generate(settings, cancel_token)
            .await
            .context("Failed to generate response")?;

        Ok(stream)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arey_core::agent::AgentConfig;
    use arey_core::agent::AgentSource;
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

        let mut agents = HashMap::new();
        agents.insert(
            "default".to_string(),
            AgentConfig::new("default", "You are a helpful assistant.", vec![])
                .to_agent(AgentSource::BuiltIn),
        );

        let config = Config {
            models,
            profiles: HashMap::new(),
            agents,
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

        let task = Task::new(
            "test".to_string(),
            config.task.model.clone(),
            &config,
            "default".to_string(),
        )
        .unwrap();
        assert_eq!(task.instruction, "test");
        assert!(task.session.is_none());
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

        let mut agents = HashMap::new();
        agents.insert(
            "default".to_string(),
            AgentConfig::new("default", "You are a helpful assistant.", vec![])
                .to_agent(AgentSource::BuiltIn),
        );

        let config = Config {
            models,
            profiles: HashMap::new(),
            agents,
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

        let mut task = Task::new(
            "test instruction".to_string(),
            config.task.model.clone(),
            &config,
            "default".to_string(),
        )
        .unwrap();

        task.load_session().await?;
        assert!(task.session.is_some());

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
