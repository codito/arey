use anyhow::{Context, Result};
use futures::StreamExt;
use futures::stream::BoxStream;

use arey_core::agent::Agent;
use arey_core::completion::{CancellationToken, ChatMessage, Completion, SenderType};
use arey_core::config::Config;
use arey_core::model::ModelConfig;
use arey_core::session::{Session, SessionConfig, SessionEvent};

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

        // Get tools from registry based on agent's effective tools
        let tool_registry = get_tools(self.config).unwrap_or_default();
        let tools = tool_registry.tools_for(self.current_agent.effective_tools());

        let session_config = SessionConfig {
            system_prompt: self.current_agent.prompt.clone(),
            tools,
            ..Default::default()
        };

        let session =
            Session::new(model_config, session_config).context("Failed to create task session")?;

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

        self.session
            .as_ref()
            .map(|s| s.metrics().clone())
            .ok_or_else(|| anyhow::anyhow!("No metrics available"))
    }

    pub async fn run(&mut self) -> Result<BoxStream<'_, Result<Completion>>> {
        let session = self
            .session
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("Session not loaded"))?;

        // Add the user instruction to the session
        session
            .add_message(ChatMessage {
                sender: SenderType::User,
                text: self.instruction.clone(),
                ..Default::default()
            })
            .context("Failed to add instruction message")?;

        let profile = self.current_agent.effective_profile();
        let settings = profile
            .to_settings()
            .context("Failed to convert profile to settings")
            .unwrap_or_default();
        let cancel_token = CancellationToken::new();

        let stream = session
            .generate(settings, cancel_token)
            .await
            .context("Failed to generate response")?;

        // Map SessionEvent to Completion
        let stream = stream.filter_map(|event| async move {
            match event {
                Ok(SessionEvent::Token(c)) => Some(Ok(c)),
                Ok(SessionEvent::CompactionStart) => {
                    tracing::info!("Context compaction started");
                    None
                }
                Ok(SessionEvent::CompactionEnd { result }) => {
                    tracing::info!(
                        "Context compaction completed: {} -> {} messages",
                        result.original_messages,
                        result.compacted_messages
                    );
                    None
                }
                Ok(SessionEvent::ReasoningStart) => {
                    tracing::debug!("Reasoning started");
                    None
                }
                Ok(SessionEvent::ReasoningEnd { .. }) => {
                    tracing::debug!("Reasoning ended");
                    None
                }
                Ok(SessionEvent::ToolStart { .. }) => {
                    tracing::info!("Tool calls started");
                    None
                }
                Ok(SessionEvent::ToolEnd { .. }) => {
                    tracing::info!("Tool calls completed");
                    None
                }
                Err(e) => Some(Err(e)),
            }
        });

        Ok(Box::pin(stream))
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
            mcp: serde_yaml::Value::Null,
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
            mcp: serde_yaml::Value::Null,
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
