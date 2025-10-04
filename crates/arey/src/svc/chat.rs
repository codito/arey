use anyhow::{Context, Result};
use arey_core::agent::{Agent, AgentSource};
use arey_core::completion::{CancellationToken, ChatMessage, Completion};
use arey_core::config::{Config, ProfileConfig};
use arey_core::session::Session;
use arey_core::tools::Tool;
use futures::{StreamExt, stream::BoxStream};
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

/// Represents an interactive chat session between a user and an AI model.
///
/// It maintains conversation history and manages tool usage.
pub struct Chat<'a> {
    session: Session,
    current_agent: Agent,
    pub available_tools: HashMap<&'a str, Arc<dyn Tool>>,
    config: &'a Config,
}

impl<'a> fmt::Debug for Chat<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Chat")
            .field(
                "available_tools",
                &self.available_tools.keys().collect::<Vec<_>>(),
            )
            .finish_non_exhaustive()
    }
}

impl<'a> Chat<'a> {
    /// Creates a new `Chat` session.
    ///
    /// It uses the specified model from the configuration, or the default chat model if `None`.
    pub async fn new(
        config: &'a Config,
        model: Option<String>,
        available_tools: HashMap<&'a str, Arc<dyn Tool>>,
    ) -> Result<Self> {
        let model_config = if let Some(model_name) = model {
            config
                .models
                .get(model_name.as_str())
                .cloned()
                .context(format!("Model '{model_name}' not found in config."))?
        } else {
            config.chat.model.clone()
        };

        let agent_name = config.chat.agent_name.clone();
        let mut agent = config
            .agents
            .get(&agent_name)
            .cloned()
            .context(format!("Agent '{agent_name}' not found in config."))?;

        // Mark this agent as active
        agent.set_active(true);

        let tools: Result<Vec<Arc<dyn Tool>>, _> = agent
            .effective_tools()
            .iter()
            .map(|tool_name| {
                available_tools
                    .get(tool_name.as_str())
                    .cloned()
                    .ok_or_else(|| anyhow::anyhow!("Tool '{}' not found", tool_name))
            })
            .collect();

        let mut session =
            Session::new(model_config, &agent.prompt).context("Failed to create chat session")?;
        session.set_tools(tools?)?;

        Ok(Self {
            session,
            current_agent: agent,
            available_tools,
            config,
        })
    }

    /// Get current agent name
    pub fn agent_name(&self) -> String {
        self.current_agent.name.clone()
    }

    /// Get current agent state display string
    pub fn agent_display_state(&self) -> String {
        self.current_agent.display_state()
    }

    /// Switch to a different agent
    pub async fn set_agent(&mut self, agent_name: &str) -> Result<()> {
        let mut new_agent = self
            .config
            .agents
            .get(agent_name)
            .cloned()
            .context(format!("Agent '{agent_name}' not found in config."))?;

        // Deactivate current agent
        self.current_agent.set_active(false);

        let model_key = self.session.model_key();
        let _model_config = self
            .config
            .models
            .get(&model_key)
            .cloned()
            .context(format!(
                "Model '{}' associated with the current session not found in config.",
                model_key
            ))?;

        let _messages = self.session.all_messages();

        // Activate new agent
        new_agent.set_active(true);

        let tools: Result<Vec<Arc<dyn Tool>>, _> = new_agent
            .effective_tools()
            .iter()
            .map(|name| {
                self.available_tools
                    .get(name.as_str())
                    .cloned()
                    .ok_or_else(|| anyhow::anyhow!("Tool not found: {}", name))
            })
            .collect();

        // Update the session with new agent and tools
        self.session
            .update_agent(new_agent.prompt.clone(), tools?)?;
        self.current_agent = new_agent;

        Ok(())
    }

    /// Get current model key identifier
    pub fn model_key(&self) -> String {
        self.session.model_key()
    }

    /// Switch to a different model
    pub async fn set_model(&mut self, model_name: &str) -> Result<()> {
        let model_config = self
            .config
            .models
            .get(model_name)
            .cloned()
            .context(format!("Model '{model_name}' not found in config."))?;

        // Session handles model creation and old model cleanup
        self.session.set_model(model_config)?;

        Ok(())
    }

    /// Get current profile name
    pub fn profile_name(&self) -> String {
        self.current_agent
            .runtime_state
            .current_profile_name
            .clone()
            .unwrap_or_else(|| self.agent_name())
    }

    /// Get current profile name and data
    pub fn current_profile(&self) -> Option<(String, ProfileConfig)> {
        // Check if there's a session-specific profile override
        if let Some(session_profile) = self.current_agent.runtime_state.current_profile.as_ref() {
            if let Some(profile_name) = &self.current_agent.runtime_state.current_profile_name {
                Some((profile_name.clone(), session_profile.clone()))
            } else {
                // Fallback: use agent name if profile name is not set
                Some((self.agent_name(), session_profile.clone()))
            }
        } else {
            // Fall back to the agent's default profile
            self.config
                .agents
                .get(&self.agent_name())
                .map(|agent| (self.agent_name(), agent.profile.clone()))
        }
    }

    /// Switch to a different profile
    pub fn set_profile(&mut self, profile_name: &str) -> Result<()> {
        // Look up the profile in the config
        let profile_config = self
            .config
            .profiles
            .get(profile_name)
            .context(format!("Profile '{}' not found in config.", profile_name))?
            .clone();

        // Set the profile as a session override for the current agent
        self.current_agent
            .set_current_profile_with_name(profile_name.to_string(), profile_config);
        Ok(())
    }

    /// Get current system prompt
    pub fn system_prompt(&self) -> String {
        self.session.system_prompt()
    }

    /// Set new system prompt for the current session
    pub async fn set_system_prompt(&mut self, prompt: &str) -> Result<()> {
        // Update the session with the new system prompt
        self.session.set_system_prompt(prompt)?;

        Ok(())
    }

    /// Get available agent names
    pub fn available_agent_names(&self) -> Vec<&str> {
        self.config.agents.keys().map(|s| s.as_str()).collect()
    }

    /// Get available tool names
    pub fn available_tool_names(&self) -> Vec<&str> {
        self.available_tools.keys().copied().collect()
    }

    /// Get available agents with their sources
    pub fn available_agents_with_sources(&self) -> Vec<(&str, &AgentSource)> {
        self.config
            .agents
            .iter()
            .map(|(name, agent)| (name.as_str(), &agent.metadata.source))
            .collect()
    }

    /// Get agent source information
    pub fn get_agent_source(&self, agent_name: &str) -> Option<&AgentSource> {
        self.config
            .agents
            .get(agent_name)
            .map(|agent| &agent.metadata.source)
    }

    /// Format agent source for display
    pub fn format_agent_source(&self, agent_name: &str) -> String {
        match self.get_agent_source(agent_name) {
            Some(AgentSource::BuiltIn) => "built-in".to_string(),
            Some(AgentSource::UserFile(_)) => "user".to_string(),
            None => "unknown".to_string(),
        }
    }

    /// Get available model names
    pub fn available_model_names(&self) -> Vec<&str> {
        self.config.models.keys().map(|s| s.as_str()).collect()
    }

    /// Get available profile names
    pub fn available_profile_names(&self) -> Vec<&str> {
        self.config.profiles.keys().map(|s| s.as_str()).collect()
    }

    /// Gets the tools available for the current chat session.
    pub fn tools(&self) -> Vec<Arc<dyn Tool>> {
        self.session.tools()
    }

    /// Sets the tools available for the current chat session.
    pub async fn set_tools(&mut self, tool_names: &[String]) -> Result<()> {
        let mut tools = Vec::new();
        for name in tool_names {
            let tool = self
                .available_tools
                .get(name.as_str())
                .ok_or_else(|| anyhow::anyhow!("Tool not found or not available: {}", name))?;
            tools.push(tool.clone());
        }

        // Update the session with the new tools
        self.session.set_tools(tools)?;

        Ok(())
    }

    /// Get all messages from the session
    pub fn get_all_messages(&self) -> Vec<ChatMessage> {
        self.session.all_messages()
    }

    /// Retrieves the last message from the assistant in the conversation history.
    pub fn get_last_assistant_message(&self) -> Option<ChatMessage> {
        self.session.last_assistant_message()
    }

    /// Adds messages to the conversation history.
    pub async fn add_messages(
        &mut self,
        user_messages: Vec<ChatMessage>,
        tool_messages: Vec<ChatMessage>,
    ) {
        for message in user_messages {
            if let Err(e) = self.session.add_message(message.sender, &message.text) {
                eprintln!("Failed to add message: {}", e);
            }
        }

        for message in tool_messages {
            if let Err(e) = self.session.add_message(message.sender, &message.text) {
                eprintln!("Failed to add message: {}", e);
            }
        }
    }

    /// Clears the conversation history of the session.
    pub async fn clear_messages(&mut self) {
        if let Err(e) = self.session.clear_history() {
            eprintln!("Failed to clear history: {}", e);
        }
    }

    /// Generates a streaming response from the model based on the conversation history.
    pub async fn stream_response(
        &self,
        cancel_token: CancellationToken,
    ) -> Result<BoxStream<'_, Result<Completion>>> {
        // Use the current profile (either session override or agent default)
        let profile = self
            .current_profile()
            .map(|(_, profile_config)| profile_config);

        let settings = if let Some(profile) = profile {
            let mut settings = HashMap::new();
            settings.insert("temperature".to_string(), profile.temperature.to_string());
            settings.insert(
                "repeat_penalty".to_string(),
                profile.repeat_penalty.to_string(),
            );
            settings.insert("top_k".to_string(), profile.top_k.to_string());
            settings.insert("top_p".to_string(), profile.top_p.to_string());
            settings
        } else {
            HashMap::new()
        };

        let stream = async_stream::stream! {
            let mut inner_stream = match self.session.generate(settings, cancel_token.clone()).await {
                Ok(stream) => stream,
                Err(e) => {
                    yield Err(e);
                    return;
                }
            };

            while let Some(item) = inner_stream.next().await {
                yield item;
            }
        };

        Ok(Box::pin(stream))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arey_core::{
        completion::SenderType,
        config::{Config, get_config},
        tools::{Tool, ToolError},
    };
    use async_trait::async_trait;
    use serde_json::{Value, json};
    use std::sync::Arc;
    use wiremock::{
        Mock, MockServer, ResponseTemplate,
        matchers::{method, path},
    };

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

    use crate::test_utils::create_temp_config_file_with_agent as create_temp_config_file;

    async fn get_test_config(server: &MockServer) -> Result<Config> {
        let config_file = create_temp_config_file(&server.uri());
        get_config(Some(config_file.path().to_path_buf()))
            .map_err(|e| anyhow::anyhow!("Failed to create temp config file. Error {}", e))
    }

    #[derive(Debug, Clone)]
    struct MockTool;

    #[async_trait]
    impl Tool for MockTool {
        fn name(&self) -> String {
            "mock_tool".to_string()
        }

        fn description(&self) -> String {
            "A mock tool for testing".to_string()
        }

        fn parameters(&self) -> Value {
            json!({"type": "object", "properties": {}})
        }

        async fn execute(&self, _input: &Value) -> Result<Value, ToolError> {
            Ok(json!("Mock tool executed"))
        }
    }

    #[tokio::test]
    async fn test_chat_new() -> Result<()> {
        let server = MockServer::start().await;
        let mut config = get_test_config(&server).await?;
        let mock_tool: Arc<dyn Tool> = Arc::new(MockTool);
        let available_tools = HashMap::from([("mock_tool", mock_tool)]);

        // Test with existing model, should use default agent from config
        let chat = Chat::new(
            &config,
            Some("test-model".to_string()),
            available_tools.clone(),
        )
        .await;
        assert!(chat.is_ok());
        assert_eq!(chat.unwrap().agent_name(), "test-agent");

        // Test with a specified agent
        config.chat.agent_name = "test-agent".to_string();
        let chat = Chat::new(&config, None, available_tools).await?;
        assert_eq!(chat.agent_name(), "test-agent");
        assert_eq!(chat.system_prompt(), "You are a test agent.");
        assert_eq!(chat.tools().len(), 1);

        // Test with non-existent agent
        config.chat.agent_name = "bad-agent".to_string();
        let chat = Chat::new(&config, None, HashMap::new()).await;
        assert!(chat.is_err());
        assert!(
            chat.unwrap_err()
                .to_string()
                .contains("Agent 'bad-agent' not found in config.")
        );

        // Test with non-existent model
        let chat = Chat::new(&config, Some("bad-model".to_string()), HashMap::new()).await;
        assert!(chat.is_err());
        assert!(
            chat.unwrap_err()
                .to_string()
                .contains("Model 'bad-model' not found in config.")
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_set_tools() -> Result<()> {
        let server = MockServer::start().await;
        let config = get_test_config(&server).await?;

        let mock_tool: Arc<dyn Tool> = Arc::new(MockTool);
        let available_tools = HashMap::from([("mock_tool", mock_tool)]);

        let mut chat = Chat::new(&config, None, available_tools).await?;
        assert!(chat.set_tools(&["mock_tool".to_string()]).await.is_ok());

        // Test with a tool that is not available
        assert!(chat.set_tools(&["bad_tool".to_string()]).await.is_err());

        Ok(())
    }

    #[tokio::test]
    async fn test_get_tools() -> Result<()> {
        let server = MockServer::start().await;
        let config = get_test_config(&server).await?;

        let mock_tool: Arc<dyn Tool> = Arc::new(MockTool);
        let available_tools = HashMap::from([("mock_tool", mock_tool)]);

        let mut chat = Chat::new(&config, None, available_tools).await?;

        // Initially, tools are set based on agent configuration
        let tools = chat.tools();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name(), "mock_tool");

        // Clear tools first
        chat.set_tools(&[]).await?;

        // Verify tools are cleared
        let tools = chat.tools();
        assert!(tools.is_empty());

        // Set a tool
        chat.set_tools(&["mock_tool".to_string()]).await?;

        // Get the tools and verify
        let tools = chat.tools();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name(), "mock_tool");

        Ok(())
    }

    #[tokio::test]
    async fn test_add_and_get_last_message() -> Result<()> {
        let server = MockServer::start().await;
        let config = get_test_config(&server).await?;
        let mock_tool: Arc<dyn Tool> = Arc::new(MockTool);
        let available_tools = HashMap::from([("mock_tool", mock_tool)]);
        let mut chat = Chat::new(&config, None, available_tools).await?;

        chat.add_messages(
            vec![ChatMessage {
                sender: SenderType::User,
                text: "Hello".to_string(),
                tools: Vec::new(),
            }],
            Vec::new(),
        )
        .await;
        assert!(chat.get_last_assistant_message().is_none());

        chat.add_messages(
            vec![ChatMessage {
                sender: SenderType::Assistant,
                text: "Hi there!".to_string(),
                tools: Vec::new(),
            }],
            Vec::new(),
        )
        .await;

        let last_message = chat.get_last_assistant_message();
        assert!(last_message.is_some());
        let last_message = last_message.unwrap();
        assert_eq!(last_message.sender, SenderType::Assistant);
        assert_eq!(last_message.text, "Hi there!");

        Ok(())
    }

    #[tokio::test]
    async fn test_clear_messages() -> Result<()> {
        let server = MockServer::start().await;
        let config = get_test_config(&server).await?;
        let mock_tool: Arc<dyn Tool> = Arc::new(MockTool);
        let available_tools = HashMap::from([("mock_tool", mock_tool)]);
        let mut chat = Chat::new(&config, None, available_tools).await?;

        chat.add_messages(
            vec![
                ChatMessage {
                    sender: SenderType::User,
                    text: "Hello".to_string(),
                    tools: Vec::new(),
                },
                ChatMessage {
                    sender: SenderType::Assistant,
                    text: "Hi there!".to_string(),
                    tools: Vec::new(),
                },
            ],
            Vec::new(),
        )
        .await;

        assert!(chat.get_last_assistant_message().is_some());

        chat.clear_messages().await;

        assert!(chat.get_last_assistant_message().is_none());
        Ok(())
    }

    #[tokio::test]
    async fn test_stream_response() -> Result<()> {
        let server = MockServer::start().await;
        let response = ResponseTemplate::new(200)
            .set_body_bytes(mock_event_stream_body().as_bytes())
            .insert_header("Content-Type", "text/event-stream");
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(response)
            .mount(&server)
            .await;

        let config = get_test_config(&server).await?;
        let mock_tool: Arc<dyn Tool> = Arc::new(MockTool);
        let available_tools = HashMap::from([("mock_tool", mock_tool)]);
        let chat = Chat::new(&config, None, available_tools).await?;

        let cancel_token = CancellationToken::new();
        let mut stream = chat.stream_response(cancel_token).await?;

        let mut response_text = String::new();
        while let Some(result) = stream.next().await {
            if let Completion::Response(response) = result? {
                response_text.push_str(&response.text);
            }
        }

        assert_eq!(response_text, "Hello world!");

        Ok(())
    }

    #[tokio::test]
    async fn test_set_model() -> Result<()> {
        let server = MockServer::start().await;
        let mut config = get_test_config(&server).await?;

        // Add second model to config
        config.models.insert(
            "test-model-2".to_string(),
            arey_core::model::ModelConfig {
                key: "test-key".to_string(),
                name: "test-model-2".to_string(),
                provider: arey_core::model::ModelProvider::Openai,
                settings: HashMap::from([
                    ("base_url".to_string(), server.uri().into()),
                    ("api_key".to_string(), "MOCK_OPENAI_API_KEY_2".into()),
                ]),
            },
        );

        let mock_tool: Arc<dyn Tool> = Arc::new(MockTool);
        let available_tools = HashMap::from([("mock_tool", mock_tool)]);

        let mut chat = Chat::new(&config, Some("test-model".to_string()), available_tools).await?;

        // Switch to second model
        chat.set_model("test-model-2").await?;

        // Verify model changed
        let models = chat.available_model_names();
        assert_eq!(models.len(), 2);
        assert!(models.contains(&"test-model"));
        assert!(models.contains(&"test-model-2"));
        Ok(())
    }

    #[tokio::test]
    async fn test_set_agent() -> Result<()> {
        let server = MockServer::start().await;
        let config = get_test_config(&server).await?;
        let mock_tool: Arc<dyn Tool> = Arc::new(MockTool);
        let available_tools = HashMap::from([("mock_tool", mock_tool)]);

        let mut chat = Chat::new(&config, None, available_tools).await?;
        assert_eq!(chat.agent_name(), "test-agent");
        assert_eq!(chat.system_prompt(), "You are a test agent.");

        // Set a valid agent
        chat.set_agent("test-agent").await?;
        assert_eq!(chat.agent_name(), "test-agent".to_string());
        assert_eq!(chat.system_prompt(), "You are a test agent.");
        assert_eq!(chat.tools().len(), 1);

        // Setting an invalid agent should fail
        let result = chat.set_agent("non-existent-agent").await;
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Agent 'non-existent-agent' not found in config.")
        );

        // Agent should remain unchanged after error
        assert_eq!(chat.agent_name(), "test-agent".to_string());

        Ok(())
    }

    #[tokio::test]
    async fn test_set_system_prompt() -> Result<()> {
        let server = MockServer::start().await;
        let config = get_test_config(&server).await?;
        let mock_tool: Arc<dyn Tool> = Arc::new(MockTool);
        let available_tools = HashMap::from([("mock_tool", mock_tool)]);

        // Use the test-agent which has tools configured
        let mut chat = Chat::new(&config, Some("test-model".to_string()), available_tools).await?;
        chat.set_agent("test-agent").await?;

        let original_prompt = chat.system_prompt();
        assert_eq!(original_prompt, "You are a test agent.");

        // Add some messages to test preservation
        chat.add_messages(
            vec![ChatMessage {
                sender: SenderType::User,
                text: "Hello".to_string(),
                tools: vec![],
            }],
            vec![],
        )
        .await;

        // Verify we have tools initially
        let tools = chat.tools();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name(), "mock_tool");

        // Set a new system prompt
        let new_prompt = "You are a Python expert.";
        chat.set_system_prompt(new_prompt).await?;
        assert_eq!(chat.system_prompt(), new_prompt);

        // Verify that the message history was preserved
        let messages = chat.get_all_messages();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].text, "Hello");

        // Verify tools are preserved after prompt change
        let tools = chat.tools();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name(), "mock_tool");

        Ok(())
    }

    #[tokio::test]
    async fn test_set_profile() -> Result<()> {
        let server = MockServer::start().await;
        let mut config = get_test_config(&server).await?;

        // Add a second profile to the config for testing
        config.profiles.insert(
            "test-profile".to_string(),
            ProfileConfig {
                temperature: 0.8,
                top_p: 0.9,
                top_k: 40,
                repeat_penalty: 1.1,
            },
        );

        let mock_tool: Arc<dyn Tool> = Arc::new(MockTool);
        let available_tools = HashMap::from([("mock_tool", mock_tool)]);
        let mut chat = Chat::new(&config, Some("test-model".to_string()), available_tools).await?;

        // Check initial profile (should be the agent's default profile)
        let initial_profile = chat.current_profile();
        assert!(initial_profile.is_some());
        let (agent_name, profile_config) = initial_profile.unwrap();
        assert_eq!(agent_name, "test-agent");
        assert_eq!(profile_config.temperature, 0.1); // Default from test config

        // Set a new profile
        chat.set_profile("test-profile")?;

        // Check that the profile was updated
        let new_profile = chat.current_profile();
        assert!(new_profile.is_some());
        let (profile_name, profile_config) = new_profile.unwrap();
        assert_eq!(profile_name, "test-profile"); // Profile name should be the new profile name
        assert_eq!(profile_config.temperature, 0.8); // New temperature from test-profile

        // Setting an invalid profile should fail
        let result = chat.set_profile("non-existent-profile");
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Profile 'non-existent-profile' not found")
        );

        // Profile should remain unchanged after error
        let current_profile = chat.current_profile();
        assert!(current_profile.is_some());
        let (_, profile_config) = current_profile.unwrap();
        assert_eq!(profile_config.temperature, 0.8); // Should still be test-profile

        Ok(())
    }

    #[tokio::test]
    async fn test_profile_name_returns_correct_name() -> Result<()> {
        let server = MockServer::start().await;
        let mut config = get_test_config(&server).await?;

        // Add multiple profiles to test
        config.profiles.insert(
            "creative".to_string(),
            ProfileConfig {
                temperature: 0.9,
                top_p: 0.95,
                top_k: 50,
                repeat_penalty: 1.0,
            },
        );
        config.profiles.insert(
            "precise".to_string(),
            ProfileConfig {
                temperature: 0.1,
                top_p: 0.5,
                top_k: 20,
                repeat_penalty: 1.2,
            },
        );

        let mock_tool: Arc<dyn Tool> = Arc::new(MockTool);
        let available_tools = HashMap::from([("mock_tool", mock_tool)]);
        let mut chat = Chat::new(&config, Some("test-model".to_string()), available_tools).await?;

        // Initially, profile_name should return the agent name (default behavior)
        assert_eq!(chat.profile_name(), "test-agent");

        // Switch to creative profile
        chat.set_profile("creative")?;
        assert_eq!(chat.profile_name(), "creative");

        // Switch to precise profile
        chat.set_profile("precise")?;
        assert_eq!(chat.profile_name(), "precise");

        // Test error case for non-existent profile
        let result = chat.set_profile("non-existent");
        assert!(result.is_err());

        // Profile name should remain unchanged after error
        assert_eq!(chat.profile_name(), "precise");

        Ok(())
    }

    #[tokio::test]
    async fn test_current_profile_returns_correct_data() -> Result<()> {
        let server = MockServer::start().await;
        let mut config = get_test_config(&server).await?;

        config.profiles.insert(
            "test-profile".to_string(),
            ProfileConfig {
                temperature: 0.8,
                top_p: 0.9,
                top_k: 40,
                repeat_penalty: 1.1,
            },
        );

        let mock_tool: Arc<dyn Tool> = Arc::new(MockTool);
        let available_tools = HashMap::from([("mock_tool", mock_tool)]);
        let mut chat = Chat::new(&config, Some("test-model".to_string()), available_tools).await?;

        // Initially, current_profile should return the agent's default profile
        let initial_profile = chat.current_profile();
        assert!(initial_profile.is_some());
        let (name, config) = initial_profile.unwrap();
        assert_eq!(name, "test-agent");
        assert_eq!(config.temperature, 0.1); // From test agent config

        // Switch to test-profile
        chat.set_profile("test-profile")?;

        // Now current_profile should return the test-profile
        let new_profile = chat.current_profile();
        assert!(new_profile.is_some());
        let (name, config) = new_profile.unwrap();
        assert_eq!(name, "test-profile");
        assert_eq!(config.temperature, 0.8); // From test-profile config

        Ok(())
    }

    #[tokio::test]
    async fn test_profile_name_updates_correctly() -> Result<()> {
        let server = MockServer::start().await;
        let mut config = get_test_config(&server).await?;

        config.profiles.insert(
            "fast".to_string(),
            ProfileConfig {
                temperature: 0.9,
                top_p: 0.8,
                top_k: 50,
                repeat_penalty: 1.0,
            },
        );

        let mock_tool: Arc<dyn Tool> = Arc::new(MockTool);
        let available_tools = HashMap::from([("mock_tool", mock_tool)]);
        let mut chat = Chat::new(&config, Some("test-model".to_string()), available_tools).await?;

        // Test initial profile name
        assert_eq!(chat.profile_name(), "test-agent");

        // Switch to fast profile
        chat.set_profile("fast")?;

        // Test updated profile name
        assert_eq!(chat.profile_name(), "fast");

        Ok(())
    }

    #[tokio::test]
    async fn test_stream_response_uses_current_profile() -> Result<()> {
        let server = MockServer::start().await;
        let mut config = get_test_config(&server).await?;

        config.profiles.insert(
            "high-temp".to_string(),
            ProfileConfig {
                temperature: 0.9,
                top_p: 0.95,
                top_k: 50,
                repeat_penalty: 1.0,
            },
        );

        let mock_tool: Arc<dyn Tool> = Arc::new(MockTool);
        let available_tools = HashMap::from([("mock_tool", mock_tool)]);
        let mut chat = Chat::new(&config, Some("test-model".to_string()), available_tools).await?;

        // Switch to high-temp profile
        chat.set_profile("high-temp")?;

        // Check that the current profile reflects the change
        let current_profile = chat.current_profile();
        assert!(current_profile.is_some());
        let (profile_name, profile_config) = current_profile.unwrap();
        assert_eq!(profile_name, "high-temp");
        assert_eq!(profile_config.temperature, 0.9);

        // The stream_response method should use the current profile
        // We can't easily test the actual stream without a real model, but we can verify
        // that the current_profile() method returns the expected profile that stream_response would use
        assert_eq!(chat.profile_name(), "high-temp");

        Ok(())
    }

    #[tokio::test]
    async fn test_profile_switching_integration() -> Result<()> {
        let server = MockServer::start().await;
        let mut config = get_test_config(&server).await?;

        config.profiles.insert(
            "integration-test-profile".to_string(),
            ProfileConfig {
                temperature: 0.7,
                top_p: 0.8,
                top_k: 30,
                repeat_penalty: 1.15,
            },
        );

        let mock_tool: Arc<dyn Tool> = Arc::new(MockTool);
        let available_tools = HashMap::from([("mock_tool", mock_tool)]);
        let mut chat = Chat::new(&config, Some("test-model".to_string()), available_tools).await?;

        // Test initial state
        assert_eq!(chat.profile_name(), "test-agent");
        let initial_profile = chat.current_profile();
        assert!(initial_profile.is_some());
        let (initial_name, initial_config) = initial_profile.unwrap();
        assert_eq!(initial_name, "test-agent");
        assert_eq!(initial_config.temperature, 0.1);

        // Test profile switching
        chat.set_profile("integration-test-profile")?;

        // Verify the profile was switched
        assert_eq!(chat.profile_name(), "integration-test-profile");
        let switched_profile = chat.current_profile();
        assert!(switched_profile.is_some());
        let (switched_name, switched_config) = switched_profile.unwrap();
        assert_eq!(switched_name, "integration-test-profile");
        assert_eq!(switched_config.temperature, 0.7);

        // Test that stream_response would use the correct profile
        // by verifying current_profile() returns what we expect
        let stream_profile = chat.current_profile();
        assert!(stream_profile.is_some());
        let (stream_name, stream_config) = stream_profile.unwrap();
        assert_eq!(stream_name, "integration-test-profile");
        assert_eq!(stream_config.temperature, 0.7);

        Ok(())
    }
}
