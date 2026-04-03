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
    session: Option<Session>,
    model_config: Option<arey_core::model::ModelConfig>,
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
    /// Creates a new `Chat` session without loading the model.
    ///
    /// Call `load_session()` to initialize the model. This allows showing
    /// a loading indicator while the model is being loaded.
    pub fn new(
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

        if config.chat.profile != ProfileConfig::default() {
            agent.set_current_profile_with_name(
                config
                    .chat
                    .profile_name
                    .clone()
                    .unwrap_or_else(|| "chat".to_string()),
                config.chat.profile.clone(),
            );
        }

        // Mark this agent as active
        agent.set_active(true);

        Ok(Self {
            session: None,
            model_config: Some(model_config),
            current_agent: agent,
            available_tools,
            config,
        })
    }

    /// Loads the session, initializing the model.
    ///
    /// This should be called before any chat operations. It can be wrapped
    /// with a loading indicator to show model loading progress.
    /// If the session is already loaded, this is a no-op.
    pub async fn load_session(&mut self) -> Result<()> {
        if self.session.is_some() {
            return Ok(());
        }

        let model_config = self.model_config.take().context("Session already loaded")?;

        let tools: Vec<Arc<dyn Tool>> = self
            .current_agent
            .effective_tools()
            .iter()
            .filter_map(|tool_name| self.available_tools.get(tool_name.as_str()).cloned())
            .collect();

        let mut session = Session::new(model_config, &self.current_agent.prompt)
            .context("Failed to create chat session")?;
        session.set_tools(tools)?;

        self.session = Some(session);
        Ok(())
    }

    /// Loads session synchronously for use in tests or non-async contexts.
    /// This is a blocking wrapper around load_session.
    #[cfg(test)]
    pub fn load_session_blocking(&mut self) -> Result<()> {
        let model_config = self.model_config.take().context("Session already loaded")?;

        let tools: Vec<Arc<dyn Tool>> = self
            .current_agent
            .effective_tools()
            .iter()
            .filter_map(|tool_name| self.available_tools.get(tool_name.as_str()).cloned())
            .collect();

        let mut session = Session::new(model_config, &self.current_agent.prompt)
            .context("Failed to create chat session")?;
        session.set_tools(tools)?;

        self.session = Some(session);
        Ok(())
    }

    /// Ensures the session is loaded, loading it if necessary.
    /// This is called automatically by methods that need the session.
    pub async fn ensure_session_loaded(&mut self) -> Result<()> {
        self.load_session().await
    }

    /// Get current agent name
    pub fn agent_name(&self) -> String {
        self.current_agent.name.clone()
    }

    /// Get current agent state display string
    pub fn agent_display_state(&self) -> String {
        self.current_agent.display_state()
    }

    fn session(&self) -> &Session {
        if self.session.is_none() {
            panic!("Session not loaded. Call load_session() first.");
        }
        self.session.as_ref().unwrap()
    }

    fn session_mut(&mut self) -> &mut Session {
        self.ensure_session_loaded_blocking();
        self.session.as_mut().unwrap()
    }

    /// Loads session synchronously for use in tests or non-async contexts.
    /// This is a blocking wrapper around load_session.
    pub fn ensure_session_loaded_blocking(&mut self) {
        if self.session.is_some() {
            return;
        }

        let model_config = match self.model_config.take() {
            Some(c) => c,
            None => return,
        };

        let tools: Vec<Arc<dyn Tool>> = self
            .current_agent
            .effective_tools()
            .iter()
            .filter_map(|tool_name| self.available_tools.get(tool_name.as_str()).cloned())
            .collect();

        if let Ok(mut session) = Session::new(model_config, &self.current_agent.prompt)
            && session.set_tools(tools).is_ok()
        {
            self.session = Some(session);
        }
    }

    /// Check if session is loaded
    pub fn is_session_loaded(&self) -> bool {
        self.session.is_some()
    }

    /// Get the model name that will be loaded
    pub fn model_key(&self) -> String {
        self.model_config
            .as_ref()
            .map(|c| c.key.clone())
            .unwrap_or_else(|| self.session().model_key())
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

        let model_key = self.session().model_key();
        let _model_config = self
            .config
            .models
            .get(&model_key)
            .cloned()
            .context(format!(
                "Model '{}' associated with the current session not found in config.",
                model_key
            ))?;

        let _messages = self.session().all_messages();

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
        self.session_mut()
            .update_agent(new_agent.prompt.clone(), tools?)?;
        self.current_agent = new_agent;

        Ok(())
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
        self.session_mut().set_model(model_config)?;

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
        self.session().system_prompt()
    }

    /// Set new system prompt for the current session
    pub async fn set_system_prompt(&mut self, prompt: &str) -> Result<()> {
        // Update the session with the new system prompt
        self.session_mut().set_system_prompt(prompt)?;

        Ok(())
    }

    /// Get the enable_thinking setting
    pub fn enable_thinking(&self) -> Option<bool> {
        self.session().enable_thinking()
    }

    /// Set the enable_thinking template parameter
    /// None = use model's default, Some(true) = enable thinking, Some(false) = disable thinking
    pub fn set_enable_thinking(&mut self, enabled: Option<bool>) {
        self.session_mut().set_enable_thinking(enabled);
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
        self.session().tools()
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
        self.session_mut().set_tools(tools)?;

        Ok(())
    }

    /// Get all messages from the session
    pub fn get_all_messages(&self) -> Vec<ChatMessage> {
        self.session().all_messages()
    }

    /// Retrieves the last message from the assistant in the conversation history.
    pub fn get_last_assistant_message(&self) -> Option<ChatMessage> {
        self.session().last_assistant_message()
    }

    /// Adds messages to the conversation history.
    pub async fn add_messages(&mut self, messages: Vec<ChatMessage>) {
        for message in messages {
            if let Err(e) = self.session_mut().add_message(message) {
                eprintln!("Failed to add message: {}", e);
            }
        }
    }

    /// Clears the conversation history of the session.
    pub async fn clear_messages(&mut self) {
        if let Err(e) = self.session_mut().clear_history().await {
            eprintln!("Failed to clear history: {}", e);
        }
    }

    /// Generates a streaming response from the model based on the conversation history.
    pub async fn stream_response(
        &mut self,
        cancel_token: CancellationToken,
    ) -> Result<BoxStream<'_, Result<Completion>>> {
        // Use the current profile (either session override or agent default)
        let profile = self
            .current_profile()
            .map(|(_, profile_config)| profile_config);

        let settings = if let Some(profile) = profile {
            profile
                .to_settings()
                .context("Failed to convert profile to settings")?
        } else {
            HashMap::new()
        };

        tracing::debug!("Settings from profile: {:?}", settings);
        if let Some(max_tokens) = settings.get("max_tokens") {
            tracing::debug!("Profile max_tokens: {}", max_tokens);
        } else {
            tracing::warn!("max_tokens NOT found in profile settings!");
        }

        let session = self.session_mut();

        let stream = async_stream::stream! {
            let mut inner_stream = match session.generate(settings, cancel_token.clone()).await {
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
        config::{Config, get_config},
        tools::{Tool, ToolError},
    };
    use async_trait::async_trait;
    use serde_json::{Value, json};
    use std::sync::Arc;
    use wiremock::MockServer;

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
        let config = get_test_config(&server).await?;
        let mock_tool: Arc<dyn Tool> = Arc::new(MockTool);
        let available_tools = HashMap::from([("mock_tool", mock_tool)]);

        // Test with existing model, should use default agent from config
        let mut chat = Chat::new(&config, Some("test-model".to_string()), available_tools)?;

        chat.load_session().await?;

        let current = chat.current_profile();
        assert!(current.is_some());
        let (name, profile_config) = current.unwrap();
        assert_eq!(name, "precise");
        assert_eq!(profile_config.temperature, 0.3);
        assert_eq!(profile_config.top_k, 20);

        Ok(())
    }
}
