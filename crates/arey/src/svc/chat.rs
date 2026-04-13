use anyhow::{Context, Result};
use arey_core::agent::{Agent, AgentSource};
use arey_core::completion::{CancellationToken, ChatMessage};
use arey_core::config::{Config, ProfileConfig};
use arey_core::model::ModelConfig;
use arey_core::registry::ToolRegistry;
use arey_core::session::{Session, SessionConfig, SessionEvent};
use arey_core::tools::Tool;
use arey_mcp::McpRegistry;
use futures::{StreamExt, stream::BoxStream};
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use tracing::debug;

/// Represents an interactive chat session between a user and an AI model.
///
/// It maintains conversation history and manages tool usage.
pub struct Chat<'a> {
    config: &'a Config,
    model_config: ModelConfig,
    session_config: SessionConfig,
    current_agent: Agent,
    session: Option<Session>,
    tool_registry: ToolRegistry,
    mcp_registry: Option<McpRegistry>,
}

impl<'a> fmt::Debug for Chat<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Chat")
            .field(
                "tool_names",
                &self
                    .session_config
                    .tools
                    .iter()
                    .map(|t| t.name())
                    .collect::<Vec<_>>(),
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
        tool_registry: arey_core::registry::ToolRegistry,
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

        // Get tools from registry based on agent's effective tools
        let tools = tool_registry.tools_for(agent.effective_tools());

        let session_config = SessionConfig {
            system_prompt: agent.prompt.clone(),
            tools,
            ..Default::default()
        };

        Ok(Self {
            session: None,
            model_config,
            session_config,
            current_agent: agent,
            tool_registry,
            config,
            mcp_registry: None,
        })
    }

    /// Builder method to set MCP server manager after creating Chat
    pub fn with_mcp_registry(mut self, mcp_registry: McpRegistry) -> Self {
        self.mcp_registry = Some(mcp_registry);
        self
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

        let session = Session::new(self.model_config.clone(), self.session_config.clone())
            .context("Failed to create chat session")?;

        self.session = Some(session);
        Ok(())
    }

    /// Loads session synchronously for use in tests or non-async contexts.
    /// This is a blocking wrapper around load_session.
    #[cfg(test)]
    pub fn load_session_blocking(&mut self) -> Result<()> {
        if self.session.is_some() {
            return Ok(());
        }

        let session = Session::new(self.model_config.clone(), self.session_config.clone())
            .context("Failed to create chat session")?;

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

        if let Ok(session) = Session::new(self.model_config.clone(), self.session_config.clone()) {
            self.session = Some(session);
        }
    }

    /// Check if session is loaded
    pub fn is_session_loaded(&self) -> bool {
        self.session.is_some()
    }

    /// Get the model name that will be loaded
    pub fn model_key(&self) -> String {
        if let Some(ref session) = self.session {
            session.model_key().to_string()
        } else {
            self.model_config.key.clone()
        }
    }

    fn update_session_config(&mut self, config: SessionConfig) {
        self.session_config = config.clone();
        if let Some(session) = self.session.as_mut() {
            session.update_config(config);
        }
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
        let _model_config = self.config.models.get(model_key).cloned().context(format!(
            "Model '{}' associated with the current session not found in config.",
            model_key
        ))?;

        // Activate new agent
        new_agent.set_active(true);

        // Ensure we clear any session overrides when switching agent
        new_agent.set_session_tools(None);
        new_agent.clear_current_profile();

        // Get new tools from registry for new agent
        let tools = self.tool_registry.tools_for(new_agent.effective_tools());
        debug!(
            "Tools for agent '{}': {:?}",
            new_agent.name,
            tools.iter().map(|f| f.name()).collect::<Vec<_>>()
        );

        // Update the session with new agent and tools
        let new_config = SessionConfig {
            system_prompt: new_agent.prompt.clone(),
            tools,
            ..self.session_config.clone()
        };
        self.update_session_config(new_config);
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
        self.session_mut().update_model(model_config)?;

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
        self.session_config.system_prompt.clone()
    }

    /// Set new system prompt for the current session
    pub async fn set_system_prompt(&mut self, prompt: &str) -> Result<()> {
        let mut config = self.session_config.clone();
        config.system_prompt = prompt.to_string();
        self.update_session_config(config);
        Ok(())
    }

    /// Get the enable_reasoning setting
    pub fn enable_reasoning(&self) -> Option<bool> {
        self.session_config.enable_reasoning
    }

    /// Set the enable_reasoning template parameter
    /// None = use model's default, Some(true) = enable reasoning, Some(false) = disable reasoning
    pub fn set_enable_reasoning(&mut self, enabled: Option<bool>) {
        let mut config = self.session_config.clone();
        config.enable_reasoning = enabled;
        self.update_session_config(config);
    }

    /// Get available agent names
    pub fn available_agent_names(&self) -> Vec<&str> {
        self.config.agents.keys().map(|s| s.as_str()).collect()
    }

    /// Get available tool names (all tools from registry, not just active ones)
    pub fn available_tool_names(&self) -> Vec<String> {
        self.tool_registry.list()
    }

    /// Get MCP server manager reference
    pub fn mcp_registry(&mut self) -> Option<&mut McpRegistry> {
        self.mcp_registry.as_mut()
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
        self.session_config.tools.clone()
    }

    /// Gets the tools for the current chat session.
    pub fn session_tools(&self) -> &[Arc<dyn Tool>] {
        &self.session_config.tools
    }

    /// Sets the tools available for the current chat session.
    pub async fn set_tools(&mut self, tool_names: &[String]) -> Result<()> {
        let tools = self.tool_registry.tools_for(tool_names);

        if tools.len() != tool_names.len() {
            let missing: Vec<_> = tool_names
                .iter()
                .filter(|name| self.tool_registry.get(name).is_none())
                .collect();
            anyhow::bail!("Tools not found or not available: {:?}", missing);
        }

        // Update current agent's session tools to keep it in sync
        self.current_agent
            .set_session_tools(Some(tool_names.to_vec()));

        let mut config = self.session_config.clone();
        config.tools = tools;
        self.update_session_config(config);
        Ok(())
    }

    /// Get all messages from the session
    pub fn get_all_messages(&self) -> Vec<ChatMessage> {
        self.session().messages()
    }

    /// Returns true if the model was reset during the last clear() call.
    #[cfg(test)]
    pub fn was_model_reset(&self) -> bool {
        self.session().was_model_reset()
    }

    /// Retrieves the last message from the assistant in the conversation history.
    pub fn get_last_assistant_message(&self) -> Option<ChatMessage> {
        self.session()
            .messages()
            .iter()
            .rev()
            .find(|m| m.sender == arey_core::completion::SenderType::Assistant)
            .cloned()
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
        if let Err(e) = self.session_mut().clear().await {
            eprintln!("Failed to clear history: {}", e);
        }
    }

    /// Generates a streaming response from the model based on the conversation history.
    pub async fn stream_response(
        &mut self,
        cancel_token: CancellationToken,
    ) -> Result<BoxStream<'_, Result<SessionEvent>>> {
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

    #[derive(Debug, Clone)]
    struct OtherMockTool;

    #[async_trait]
    impl Tool for OtherMockTool {
        fn name(&self) -> String {
            "other_mock_tool".to_string()
        }

        fn description(&self) -> String {
            "Another mock tool for testing".to_string()
        }

        fn parameters(&self) -> Value {
            json!({"type": "object", "properties": {}})
        }

        async fn execute(&self, _input: &Value) -> Result<Value, ToolError> {
            Ok(json!("Other mock tool executed"))
        }
    }

    #[tokio::test]
    async fn test_chat_new() -> Result<()> {
        let server = MockServer::start().await;
        let config = get_test_config(&server).await?;
        let mock_tool: Arc<dyn Tool> = Arc::new(MockTool);
        let mut tool_registry = ToolRegistry::new();
        tool_registry.register(mock_tool)?;

        // Test with existing model, should use default agent from config
        let mut chat = Chat::new(&config, Some("test-model".to_string()), tool_registry)?;

        chat.load_session().await?;

        let current = chat.current_profile();
        assert!(current.is_some());
        let (name, profile_config) = current.unwrap();
        assert_eq!(name, "precise");
        assert_eq!(profile_config.temperature, 0.3);
        assert_eq!(profile_config.top_k, 20);

        Ok(())
    }

    #[tokio::test]
    async fn test_available_tool_names_returns_all_registry_tools() -> Result<()> {
        let server = MockServer::start().await;
        let config = get_test_config(&server).await?;

        // Create registry with multiple tools
        let mock_tool1: Arc<dyn Tool> = Arc::new(MockTool);
        let mock_tool2: Arc<dyn Tool> = Arc::new(OtherMockTool);
        let mut tool_registry = ToolRegistry::new();
        tool_registry.register(mock_tool1)?;
        tool_registry.register(mock_tool2)?;

        // Chat with empty effective_tools (no tools enabled for agent)
        let chat = Chat::new(&config, Some("test-model".to_string()), tool_registry)?;

        // available_tool_names should return ALL tools from registry, not just active ones
        let available = chat.available_tool_names();
        assert_eq!(available.len(), 2);
        assert!(available.contains(&"mock_tool".to_string()));
        assert!(available.contains(&"other_mock_tool".to_string()));

        Ok(())
    }

    #[tokio::test]
    async fn test_chat_with_mcp_builder() -> Result<()> {
        let server = MockServer::start().await;
        let config = get_test_config(&server).await?;

        // Create Chat without MCP
        let mut chat = Chat::new(&config, Some("test-model".to_string()), ToolRegistry::new())?;

        // Initially no MCP manager
        let mcp_before = chat.mcp_registry();
        assert!(mcp_before.is_none());

        // Create a mock McpRegistry for testing
        let mcp_registry = McpRegistry::new();

        // Use builder to add MCP registry
        chat = chat.with_mcp_registry(mcp_registry);

        // Verify MCP registry is now set
        let mcp_after = chat.mcp_registry();
        assert!(mcp_after.is_some());

        Ok(())
    }
}
