//! An agent is a combination of a system prompt, tools, and model parameters.
//! This provides a way to create specialized assistants for specific tasks.

use crate::config::ProfileConfig;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// The source from which an agent was loaded.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AgentSource {
    /// Built-in agent that ships with the application.
    BuiltIn,
    /// User-defined agent loaded from a file.
    UserFile(PathBuf),
}

/// Metadata about an agent's origin and loading information.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AgentMetadata {
    /// The source of this agent.
    pub source: AgentSource,
    /// The file path, if the agent was loaded from a file.
    pub file_path: Option<PathBuf>,
}

impl Default for AgentMetadata {
    fn default() -> Self {
        Self {
            source: AgentSource::BuiltIn,
            file_path: None,
        }
    }
}

/// Runtime state for an agent during a session.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct AgentRuntimeState {
    /// The current model being used (can override the agent's default).
    pub current_model: Option<String>,
    /// The current profile name being used (can override the agent's default).
    pub current_profile_name: Option<String>,
    /// The current profile configuration being used (can override the agent's default).
    pub current_profile: Option<ProfileConfig>,
    /// Session-specific tool overrides.
    pub session_tools: Option<Vec<String>>,
    /// Whether this agent is active in the current session.
    pub is_active: bool,
}

/// A unified agent representation that combines configuration, metadata, and runtime state.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Agent {
    /// The name of the agent.
    pub name: String,
    /// The system prompt that defines the agent's persona and instructions.
    pub prompt: String,
    /// A list of tool names that this agent is allowed to use.
    #[serde(default)]
    pub tools: Vec<String>,
    /// The generation profile for the agent, controlling parameters like temperature.
    #[serde(default)]
    pub profile: ProfileConfig,
    /// Metadata about the agent's origin and loading information.
    #[serde(skip)]
    pub metadata: AgentMetadata,
    /// Runtime state for the agent during a session.
    #[serde(skip)]
    pub runtime_state: AgentRuntimeState,
}

impl Agent {
    /// Creates a new Agent with the given configuration.
    pub fn new(
        name: String,
        prompt: String,
        tools: Vec<String>,
        profile: ProfileConfig,
        metadata: AgentMetadata,
    ) -> Self {
        Self {
            name,
            prompt,
            tools,
            profile,
            metadata,
            runtime_state: AgentRuntimeState::default(),
        }
    }

    /// Creates a new Agent with default metadata.
    pub fn with_default_metadata(
        name: String,
        prompt: String,
        tools: Vec<String>,
        profile: ProfileConfig,
        source: AgentSource,
    ) -> Self {
        let metadata = AgentMetadata {
            source,
            file_path: None,
        };
        Self::new(name, prompt, tools, profile, metadata)
    }

    /// Gets the effective tools for this agent, considering session overrides.
    pub fn effective_tools(&self) -> &[String] {
        self.runtime_state
            .session_tools
            .as_ref()
            .unwrap_or(&self.tools)
    }

    /// Gets the effective profile for this agent, considering session overrides.
    pub fn effective_profile(&self) -> &ProfileConfig {
        self.runtime_state
            .current_profile
            .as_ref()
            .unwrap_or(&self.profile)
    }

    /// Gets the effective model for this agent, considering session overrides.
    pub fn effective_model(&self) -> Option<&str> {
        self.runtime_state.current_model.as_deref()
    }

    /// Sets the current model for this session.
    pub fn set_current_model(&mut self, model: Option<String>) {
        self.runtime_state.current_model = model;
    }

    /// Sets the current profile for this session.
    pub fn set_current_profile(&mut self, profile: Option<ProfileConfig>) {
        self.runtime_state.current_profile = profile;
    }

    /// Sets the current profile with name for this session.
    pub fn set_current_profile_with_name(&mut self, profile_name: String, profile: ProfileConfig) {
        self.runtime_state.current_profile_name = Some(profile_name);
        self.runtime_state.current_profile = Some(profile);
    }

    /// Clears the current profile and returns to the agent's default profile.
    pub fn clear_current_profile(&mut self) {
        self.runtime_state.current_profile_name = None;
        self.runtime_state.current_profile = None;
    }

    /// Sets the session tools.
    pub fn set_session_tools(&mut self, tools: Option<Vec<String>>) {
        self.runtime_state.session_tools = tools;
    }

    /// Sets whether this agent is active in the current session.
    pub fn set_active(&mut self, is_active: bool) {
        self.runtime_state.is_active = is_active;
    }

    /// Checks if this agent is a built-in agent.
    pub fn is_builtin(&self) -> bool {
        matches!(self.metadata.source, AgentSource::BuiltIn)
    }

    /// Checks if this agent is a user-defined agent.
    pub fn is_user_defined(&self) -> bool {
        matches!(self.metadata.source, AgentSource::UserFile(_))
    }

    /// Gets a display string showing the agent's current state.
    pub fn display_state(&self) -> String {
        self.name.clone()
    }
}

/// Legacy AgentConfig type for backward compatibility.
/// This will be deprecated in favor of the unified Agent struct.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct AgentConfig {
    /// The name of the agent. If configured in `arey.yml`, this is typically the key from the `agents` map.
    #[serde(default)]
    pub name: String,
    /// The system prompt that defines the agent's persona and instructions.
    pub prompt: String,
    /// A list of tool names that this agent is allowed to use.
    #[serde(default)]
    pub tools: Vec<String>,
    /// The generation profile for the agent, controlling parameters like temperature.
    #[serde(default)]
    pub profile: ProfileConfig,
}

impl AgentConfig {
    /// Creates a new `AgentConfig` programmatically.
    pub fn new(name: &str, prompt: &str, tools: Vec<String>) -> Self {
        Self {
            name: name.to_string(),
            prompt: prompt.to_string(),
            tools,
            profile: ProfileConfig::default(),
        }
    }

    /// Sets the profile for the agent's configuration using a builder pattern.
    pub fn with_profile(mut self, profile: ProfileConfig) -> Self {
        self.profile = profile;
        self
    }

    /// Converts this AgentConfig to a unified Agent with default metadata.
    pub fn to_agent(self, source: AgentSource) -> Agent {
        Agent::with_default_metadata(self.name, self.prompt, self.tools, self.profile, source)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_yaml;

    #[test]
    fn test_agent_new() {
        let metadata = AgentMetadata {
            source: AgentSource::BuiltIn,
            file_path: None,
        };

        let agent = Agent::new(
            "coder".to_string(),
            "You are a coder.".to_string(),
            vec!["search".to_string(), "file".to_string()],
            ProfileConfig::default(),
            metadata,
        );

        assert_eq!(agent.name, "coder");
        assert_eq!(agent.prompt, "You are a coder.");
        assert_eq!(agent.tools, vec!["search".to_string(), "file".to_string()]);
        assert_eq!(agent.profile, ProfileConfig::default());
        assert!(agent.is_builtin());
        assert!(!agent.is_user_defined());
    }

    #[test]
    fn test_agent_with_default_metadata() {
        let agent = Agent::with_default_metadata(
            "writer".to_string(),
            "You are a writer.".to_string(),
            vec![],
            ProfileConfig::default(),
            AgentSource::UserFile(PathBuf::from("test.yml")),
        );

        assert_eq!(agent.name, "writer");
        assert_eq!(agent.prompt, "You are a writer.");
        assert!(agent.tools.is_empty());
        assert!(agent.is_user_defined());
        assert!(!agent.is_builtin());
    }

    #[test]
    fn test_agent_effective_methods() {
        let mut agent = Agent::with_default_metadata(
            "test".to_string(),
            "Test prompt".to_string(),
            vec!["search".to_string()],
            ProfileConfig::default(),
            AgentSource::BuiltIn,
        );

        // Test default behavior
        assert_eq!(agent.effective_tools(), &["search".to_string()]);
        assert_eq!(agent.effective_profile(), &ProfileConfig::default());
        assert_eq!(agent.effective_model(), None);

        // Test with session overrides
        agent.set_current_model(Some("gpt-4".to_string()));
        let custom_profile = ProfileConfig {
            temperature: 0.9,
            ..Default::default()
        };
        agent.set_current_profile(Some(custom_profile.clone()));
        agent.set_session_tools(Some(vec!["file".to_string(), "edit".to_string()]));

        assert_eq!(agent.effective_model(), Some("gpt-4"));
        assert_eq!(agent.effective_profile(), &custom_profile);
        assert_eq!(
            agent.effective_tools(),
            &["file".to_string(), "edit".to_string()]
        );
    }

    #[test]
    fn test_agent_display_state() {
        let mut agent = Agent::with_default_metadata(
            "coder".to_string(),
            "Test prompt".to_string(),
            vec![],
            ProfileConfig::default(),
            AgentSource::BuiltIn,
        );

        // Test basic state
        assert_eq!(agent.display_state(), "coder");

        // Test with model override (model is no longer shown in display_state as it's shown separately in status)
        agent.set_current_model(Some("gpt-4".to_string()));
        assert_eq!(agent.display_state(), "coder");

        // Test with custom temperature (temperature is no longer shown in display_state)
        let mut agent = Agent::with_default_metadata(
            "creative".to_string(),
            "Test prompt".to_string(),
            vec![],
            ProfileConfig {
                temperature: 0.9,
                ..Default::default()
            },
            AgentSource::BuiltIn,
        );
        assert_eq!(agent.display_state(), "creative");

        // Test with both model and custom temperature (neither shown in display_state)
        agent.set_current_model(Some("claude".to_string()));
        assert_eq!(agent.display_state(), "creative");
    }

    #[test]
    fn test_agent_runtime_state() {
        let mut agent = Agent::with_default_metadata(
            "test".to_string(),
            "Test prompt".to_string(),
            vec![],
            ProfileConfig::default(),
            AgentSource::BuiltIn,
        );

        // Test default state
        assert!(!agent.runtime_state.is_active);
        assert!(agent.runtime_state.current_model.is_none());
        assert!(agent.runtime_state.current_profile.is_none());
        assert!(agent.runtime_state.session_tools.is_none());

        // Test setting state
        agent.set_active(true);
        assert!(agent.runtime_state.is_active);

        agent.set_current_model(Some("test-model".to_string()));
        assert_eq!(
            agent.runtime_state.current_model,
            Some("test-model".to_string())
        );
    }

    // Legacy AgentConfig tests for backward compatibility
    #[test]
    fn test_agent_config_new() {
        let agent = AgentConfig::new(
            "coder",
            "You are a coder.",
            vec!["search".to_string(), "file".to_string()],
        );
        assert_eq!(agent.name, "coder");
        assert_eq!(agent.prompt, "You are a coder.");
        assert_eq!(agent.tools, vec!["search".to_string(), "file".to_string()]);
        assert_eq!(agent.profile, ProfileConfig::default());
    }

    #[test]
    fn test_agent_config_with_profile() {
        let profile = ProfileConfig {
            temperature: 0.9,
            ..Default::default()
        };
        let agent =
            AgentConfig::new("writer", "You are a writer.", vec![]).with_profile(profile.clone());
        assert_eq!(agent.profile, profile);
    }

    #[test]
    fn test_agent_config_to_agent() {
        let config = AgentConfig::new("converter", "Test prompt", vec!["tool1".to_string()]);

        let agent = config.clone().to_agent(AgentSource::BuiltIn);

        assert_eq!(agent.name, config.name);
        assert_eq!(agent.prompt, config.prompt);
        assert_eq!(agent.tools, config.tools);
        assert_eq!(agent.profile, config.profile);
        assert!(agent.is_builtin());
    }

    #[test]
    fn test_agent_config_deserialization() {
        let yaml = r#"
            prompt: "Test Agent Prompt"
            tools:
              - tool1
              - tool2
            profile:
              temperature: 0.8
              top_k: 50
        "#;
        let agent: AgentConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(agent.prompt, "Test Agent Prompt");
        assert_eq!(agent.tools, vec!["tool1".to_string(), "tool2".to_string()]);
        assert_eq!(agent.profile.temperature, 0.8);
        assert_eq!(agent.profile.top_k, 50);
        // default values
        assert_eq!(agent.profile.repeat_penalty, 1.176);
        assert_eq!(agent.profile.top_p, 0.1);
    }

    #[test]
    fn test_agent_config_deserialization_defaults() {
        let yaml = r#"
            prompt: "Minimal Agent"
        "#;
        let agent: AgentConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(agent.prompt, "Minimal Agent");
        assert_eq!(agent.tools, Vec::<String>::new());
        assert_eq!(agent.profile, ProfileConfig::default());
        assert_eq!(agent.name, ""); // Name is not in yaml, defaults to empty
    }
}
