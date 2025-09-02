//! An agent is a combination of a system prompt, tools, and model parameters.
//! This provides a way to create specialized assistants for specific tasks.

use crate::config::ProfileConfig;
use serde::{Deserialize, Serialize};

/// Configuration for an agent, defining its behavior and capabilities.
/// Agents are designed to be configured in a user's `arey.yml` file.
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_yaml;

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
