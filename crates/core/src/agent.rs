//! An agent is a combination of a system prompt, tools, and model parameters.
//! This provides a way to create specialized assistants for specific tasks.

use crate::{config::ProfileConfig, tools::Tool};
use std::{collections::HashMap, sync::Arc};

/// Represents a system prompt with pre-calculated token count for efficient trimming.
#[derive(Clone, Debug)]
pub struct SystemPrompt {
    text: String,
    tokens: usize, // Pre-calculated tokens for trimming
}

impl SystemPrompt {
    /// Creates a new `SystemPrompt`.
    ///
    /// The token count is estimated using a simple heuristic (chars / 4).
    /// This should be replaced with a proper tokenizer in the future.
    pub fn new(text: &str) -> Self {
        let tokens = text.len() / 4; // Use actual tokenizer in prod
        Self {
            text: text.to_string(),
            tokens,
        }
    }

    /// Returns the token count of the system prompt.
    pub fn tokens(&self) -> usize {
        self.tokens
    }

    /// Returns the text of the system prompt.
    pub fn text(&self) -> &str {
        &self.text
    }
}

/// Configuration for an agent, bundling system prompt, tools, and conversation preferences.
#[derive(Clone, Debug)]
pub struct AgentConfig {
    name: String,
    system_prompt: SystemPrompt,
    tools: HashMap<String, Arc<dyn Tool>>,
    profile: ProfileConfig,
}

impl AgentConfig {
    /// Creates a new `AgentConfig`.
    pub fn new(name: &str, prompt: &str, tools: Vec<Arc<dyn Tool>>) -> Self {
        let tools_map = tools.into_iter().map(|t| (t.name(), t)).collect();

        Self {
            name: name.to_string(),
            system_prompt: SystemPrompt::new(prompt),
            tools: tools_map,
            profile: ProfileConfig::default(),
        }
    }

    /// Sets the profile for the agent's configuration using a builder pattern.
    pub fn with_profile(mut self, profile: ProfileConfig) -> Self {
        self.profile = profile;
        self
    }
}
