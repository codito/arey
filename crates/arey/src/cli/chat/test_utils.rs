#![cfg(test)]

//! Test utilities for chat modules

#![allow(dead_code)]

use anyhow::Result;
use arey_core::agent::Agent;
use arey_core::config::{Config, ModeConfig, ProfileConfig};
use arey_core::model::ModelConfig;
use arey_core::tools::{Tool, ToolError};
use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;

/// Creates a basic test configuration programmatically
pub fn create_test_config() -> Result<Config> {
    let mut models = HashMap::new();

    // Create test model configurations
    let test_model_1 = ModelConfig {
        key: "test-model-1".to_string(),
        name: "test-model-1".to_string(),
        provider: arey_core::model::ModelProvider::Test,
        settings: HashMap::new(),
    };

    let test_model_2 = ModelConfig {
        key: "test-model-2".to_string(),
        name: "test-model-2".to_string(),
        provider: arey_core::model::ModelProvider::Test,
        settings: HashMap::new(),
    };

    models.insert("test-model-1".to_string(), test_model_1);
    models.insert("test-model-2".to_string(), test_model_2);

    let mut profiles = HashMap::new();
    profiles.insert(
        "test-profile".to_string(),
        ProfileConfig {
            temperature: 0.8,
            repeat_penalty: 1.1,
            top_k: 40,
            top_p: 0.9,
        },
    );

    let chat_mode = ModeConfig {
        model: ModelConfig {
            key: "test-model-1".to_string(),
            name: "test-model-1".to_string(),
            provider: arey_core::model::ModelProvider::Test,
            settings: HashMap::new(),
        },
        agent_name: "default".to_string(),
        profile: ProfileConfig::default(),
        profile_name: None,
    };

    let task_mode = ModeConfig {
        model: ModelConfig {
            key: "test-model-1".to_string(),
            name: "test-model-1".to_string(),
            provider: arey_core::model::ModelProvider::Test,
            settings: HashMap::new(),
        },
        agent_name: "default".to_string(),
        profile: ProfileConfig::default(),
        profile_name: None,
    };

    let mut agents = HashMap::new();
    agents.insert(
        "default".to_string(),
        Agent::new(
            "default".to_string(),
            "You are a helpful assistant.".to_string(),
            vec![],
            ProfileConfig::default(),
            Default::default(),
        ),
    );

    Ok(Config {
        models,
        profiles,
        agents,
        chat: chat_mode,
        task: task_mode,
        theme: "ansi".to_string(),
        tools: HashMap::new(),
    })
}

/// Creates a test config with a custom agent
pub fn create_test_config_with_custom_agent() -> Result<Config> {
    let mut config = create_test_config()?;

    // Add custom test agent
    config.agents.insert(
        "test-agent".to_string(),
        Agent::new(
            "test-agent".to_string(),
            "You are a test agent.".to_string(),
            vec![],
            ProfileConfig::default(),
            Default::default(),
        ),
    );

    // Update chat and task to use test-agent
    config.chat.agent_name = "test-agent".to_string();
    config.task.agent_name = "test-agent".to_string();

    Ok(config)
}

/// Creates a test config with a model that returns tool calls
pub fn create_test_config_with_tool_call_model() -> Result<Config> {
    let mut config = create_test_config()?;

    // Add tool-call model
    let tool_call_model = ModelConfig {
        key: "tool-call-model".to_string(),
        name: "tool-call-model".to_string(),
        provider: arey_core::model::ModelProvider::Test,
        settings: {
            let mut settings = HashMap::new();
            settings.insert(
                "response_mode".to_string(),
                serde_yaml::Value::String("tool_call".to_string()),
            );
            settings
        },
    };

    config
        .models
        .insert("tool-call-model".to_string(), tool_call_model);
    config.chat.model = ModelConfig {
        key: "tool-call-model".to_string(),
        name: "tool-call-model".to_string(),
        provider: arey_core::model::ModelProvider::Test,
        settings: {
            let mut settings = HashMap::new();
            settings.insert(
                "response_mode".to_string(),
                serde_yaml::Value::String("tool_call".to_string()),
            );
            settings
        },
    };
    config.task.model = ModelConfig {
        key: "tool-call-model".to_string(),
        name: "tool-call-model".to_string(),
        provider: arey_core::model::ModelProvider::Test,
        settings: {
            let mut settings = HashMap::new();
            settings.insert(
                "response_mode".to_string(),
                serde_yaml::Value::String("tool_call".to_string()),
            );
            settings
        },
    };

    Ok(config)
}

/// Creates a test config with a model that returns errors
pub fn create_test_config_with_error_model() -> Result<Config> {
    let mut config = create_test_config()?;

    // Add error model
    let error_model = ModelConfig {
        key: "error-model".to_string(),
        name: "error-model".to_string(),
        provider: arey_core::model::ModelProvider::Test,
        settings: {
            let mut settings = HashMap::new();
            settings.insert(
                "response_mode".to_string(),
                serde_yaml::Value::String("error".to_string()),
            );
            settings
        },
    };

    config.models.insert("error-model".to_string(), error_model);
    config.chat.model = ModelConfig {
        key: "error-model".to_string(),
        name: "error-model".to_string(),
        provider: arey_core::model::ModelProvider::Test,
        settings: {
            let mut settings = HashMap::new();
            settings.insert(
                "response_mode".to_string(),
                serde_yaml::Value::String("error".to_string()),
            );
            settings
        },
    };
    config.task.model = ModelConfig {
        key: "error-model".to_string(),
        name: "error-model".to_string(),
        provider: arey_core::model::ModelProvider::Test,
        settings: {
            let mut settings = HashMap::new();
            settings.insert(
                "response_mode".to_string(),
                serde_yaml::Value::String("error".to_string()),
            );
            settings
        },
    };

    Ok(config)
}

/// Mock tool for testing
#[derive(Debug)]
pub struct MockTool;

#[async_trait]
impl Tool for MockTool {
    fn name(&self) -> String {
        "mock_tool".to_string()
    }

    fn description(&self) -> String {
        "A mock tool for testing".to_string()
    }

    fn parameters(&self) -> Value {
        Value::Object(serde_json::Map::new())
    }

    async fn execute(&self, _args: &Value) -> Result<Value, ToolError> {
        Ok(Value::String("mock tool output".to_string()))
    }
}
