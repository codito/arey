//! Test utilities for arey-core crate
//!
//! This module provides common test helpers and utilities to avoid code duplication
//! and ensure consistent test setup across the codebase.

use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use tempfile::Builder;

/// Creates a temporary config file with the given content.
/// Uses tempfile::Builder to ensure unique directories for parallel tests.
///
/// # Arguments
/// * `content` - The YAML content to write to the config file
///
/// # Returns
/// * `PathBuf` - Path to the created temporary config file
///
/// # Panics
/// Panics if temp directory creation or file writing fails.
pub fn create_temp_config(content: &str) -> PathBuf {
    let temp_dir = Builder::new()
        .prefix("arey-test")
        .rand_bytes(8)
        .tempdir()
        .unwrap();
    let config_path = temp_dir.path().join("arey.yml");
    std::fs::create_dir_all(temp_dir.path()).unwrap();
    File::create(&config_path)
        .unwrap()
        .write_all(content.as_bytes())
        .unwrap();
    // Keep the temp directory alive by leaking it (this is just for tests)
    let _ = Box::leak(Box::new(temp_dir));
    config_path
}

/// Creates a temporary directory for agent files.
/// Uses tempfile::Builder to ensure unique directories for parallel tests.
///
/// # Returns
/// * `PathBuf` - Path to the created temporary agents directory
///
/// # Panics
/// Panics if temp directory creation fails.
pub fn create_temp_agents_dir() -> PathBuf {
    let temp_dir = Builder::new()
        .prefix("arey-test-agents")
        .rand_bytes(8)
        .tempdir()
        .unwrap();
    let agents_dir = temp_dir.path().join("agents");
    std::fs::create_dir_all(&agents_dir).unwrap();
    // Keep the temp directory alive by leaking it (this is just for tests)
    let _ = Box::leak(Box::new(temp_dir));
    agents_dir
}

/// Creates a test agent file with the given content.
///
/// # Arguments
/// * `agents_dir` - Path to the agents directory
/// * `agent_name` - Name of the agent (will become the filename)
/// * `content` - YAML content for the agent configuration
///
/// # Panics
/// Panics if file writing fails.
pub fn create_test_agent(agents_dir: &std::path::Path, agent_name: &str, content: &str) {
    let agent_path = agents_dir.join(format!("{}.yml", agent_name));
    std::fs::write(&agent_path, content).unwrap();
}

/// Default test model configuration for testing.
pub fn dummy_model_config(name: &str) -> crate::model::ModelConfig {
    crate::model::ModelConfig {
        name: name.to_string(),
        key: "".to_string(), // Default empty
        provider: crate::model::ModelProvider::Gguf,
        settings: std::collections::HashMap::from([(
            "n_ctx".to_string(),
            serde_yaml::Value::Number(4096.into()),
        )]),
    }
}
