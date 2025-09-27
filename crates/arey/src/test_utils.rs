//! Test utilities for arey crate
//!
//! This module provides common test helpers for creating temporary config files
//! and agent setups to avoid code duplication and ensure consistent test setup.

use std::fs;
use tempfile::{Builder, NamedTempFile};

/// Creates a temporary config file with a test OpenAI-compatible server.
/// Uses tempfile::Builder to ensure unique directories for parallel tests.
///
/// # Arguments
/// * `server_uri` - The base URI for the test server
///
/// # Returns
/// * `NamedTempFile` - Temporary config file with test configuration
///
/// # Panics
/// Panics if temp directory creation or file writing fails.
pub fn create_temp_config_file(server_uri: &str) -> NamedTempFile {
    let temp_dir = Builder::new()
        .prefix("arey-test")
        .rand_bytes(8)
        .tempdir()
        .unwrap();
    let config_path = temp_dir.path().join("arey.yml");
    let config_content = format!(
        r#"
models:
  test-model:
    name: test-model
    provider: openai
    base_url: "{server_uri}"
    api_key: "MOCK_OPENAI_API_KEY"
profiles: {{}}
chat:
    model: test-model
task:
  model: test-model
"#,
    );
    fs::write(&config_path, config_content).unwrap();

    // Create a NamedTempFile that references our config file
    let file = NamedTempFile::new_in(temp_dir.path()).unwrap();
    fs::copy(&config_path, file.path()).unwrap();

    // Keep the temp directory alive by leaking it (this is just for tests)
    let _ = Box::leak(Box::new(temp_dir));
    file
}

/// Creates a temporary config file with test agent configuration.
/// Uses tempfile::Builder to ensure unique directories for parallel tests.
///
/// # Arguments
/// * `server_uri` - The base URI for the test server
///
/// # Returns
/// * `NamedTempFile` - Temporary config file with test agent configuration
///
/// # Panics
/// Panics if temp directory creation or file writing fails.
pub fn create_temp_config_file_with_agent(server_uri: &str) -> NamedTempFile {
    let temp_dir = Builder::new()
        .prefix("arey-test")
        .rand_bytes(8)
        .tempdir()
        .unwrap();
    let config_dir = temp_dir.path();

    // Create agents directory in the same directory as the config file
    let agents_dir = config_dir.join("agents");
    fs::create_dir_all(&agents_dir).unwrap();

    // Create the test agent file
    let test_agent_path = agents_dir.join("test-agent.yml");
    let test_agent_content = r#"
name: "test-agent"
prompt: "You are a test agent."
tools: [ "mock_tool" ]
profile:
  temperature: 0.1
"#;
    fs::write(&test_agent_path, test_agent_content).unwrap();

    // Create the main config file
    let config_path = config_dir.join("arey.yml");
    let config_content = format!(
        r#"
models:
  test-model:
    provider: openai
    base_url: "{server_uri}"
    api_key: "MOCK_OPENAI_API_KEY"
chat:
  model: test-model
  agent: "test-agent"
task:
  model: test-model
  agent: "test-agent"
profiles:
  test-profile:
    temperature: 0.5
    top_p: 0.9
    repeat_penalty: 1.1
    top_k: 40
"#,
    );
    fs::write(&config_path, config_content).unwrap();

    // Create a NamedTempFile that references our config file
    let file = NamedTempFile::new_in(temp_dir.path()).unwrap();
    fs::copy(&config_path, file.path()).unwrap();

    // Keep the temp directory alive by leaking it (this is just for tests)
    let _ = Box::leak(Box::new(temp_dir));
    file
}
