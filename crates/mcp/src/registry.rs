use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use tokio::sync::RwLock;

use arey_core::tools::Tool;

use crate::client::McpClient;
pub use crate::config::{McpConfig, McpServerConfig, McpServerStatus};

pub struct McpRegistry {
    servers: HashMap<String, McpServerState>,
    enabled: RwLock<HashMap<String, bool>>,
}

struct McpServerState {
    client: McpClient,
    #[allow(dead_code)]
    config: McpServerConfig,
}

impl McpRegistry {
    pub fn new() -> Self {
        Self {
            servers: HashMap::new(),
            enabled: RwLock::new(HashMap::new()),
        }
    }

    pub async fn add_server(&mut self, name: String, config: &McpServerConfig) -> Result<()> {
        let client = McpClient::new(name.clone(), config).await?;

        self.servers.insert(
            name.clone(),
            McpServerState {
                client,
                config: config.clone(),
            },
        );

        if config.enabled {
            self.enable(&name).await?;
        }

        Ok(())
    }

    pub async fn remove_server(&mut self, name: &str) -> Result<()> {
        self.disable(name).await?;
        self.servers.remove(name);
        Ok(())
    }

    pub async fn enable(&mut self, name: &str) -> Result<()> {
        if !self.servers.contains_key(name) {
            anyhow::bail!("MCP server '{}' not found", name);
        }

        let mut enabled = self.enabled.write().await;
        enabled.insert(name.to_string(), true);
        Ok(())
    }

    pub async fn disable(&mut self, name: &str) -> Result<()> {
        let mut enabled = self.enabled.write().await;
        enabled.insert(name.to_string(), false);
        Ok(())
    }

    pub async fn list(&self) -> Vec<McpServerStatus> {
        let enabled = self.enabled.read().await;

        self.servers
            .iter()
            .map(|(name, state)| {
                let tool_count = if enabled.get(name) == Some(&true) {
                    state.client.tools().len()
                } else {
                    0
                };

                McpServerStatus {
                    name: name.clone(),
                    running: true,
                    enabled: *enabled.get(name).unwrap_or(&false),
                    tool_count,
                }
            })
            .collect()
    }

    pub fn get_tools(&self) -> Vec<Arc<dyn Tool>> {
        self.get_all_tools()
    }

    pub fn get_all_tools(&self) -> Vec<Arc<dyn Tool>> {
        self.servers
            .values()
            .flat_map(|state| state.client.tools())
            .collect()
    }

    pub async fn get_enabled_tools(&self) -> Vec<Arc<dyn Tool>> {
        let enabled = self.enabled.read().await;

        self.servers
            .iter()
            .filter(|(name, _)| enabled.get(*name) == Some(&true))
            .flat_map(|(_, state)| state.client.tools())
            .collect()
    }

    pub fn server_names(&self) -> Vec<String> {
        self.servers.keys().cloned().collect()
    }

    pub async fn is_enabled(&self, name: &str) -> bool {
        let enabled = self.enabled.read().await;
        *enabled.get(name).unwrap_or(&false)
    }
}

impl Default for McpRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl McpRegistry {
    /// Creates McpRegistry from config and starts enabled MCP servers.
    /// Returns None if no servers are configured or enabled.
    pub async fn from_config(config: &arey_core::config::Config) -> Result<Option<Self>> {
        let mcp_value = &config.mcp;

        let is_mcp_empty = match mcp_value {
            serde_yaml::Value::Null => true,
            serde_yaml::Value::Sequence(seq) => seq.is_empty(),
            _ => false,
        };

        if is_mcp_empty {
            return Ok(None);
        }

        let mcp_config: McpConfig = match serde_yaml::from_value(mcp_value.clone()) {
            Ok(cfg) => cfg,
            Err(_) => return Ok(None),
        };

        if mcp_config.servers.is_empty() {
            return Ok(None);
        }

        let mut registry = Self::new();
        let mut started_any = false;

        for (name, server_config) in mcp_config.servers {
            if server_config.enabled {
                started_any = true;
                match registry.add_server(name.clone(), &server_config).await {
                    Ok(_) => {
                        tracing::info!("Started MCP server: {}", name);
                    }
                    Err(e) => {
                        tracing::warn!("Failed to start MCP server '{}': {}", name, e);
                    }
                }
            }
        }

        if started_any {
            Ok(Some(registry))
        } else {
            Ok(None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mock McpClient for testing
    struct MockMcpClient {
        name: String,
        tools: Vec<Arc<dyn Tool>>,
    }

    impl MockMcpClient {
        #[allow(dead_code)]
        fn new(name: &str, tool_count: usize) -> Self {
            let tools: Vec<Arc<dyn Tool>> = (0..tool_count)
                .map(|i| {
                    Arc::new(MockTool {
                        name: format!("{}_tool_{}", name, i),
                        description: format!("Mock tool {} from {}", i, name),
                    }) as Arc<dyn Tool>
                })
                .collect();

            Self {
                name: name.to_string(),
                tools,
            }
        }

        #[allow(dead_code)]
        fn tools(&self) -> Vec<Arc<dyn Tool>> {
            self.tools.clone()
        }
    }

    #[allow(dead_code)]
    struct MockTool {
        name: String,
        description: String,
    }

    #[async_trait::async_trait]
    impl Tool for MockTool {
        fn name(&self) -> String {
            self.name.clone()
        }

        fn description(&self) -> String {
            self.description.clone()
        }

        fn parameters(&self) -> serde_json::Value {
            serde_json::json!({})
        }

        async fn execute(
            &self,
            _arguments: &serde_json::Value,
        ) -> Result<serde_json::Value, arey_core::tools::ToolError> {
            Ok(serde_json::json!({ "mock": true }))
        }
    }

    // Note: Since McpClient::new requires spawning a real process,
    // we test the manager methods that don't require a running server
    // by testing the logic that doesn't depend on actual MCP connections.
    // Full integration tests would require mock MCP servers.

    #[tokio::test]
    async fn test_manager_new() {
        let manager = McpRegistry::new();
        let names = manager.server_names();
        assert!(names.is_empty());
    }

    #[tokio::test]
    async fn test_manager_disable_nonexistent() {
        let mut manager = McpRegistry::new();
        let result = manager.disable("nonexistent").await;
        // Should succeed but do nothing
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_manager_disable_enables_then_disables() {
        let mut manager = McpRegistry::new();

        // Enable on non-existent should error
        let result = manager.enable("nonexistent").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_manager_list_empty() {
        let manager = McpRegistry::new();
        let list = manager.list().await;
        assert!(list.is_empty());
    }

    #[tokio::test]
    async fn test_manager_get_tools_empty() {
        let manager = McpRegistry::new();
        let tools = manager.get_tools();
        assert!(tools.is_empty());
    }

    #[tokio::test]
    async fn test_manager_get_enabled_tools_empty() {
        let manager = McpRegistry::new();
        let tools = manager.get_enabled_tools().await;
        assert!(tools.is_empty());
    }

    #[tokio::test]
    async fn test_manager_is_enabled_nonexistent() {
        let manager = McpRegistry::new();
        let enabled = manager.is_enabled("nonexistent").await;
        assert!(!enabled);
    }

    #[tokio::test]
    async fn test_manager_remove_nonexistent() {
        let mut manager = McpRegistry::new();
        let result = manager.remove_server("nonexistent").await;
        // Should succeed (disable does nothing, remove returns None)
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_manager_disable_twice() {
        let mut manager = McpRegistry::new();

        // Disable twice should be idempotent
        let result1 = manager.disable("test").await;
        assert!(result1.is_ok());

        let result2 = manager.disable("test").await;
        assert!(result2.is_ok());

        // Should still be disabled
        let enabled = manager.is_enabled("test").await;
        assert!(!enabled);
    }

    #[tokio::test]
    async fn test_manager_enable_twice() {
        let mut manager = McpRegistry::new();

        // Enable twice should be idempotent
        // But enable on nonexistent should error
        let result1 = manager.enable("test").await;
        assert!(result1.is_err()); // Server doesn't exist

        let result2 = manager.enable("test").await;
        assert!(result2.is_err()); // Still doesn't exist
    }

    #[tokio::test]
    async fn test_manager_server_names_after_disable() {
        let mut manager = McpRegistry::new();
        let names_before = manager.server_names();
        assert!(names_before.is_empty());

        manager.disable("any").await.unwrap();

        let names_after = manager.server_names();
        assert!(names_after.is_empty()); // No servers added
    }
}

#[cfg(test)]
mod config_tests {
    use super::*;
    use arey_core::agent::Agent;
    use arey_core::config::{Config, ModeConfig, ProfileConfig};
    use arey_core::model::ModelConfig;
    use serde_yaml::Value;
    use std::collections::HashMap;
    use yare::parameterized;

    fn create_test_config_with_mcp(mcp_value: Value) -> Config {
        let mut models = HashMap::new();
        models.insert(
            "test-model".to_string(),
            ModelConfig {
                name: "test-model".to_string(),
                key: "test-model".to_string(),
                ..Default::default()
            },
        );

        let mut profiles = HashMap::new();
        profiles.insert("default".to_string(), ProfileConfig::default());

        let mut agents = HashMap::new();
        agents.insert(
            "default".to_string(),
            Agent::new(
                "default".to_string(),
                "You are helpful.".to_string(),
                vec![],
                ProfileConfig::default(),
                Default::default(),
            ),
        );

        let chat_mode = ModeConfig {
            model: models.get("test-model").cloned().unwrap(),
            agent_name: "default".to_string(),
            profile: ProfileConfig::default(),
            profile_name: Some("default".to_string()),
        };

        Config {
            models,
            profiles,
            agents,
            chat: chat_mode.clone(),
            task: chat_mode,
            theme: "light".to_string(),
            tools: HashMap::new(),
            mcp: mcp_value,
        }
    }

    #[tokio::test]
    async fn test_mcp_registry_from_config_null() {
        let config = create_test_config_with_mcp(Value::Null);
        let result = McpRegistry::from_config(&config).await;
        assert!(result.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_mcp_registry_from_config_empty_servers() {
        let config = create_test_config_with_mcp(serde_yaml::from_str("servers: {}").unwrap());
        let result = McpRegistry::from_config(&config).await;
        assert!(result.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_mcp_registry_from_config_invalid_yaml() {
        let config = create_test_config_with_mcp(serde_yaml::from_str("servers: []").unwrap());
        let result = McpRegistry::from_config(&config).await;
        assert!(result.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_mcp_registry_from_config_sequence() {
        let config =
            create_test_config_with_mcp(serde_yaml::from_str("- server1\n- server2").unwrap());
        let result = McpRegistry::from_config(&config).await;
        assert!(result.unwrap().is_none());
    }

    #[test]
    fn test_mcp_registry_from_config_disabled_server() {
        let yaml = r#"
servers:
  test-server:
    command: npx
    args: [-y, test-server]
    enabled: false
"#;
        let config = create_test_config_with_mcp(serde_yaml::from_str(yaml).unwrap());
        let result = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(McpRegistry::from_config(&config));
        let mgr = result.unwrap();
        if let Some(m) = mgr {
            assert_eq!(m.server_names().len(), 0);
        }
    }

    #[parameterized(
        null = { serde_yaml::Value::Null },
        empty_servers = { serde_yaml::from_str("servers: {}").unwrap() },
        invalid_servers = { serde_yaml::from_str("servers: []").unwrap() },
        sequence = { serde_yaml::from_str("- server1\n- server2").unwrap() },
    )]
    #[test_macro(tokio::test)]
    async fn test_mcp_registry_from_config_invalid(mcp_value: serde_yaml::Value) {
        let config = create_test_config_with_mcp(mcp_value);
        let result = McpRegistry::from_config(&config).await;
        assert!(result.unwrap().is_none());
    }

    #[test]
    fn test_mcp_config_parsing_enabled_server() {
        let yaml = r#"
servers:
  fs:
    command: npx
    args:
      - -y
      - "@modelcontextprotocol/server-filesystem"
      - /tmp
    enabled: true
  memory:
    command: npx
    args:
      - -y
      - "@modelcontextprotocol/server-memory"
    enabled: false
"#;
        let mcp_config: McpConfig = serde_yaml::from_str(yaml).unwrap();

        assert_eq!(mcp_config.servers.len(), 2);

        let fs = mcp_config.servers.get("fs").unwrap();
        assert_eq!(fs.command, "npx");
        assert!(fs.enabled);

        let memory = mcp_config.servers.get("memory").unwrap();
        assert!(!memory.enabled);
    }

    #[test]
    fn test_mcp_server_config_defaults() {
        let yaml = r#"
command: npx
args: [-y, server]
"#;
        let config: McpServerConfig = serde_yaml::from_str(yaml).unwrap();

        assert_eq!(config.command, "npx");
        assert_eq!(config.args, vec!["-y", "server"]);
        assert!(config.env.is_empty());
        assert!(config.enabled); // default is true
    }

    #[test]
    fn test_mcp_server_config_with_env() {
        let yaml = r#"
command: npx
args: [-y, server]
env:
  HOME: /home/user
  DEBUG: "true"
enabled: true
"#;
        let config: McpServerConfig = serde_yaml::from_str(yaml).unwrap();

        assert_eq!(config.env.get("HOME"), Some(&"/home/user".to_string()));
        assert_eq!(config.env.get("DEBUG"), Some(&"true".to_string()));
    }

    #[test]
    fn test_mcp_server_config_explicit_enabled() {
        let yaml = r#"
command: npx
args: []
enabled: true
"#;
        let config: McpServerConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(config.enabled);
    }

    #[test]
    fn test_mcp_server_config_explicit_disabled() {
        let yaml = r#"
command: npx
args: []
enabled: false
"#;
        let config: McpServerConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(!config.enabled);
    }
}
