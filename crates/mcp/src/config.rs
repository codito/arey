use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct McpServerConfig {
    pub command: String,
    pub args: Vec<String>,
    #[serde(default)]
    pub env: HashMap<String, String>,
    #[serde(default = "default_enabled")]
    pub enabled: bool,
}

fn default_enabled() -> bool {
    true
}

#[derive(Debug, Deserialize, Serialize, Clone, Default)]
pub struct McpConfig {
    #[serde(default)]
    pub servers: HashMap<String, McpServerConfig>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct McpServerStatus {
    pub name: String,
    pub running: bool,
    pub enabled: bool,
    pub tool_count: usize,
}

impl Default for McpServerConfig {
    fn default() -> Self {
        Self {
            command: String::new(),
            args: Vec::new(),
            env: HashMap::new(),
            enabled: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use yare::parameterized;

    #[parameterized(
        no_enabled_field = { "command: npx\nargs: []", true },
        explicit_true = { "command: npx\nargs: []\nenabled: true", true },
        explicit_false = { "command: npx\nargs: []\nenabled: false", false },
    )]
    fn test_mcp_server_config_enabled(yaml: &str, expected: bool) {
        let config: McpServerConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.enabled, expected);
    }

    #[test]
    fn test_mcp_server_config_default() {
        let config = McpServerConfig::default();
        assert!(config.command.is_empty());
        assert!(config.args.is_empty());
        assert!(config.env.is_empty());
        assert!(config.enabled);
    }

    #[test]
    fn test_mcp_server_config_deserialization() {
        let yaml = r#"command: npx
args:
  - -y
  - "@modelcontextprotocol/server-filesystem"
  - /home/user
env:
  HOME: /home/user
enabled: true"#;

        let config: McpServerConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.command, "npx");
        assert_eq!(config.args.len(), 3);
        assert_eq!(config.args[0], "-y");
        assert_eq!(config.env.get("HOME"), Some(&"/home/user".to_string()));
        assert!(config.enabled);
    }

    #[test]
    fn test_mcp_config_default() {
        let config = McpConfig::default();
        assert!(config.servers.is_empty());
    }

    #[test]
    fn test_mcp_config_deserialization() {
        let yaml = r#"servers:
  filesystem:
    command: npx
    args:
      - -y
      - "@modelcontextprotocol/server-filesystem"
      - /home
    enabled: true
  memory:
    command: npx
    args:
      - -y
      - "@modelcontextprotocol/server-memory"
    enabled: false"#;

        let config: McpConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.servers.len(), 2);

        let fs = config.servers.get("filesystem").unwrap();
        assert_eq!(fs.command, "npx");
        assert!(fs.enabled);

        let memory = config.servers.get("memory").unwrap();
        assert!(!memory.enabled);
    }

    #[parameterized(
        empty = { "servers: {}", 0 },
        one_server = { "servers:\n  fs:\n    command: npx\n    args: []", 1 },
        three_servers = { "servers:\n  fs:\n    command: npx\n    args: []\n  memory:\n    command: npx\n    args: []\n  github:\n    command: npx\n    args: []", 3 },
    )]
    fn test_mcp_config_server_count(yaml: &str, expected_count: usize) {
        let config: McpConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.servers.len(), expected_count);
    }

    #[test]
    fn test_mcp_config_serialization() {
        let mut servers = HashMap::new();
        servers.insert(
            "test".to_string(),
            McpServerConfig {
                command: "npx".to_string(),
                args: vec!["-y".to_string(), "server".to_string()],
                env: HashMap::new(),
                enabled: true,
            },
        );

        let config = McpConfig { servers };
        let yaml = serde_yaml::to_string(&config).unwrap();
        assert!(yaml.contains("test:"));
        assert!(yaml.contains("command: npx"));
    }

    #[test]
    fn test_mcp_server_status_fields() {
        let status = McpServerStatus {
            name: "test-server".to_string(),
            running: true,
            enabled: true,
            tool_count: 5,
        };

        assert_eq!(status.name, "test-server");
        assert!(status.running);
        assert!(status.enabled);
        assert_eq!(status.tool_count, 5);
    }

    #[test]
    fn test_mcp_server_config_clone() {
        let config = McpServerConfig {
            command: "npx".to_string(),
            args: vec!["-y".to_string()],
            env: HashMap::from([("KEY".to_string(), "value".to_string())]),
            enabled: false,
        };

        let cloned = config.clone();
        assert_eq!(cloned.command, config.command);
        assert_eq!(cloned.args, config.args);
        assert_eq!(cloned.env.get("KEY"), Some(&"value".to_string()));
        assert_eq!(cloned.enabled, config.enabled);
    }

    #[test]
    fn test_mcp_server_config_with_env() {
        let yaml = r#"command: npx
args:
  - -y
  - server
env:
  HOME: /home/user
  PATH: /usr/bin
enabled: true"#;

        let config: McpServerConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.env.len(), 2);
        assert_eq!(config.env.get("HOME"), Some(&"/home/user".to_string()));
        assert_eq!(config.env.get("PATH"), Some(&"/usr/bin".to_string()));
    }

    #[test]
    fn test_mcp_config_multiple_servers() {
        let yaml = r#"servers:
  fs:
    command: npx
    args: [-y, fs-server]
    enabled: true
  memory:
    command: npx
    args: [-y, memory-server]
    enabled: false
  github:
    command: npx
    args: [-y, github-server]
    enabled: true"#;

        let config: McpConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.servers.len(), 3);

        assert!(config.servers.get("fs").unwrap().enabled);
        assert!(!config.servers.get("memory").unwrap().enabled);
        assert!(config.servers.get("github").unwrap().enabled);
    }

    #[test]
    fn test_mcp_config_empty_servers() {
        let yaml = "servers: {}";
        let config: McpConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(config.servers.is_empty());
    }

    #[test]
    fn test_mcp_server_status_equality() {
        let status1 = McpServerStatus {
            name: "test".to_string(),
            running: true,
            enabled: true,
            tool_count: 5,
        };

        let status2 = McpServerStatus {
            name: "test".to_string(),
            running: true,
            enabled: true,
            tool_count: 5,
        };

        assert_eq!(status1, status2);
    }

    #[test]
    fn test_mcp_server_status_display() {
        let status = McpServerStatus {
            name: "test-server".to_string(),
            running: true,
            enabled: false,
            tool_count: 3,
        };

        assert_eq!(status.name, "test-server");
        assert!(status.running);
        assert!(!status.enabled);
        assert_eq!(status.tool_count, 3);
    }
}
