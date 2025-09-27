use std::{
    collections::HashMap,
    fs::{self, File},
    io::Write,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::instrument;

use crate::{
    agent::{Agent, AgentMetadata, AgentSource},
    assets::{get_config_dir, get_default_config},
    model::ModelConfig,
};

#[derive(Error, Debug)]
pub enum AreyConfigError {
    #[error("File system error: {0}")]
    IO(#[from] std::io::Error),
    #[error("YAML parsing error: {0}")]
    YAMLError(#[from] serde_yaml::Error),
    #[error("Configuration error: {0}")]
    Config(String),
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
#[serde(default)]
pub struct ProfileConfig {
    pub temperature: f32,
    pub repeat_penalty: f32,
    pub top_k: i32,
    pub top_p: f32,
}

impl Default for ProfileConfig {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            repeat_penalty: 1.176,
            top_k: 40,
            top_p: 0.1,
        }
    }
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ModeConfig {
    pub model: ModelConfig,
    pub agent_name: String,
    #[serde(default)]
    pub profile: ProfileConfig,
    #[serde(skip)]
    pub profile_name: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Config {
    pub models: HashMap<String, ModelConfig>,
    pub profiles: HashMap<String, ProfileConfig>,
    #[serde(default)]
    pub agents: HashMap<String, Agent>,
    pub chat: ModeConfig,
    pub task: ModeConfig,
    #[serde(default = "default_theme")]
    pub theme: String,
    #[serde(default)]
    pub tools: HashMap<String, serde_yaml::Value>,
}

fn default_theme() -> String {
    "light".to_string()
}

#[derive(Deserialize, Debug)]
#[serde(untagged)]
enum StringOrObject<T> {
    String(String),
    Object(T),
}

#[derive(Deserialize, Debug)]
struct RawModeConfig {
    model: StringOrObject<ModelConfig>,
    #[serde(default)]
    agent: Option<String>,
    #[serde(default)]
    profile: Option<StringOrObject<ProfileConfig>>,
}

#[derive(Deserialize, Debug)]
struct AgentFileConfig {
    name: String,
    prompt: String,
    #[serde(default)]
    tools: Vec<String>,
    #[serde(default)]
    profile: Option<ProfileConfig>,
}

fn create_default_agent(agents_dir: &Path) -> Result<(), AreyConfigError> {
    if !agents_dir.exists() {
        fs::create_dir_all(agents_dir)?;
    }

    // Create default agent if it doesn't exist, using the content from data file
    let default_agent_content = include_str!("../data/agents/default.yml");

    let default_path = agents_dir.join("default.yml");
    if !default_path.exists() {
        fs::write(&default_path, default_agent_content)?;
    }

    Ok(())
}

fn load_default_agent_from_data() -> Result<Agent, AreyConfigError> {
    let default_content = include_str!("../data/agents/default.yml");
    let agent_file: AgentFileConfig =
        serde_yaml::from_str(default_content).map_err(AreyConfigError::YAMLError)?;

    let metadata = AgentMetadata {
        source: AgentSource::BuiltIn,
        file_path: None,
    };

    let agent = Agent::new(
        agent_file.name,
        agent_file.prompt,
        agent_file.tools,
        agent_file.profile.unwrap_or_default(),
        metadata,
    );

    Ok(agent)
}

fn get_built_in_agents() -> HashMap<String, Agent> {
    let mut agents = HashMap::new();

    // Load default agent from data file
    if let Ok(agent) = load_default_agent_from_data() {
        agents.insert("default".to_string(), agent);
    }

    agents
}

#[instrument]
fn load_agents_from_directory(
    agents_dir: &Path,
) -> Result<HashMap<String, Agent>, AreyConfigError> {
    let mut agents = HashMap::new();

    if !agents_dir.exists() {
        return Ok(agents);
    }

    let entries = fs::read_dir(agents_dir).map_err(AreyConfigError::IO)?;

    for entry in entries {
        let entry = entry.map_err(AreyConfigError::IO)?;
        let path = entry.path();

        if path.extension().and_then(|s| s.to_str()) == Some("yml")
            || path.extension().and_then(|s| s.to_str()) == Some("yaml")
        {
            let content = fs::read_to_string(&path).map_err(AreyConfigError::IO)?;

            let agent_config: AgentFileConfig =
                serde_yaml::from_str(&content).map_err(AreyConfigError::YAMLError)?;

            let metadata = AgentMetadata {
                source: AgentSource::UserFile(path.clone()),
                file_path: Some(path),
            };

            let agent = Agent::new(
                agent_config.name.clone(),
                agent_config.prompt,
                agent_config.tools,
                agent_config.profile.unwrap_or_default(),
                metadata,
            );

            agents.insert(agent_config.name, agent);
        }
    }

    Ok(agents)
}

fn merge_agents(
    built_in: HashMap<String, Agent>,
    user_agents: HashMap<String, Agent>,
) -> HashMap<String, Agent> {
    let mut merged_agents = HashMap::new();

    // First add built-in agents
    for (name, agent) in built_in {
        merged_agents.insert(name, agent);
    }

    // User agents override built-in agents
    for (name, agent) in user_agents {
        merged_agents.insert(name, agent);
    }

    merged_agents
}

#[derive(Deserialize, Debug)]
struct RawConfig {
    models: HashMap<String, ModelConfig>,
    profiles: HashMap<String, ProfileConfig>,
    chat: RawModeConfig,
    task: RawModeConfig,
    theme: Option<String>,
    #[serde(default)]
    tools: HashMap<String, serde_yaml::Value>,
}

impl RawConfig {
    #[instrument]
    fn to_config(&self, config_dir: &Path) -> Result<Config, AreyConfigError> {
        let mut models_with_names = HashMap::new();
        for (k, v) in &self.models {
            // Update model name if not set
            let model_name = if v.name.is_empty() {
                k.clone()
            } else {
                v.name.clone()
            };
            let model = ModelConfig {
                name: model_name,
                key: k.clone(),
                ..v.clone()
            };
            models_with_names.insert(k.clone(), model);
        }

        let resolve_model =
            |model_entry: &StringOrObject<ModelConfig>| -> Result<ModelConfig, AreyConfigError> {
                match model_entry {
                    StringOrObject::String(s) => models_with_names
                        .get(s)
                        .cloned()
                        .ok_or_else(|| AreyConfigError::Config(format!("Model '{s}' not found"))),
                    StringOrObject::Object(m) => {
                        let mut resolved = m.clone();
                        if resolved.key.is_empty() {
                            resolved.key = resolved.name.clone();
                        }
                        Ok(resolved)
                    }
                }
            };

        let resolve_profile = |profile_entry: &Option<StringOrObject<ProfileConfig>>| -> Result<ProfileConfig, AreyConfigError> {
            match profile_entry {
                Some(StringOrObject::String(s)) => self.profiles
                    .get(s)
                    .cloned()
                    .ok_or_else(|| AreyConfigError::Config(format!("Profile '{s}' not found"))),
                Some(StringOrObject::Object(p)) => Ok(p.clone()),
                None => Ok(ProfileConfig::default()),
            }
        };

        let get_profile_name =
            |profile_entry: &Option<StringOrObject<ProfileConfig>>| -> Option<String> {
                match profile_entry {
                    Some(StringOrObject::String(s)) => Some(s.clone()),
                    _ => None,
                }
            };

        // Load agents from multiple sources
        let agents_dir = config_dir.join("agents");

        // Create default agent if directory doesn't exist
        if let Err(e) = create_default_agent(&agents_dir) {
            tracing::warn!("Failed to create default agent: {}", e);
        }

        let built_in_agents = get_built_in_agents();
        let user_agents = load_agents_from_directory(&agents_dir)?;

        let mut merged_agents = merge_agents(built_in_agents, user_agents);

        const DEFAULT_AGENT_NAME: &str = "default";
        if !merged_agents.contains_key(DEFAULT_AGENT_NAME) {
            let default_metadata = AgentMetadata {
                source: AgentSource::BuiltIn,
                file_path: None,
            };
            let default_agent = Agent::new(
                DEFAULT_AGENT_NAME.to_string(),
                "You are a helpful assistant.".to_string(),
                Vec::new(),
                ProfileConfig::default(),
                default_metadata,
            );
            merged_agents.insert(DEFAULT_AGENT_NAME.to_string(), default_agent);
        }

        let chat_model = resolve_model(&self.chat.model)?;
        let chat_agent_name = self
            .chat
            .agent
            .clone()
            .unwrap_or_else(|| DEFAULT_AGENT_NAME.to_string());
        if !merged_agents.contains_key(&chat_agent_name) {
            return Err(AreyConfigError::Config(format!(
                "Agent '{chat_agent_name}' specified in chat mode not found"
            )));
        }

        let task_model = resolve_model(&self.task.model)?;
        let task_agent_name = self
            .task
            .agent
            .clone()
            .unwrap_or_else(|| DEFAULT_AGENT_NAME.to_string());
        if !merged_agents.contains_key(&task_agent_name) {
            return Err(AreyConfigError::Config(format!(
                "Agent '{task_agent_name}' specified in task mode not found"
            )));
        }

        let task_profile = resolve_profile(&self.task.profile)?;
        let task_profile_name = get_profile_name(&self.task.profile);

        Ok(Config {
            models: models_with_names,
            profiles: self.profiles.clone(),
            agents: merged_agents,
            chat: ModeConfig {
                model: chat_model,
                agent_name: chat_agent_name,
                profile: ProfileConfig::default(),
                profile_name: None,
            },
            task: ModeConfig {
                model: task_model,
                agent_name: task_agent_name,
                profile: task_profile,
                profile_name: task_profile_name,
            },
            theme: self.theme.clone().unwrap_or_else(default_theme),
            tools: self.tools.clone(),
        })
    }
}

#[instrument(skip(config_path))]
pub fn create_or_get_config_file(
    config_path: Option<PathBuf>,
) -> Result<(bool, PathBuf), AreyConfigError> {
    let actual_path = config_path.unwrap_or_else(|| {
        let config_dir = get_config_dir();
        config_dir.join("arey.yml")
    });

    let parent_dir = actual_path.parent().ok_or_else(|| {
        AreyConfigError::IO(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "Config path has no parent directory",
        ))
    })?;

    if !parent_dir.exists() {
        fs::create_dir_all(parent_dir)?;
    }

    if actual_path.exists() {
        Ok((true, actual_path))
    } else {
        File::create(&actual_path)?.write_all(get_default_config().as_bytes())?;
        Ok((false, actual_path))
    }
}

#[instrument(skip(config_path))]
pub fn get_config(config_path: Option<PathBuf>) -> Result<Config, AreyConfigError> {
    let (_, config_file) = create_or_get_config_file(config_path)?;
    let config_dir = config_file.parent().ok_or_else(|| {
        AreyConfigError::IO(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "Config path has no parent directory",
        ))
    })?;

    let content = fs::read_to_string(&config_file)?;
    let mut val: serde_yaml::Value = serde_yaml::from_str(&content)?;
    val.apply_merge()?; // Apply merge keys. Model configs can use it.
    let raw: RawConfig = serde_yaml::from_value(val)?;
    raw.to_config(config_dir)
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, fs, path::PathBuf};

    use tempfile::tempdir;

    use super::*;
    use crate::agent::AgentConfig;
    use crate::test_utils::{create_temp_config, dummy_model_config};

    // Dummy config content for tests
    const DUMMY_CONFIG_CONTENT: &str = r#"
models:
  dummy-7b:
    name: dummy-7b
    type: gguf
    n_ctx: 4096
    path: /path/to/dummy_model.gguf
  dummy-13b:
    name: dummy-13b
    type: gguf
    n_ctx: 8192
    path: /path/to/another_dummy.gguf
profiles:
  default:
    temperature: 0.7
    repeat_penalty: 1.176
    top_k: 40
    top_p: 0.1
  creative:
    temperature: 0.9
    repeat_penalty: 1.1
    top_k: 50
    top_p: 0.9
  concise:
    temperature: 0.8
    repeat_penalty: 1.2
    top_k: 30
    top_p: 0.05
chat:
  model: dummy-7b
task:
  model: dummy-13b
  profile: concise
theme: dark
"#;

    #[test]
    fn test_profile_config_default() {
        let default_profile = ProfileConfig::default();
        assert_eq!(default_profile.temperature, 0.7);
        assert_eq!(default_profile.repeat_penalty, 1.176);
        assert_eq!(default_profile.top_k, 40);
        assert_eq!(default_profile.top_p, 0.1);
    }

    #[test]
    fn test_raw_config_to_config_valid() {
        let mut models = HashMap::new();
        models.insert("dummy-7b".to_string(), dummy_model_config("dummy-7b"));
        models.insert("dummy-13b".to_string(), dummy_model_config("dummy-13b"));

        let mut profiles = HashMap::new();
        profiles.insert("default".to_string(), ProfileConfig::default());
        profiles.insert(
            "creative".to_string(),
            ProfileConfig {
                temperature: 0.9,
                ..Default::default()
            },
        );
        profiles.insert(
            "concise".to_string(),
            ProfileConfig {
                temperature: 0.8,
                ..Default::default()
            },
        );

        let raw_config = RawConfig {
            models: models.clone(),
            profiles: profiles.clone(),
            chat: RawModeConfig {
                model: StringOrObject::String("dummy-7b".to_string()),
                agent: None,
                profile: None,
            },
            task: RawModeConfig {
                model: StringOrObject::String("dummy-13b".to_string()),
                agent: None,
                profile: Some(StringOrObject::String("concise".to_string())),
            },
            theme: Some("dark".to_string()),
            tools: HashMap::new(),
        };

        let temp_dir = tempfile::TempDir::new().unwrap();
        let config = raw_config.to_config(temp_dir.path()).unwrap();

        assert_eq!(config.models.len(), 2);
        assert_eq!(config.profiles.len(), 3);
        assert_eq!(config.chat.model.name, "dummy-7b");
        assert_eq!(config.chat.agent_name, "default");
        let chat_agent = config.agents.get(&config.chat.agent_name).unwrap();
        assert_eq!(chat_agent.profile.temperature, 0.7);
        assert_eq!(config.task.model.name, "dummy-13b");
        assert_eq!(config.task.profile.temperature, 0.8);
        assert_eq!(config.theme, "dark");
    }

    #[test]
    fn test_raw_config_to_config_missing_model_reference() {
        let mut models = HashMap::new();
        models.insert("dummy-7b".to_string(), dummy_model_config("dummy-7b"));

        let raw_config = RawConfig {
            models,
            profiles: HashMap::new(),
            chat: RawModeConfig {
                model: StringOrObject::String("non-existent-model".to_string()),
                agent: None,
                profile: None,
            },
            task: RawModeConfig {
                model: StringOrObject::String("dummy-7b".to_string()),
                agent: None,
                profile: None,
            },
            theme: None,
            tools: HashMap::new(),
        };

        let err = raw_config.to_config(&PathBuf::from("/tmp")).unwrap_err();
        assert!(
            matches!(err, AreyConfigError::Config(msg) if msg.contains("Model 'non-existent-model' not found"))
        );
    }

    #[test]
    fn test_raw_config_to_config_missing_profile_reference() {
        let mut models = HashMap::new();
        models.insert("dummy-7b".to_string(), dummy_model_config("dummy-7b"));

        let raw_config = RawConfig {
            models,
            profiles: HashMap::new(),
            chat: RawModeConfig {
                model: StringOrObject::String("dummy-7b".to_string()),
                agent: None,
                profile: None,
            },
            task: RawModeConfig {
                model: StringOrObject::String("dummy-7b".to_string()),
                agent: None,
                profile: Some(StringOrObject::String("non-existent-profile".to_string())),
            },
            theme: None,
            tools: HashMap::new(),
        };

        let err = raw_config.to_config(&PathBuf::from("/tmp")).unwrap_err();
        assert!(
            matches!(err, AreyConfigError::Config(msg) if msg.contains("Profile 'non-existent-profile' not found"))
        );
    }

    #[test]
    fn test_raw_config_to_config_inline_model_and_profile() {
        let raw_config = RawConfig {
            models: HashMap::new(),   // No named models
            profiles: HashMap::new(), // No named profiles
            chat: RawModeConfig {
                model: StringOrObject::Object(dummy_model_config("inline-chat-model")),
                agent: None,
                profile: None,
            },
            task: RawModeConfig {
                model: StringOrObject::Object(dummy_model_config("inline-task-model")),
                agent: None,
                profile: None, // Should use default profile
            },
            theme: Some("light".to_string()),
            tools: HashMap::new(),
        };

        let temp_dir = tempfile::TempDir::new().unwrap();
        let config = raw_config.to_config(temp_dir.path()).unwrap();

        assert_eq!(config.chat.model.name, "inline-chat-model");
        assert_eq!(config.chat.model.key, "inline-chat-model"); // Fallback set
        let chat_agent = config.agents.get(&config.chat.agent_name).unwrap();
        assert_eq!(chat_agent.profile.temperature, 0.7);
        assert_eq!(config.task.model.name, "inline-task-model");
        assert_eq!(config.task.model.key, "inline-task-model"); // Fallback set
        assert_eq!(config.task.profile.temperature, 0.7); // Default temperature
        assert_eq!(config.theme, "light");
    }

    #[test]
    fn test_model_key_separate_from_name() {
        let yaml = r#"
models:
  my-key:
    name: custom-name  # Explicit name
    type: openai
profiles: {}
chat:
  model: my-key
task:
  model:
    name: dummy-task
    type: test
"#;
        let config_file = create_temp_config(yaml);
        let config = get_config(Some(config_file)).unwrap();
        let model = config.models.get("my-key").unwrap();
        assert_eq!(model.key, "my-key"); // Always the config key
        assert_eq!(model.name, "custom-name"); // Explicit override
        assert_eq!(config.chat.model.key, "my-key"); // Propagated
    }

    #[test]
    fn test_inline_model_key_fallback() {
        let yaml = r#"
models: {}
profiles: {}
chat:
  model:
    name: inline-model
    type: test
task:
  model:
    name: dummy-task
    type: test
"#;
        let config_file = create_temp_config(yaml);
        let config = get_config(Some(config_file)).unwrap();
        assert_eq!(config.chat.model.name, "inline-model");
        assert_eq!(config.chat.model.key, "inline-model"); // Fallback to name
    }

    #[test]
    fn test_create_or_get_config_file_when_exists() {
        let config_path = create_temp_config(DUMMY_CONFIG_CONTENT);

        let (exists, file_path) = create_or_get_config_file(Some(config_path.clone())).unwrap();

        assert!(exists);
        assert_eq!(file_path, config_path);
        assert!(file_path.exists());
    }

    #[test]
    fn test_create_or_get_config_file_when_not_exist() {
        let config_dir = tempdir().unwrap();
        let config_file = config_dir.path().join("arey.yml");

        let (exists, file_path) = create_or_get_config_file(Some(config_file.clone())).unwrap();

        println!("{config_dir:?} {config_file:?}");
        assert!(!exists);
        assert_eq!(file_path, config_file);
        assert!(file_path.exists());
    }

    #[test]
    fn test_get_config_return_config_for_valid_schema() {
        let config_file = create_temp_config(DUMMY_CONFIG_CONTENT);
        let config = get_config(Some(config_file)).unwrap();

        assert_eq!(config.models.len(), 2);
        assert_eq!(config.profiles.len(), 3);
        assert_eq!(config.chat.model.name, "dummy-7b");
        let chat_agent = config.agents.get(&config.chat.agent_name).unwrap();
        assert_eq!(chat_agent.profile.temperature, 0.7);
        assert_eq!(config.task.model.name, "dummy-13b");
        assert_eq!(config.task.profile.temperature, 0.8);
        assert_eq!(config.theme, "dark");

        let dummy7b = config.models.get("dummy-7b").unwrap();
        assert_eq!(dummy7b.settings.len(), 2);
        assert_eq!(dummy7b.key, "dummy-7b"); // New: Verify key is set
        assert_eq!(dummy7b.name, "dummy-7b"); // Existing: name fallback
    }

    #[test]
    fn test_get_config_throws_for_invalid_yaml() {
        let config_file = create_temp_config("invalid yaml content: - [");
        let err = get_config(Some(config_file)).unwrap_err();
        assert!(matches!(err, AreyConfigError::YAMLError(_)));
        assert!(format!("{err}").contains("YAML parsing error"));
    }

    #[test]
    fn test_get_config_throws_for_missing_referenced_model() {
        let invalid_config_content = r#"
models: {} # Empty models map
profiles: {} # Empty profiles map
chat:
  model: non-existent-model # References a model not in the map
task:
  model: non-existent-model
"#;
        let config_file = create_temp_config(invalid_config_content);
        let err = get_config(Some(config_file)).unwrap_err();
        assert!(
            matches!(err, AreyConfigError::Config(msg) if msg.contains("Model 'non-existent-model' not found"))
        );
    }

    #[test]
    fn test_get_config_with_merge_keys() {
        const MERGE_KEY_CONFIG: &str = r#"
models:
  gguf-base: &gguf-base
    type: gguf
    n_ctx: 4096
    common-key: "common-value"

  model-a:
    <<: *gguf-base
    name: model-a
    path: /path/to/model-a.gguf

  model-b:
    <<: *gguf-base
    name: model-b
    path: /path/to/model-b.gguf
    n_ctx: 8192 # override

profiles:
  default:
    temperature: 0.7
    repeat_penalty: 1.176
    top_k: 40
    top_p: 0.1

chat:
  model: model-a
task:
  model: model-b
"#;
        let config_file = create_temp_config(MERGE_KEY_CONFIG);
        let config = get_config(Some(config_file)).unwrap();

        assert_eq!(config.models.len(), 3); // gguf-base, model-a, model-b

        let gguf_base = config.models.get("gguf-base").unwrap();
        assert_eq!(gguf_base.name, "gguf-base");
        assert_eq!(gguf_base.key, "gguf-base"); // New: Key set
        assert_eq!(gguf_base.settings.len(), 2);
        assert_eq!(
            gguf_base.settings.get("n_ctx").unwrap(),
            &serde_yaml::Value::Number(4096.into())
        );
        assert_eq!(
            gguf_base.settings.get("common-key").unwrap(),
            &serde_yaml::Value::String("common-value".to_string())
        );

        let model_a = config.models.get("model-a").unwrap();
        assert_eq!(model_a.name, "model-a");
        assert_eq!(model_a.key, "model-a"); // New: Key set
        let model_a_settings = &model_a.settings;
        assert_eq!(model_a_settings.len(), 3);
        assert_eq!(
            model_a_settings.get("n_ctx").unwrap(),
            &serde_yaml::Value::Number(4096.into())
        );
        assert_eq!(
            model_a_settings.get("common-key").unwrap(),
            &serde_yaml::Value::String("common-value".to_string())
        );
        assert_eq!(
            model_a_settings.get("path").unwrap(),
            &serde_yaml::Value::String("/path/to/model-a.gguf".to_string())
        );

        let model_b = config.models.get("model-b").unwrap();
        assert_eq!(model_b.name, "model-b");
        assert_eq!(model_b.key, "model-b"); // New: Key set
        let model_b_settings = &model_b.settings;
        assert_eq!(model_b_settings.len(), 3);
        assert_eq!(
            model_b_settings.get("n_ctx").unwrap(),
            &serde_yaml::Value::Number(8192.into())
        ); // overridden
        assert_eq!(
            model_b_settings.get("common-key").unwrap(),
            &serde_yaml::Value::String("common-value".to_string())
        );
        assert_eq!(
            model_b_settings.get("path").unwrap(),
            &serde_yaml::Value::String("/path/to/model-b.gguf".to_string())
        );
    }

    #[test]
    fn test_get_config_with_agents() {
        const DUMMY_CONFIG_WITH_AGENTS: &str = r#"
models:
  dummy-7b:
    name: dummy-7b
    type: gguf
profiles:
  concise:
    temperature: 0.1
chat:
  model: dummy-7b
  agent: default
task:
  model: dummy-7b
  agent: default
"#;
        let config_file = create_temp_config(DUMMY_CONFIG_WITH_AGENTS);
        let config = get_config(Some(config_file)).unwrap();

        // Should have default agent (from default.yml) + any user agents
        assert!(config.agents.contains_key("default"));

        // Check that default agent has proper metadata
        let default_agent = config.agents.get("default").unwrap();
        assert!(matches!(
            default_agent.metadata.source,
            AgentSource::UserFile(_)
        ));

        assert_eq!(config.chat.agent_name, "default");
        assert_eq!(config.task.agent_name, "default");
    }

    #[test]
    fn test_get_config_with_chat_agent() {
        const DUMMY_CONFIG_WITH_CHAT_AGENT: &str = r#"
models:
  dummy-7b:
    name: dummy-7b
    type: gguf
profiles: {}
chat:
  model: dummy-7b
  agent: default
task:
  model: dummy-7b
"#;
        let config_file = create_temp_config(DUMMY_CONFIG_WITH_CHAT_AGENT);
        let config = get_config(Some(config_file)).unwrap();

        assert!(config.agents.contains_key("default"));
        assert_eq!(config.chat.agent_name, "default");
        let chat_agent = config.agents.get("default").unwrap();
        assert_eq!(chat_agent.prompt, "You are a helpful assistant.");
    }

    #[test]
    fn test_get_config_with_missing_agent() {
        let config_content = r#"
models:
  dummy-7b:
    name: dummy-7b
    type: gguf
profiles: {}
agents: {}
chat:
  model: dummy-7b
  agent: non-existent
task:
  model: dummy-7b
"#;
        let config_file = create_temp_config(config_content);
        let err = get_config(Some(config_file)).unwrap_err();
        assert!(matches!(
            err,
            AreyConfigError::Config(msg)
                if msg.contains("Agent 'non-existent' specified in chat mode not found")
        ));
    }

    #[test]
    fn test_get_config_with_default_agent() {
        const DUMMY_CONFIG_NO_CHAT_AGENT: &str = r#"
models:
  dummy-7b:
    name: dummy-7b
    type: gguf
profiles: {}
agents: {}
chat:
  model: dummy-7b
task:
  model: dummy-7b
"#;
        let config_file = create_temp_config(DUMMY_CONFIG_NO_CHAT_AGENT);
        let config = get_config(Some(config_file)).unwrap();

        assert_eq!(config.chat.agent_name, "default");
        assert!(config.agents.contains_key("default"));
        let default_agent = config.agents.get("default").unwrap();
        assert_eq!(default_agent.prompt, "You are a helpful assistant.");
    }

    #[test]
    fn test_get_built_in_agents() {
        let agents = get_built_in_agents();

        assert!(agents.contains_key("default"));

        let default_agent = agents.get("default").unwrap();
        assert_eq!(default_agent.name, "default");
        assert_eq!(default_agent.prompt, "You are a helpful assistant.");
        assert!(default_agent.tools.is_empty());
    }

    #[test]
    fn test_load_agents_from_directory() {
        let temp_dir = tempfile::tempdir().unwrap();
        let agents_dir = temp_dir.path().join("agents");
        fs::create_dir_all(&agents_dir).unwrap();

        // Create a test agent file
        let test_agent_content = r#"
name: "test-agent"
prompt: "I am a test agent."
tools:
  - search
profile:
  temperature: 0.5
"#;
        let agent_file = agents_dir.join("test-agent.yml");
        fs::write(&agent_file, test_agent_content).unwrap();

        let agents = load_agents_from_directory(&agents_dir).unwrap();

        assert_eq!(agents.len(), 1);
        assert!(agents.contains_key("test-agent"));

        let agent = agents.get("test-agent").unwrap();
        assert_eq!(agent.name, "test-agent");
        assert_eq!(agent.prompt, "I am a test agent.");
        assert_eq!(agent.tools, vec!["search"]);
        assert_eq!(agent.profile.temperature, 0.5);
        assert_eq!(
            agent.metadata.source,
            AgentSource::UserFile(agent_file.clone())
        );
    }

    #[test]
    fn test_load_agents_from_nonexistent_directory() {
        let nonexistent_dir = PathBuf::from("/nonexistent/directory");
        let agents = load_agents_from_directory(&nonexistent_dir).unwrap();
        assert!(agents.is_empty());
    }

    #[test]
    fn test_merge_agents_precedence() {
        let mut built_in = HashMap::new();
        built_in.insert(
            "default".to_string(),
            AgentConfig::new("default", "Built-in default", vec!["tool1".to_string()])
                .to_agent(AgentSource::BuiltIn),
        );

        let mut user_agents = HashMap::new();
        user_agents.insert("default".to_string(), {
            let mut agent = AgentConfig::new("default", "User default", vec!["tool2".to_string()])
                .to_agent(AgentSource::UserFile(PathBuf::from("test.yml")));
            agent.profile.temperature = 0.1;
            agent
        });

        user_agents.insert(
            "user-only".to_string(),
            AgentConfig::new("user-only", "User only agent", vec![])
                .to_agent(AgentSource::UserFile(PathBuf::from("user-only.yml"))),
        );

        let merged_agents = merge_agents(built_in, user_agents);

        // User agents should override built-in
        assert_eq!(merged_agents.len(), 2);

        let default_agent = merged_agents.get("default").unwrap();
        assert_eq!(default_agent.prompt, "User default"); // User overrides built-in
        assert_eq!(default_agent.tools, vec!["tool2"]);

        assert!(matches!(
            default_agent.metadata.source,
            AgentSource::UserFile(_)
        ));

        // User-only agent should be preserved
        assert!(merged_agents.contains_key("user-only"));
    }

    #[test]
    fn test_multi_source_agent_loading() {
        let temp_dir = tempfile::tempdir().unwrap();
        let config_dir = temp_dir.path();
        let agents_dir = config_dir.join("agents");
        fs::create_dir_all(&agents_dir).unwrap();

        // Create a user agent file
        let user_agent_content = r#"
name: "user-coder"
prompt: "User-defined coder agent"
tools:
  - search
  - file
profile:
  temperature: 0.2
"#;
        let agent_file = agents_dir.join("user-coder.yml");
        fs::write(&agent_file, user_agent_content).unwrap();

        // Create config without legacy agents section
        let config_content = r#"
models:
  test-model:
    name: test-model
    type: gguf
profiles: {}
chat:
  model: test-model
task:
  model: test-model
"#;
        let config_file = config_dir.join("arey.yml");
        fs::write(&config_file, config_content).unwrap();

        let config = get_config(Some(config_file)).unwrap();

        // Should have: default agent (from default.yml) + user agent
        assert!(config.agents.len() >= 2);

        // Check default agent is present
        assert!(config.agents.contains_key("default"));

        // Check user agent is present
        assert!(config.agents.contains_key("user-coder"));
        let user_coder = config.agents.get("user-coder").unwrap();
        assert_eq!(user_coder.prompt, "User-defined coder agent");

        // Check metadata is properly tracked within each agent
        let default_agent = config.agents.get("default").unwrap();
        let user_coder_agent = config.agents.get("user-coder").unwrap();

        // The default agent is loaded from the created default.yml file
        // so it should have UserFile source
        assert!(matches!(
            default_agent.metadata.source,
            AgentSource::UserFile(_)
        ));
        assert!(matches!(
            user_coder_agent.metadata.source,
            AgentSource::UserFile(_)
        ));
    }

    #[test]
    fn test_agent_file_format_validation() {
        let valid_content = r#"
name: "test-agent"
prompt: "Test prompt"
tools:
  - search
profile:
  temperature: 0.7
"#;

        let agent: Result<AgentFileConfig, _> = serde_yaml::from_str(valid_content);
        assert!(agent.is_ok());

        let agent_config = agent.unwrap();
        assert_eq!(agent_config.name, "test-agent");
        assert_eq!(agent_config.prompt, "Test prompt");
        assert_eq!(agent_config.tools, vec!["search"]);
        assert!(agent_config.profile.is_some());
    }

    #[test]
    fn test_agent_file_with_missing_optional_fields() {
        let minimal_content = r#"
name: "minimal-agent"
prompt: "Minimal prompt"
"#;

        let agent: Result<AgentFileConfig, _> = serde_yaml::from_str(minimal_content);
        assert!(agent.is_ok());

        let agent_config = agent.unwrap();
        assert_eq!(agent_config.name, "minimal-agent");
        assert_eq!(agent_config.prompt, "Minimal prompt");
        assert!(agent_config.tools.is_empty());
        assert!(agent_config.profile.is_none());
    }
}
