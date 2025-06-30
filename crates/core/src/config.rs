use std::{
    collections::HashMap,
    fs::{self, File},
    io::Write,
    path::PathBuf,
};

use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::instrument;

use crate::{
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

#[derive(Debug, Deserialize, Serialize, Clone)]
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
    #[serde(default)]
    pub profile: ProfileConfig,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Config {
    pub models: HashMap<String, ModelConfig>,
    pub profiles: HashMap<String, ProfileConfig>,
    pub chat: ModeConfig,
    pub task: ModeConfig,
    #[serde(default = "default_theme")]
    pub theme: String,
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
struct RawConfig {
    models: HashMap<String, ModelConfig>,
    profiles: HashMap<String, ProfileConfig>,
    chat: RawModeConfig,
    task: RawModeConfig,
    theme: Option<String>,
}

#[derive(Deserialize, Debug)]
struct RawModeConfig {
    model: StringOrObject<ModelConfig>,
    #[serde(default)]
    profile: Option<StringOrObject<ProfileConfig>>,
}

impl RawConfig {
    #[instrument]
    fn to_config(&self) -> Result<Config, AreyConfigError> {
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
                    StringOrObject::Object(m) => Ok(m.clone()),
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

        let chat_model = resolve_model(&self.chat.model)?;
        let chat_profile = resolve_profile(&self.chat.profile)?;
        let task_model = resolve_model(&self.task.model)?;
        let task_profile = resolve_profile(&self.task.profile)?;

        Ok(Config {
            models: models_with_names,
            profiles: self.profiles.clone(),
            chat: ModeConfig {
                model: chat_model,
                profile: chat_profile,
            },
            task: ModeConfig {
                model: task_model,
                profile: task_profile,
            },
            theme: self.theme.clone().unwrap_or_else(default_theme),
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
    let content = fs::read_to_string(&config_file)?;
    let raw: RawConfig = serde_yaml::from_str(&content)?;
    raw.to_config()
}

#[cfg(test)]
mod tests {
    use std::{
        collections::HashMap,
        fs::{self, File},
        io::Write,
        path::PathBuf,
    };

    use tempfile::{NamedTempFile, env::temp_dir, tempdir};

    use super::*;

    fn create_temp_config(content: &str) -> PathBuf {
        let temp_dir = temp_dir();
        let config_path = NamedTempFile::new().unwrap().path().to_owned();
        fs::create_dir_all(&temp_dir).unwrap();
        File::create(&config_path)
            .unwrap()
            .write_all(content.as_bytes())
            .unwrap();
        config_path
    }

    fn dummy_model_config(name: &str) -> crate::model::ModelConfig {
        crate::model::ModelConfig {
            name: name.to_string(),
            provider: crate::model::ModelProvider::Gguf,
            settings: HashMap::from([(
                "n_ctx".to_string(),
                serde_yaml::Value::Number(4096.into()),
            )]),
        }
    }

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
  profile: default
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
                temperature: 0.5,
                ..Default::default()
            },
        );

        let raw_config = RawConfig {
            models: models.clone(),
            profiles: profiles.clone(),
            chat: crate::config::RawModeConfig {
                model: StringOrObject::String("dummy-7b".to_string()),
                profile: Some(StringOrObject::String("default".to_string())),
            },
            task: crate::config::RawModeConfig {
                model: StringOrObject::String("dummy-13b".to_string()),
                profile: Some(StringOrObject::String("concise".to_string())),
            },
            theme: Some("dark".to_string()),
        };

        let config = raw_config.to_config().unwrap();

        assert_eq!(config.models.len(), 2);
        assert_eq!(config.profiles.len(), 3);
        assert_eq!(config.chat.model.name, "dummy-7b");
        assert_eq!(config.chat.profile.temperature, 0.7);
        assert_eq!(config.task.model.name, "dummy-13b");
        assert_eq!(config.task.profile.temperature, 0.5);
        assert_eq!(config.theme, "dark");
    }

    #[test]
    fn test_raw_config_to_config_missing_model_reference() {
        let mut models = HashMap::new();
        models.insert("dummy-7b".to_string(), dummy_model_config("dummy-7b"));

        let raw_config = RawConfig {
            models,
            profiles: HashMap::new(),
            chat: crate::config::RawModeConfig {
                model: StringOrObject::String("non-existent-model".to_string()),
                profile: None,
            },
            task: crate::config::RawModeConfig {
                model: StringOrObject::String("dummy-7b".to_string()),
                profile: None,
            },
            theme: None,
        };

        let err = raw_config.to_config().unwrap_err();
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
            chat: crate::config::RawModeConfig {
                model: StringOrObject::String("dummy-7b".to_string()),
                profile: Some(StringOrObject::String("non-existent-profile".to_string())),
            },
            task: crate::config::RawModeConfig {
                model: StringOrObject::String("dummy-7b".to_string()),
                profile: None,
            },
            theme: None,
        };

        let err = raw_config.to_config().unwrap_err();
        assert!(
            matches!(err, AreyConfigError::Config(msg) if msg.contains("Profile 'non-existent-profile' not found"))
        );
    }

    #[test]
    fn test_raw_config_to_config_inline_model_and_profile() {
        let raw_config = RawConfig {
            models: HashMap::new(),   // No named models
            profiles: HashMap::new(), // No named profiles
            chat: crate::config::RawModeConfig {
                model: StringOrObject::Object(dummy_model_config("inline-chat-model")),
                profile: Some(StringOrObject::Object(ProfileConfig {
                    temperature: 0.8,
                    ..Default::default()
                })),
            },
            task: crate::config::RawModeConfig {
                model: StringOrObject::Object(dummy_model_config("inline-task-model")),
                profile: None, // Should use default profile
            },
            theme: Some("light".to_string()),
        };

        let config = raw_config.to_config().unwrap();

        assert_eq!(config.chat.model.name, "inline-chat-model");
        assert_eq!(config.chat.profile.temperature, 0.8);
        assert_eq!(config.task.model.name, "inline-task-model");
        assert_eq!(config.task.profile.temperature, 0.7); // Default temperature
        assert_eq!(config.theme, "light");
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
        assert_eq!(config.chat.profile.temperature, 0.7);
        assert_eq!(config.task.model.name, "dummy-13b");
        assert_eq!(config.task.profile.temperature, 0.8);
        assert_eq!(config.theme, "dark");

        let dummy7b = config.models.get("dummy-7b").unwrap();
        assert_eq!(dummy7b.settings.len(), 2);
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
  profile: test
task:
  model: non-existent-model
  profile: test
"#;
        let config_file = create_temp_config(invalid_config_content);
        let err = get_config(Some(config_file)).unwrap_err();
        assert!(
            matches!(err, AreyConfigError::Config(msg) if msg.contains("Model 'non-existent-model' not found"))
        );
    }

    #[test]
    fn test_get_config_throws_for_missing_referenced_profile() {
        let invalid_config_content = r#"
models:
  dummy-7b:
    name: dummy-7b
    type: gguf
    n_ctx: 4096
    path: /path/to/dummy_model.gguf
profiles: {} # Empty profiles map
chat:
  model: dummy-7b
  profile: non-existent-profile # References a profile not in the map
task:
  model: dummy-7b
  profile: default # This one is fine, will use default if not found in map
"#;
        let config_file = create_temp_config(invalid_config_content);
        let err = get_config(Some(config_file)).unwrap_err();
        assert!(
            matches!(err, AreyConfigError::Config(msg) if msg.contains("Profile 'non-existent-profile' not found"))
        );
    }
}
