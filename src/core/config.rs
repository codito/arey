use std::{
    collections::HashMap,
    fs::{self, File},
    io::Write,
    path::PathBuf,
};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    core::model::ModelConfig,
    platform::assets::{get_config_dir, get_default_config},
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
}

#[derive(Deserialize)]
#[serde(untagged)]
enum StringOrObject<T> {
    String(String),
    Object(T),
}

#[derive(Deserialize)]
struct RawConfig {
    models: HashMap<String, ModelConfig>,
    profiles: HashMap<String, ProfileConfig>,
    chat: RawModeConfig,
    task: RawModeConfig,
}

#[derive(Deserialize)]
struct RawModeConfig {
    model: StringOrObject<ModelConfig>,
    #[serde(default)]
    profile: Option<StringOrObject<ProfileConfig>>,
}

impl RawConfig {
    fn to_config(self) -> Result<Config, AreyConfigError> {
        let mut models_with_names = HashMap::new();
        for (k, mut v) in self.models {
            // Update model name if not set
            if v.name.is_empty() {
                v.name = k.clone();
            }
            models_with_names.insert(k, v);
        }

        let resolve_model =
            |model_entry: StringOrObject<ModelConfig>| -> Result<ModelConfig, AreyConfigError> {
                match model_entry {
                    StringOrObject::String(s) => models_with_names
                        .get(&s)
                        .cloned()
                        .ok_or_else(|| AreyConfigError::Config(format!("Model '{}' not found", s))),
                    StringOrObject::Object(m) => Ok(m),
                }
            };

        let resolve_profile = |profile_entry: Option<StringOrObject<ProfileConfig>>| -> Result<ProfileConfig, AreyConfigError> {
            match profile_entry {
                Some(StringOrObject::String(s)) => self.profiles
                    .get(&s)
                    .cloned()
                    .ok_or_else(|| AreyConfigError::Config(format!("Profile '{}' not found", s))),
                Some(StringOrObject::Object(p)) => Ok(p),
                None => Ok(ProfileConfig::default()),
            }
        };

        let chat_model = resolve_model(self.chat.model)?;
        let chat_profile = resolve_profile(self.chat.profile)?;
        let task_model = resolve_model(self.task.model)?;
        let task_profile = resolve_profile(self.task.profile)?;

        Ok(Config {
            models: models_with_names,
            profiles: self.profiles, // `self.profiles` can be moved here now
            chat: ModeConfig {
                model: chat_model,
                profile: chat_profile,
            },
            task: ModeConfig {
                model: task_model,
                profile: task_profile,
            },
        })
    }
}

pub fn create_or_get_config_file() -> Result<(bool, PathBuf), AreyConfigError> {
    let config_dir = get_config_dir();
    if !config_dir.exists() {
        fs::create_dir_all(&config_dir)?;
    }

    let config_file = config_dir.join("arey.yml");
    if config_file.exists() {
        Ok((true, config_file))
    } else {
        File::create(&config_file)?.write_all(get_default_config().as_bytes())?;
        Ok((false, config_file))
    }
}

pub fn get_config() -> Result<Config, AreyConfigError> {
    let (_, config_path) = create_or_get_config_file()?;
    let content = fs::read_to_string(&config_path)?;
    let raw: RawConfig = serde_yaml::from_str(&content)?;
    raw.to_config()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_yaml::Value;

    // Helper for creating a dummy ModelConfig object
    fn dummy_model_config(name: &str) -> crate::core::model::ModelConfig {
        crate::core::model::ModelConfig {
            name: name.to_string(),
            r#type: crate::core::model::ModelProvider::Gguf,
            capabilities: vec![crate::core::model::ModelCapability::Completion],
            settings: HashMap::from([
                ("n_ctx".to_string(), Value::Number(4096.into())),
            ]),
        }
    }

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
        profiles.insert("creative".to_string(), ProfileConfig { temperature: 0.9, ..Default::default() });
        profiles.insert("concise".to_string(), ProfileConfig { temperature: 0.5, ..Default::default() });

        let raw_config = RawConfig {
            models: models.clone(),
            profiles: profiles.clone(),
            chat: crate::core::config::RawModeConfig {
                model: StringOrObject::String("dummy-7b".to_string()),
                profile: Some(StringOrObject::String("default".to_string())),
            },
            task: crate::core::config::RawModeConfig {
                model: StringOrObject::String("dummy-13b".to_string()),
                profile: Some(StringOrObject::String("concise".to_string())),
            },
        };

        let config = raw_config.to_config().unwrap();

        assert_eq!(config.models.len(), 2);
        assert_eq!(config.profiles.len(), 3);
        assert_eq!(config.chat.model.name, "dummy-7b");
        assert_eq!(config.chat.profile.temperature, 0.7);
        assert_eq!(config.task.model.name, "dummy-13b");
        assert_eq!(config.task.profile.temperature, 0.5);
    }

    #[test]
    fn test_raw_config_to_config_missing_model_reference() {
        let mut models = HashMap::new();
        models.insert("dummy-7b".to_string(), dummy_model_config("dummy-7b"));

        let raw_config = RawConfig {
            models,
            profiles: HashMap::new(),
            chat: crate::core::config::RawModeConfig {
                model: StringOrObject::String("non-existent-model".to_string()),
                profile: None,
            },
            task: crate::core::config::RawModeConfig {
                model: StringOrObject::String("dummy-7b".to_string()),
                profile: None,
            },
        };

        let err = raw_config.to_config().unwrap_err();
        assert!(matches!(err, AreyConfigError::Config(msg) if msg.contains("Model 'non-existent-model' not found")));
    }

    #[test]
    fn test_raw_config_to_config_missing_profile_reference() {
        let mut models = HashMap::new();
        models.insert("dummy-7b".to_string(), dummy_model_config("dummy-7b"));

        let raw_config = RawConfig {
            models,
            profiles: HashMap::new(),
            chat: crate::core::config::RawModeConfig {
                model: StringOrObject::String("dummy-7b".to_string()),
                profile: Some(StringOrObject::String("non-existent-profile".to_string())),
            },
            task: crate::core::config::RawModeConfig {
                model: StringOrObject::String("dummy-7b".to_string()),
                profile: None,
            },
        };

        let err = raw_config.to_config().unwrap_err();
        assert!(matches!(err, AreyConfigError::Config(msg) if msg.contains("Profile 'non-existent-profile' not found")));
    }

    #[test]
    fn test_raw_config_to_config_inline_model_and_profile() {
        let raw_config = RawConfig {
            models: HashMap::new(), // No named models
            profiles: HashMap::new(), // No named profiles
            chat: crate::core::config::RawModeConfig {
                model: StringOrObject::Object(dummy_model_config("inline-chat-model")),
                profile: Some(StringOrObject::Object(ProfileConfig { temperature: 0.8, ..Default::default() })),
            },
            task: crate::core::config::RawModeConfig {
                model: StringOrObject::Object(dummy_model_config("inline-task-model")),
                profile: None, // Should use default profile
            },
        };

        let config = raw_config.to_config().unwrap();

        assert_eq!(config.chat.model.name, "inline-chat-model");
        assert_eq!(config.chat.profile.temperature, 0.8);
        assert_eq!(config.task.model.name, "inline-task-model");
        assert_eq!(config.task.profile.temperature, 0.7); // Default temperature
    }
}
