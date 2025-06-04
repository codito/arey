use std::{
    collections::HashMap,
    fs::{self, File},
    io::Write,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    core::model::{ModelConfig, ModelProvider},
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
    #[serde(flatten)]
    model: StringOrObject<ModelConfig>,
    #[serde(default, flatten)]
    profile: Option<StringOrObject<ProfileConfig>>,
}

impl RawConfig {
    fn to_config(self) -> Result<Config, AreyConfigError> {
        let models = self.models;
        let profiles = self.profiles;
        let mut models_with_names = HashMap::new();

        for (k, mut v) in models {
            // Update model name if not set
            if v.name.is_empty() {
                v.name = k.clone();
            }
            models_with_names.insert(k, v);
        }

        let resolve_model =
            |model: StringOrObject<ModelConfig>| -> Result<ModelConfig, AreyConfigError> {
                match model {
                    StringOrObject::String(s) => models_with_names
                        .get(&s)
                        .cloned()
                        .ok_or_else(|| AreyConfigError::Config(format!("Model '{}' not found", s))),
                    StringOrObject::Object(m) => Ok(m),
                }
            };

        let resolve_profile = |profile: Option<StringOrObject<ProfileConfig>>| -> Result<ProfileConfig, AreyConfigError> {
            match profile {
                Some(StringOrObject::String(s)) => profiles
                    .get(&s)
                    .cloned()
                    .ok_or_else(|| AreyConfigError::Config(format!("Profile '{}' not found", s))),
                Some(StringOrObject::Object(p)) => Ok(p),
                None => Ok(ProfileConfig::default()),
            }
        };

        Ok(Config {
            models: models_with_names,
            profiles,
            chat: ModeConfig {
                model: resolve_model(self.chat.model)?,
                profile: resolve_profile(self.chat.profile)?,
            },
            task: ModeConfig {
                model: resolve_model(self.task.model)?,
                profile: resolve_profile(self.task.profile)?,
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
