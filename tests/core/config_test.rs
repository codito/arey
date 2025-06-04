use crate::core::config::{AreyConfigError, ProfileConfig, RawConfig, StringOrObject};
use std::collections::HashMap;
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
