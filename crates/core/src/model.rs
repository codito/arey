use serde::{Deserialize, Serialize, de::IntoDeserializer};
use std::collections::HashMap;

/// Model configuration for the tool.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct ModelConfig {
    #[serde(default)]
    pub name: String,
    #[serde(alias = "type")]
    pub provider: ModelProvider,
    #[serde(default, flatten)]
    pub settings: HashMap<String, serde_yaml::Value>,
}

impl ModelConfig {
    /// Get a setting value by key, deserializing it into the requested type.
    pub fn get_setting<'a, T>(&'a self, key: &str) -> Option<T>
    where
        T: Deserialize<'a>,
    {
        self.settings
            .get(key)
            .and_then(|v| T::deserialize(v.clone().into_deserializer()).ok())
    }
}

/// Supported model provider integrations (serialized as lowercase strings).
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ModelProvider {
    Gguf,
    Openai,
    Test,
}

impl From<ModelProvider> for String {
    fn from(val: ModelProvider) -> Self {
        val.as_str().into()
    }
}

impl ModelProvider {
    pub fn as_str(&self) -> &'static str {
        match &self {
            ModelProvider::Gguf => "gguf",
            ModelProvider::Openai => "openai",
            ModelProvider::Test => "test",
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct ModelMetrics {
    pub init_latency_ms: f32,
}

#[derive(thiserror::Error, Debug)]
pub enum ModelInitError {
    // #[error("Unsupported model type: {0}")]
    // UnsupportedType(String),
    // #[error("Initialization error: {0}")]
    // InitError(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_yaml;

    #[test]
    fn test_model_provider_strings() {
        assert_eq!(ModelProvider::Gguf.as_str(), "gguf");
        assert_eq!(ModelProvider::Openai.as_str(), "openai");
        assert_eq!(ModelProvider::Test.as_str(), "test");

        let s: String = ModelProvider::Gguf.into();
        assert_eq!(s, "gguf");
        let s: String = ModelProvider::Openai.into();
        assert_eq!(s, "openai");
        let s: String = ModelProvider::Test.into();
        assert_eq!(s, "test");
    }

    #[test]
    fn test_model_config_deserialization() {
        let yaml = r#"
            name: test_model
            provider: openai
            api_key: test_key
        "#;
        let config: ModelConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.name, "test_model");
        assert_eq!(config.provider, ModelProvider::Openai);
        assert_eq!(
            config.settings.get("api_key").unwrap().as_str().unwrap(),
            "test_key"
        );

        let yaml_alias = r#"
            name: test_model_alias
            type: gguf
        "#;
        let config_alias: ModelConfig = serde_yaml::from_str(yaml_alias).unwrap();
        assert_eq!(config_alias.provider, ModelProvider::Gguf);
    }

    #[test]
    fn test_model_metrics_default() {
        let metrics: ModelMetrics = Default::default();
        assert_eq!(metrics.init_latency_ms, 0.0);
    }

    #[test]
    fn test_model_config_get_setting() {
        let yaml = r#"
            name: test_model
            provider: openai
            str_key: "hello"
            int_key: 123
            neg_int_key: -456
        "#;
        let config: ModelConfig = serde_yaml::from_str(yaml).unwrap();

        // Test getting a string
        assert_eq!(
            config.get_setting::<String>("str_key"),
            Some("hello".to_string())
        );

        // Test getting different integer types
        assert_eq!(config.get_setting::<i64>("int_key"), Some(123));
        assert_eq!(config.get_setting::<u32>("int_key"), Some(123));
        assert_eq!(config.get_setting::<i32>("neg_int_key"), Some(-456));

        // Test getting a missing key
        assert_eq!(config.get_setting::<String>("missing_key"), None);

        // Test type mismatch
        assert_eq!(config.get_setting::<i64>("str_key"), None);
        assert_eq!(config.get_setting::<String>("int_key"), None);
    }
}
