use serde::{Deserialize, Serialize, de::IntoDeserializer};
use std::collections::HashMap;

/// Template override configuration.
///
/// Supports two formats:
/// - Simple string: `template: "default"` or `template: "model"`
/// - Detailed map: `template: { name: "default", args: { enable_thinking: true }}`
///
/// The `args` map allows passing arbitrary kwargs to the template without
/// hardcoding them in the application.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(untagged)]
pub enum TemplateOverride {
    /// Simple form: just the template name
    /// - "default": Use built-in default template
    /// - "model": Use model's built-in template (explicit)
    Simple(String),
    /// Detailed form with name and optional args
    Detailed {
        /// Template name: "default", "model", or custom identifier
        #[serde(rename = "name")]
        template_name: String,
        /// Optional template arguments (kwargs)
        /// Example: { enable_thinking: true, add_generation_prompt: false }
        #[serde(default, rename = "args")]
        template_args: Option<HashMap<String, serde_yaml::Value>>,
    },
}

impl TemplateOverride {
    /// Get the template name.
    pub fn name(&self) -> &str {
        match self {
            TemplateOverride::Simple(s) => s.as_str(),
            TemplateOverride::Detailed { template_name, .. } => template_name.as_str(),
        }
    }

    /// Get the template arguments as a HashMap.
    /// Returns empty map if args is None.
    pub fn args(&self) -> HashMap<String, serde_yaml::Value> {
        match self {
            TemplateOverride::Detailed { template_args, .. } => {
                template_args.clone().unwrap_or_default()
            }
            TemplateOverride::Simple(_) => HashMap::new(),
        }
    }
}

/// Model configuration for the tool.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct ModelConfig {
    /// The original key from the config file's models HashMap (e.g., "dummy-7b").
    /// Set during parsing; not part of YAML.
    #[serde(default, skip)]
    pub key: String,
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

    /// Get the original config key.
    pub fn get_key(&self) -> &str {
        &self.key
    }

    /// Get the template override configuration.
    ///
    /// Returns `Some(TemplateOverride)` if the user specified a `template` key in settings.
    /// Returns `None` if no template override is specified (use model's default).
    pub fn template_override(&self) -> Option<TemplateOverride> {
        self.get_setting::<TemplateOverride>("template")
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            name: String::new(),
            key: String::new(),
            provider: ModelProvider::Test,
            settings: HashMap::new(),
        }
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
    #[error("Unsupported model type: {0}")]
    UnsupportedType(String),
    #[error("Initialization error: {0}")]
    InitError(String),
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
        assert_eq!(config.key, ""); // Empty after direct deserialization (pre-parsing)
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
        assert_eq!(config_alias.key, ""); // Empty pre-parsing
    }

    #[test]
    fn test_model_config_get_key() {
        let config = ModelConfig {
            name: "test".to_string(),
            key: "config-key".to_string(),
            provider: ModelProvider::Openai,
            settings: HashMap::new(),
        };
        assert_eq!(config.get_key(), "config-key");

        // Default case
        let default_config: ModelConfig = Default::default();
        assert_eq!(default_config.get_key(), "");
    }

    #[test]
    fn test_model_config_with_explicit_key_simulation() {
        // Simulate post-parsing where key is set separately from name
        let config = ModelConfig {
            name: "custom-name".to_string(),
            key: "my-key".to_string(),
            provider: ModelProvider::Gguf,
            settings: HashMap::new(),
        };
        assert_eq!(config.name, "custom-name");
        assert_eq!(config.key, "my-key");
        assert_eq!(config.get_key(), "my-key");
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

    #[test]
    fn test_template_override_simple() {
        let yaml = r#"
            name: test_model
            provider: gguf
            template: "default"
        "#;
        let config: ModelConfig = serde_yaml::from_str(yaml).unwrap();
        let tmpl = config.template_override().unwrap();
        assert_eq!(tmpl.name(), "default");
        assert!(tmpl.args().is_empty());
    }

    #[test]
    fn test_template_override_detailed() {
        let yaml = r#"
            name: test_model
            provider: gguf
            template:
              name: qwen3
              args:
                enable_thinking: true
                add_generation_prompt: false
        "#;
        let config: ModelConfig = serde_yaml::from_str(yaml).unwrap();
        let tmpl = config.template_override().unwrap();
        assert_eq!(tmpl.name(), "qwen3");
        let args = tmpl.args();
        assert_eq!(
            args.get("enable_thinking").and_then(|v| v.as_bool()),
            Some(true)
        );
        assert_eq!(
            args.get("add_generation_prompt").and_then(|v| v.as_bool()),
            Some(false)
        );
    }

    #[test]
    fn test_template_override_none() {
        let yaml = r#"
            name: test_model
            provider: gguf
        "#;
        let config: ModelConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(config.template_override().is_none());
    }
}
