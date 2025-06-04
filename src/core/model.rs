use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[cfg(test)]
mod tests {
    use super::*;
    use serde_yaml;

    #[test]
    fn test_model_config_default_capabilities() {
        let config = ModelConfig {
            name: "test".into(),
            r#type: ModelProvider::Gguf,
            capabilities: default_capabilities(),
            settings: HashMap::new(),
        };
        assert_eq!(config.capabilities.len(), 1);
        assert!(matches!(config.capabilities[0], ModelCapability::Completion));
    }

    #[test]
    fn test_model_provider_serialization() {
        let gguf = ModelProvider::Gguf;
        let serialized = serde_yaml::to_string(&gguf).unwrap();
        assert_eq!(serialized.trim(), "gguf");

        let openai = ModelProvider::Openai;
        let serialized = serde_yaml::to_string(&openai).unwrap();
        assert_eq!(serialized.trim(), "openai");

        let ollama = ModelProvider::Ollama;
        let serialized = serde_yaml::to_string(&ollama).unwrap();
        assert_eq!(serialized.trim(), "ollama");
    }

    #[test]
    fn test_model_capability_serialization() {
        let completion = ModelCapability::Completion;
        let serialized = serde_yaml::to_string(&completion).unwrap();
        assert_eq!(serialized.trim(), "completion");

        let tools = ModelCapability::ToolsLlama;
        let serialized = serde_yaml::to_string(&tools).unwrap();
        assert_eq!(serialized.trim(), "tools:llama");
    }

    #[test]
    fn test_model_config_yaml_parsing() {
        let yaml = r#"
            name: test-model
            type: gguf
            n_ctx: 4096
        "#;
        
        let config: ModelConfig = serde_yaml::from_str(yaml).unwrap();
        
        assert_eq!(config.name, "test-model");
        assert!(matches!(config.r#type, ModelProvider::Gguf));
        assert_eq!(*config.settings.get("n_ctx").unwrap(), serde_yaml::Value::Number(4096.into()));
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelProvider {
    Gguf,
    Openai,
    Ollama,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelCapability {
    Completion,
    Embedding,
    Chat,
    Code,
    Math,
    #[serde(rename = "tools:openai")]
    ToolsOpenai,
    #[serde(rename = "tools:llama")]
    ToolsLlama,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub name: String,
    pub r#type: ModelProvider,
    #[serde(default = "default_capabilities")]
    pub capabilities: Vec<ModelCapability>,
    #[serde(flatten)]
    pub settings: HashMap<String, serde_yaml::Value>,
}

fn default_capabilities() -> Vec<ModelCapability> {
    vec![ModelCapability::Completion]
}

pub struct ModelMetrics {
    pub init_latency_ms: f32,
}
