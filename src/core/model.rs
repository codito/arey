use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[cfg(test)]
mod tests {
    use super::*;

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
