use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Model configuration for the tool.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct ModelConfig {
    #[serde(default)]
    pub name: String,
    #[serde(alias = "type")]
    pub provider: ModelProvider,
    #[serde(default)]
    pub settings: HashMap<String, serde_yaml::Value>,
}

/// Supported model provider integrations (serialized as lowercase strings).
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ModelProvider {
    Gguf,
    Openai,
    Ollama,
}

impl std::fmt::Display for ModelProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelProvider::Gguf => write!(f, "gguf"),
            ModelProvider::Openai => write!(f, "openai"),
            ModelProvider::Ollama => write!(f, "ollama"),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct ModelMetrics {
    pub init_latency_ms: f32,
}
