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

/// Supported model provider integrations.
/// make this serializable to str AI!
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub enum ModelProvider {
    Gguf,
    Openai,
    Ollama,
}

#[derive(Debug, Clone, Default)]
pub struct ModelMetrics {
    pub init_latency_ms: f32,
}
