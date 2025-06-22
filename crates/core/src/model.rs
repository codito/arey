use serde::{Deserialize, Serialize};
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

/// Supported model provider integrations (serialized as lowercase strings).
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ModelProvider {
    Gguf,
    Openai,
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
