use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub type ModelProvider = String;

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
    pub capabilities: Vec<ModelCapability>,
    #[serde(flatten)]
    pub settings: HashMap<String, serde_yaml::Value>,
}

pub struct ModelMetrics {
    pub init_latency_ms: f32,
}
