use crate::core::completion::CompletionModel;
use crate::core::model::ModelConfig;
use crate::platform::{llama, ollama, openai};
use anyhow::Result;
use std::sync::Arc;
use tokio::sync::Mutex;

#[derive(thiserror::Error, Debug)]
pub enum ModelInitError {
    #[error("Unsupported model type: {0}")]
    UnsupportedType(String),
    #[error("Initialization error: {0}")]
    InitError(String),
}

pub fn get_completion_llm(
    model_config: crate::core::model::ModelConfig
) -> Result<Arc<Mutex<dyn CompletionModel + Send + Sync>>> {
    match model_config.r#type {
        crate::core::model::ModelProvider::Gguf => {
            let model = llama::LlamaBaseModel::new(model_config)?;
            Ok(Arc::new(Mutex::new(Box::new(model))))
        }
        crate::core::model::ModelProvider::Ollama => {
            let model = ollama::OllamaBaseModel::new(model_config)?;
            Ok(Arc::new(Mutex::new(Box::new(model))))
        }
        crate::core::model::ModelProvider::Openai => {
            let model = openai::OpenAIBaseModel::new(model_config)?;
            Ok(Arc::new(Mutex::new(Box::new(model))))
        }
    }
}
