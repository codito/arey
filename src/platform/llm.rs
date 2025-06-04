use crate::core::ModelProvider;
use crate::core::completion::CompletionModel;
use crate::core::model::ModelConfig;
use crate::platform::{llama, ollama, openai};
use anyhow::Result;

#[derive(thiserror::Error, Debug)]
pub enum ModelInitError {
    #[error("Unsupported model type: {0}")]
    UnsupportedType(String),
    #[error("Initialization error: {0}")]
    InitError(String),
}

pub fn get_completion_llm(model_config: ModelConfig) -> Result<Box<dyn CompletionModel>> {
    match model_config.r#type {
        ModelProvider::Gguf => {
            let model = llama::LlamaBaseModel::new(model_config)?;
            Ok(Box::new(model))
        }
        ModelProvider::Ollama => {
            let model = ollama::OllamaBaseModel::new(model_config)?;
            Ok(Box::new(model))
        }
        ModelProvider::Openai => {
            let model = openai::OpenAIBaseModel::new(model_config)?;
            Ok(Box::new(model))
        }
    }
}
