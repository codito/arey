use crate::core::completion::CompletionModel;
use crate::core::model::{ModelConfig, ModelMetrics};
use crate::platform::{llama, ollama, openai};
use anyhow::Result;

pub type ModelType = ffigen::types::ModelProvider;

#[derive(thiserror::Error, Debug)]
pub enum ModelInitError {
    #[error("Unsupported model type: {0}")]
    UnsupportedType(String),
    #[error("Initialization error: {0}")]
    InitError(String),
}

pub fn get_completion_llm(model_config: ModelConfig) -> Result<Box<dyn CompletionModel>> {
    match model_config.r#type {
        ModelType::Gguf => {
            let model = llama::LlamaBaseModel::new(model_config)?;
            Ok(Box::new(model))
        }
        ModelType::Ollama => {
            let model = ollama::OllamaBaseModel::new(model_config)?;
            Ok::<Box<dyn CompletionModel>>(Box::new(model))
        }
        ModelType::Openai => {
            let model = openai::OpenAIBaseModel::new(model_config)?;
            Ok(Box::new(model))
        }
        _ => Err(ModelInitError::UnsupportedType(format!("{:?}", model_config.r#type))
            .into()),
    }
}
