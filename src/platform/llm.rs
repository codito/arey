use crate::core::completion::CompletionModel;
use crate::core::model::ModelProvider;
use crate::platform::{llama, openai};
use anyhow::Result;

#[derive(thiserror::Error, Debug)]
pub enum ModelInitError {
    // #[error("Unsupported model type: {0}")]
    // UnsupportedType(String),
    // #[error("Initialization error: {0}")]
    // InitError(String),
}

pub fn get_completion_llm(
    model_config: crate::core::model::ModelConfig,
) -> Result<Box<dyn CompletionModel + Send + Sync>> {
    match model_config.provider {
        ModelProvider::Gguf => {
            let model = llama::LlamaBaseModel::new(model_config)?;
            Ok(Box::new(model))
        }
        ModelProvider::Openai => {
            let model = openai::OpenAIBaseModel::new(model_config)?;
            Ok(Box::new(model))
        }
    }
}
