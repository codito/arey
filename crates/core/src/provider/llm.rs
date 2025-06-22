use crate::completion::CompletionModel;
use crate::model::ModelProvider;
use crate::provider::{llama, openai};
use anyhow::Result;

pub fn get_completion_llm(
    model_config: crate::model::ModelConfig,
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
