use crate::completion::CompletionModel;
use crate::model::ModelProvider;
use crate::provider::{gguf, openai};
use anyhow::Result;
use tracing::instrument;

#[instrument(skip(model_config))]
pub fn get_completion_llm(
    model_config: crate::model::ModelConfig,
) -> Result<Box<dyn CompletionModel + Send + Sync>> {
    match model_config.provider {
        ModelProvider::Gguf => {
            let model = gguf::GgufBaseModel::new(model_config)?;
            Ok(Box::new(model))
        }
        ModelProvider::Openai => {
            let model = openai::OpenAIBaseModel::new(model_config)?;
            Ok(Box::new(model))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{ModelConfig, ModelProvider};
    use std::collections::HashMap;

    #[test]
    fn test_get_completion_llm_openai_provider() {
        let mut settings = HashMap::new();
        settings.insert("base_url".to_string(), "http://localhost:1234".into());
        settings.insert("api_key".to_string(), "sk-dummy".into());
        let model_config = ModelConfig {
            name: "test-openai".to_string(),
            provider: ModelProvider::Openai,
            settings,
        };
        let model = get_completion_llm(model_config);
        assert!(model.is_ok());
    }

    #[test]
    fn test_get_completion_llm_gguf_provider_error() {
        // Gguf model requires a 'path' setting, so this should fail.
        let model_config = ModelConfig {
            name: "极速test-gguf".to_string(),
            provider: ModelProvider::Gguf,
            settings: HashMap::new(),
        };
        let model = get_completion_llm(model_config);
        assert!(model.is_err());
        assert!(
            model
                .err()
                .unwrap()
                .to_string()
                .contains("'path' setting is required for gguf model")
        );
    }
}
