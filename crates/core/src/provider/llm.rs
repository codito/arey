use crate::completion::CompletionModel;
use crate::model::ModelProvider;
use crate::provider::{gguf, openai, test_provider};
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
        ModelProvider::Test => {
            let model = test_provider::TestProviderModel::new(model_config)?;
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
            key: "test-key".to_string(),
            name: "test-openai".to_string(),
            provider: ModelProvider::Openai,
            settings,
        };
        let model = get_completion_llm(model_config);
        assert!(model.is_ok());
    }

    #[test]
    fn test_get_completion_llm_gguf_provider_error() {
        let model_config = ModelConfig {
            key: "test-key".to_string(),
            name: "test-gguf".to_string(),
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
                .contains("GGUF model path not found")
        );
    }

    #[test]
    fn test_get_completion_llm_test_provider() {
        let model_config = ModelConfig {
            key: "test-key".to_string(),
            name: "test-test".to_string(),
            provider: ModelProvider::Test,
            settings: HashMap::new(),
        };
        let model = get_completion_llm(model_config);
        assert!(model.is_ok());
    }

    #[test]
    fn test_get_completion_llm_gguf_v2_error() {
        let mut settings = HashMap::new();
        settings.insert(
            "implementation".to_string(),
            serde_yaml::Value::String("v2".to_string()),
        );
        let model_config = ModelConfig {
            key: "test-key".to_string(),
            name: "test-gguf-v2".to_string(),
            provider: ModelProvider::Gguf,
            settings,
        };
        let model = get_completion_llm(model_config);
        assert!(model.is_err());
        assert!(
            model
                .err()
                .unwrap()
                .to_string()
                .contains("GGUF model path not found")
        );
    }

    #[test]
    fn test_get_completion_llm_gguf_v2() {
        use std::collections::HashMap;
        let mut settings = HashMap::new();
        settings.insert(
            "implementation".to_string(),
            serde_yaml::Value::String("v2".to_string()),
        );
        settings.insert(
            "path".to_string(),
            serde_yaml::Value::String("/nonexistent/model.gguf".to_string()),
        );
        settings.insert(
            "n_ctx".to_string(),
            serde_yaml::Value::String("512".to_string()),
        );
        let model_config = ModelConfig {
            key: "test-key".to_string(),
            name: "test-gguf-v2".to_string(),
            provider: ModelProvider::Gguf,
            settings,
        };
        let model = get_completion_llm(model_config);
        // Should fail because model file doesn't exist, but shouldn't fail on "GGUF model path not found"
        assert!(model.is_err());
        let err = model.err().unwrap().to_string();
        assert!(!err.contains("GGUF model path not found"), "Got: {}", err);
    }
}
