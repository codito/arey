use anyhow::{Context, Result, anyhow};
use arey_core::tools::{Tool, ToolError};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::sync::Arc;

pub mod providers;
use providers::searxng::SearxngProvider;

/// The standardized output format for a single search result.
#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
pub struct SearchResult {
    pub title: String,
    pub url: String,
    pub snippet: String,
}

/// A trait for search engine providers.
#[async_trait]
pub trait SearchProvider: Send + Sync {
    /// Performs a search with the given query.
    async fn search(&self, query: &str) -> Result<Vec<SearchResult>>;
}

/// Configuration for the Search tool.
#[derive(Deserialize, Debug)]
pub struct SearchToolConfig {
    provider: String,
    #[serde(flatten)]
    provider_config: serde_yaml::Value,
}

/// A tool for searching the web.
pub struct SearchTool {
    provider: Arc<dyn SearchProvider>,
}

impl SearchTool {
    /// Creates a new SearchTool from a configuration value.
    pub fn from_config(config_value: &serde_yaml::Value) -> Result<Self> {
        let config: SearchToolConfig = serde_yaml::from_value(config_value.clone())
            .context("Failed to parse search tool config")?;
        let provider: Arc<dyn SearchProvider> = match config.provider.as_str() {
            "searxng" => Arc::new(
                SearxngProvider::from_config(&config.provider_config)
                    .context("Failed to configure searxng provider")?,
            ),
            _ => return Err(anyhow!("Unsupported search provider: {}", config.provider)),
        };

        Ok(Self { provider })
    }
}

#[async_trait]
impl Tool for SearchTool {
    fn name(&self) -> String {
        "search".to_string()
    }

    fn description(&self) -> String {
        "Searches the web for a given query.".to_string()
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query."
                }
            },
            "required": ["query"]
        })
    }

    async fn execute(&self, arguments: &Value) -> Result<Value, ToolError> {
        let query = arguments["query"]
            .as_str()
            .ok_or_else(|| ToolError::ExecutionError("Missing 'query' parameter".to_string()))?;

        let results = self
            .provider
            .search(query)
            .await
            .with_context(|| format!("Search failed for query: {}", query))
            .map_err(|e| ToolError::ExecutionError(e.to_string()))?;

        serde_json::to_value(results).map_err(|e| ToolError::ExecutionError(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_yaml;

    #[test]
    fn test_search_tool_from_config_searxng() {
        let yaml_config = r#"
            provider: searxng
            base_url: "http://localhost:8888"
        "#;
        let config_value: serde_yaml::Value = serde_yaml::from_str(yaml_config).unwrap();
        let tool = SearchTool::from_config(&config_value);
        assert!(tool.is_ok());
    }

    #[test]
    fn test_search_tool_from_config_unsupported() {
        let yaml_config = r#"
            provider: foobar
            api_key: "12345"
        "#;
        let config_value: serde_yaml::Value = serde_yaml::from_str(yaml_config).unwrap();
        let tool = SearchTool::from_config(&config_value);
        assert!(tool.is_err());
        assert_eq!(
            tool.unwrap_err().to_string(),
            "Unsupported search provider: foobar"
        );
    }
}
