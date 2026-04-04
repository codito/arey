use anyhow::{Context, Result, anyhow};
use arey_core::tools::{Tool, ToolError};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::sync::Arc;
use tracing::debug;

pub mod providers;
use providers::{SearchOptions, SearxngProvider};

/// The standardized output format for a single search result.
#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
pub struct SearchResult {
    pub title: String,
    pub url: String,
    pub snippet: String,
}

/// A trait for search engine providers.
#[async_trait]
pub trait SearchProvider: Send + Sync + std::fmt::Debug {
    async fn search(&self, query: &str, options: &SearchOptions) -> Result<Vec<SearchResult>>;
}

/// Configuration for the Search tool.
#[derive(Deserialize, Debug)]
pub struct SearchToolConfig {
    provider: String,
    #[serde(flatten)]
    provider_config: serde_yaml::Value,
}

/// A tool for searching the web.
#[derive(Debug)]
pub struct SearchTool {
    provider: Arc<dyn SearchProvider>,
}

impl SearchTool {
    pub fn from_config(config_value: &serde_yaml::Value) -> Result<Self> {
        debug!("Initializing search tool from config: {:?}", config_value);
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
                },
                "format": {
                    "type": "string",
                    "enum": ["json", "markdown"],
                    "default": "markdown",
                    "description": "Output format. Use 'markdown' for human-readable results, 'json' for structured data."
                },
                "language": {
                    "type": "string",
                    "description": "Language code (e.g., en, de, fr). Uses instance default if not specified."
                },
                "time_range": {
                    "type": "string",
                    "enum": ["day", "month", "year"],
                    "description": "Limit results to recency. Only works with engines that support it."
                },
                "categories": {
                    "type": "string",
                    "description": "Comma-separated categories: general, images, videos, news, map, music, it, science, files, social_media"
                },
                "safesearch": {
                    "type": "integer",
                    "enum": [0, 1, 2],
                    "description": "SafeSearch level: 0=none, 1=moderate, 2=strict"
                },
                "engines": {
                    "type": "string",
                    "description": "Comma-separated engine names (e.g., google, bing, duckduckgo)"
                },
                "pageno": {
                    "type": "integer",
                    "default": 1,
                    "description": "Page number for pagination"
                }
            },
            "required": ["query"]
        })
    }

    async fn execute(&self, arguments: &Value) -> Result<Value, ToolError> {
        let query = arguments["query"]
            .as_str()
            .ok_or_else(|| ToolError::ExecutionError("Missing 'query' parameter".to_string()))?;

        let format = arguments["format"].as_str().unwrap_or("markdown");

        let options = SearchOptions {
            language: arguments["language"].as_str().map(String::from),
            time_range: arguments["time_range"].as_str().map(String::from),
            categories: arguments["categories"].as_str().map(String::from),
            safesearch: arguments["safesearch"].as_u64().map(|v| v as u8),
            engines: arguments["engines"].as_str().map(String::from),
            pageno: arguments["pageno"].as_u64().map(|v| v as u32),
        };

        debug!(query, ?options, "Executing search");
        let results = self
            .provider
            .search(query, &options)
            .await
            .with_context(|| format!("Search failed for query: {query}"))
            .map_err(|e| ToolError::ExecutionError(e.to_string()))?;

        debug!(?results, "Search results received");

        let output = if format == "markdown" {
            Value::String(format_results_markdown(query, &results))
        } else {
            serde_json::to_value(results).map_err(|e| ToolError::ExecutionError(e.to_string()))?
        };

        Ok(output)
    }
}

fn format_results_markdown(query: &str, results: &[SearchResult]) -> String {
    if results.is_empty() {
        return format!("## Search Results: \"{}\"\n\nNo results found.", query);
    }

    let mut output = format!("## Search Results: \"{}\"\n\n", query);
    output.push_str(&format!("Found {} result(s)\n\n", results.len()));

    for (i, result) in results.iter().enumerate() {
        output.push_str(&format!("**{}. {}**\n", i + 1, result.title));
        output.push_str(&format!("{}\n", result.url));
        if !result.snippet.is_empty() {
            output.push_str(&format!("{}\n", result.snippet));
        }
        output.push_str("\n---\n\n");
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn test_format_results_markdown_empty() {
        let results = vec![];
        let markdown = format_results_markdown("test query", &results);
        assert!(markdown.contains("No results found"));
    }

    #[test]
    fn test_format_results_markdown_with_results() {
        let results = vec![
            SearchResult {
                title: "Result 1".to_string(),
                url: "https://example.com/1".to_string(),
                snippet: "Snippet 1".to_string(),
            },
            SearchResult {
                title: "Result 2".to_string(),
                url: "https://example.com/2".to_string(),
                snippet: "Snippet 2".to_string(),
            },
        ];
        let markdown = format_results_markdown("test", &results);
        assert!(markdown.contains("Search Results"));
        assert!(markdown.contains("Result 1"));
        assert!(markdown.contains("https://example.com/1"));
        assert!(markdown.contains("Snippet 1"));
    }

    #[test]
    fn test_search_options_merge() {
        let defaults = SearchOptions {
            language: Some("en".to_string()),
            categories: Some("science".to_string()),
            ..Default::default()
        };
        let overrides = SearchOptions {
            time_range: Some("day".to_string()),
            language: Some("de".to_string()),
            ..Default::default()
        };
        let merged = overrides.merge(&defaults);
        assert_eq!(merged.language, Some("de".to_string()));
        assert_eq!(merged.time_range, Some("day".to_string()));
        assert_eq!(merged.categories, Some("science".to_string()));
    }
}
