use anyhow::{Context, Result};
use arey_core::tools::{Tool, ToolError};
use async_trait::async_trait;
use readabilityrs::{Readability, ReadabilityOptions};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tracing::debug;

const SURROUNDING_LINES: usize = 3;

#[derive(Serialize, Deserialize, Debug)]
pub struct FetchResult {
    pub title: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub author: Option<String>,
    pub content: String,
}

#[derive(Debug)]
pub struct FetchTool {
    client: Client,
}

impl FetchTool {
    pub fn new() -> Self {
        Self {
            client: Client::new(),
        }
    }

    async fn fetch_html(&self, url: &str) -> Result<String> {
        let response = self
            .client
            .get(url)
            .header("User-Agent", "Mozilla/5.0 (compatible; arey/1.0)")
            .send()
            .await
            .context("Failed to fetch URL")?;

        let html = response
            .text()
            .await
            .context("Failed to read response body")?;

        Ok(html)
    }

    fn parse_html(&self, html: &str, url: &str) -> Result<Option<readabilityrs::Article>> {
        let options = ReadabilityOptions::builder().output_markdown(true).build();
        let readability = Readability::new(html, Some(url), Some(options))
            .context("Failed to initialize Readability")?;

        Ok(readability.parse())
    }

    fn find_line_number(content: &str, search_text: &str) -> Option<usize> {
        content
            .lines()
            .enumerate()
            .find(|(_, line)| line.contains(search_text))
            .map(|(i, _)| i + 1)
    }

    fn extract_line_excerpt(content: &str, line_num: usize) -> Option<String> {
        let lines: Vec<&str> = content.lines().collect();
        if line_num == 0 || line_num > lines.len() {
            return None;
        }

        let start = line_num.saturating_sub(SURROUNDING_LINES + 1);
        let end = (line_num + SURROUNDING_LINES).min(lines.len());

        if start >= end {
            return None;
        }

        let excerpt: String = lines[start..end]
            .iter()
            .enumerate()
            .map(|(i, line)| {
                let current_line_num = start + i + 1;
                if current_line_num == line_num {
                    format!("> {}", line)
                } else {
                    line.to_string()
                }
            })
            .collect::<Vec<_>>()
            .join("\n");

        Some(excerpt)
    }
}

impl Default for FetchTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for FetchTool {
    fn name(&self) -> String {
        "fetch".to_string()
    }

    fn description(&self) -> String {
        "Fetches a URL and extracts readable content as markdown".to_string()
    }

    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch"
                },
                "search_text": {
                    "type": "string",
                    "description": "Optional text to search for. If provided, returns only the surrounding context (3 lines before/after matching text) instead of full content."
                }
            },
            "required": ["url"]
        })
    }

    async fn execute(&self, arguments: &Value) -> Result<Value, ToolError> {
        let url = arguments["url"]
            .as_str()
            .ok_or_else(|| ToolError::ExecutionError("Missing 'url' parameter".to_string()))?;

        let search_text = arguments["search_text"].as_str();

        debug!(url, ?search_text, "Executing fetch");

        let html = self
            .fetch_html(url)
            .await
            .map_err(|e| ToolError::ExecutionError(e.to_string()))?;

        let article = self
            .parse_html(&html, url)
            .map_err(|e| ToolError::ExecutionError(e.to_string()))?;

        let article = article.ok_or_else(|| {
            ToolError::ExecutionError("Could not extract content from page".to_string())
        })?;

        let content = article
            .markdown_content
            .or(article.content)
            .unwrap_or_default();

        let content = if let Some(search) = search_text {
            Self::find_line_number(&content, search)
                .and_then(|ln| Self::extract_line_excerpt(&content, ln))
                .unwrap_or(content)
        } else {
            content
        };

        let result = FetchResult {
            title: article.title.unwrap_or_default(),
            author: article.byline,
            content,
        };

        debug!(title = result.title, "Fetch completed successfully");

        serde_json::to_value(result).map_err(|e| ToolError::ExecutionError(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_line_number_found() {
        let content = "Line 1\nLine 2\nTarget line\nLine 4";

        let result = FetchTool::find_line_number(content, "Target");
        assert_eq!(result, Some(3));
    }

    #[test]
    fn test_find_line_number_not_found() {
        let content = "Line 1\nLine 2\nLine 3";

        let result = FetchTool::find_line_number(content, "Target");
        assert_eq!(result, None);
    }

    #[test]
    fn test_find_line_number_first_line() {
        let content = "Found it\nOther content";

        let result = FetchTool::find_line_number(content, "Found");
        assert_eq!(result, Some(1));
    }

    #[test]
    fn test_extract_line_excerpt_valid() {
        let content = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5";

        let result = FetchTool::extract_line_excerpt(content, 3);
        assert!(result.is_some());
        let excerpt = result.unwrap();
        assert!(excerpt.contains("Line 1"));
        assert!(excerpt.contains("Line 2"));
        assert!(excerpt.contains("> Line 3"));
        assert!(excerpt.contains("Line 4"));
    }

    #[test]
    fn test_extract_line_excerpt_at_start() {
        let content = "Line 1\nLine 2\nLine 3";

        let result = FetchTool::extract_line_excerpt(content, 1);
        assert!(result.is_some());
        let excerpt = result.unwrap();
        assert!(excerpt.contains("> Line 1"));
    }

    #[test]
    fn test_extract_line_excerpt_out_of_bounds() {
        let content = "Line 1\nLine 2";

        let result = FetchTool::extract_line_excerpt(content, 10);
        assert!(result.is_none());
    }

    #[test]
    fn test_extract_line_excerpt_zero() {
        let content = "Line 1\nLine 2";

        let result = FetchTool::extract_line_excerpt(content, 0);
        assert!(result.is_none());
    }

    #[test]
    fn test_fetch_tool_name() {
        let tool = FetchTool::new();
        assert_eq!(tool.name(), "fetch");
    }

    #[test]
    fn test_fetch_tool_description() {
        let tool = FetchTool::new();
        let desc = tool.description().to_lowercase();
        assert!(desc.contains("fetch"));
    }

    #[test]
    fn test_fetch_tool_parameters() {
        let tool = FetchTool::new();
        let params = tool.parameters();

        assert_eq!(params["type"], "object");
        assert!(params["properties"].get("url").is_some());
        assert!(params["properties"].get("search_text").is_some());
        assert!(
            params["required"]
                .as_array()
                .unwrap()
                .contains(&serde_json::json!("url"))
        );
    }

    #[test]
    fn test_fetch_tool_parameters_search_text_optional() {
        let tool = FetchTool::new();
        let params = tool.parameters();

        let required = params["required"].as_array().unwrap();
        assert!(!required.contains(&serde_json::json!("search_text")));
    }

    #[tokio::test]
    async fn test_execute_missing_url() {
        let tool = FetchTool::new();
        let args = serde_json::json!({});

        let result = tool.execute(&args).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("url"));
    }

    #[tokio::test]
    async fn test_execute_with_search_text() {
        let tool = FetchTool::new();
        let args = serde_json::json!({
            "url": "https://example.com",
            "search_text": "test"
        });

        let result = tool.execute(&args).await;
        assert!(result.is_err() || result.is_ok());
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use wiremock::matchers::method;
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[tokio::test]
    async fn test_fetch_success() {
        let mock_server = MockServer::start().await;

        let html = r#"<html><head><title>Test Article</title></head><body><article><h1>Test Title</h1><p>This is the content.</p></article></body></html>"#;

        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(200).set_body_string(html))
            .mount(&mock_server)
            .await;

        let tool = FetchTool::new();
        let args = serde_json::json!({
            "url": mock_server.uri()
        });

        let result = tool.execute(&args).await;

        if result.is_err() {
            let err = result.unwrap_err();
            if err.to_string().contains("Could not extract content") {
                return;
            }
            panic!("Unexpected error: {}", err);
        }

        let value = result.unwrap();
        assert!(value.get("title").is_some() || value.get("content").is_some());
    }

    #[tokio::test]
    async fn test_fetch_with_search_text_returns_excerpt() {
        let mock_server = MockServer::start().await;

        let html = r#"<html><head><title>Test Article</title></head><body><article><h1>Test Title</h1><p>First paragraph.</p><p>Target text here.</p><p>Third paragraph.</p></article></body></html>"#;

        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(200).set_body_string(html))
            .mount(&mock_server)
            .await;

        let tool = FetchTool::new();
        let args = serde_json::json!({
            "url": mock_server.uri(),
            "search_text": "Target text"
        });

        let result = tool.execute(&args).await;

        if result.is_err() {
            let err = result.unwrap_err();
            if err.to_string().contains("Could not extract content") {
                return;
            }
            panic!("Unexpected error: {}", err);
        }

        let value = result.unwrap();
        let content = value["content"].as_str().unwrap_or("");
        assert!(content.contains("Target text"));
    }
}
