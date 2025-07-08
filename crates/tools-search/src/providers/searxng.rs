use anyhow::{Context, Result, anyhow};
use async_trait::async_trait;
use reqwest::Client;
use serde::Deserialize;
use url::Url;

use crate::{SearchProvider, SearchResult};

// Struct for deserializing the response from the SearxNG API
#[derive(Deserialize, Debug)]
struct SearxngApiResult {
    title: String,
    url: String,
    content: String,
}

#[derive(Deserialize, Debug)]
struct SearxngResponse {
    results: Vec<SearxngApiResult>,
}

#[derive(Deserialize, Debug)]
struct SearxngConfig {
    base_url: String,
}

/// A search provider that uses a SearxNG instance.
#[derive(Debug)]
pub struct SearxngProvider {
    base_url: Url,
    client: Client,
}

impl SearxngProvider {
    /// Creates a new SearxngProvider from a configuration value.
    pub fn from_config(config_value: &serde_yaml::Value) -> Result<Self> {
        let config: SearxngConfig = serde_yaml::from_value(config_value.clone())
            .context("Failed to parse searxng provider config")?;
        Ok(Self {
            base_url: Url::parse(&config.base_url)
                .with_context(|| format!("Invalid base_url for SearxNG: {}", config.base_url))?,
            client: Client::new(),
        })
    }
}

#[async_trait]
impl SearchProvider for SearxngProvider {
    async fn search(&self, query: &str) -> Result<Vec<SearchResult>> {
        let mut url = self.base_url.clone();
        url.set_path("search");
        url.query_pairs_mut()
            .append_pair("q", query)
            .append_pair("format", "json");

        let response = self
            .client
            .get(url)
            .send()
            .await
            .context("Failed to send request to SearxNG API")?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(anyhow!(
                "SearxNG API request failed with status {}: {}",
                status,
                text
            ));
        }

        let searxng_response: SearxngResponse = response
            .json()
            .await
            .context("Failed to parse JSON response from SearxNG API")?;

        let results: Vec<SearchResult> = searxng_response
            .results
            .into_iter()
            .map(|r| SearchResult {
                title: r.title,
                url: r.url,
                snippet: r.content,
            })
            .collect();

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use wiremock::matchers::{method, path, query_param};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn mock_searxng_response() -> serde_json::Value {
        json!({
            "query": "test query",
            "results": [
                {
                    "title": "Test Result 1",
                    "url": "https://example.com/1",
                    "content": "This is the first test result.",
                    "bogus_field": "some value"
                },
                {
                    "title": "Test Result 2",
                    "url": "https://example.com/2",
                    "content": "This is the second test result."
                }
            ]
        })
    }

    #[tokio::test]
    async fn test_searxng_provider_from_config() {
        let config_yaml = r#"
            base_url: "https://example.com"
        "#;
        let config_value: serde_yaml::Value = serde_yaml::from_str(config_yaml).unwrap();
        let provider = SearxngProvider::from_config(&config_value);
        assert!(provider.is_ok());
        assert_eq!(
            provider.unwrap().base_url.to_string(),
            "https://example.com/"
        );
    }

    #[tokio::test]
    async fn test_searxng_provider_from_invalid_config() {
        let config_yaml = r#"
            wrong_key: "https://example.com"
        "#;
        let config_value: serde_yaml::Value = serde_yaml::from_str(config_yaml).unwrap();
        let provider = SearxngProvider::from_config(&config_value);
        assert!(provider.is_err());
        assert!(
            provider
                .unwrap_err()
                .to_string()
                .contains("Failed to parse searxng provider config")
        );
    }

    #[tokio::test]
    async fn test_searxng_provider_search_success() {
        // Arrange
        let server = MockServer::start().await;
        let mock_response = ResponseTemplate::new(200).set_body_json(mock_searxng_response());
        Mock::given(method("GET"))
            .and(path("/search"))
            .and(query_param("q", "test query"))
            .and(query_param("format", "json"))
            .respond_with(mock_response)
            .mount(&server)
            .await;

        let config_yaml = format!("base_url: {}", server.uri());
        let config_value: serde_yaml::Value = serde_yaml::from_str(&config_yaml).unwrap();
        let provider = SearxngProvider::from_config(&config_value).unwrap();

        // Act
        let results = provider.search("test query").await.unwrap();

        // Assert
        assert_eq!(results.len(), 2);
        assert_eq!(
            results[0],
            SearchResult {
                title: "Test Result 1".to_string(),
                url: "https://example.com/1".to_string(),
                snippet: "This is the first test result.".to_string(),
            }
        );
    }

    #[tokio::test]
    async fn test_searxng_provider_api_error() {
        // Arrange
        let server = MockServer::start().await;
        let mock_response = ResponseTemplate::new(500).set_body_string("Internal Server Error");
        Mock::given(method("GET"))
            .and(path("/search"))
            .respond_with(mock_response)
            .mount(&server)
            .await;

        let config_yaml = format!("base_url: {}", server.uri());
        let config_value: serde_yaml::Value = serde_yaml::from_str(&config_yaml).unwrap();
        let provider = SearxngProvider::from_config(&config_value).unwrap();

        // Act
        let result = provider.search("any query").await;

        // Assert
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.to_string()
                .contains("SearxNG API request failed with status 500")
        );
        assert!(err.to_string().contains("Internal Server Error"));
    }
}
