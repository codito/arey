use anyhow::{Context, Result, anyhow};
use async_trait::async_trait;
use reqwest::Client;
use serde::Deserialize;
use tracing::debug;
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

#[derive(Deserialize, Debug, Default)]
struct SearxngConfig {
    base_url: String,
    #[serde(default)]
    default_language: Option<String>,
    #[serde(default)]
    default_categories: Option<String>,
    #[serde(default)]
    default_results: Option<usize>,
}

#[derive(Debug, Clone, Default)]
pub struct SearchOptions {
    pub language: Option<String>,
    pub time_range: Option<String>,
    pub categories: Option<String>,
    pub safesearch: Option<u8>,
    pub engines: Option<String>,
    pub pageno: Option<u32>,
}

impl SearchOptions {
    pub fn merge(&self, defaults: &SearchOptions) -> Self {
        Self {
            language: self.language.clone().or(defaults.language.clone()),
            time_range: self.time_range.clone().or(defaults.time_range.clone()),
            categories: self.categories.clone().or(defaults.categories.clone()),
            safesearch: self.safesearch.or(defaults.safesearch),
            engines: self.engines.clone().or(defaults.engines.clone()),
            pageno: self.pageno.or(defaults.pageno),
        }
    }
}

/// A search provider that uses a SearxNG instance.
#[derive(Debug)]
pub struct SearxngProvider {
    base_url: Url,
    client: Client,
    default_options: SearchOptions,
    default_results: usize,
}

impl SearxngProvider {
    /// Creates a new SearxngProvider from a configuration value.
    pub fn from_config(config_value: &serde_yaml::Value) -> Result<Self> {
        debug!("Creating SearxngProvider from config: {:?}", config_value);
        let config: SearxngConfig = serde_yaml::from_value(config_value.clone())
            .context("Failed to parse searxng provider config")?;

        let default_options = SearchOptions {
            language: config.default_language,
            categories: config.default_categories,
            ..Default::default()
        };

        Ok(Self {
            base_url: Url::parse(&config.base_url)
                .with_context(|| format!("Invalid base_url for SearxNG: {}", config.base_url))?,
            client: Client::new(),
            default_options,
            default_results: config.default_results.unwrap_or(10),
        })
    }

    pub fn with_options(&self, options: SearchOptions) -> SearchOptions {
        options.merge(&self.default_options)
    }
}

#[async_trait]
impl SearchProvider for SearxngProvider {
    async fn search(&self, query: &str, options: &SearchOptions) -> Result<Vec<SearchResult>> {
        let mut url = self.base_url.clone();
        url.set_path("search");

        let merged = options.merge(&self.default_options);

        let mut query_params: Vec<(&str, String)> =
            vec![("q", query.to_string()), ("format", "json".to_string())];

        if let Some(ref lang) = merged.language {
            query_params.push(("language", lang.clone()));
        }
        if let Some(ref time) = merged.time_range {
            query_params.push(("time_range", time.clone()));
        }
        if let Some(ref cats) = merged.categories {
            query_params.push(("categories", cats.clone()));
        }
        if let Some(safe) = merged.safesearch {
            query_params.push(("safesearch", safe.to_string()));
        }
        if let Some(ref eng) = merged.engines {
            query_params.push(("engines", eng.clone()));
        }
        if let Some(page) = merged.pageno {
            query_params.push(("pageno", page.to_string()));
        }

        for (key, value) in query_params {
            url.query_pairs_mut().append_pair(key, &value);
        }

        let url_for_debug = url.clone();
        debug!(url = %url_for_debug, "Sending request to SearxNG");

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

        debug!(response = ?searxng_response, "Received response from SearxNG");

        let results: Vec<SearchResult> = searxng_response
            .results
            .into_iter()
            .take(self.default_results)
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

        let results = provider
            .search("test query", &SearchOptions::default())
            .await
            .unwrap();

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

        let result = provider
            .search("any query", &SearchOptions::default())
            .await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.to_string()
                .contains("SearxNG API request failed with status 500")
        );
        assert!(err.to_string().contains("Internal Server Error"));
    }
}
