use anyhow::{Context, Result, anyhow};
use arey_core::{
    completion::{CancellationToken, ChatMessage, Completion, CompletionMetrics, SenderType},
    config::{AreyConfigError, Config},
    model::ModelConfig,
    session::Session,
};
use chrono::Local;
#[allow(unused_imports)]
use futures::{Stream, StreamExt};
use markdown::{Constructs, ParseOptions, to_mdast};
use serde_yaml::Value;
use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
};

/// Holds the result of a `PlayFile` execution.
pub struct PlayResult {
    pub response: String,
    pub metrics: CompletionMetrics,
    pub finish_reason: Option<String>,
    // pub logs: Option<String>,
}

/// Represents a markdown file with frontmatter for configuration and a prompt.
pub struct PlayFile {
    pub file_path: PathBuf,
    pub model_config: Option<ModelConfig>,
    pub model_settings: HashMap<String, Value>,
    pub prompt: String,
    pub completion_profile: HashMap<String, Value>,
    pub output_settings: HashMap<String, String>,
    pub session: Option<Session>,
    pub result: Option<PlayResult>,
}

fn get_default_play_file() -> String {
    include_str!("../../data/play.md").to_string()
}

fn extract_frontmatter(content: &str) -> Result<(Option<Value>, String)> {
    let parse_opt = ParseOptions {
        constructs: Constructs {
            frontmatter: true,
            ..Constructs::default()
        },
        ..ParseOptions::gfm()
    };
    let ast = to_mdast(content, &parse_opt).map_err(|e| anyhow!("Markdown parse error: {e}"))?;

    let mut parsed_yaml = None;
    let mut body = String::new();

    if let markdown::mdast::Node::Root(root) = &ast {
        for node in root.children.iter() {
            if let markdown::mdast::Node::Yaml(yaml) = node {
                let yaml_value: Value = serde_yaml::from_str(&yaml.value)
                    .map_err(|e| anyhow!("YAML parse error: {e}"))?;
                parsed_yaml = Some(yaml_value);
            } else {
                body.push_str(&format!("{}\n", node.to_string()));
            }
        }
    }

    // If no YAML frontmatter was found, use the entire content as body
    if parsed_yaml.is_none() {
        body = content.to_string();
    }
    Ok((parsed_yaml, body.trim().to_owned()))
}

impl PlayFile {
    /// Parses a play file from the given path.
    ///
    /// It reads the file, extracts frontmatter for configuration, and the rest of the content
    /// as the prompt.
    pub fn new(file_path: impl AsRef<Path>, config: &Config) -> Result<Self> {
        let file_path = file_path.as_ref().to_path_buf();
        let content = fs::read_to_string(&file_path)
            .with_context(|| format!("Failed to read play file: {}", file_path.display()))?;

        let (metadata, prompt_str) = extract_frontmatter(&content)?;
        let metadata = metadata.ok_or_else(|| anyhow!("No frontmatter found"))?;

        let model_name = metadata
            .get("model")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow!("Missing 'model' in metadata"))?;

        let model_config = config.models.get(model_name).cloned();

        Ok(Self {
            file_path,
            model_config,
            model_settings: metadata
                .get("settings")
                .and_then(Value::as_mapping)
                .map(|m| {
                    m.iter()
                        .map(|(k, v)| (k.as_str().unwrap().to_owned(), v.clone()))
                        .collect()
                })
                .unwrap_or_default(),
            prompt: prompt_str,
            completion_profile: metadata
                .get("profile")
                .and_then(Value::as_mapping)
                .map(|m| {
                    m.iter()
                        .map(|(k, v)| (k.as_str().unwrap().to_owned(), v.clone()))
                        .collect()
                })
                .unwrap_or_default(),
            output_settings: metadata
                .get("output")
                .and_then(Value::as_mapping)
                .map(|m| {
                    m.iter()
                        .filter_map(|(k, v)| {
                            v.as_str()
                                .map(|s| (k.as_str().unwrap().to_owned(), s.to_owned()))
                        })
                        .collect()
                })
                .unwrap_or_default(),
            session: None,
            result: None,
        })
    }

    /// Creates a new play file from a template if it doesn't exist.
    ///
    /// If `file_path` is `Some` and the path does not exist, it creates the file.
    /// If `file_path` is `None`, it creates a temporary play file.
    /// It returns the path to the created file.
    pub fn create_missing(file_path: Option<&Path>) -> Result<PathBuf> {
        let tmp_file = match file_path {
            Some(p) if p.exists() => p.to_path_buf(),
            Some(p) => {
                fs::write(p, get_default_play_file()).context("Failed to create play file")?;
                p.to_path_buf()
            }
            None => {
                let temp_name = format!("arey_play_{}.md", Local::now().format("%Y%m%d_%H%M%S"));
                let temp_path = std::env::temp_dir().join(temp_name);
                fs::write(&temp_path, get_default_play_file())?;
                temp_path
            }
        };
        Ok(tmp_file)
    }

    /// Ensures that a chat session is initialized for the `PlayFile`.
    ///
    /// If a session does not already exist, it creates one using the model configuration
    /// from the play file.
    pub async fn ensure_session(&mut self) -> Result<()> {
        if self.session.is_none() {
            let model_config = self.model_config.as_ref().ok_or_else(|| {
                AreyConfigError::Config("Missing model configuration".to_string())
            })?;

            let session = Session::new(model_config.clone(), "")?;

            // TODO: Apply model_settings overrides
            self.session = Some(session);
        }
        Ok(())
    }

    /// Generates a response based on the prompt in the play file.
    ///
    /// It returns a stream of `Completion` events.
    pub async fn generate(&mut self) -> Result<impl Stream<Item = Result<Completion>>> {
        self.ensure_session().await?;
        let session = self.session.as_mut().unwrap();
        let prompt = self.prompt.clone();
        let completion_profile = self.completion_profile.clone();

        // Add the message first
        session.add_message(ChatMessage {
            sender: SenderType::User,
            text: prompt,
            ..Default::default()
        })?;

        let settings: HashMap<String, String> = completion_profile
            .iter()
            .filter_map(|(k, v)| {
                v.as_str()
                    .map(|s| (k.clone(), s.to_owned()))
                    .or_else(|| v.as_bool().map(|b| (k.clone(), b.to_string())))
                    .or_else(|| v.as_i64().map(|n| (k.clone(), n.to_string())))
                    .or_else(|| v.as_f64().map(|f| (k.clone(), f.to_string())))
            })
            .collect();
        let cancel_token = CancellationToken::new();

        let stream = session.generate(settings, cancel_token).await?;

        Ok(stream)
    }
}

#[cfg(test)]
mod test {
    use arey_core::config::get_config;
    use serde_json::json;
    use wiremock::{
        Mock, MockServer, ResponseTemplate,
        matchers::{method, path},
    };

    use super::*;
    use tempfile::tempdir;

    const SAMPLE_PLAY_CONTENT: &str = r#"---
model: test-model
settings:
  n_ctx: 4096
profile:
  temperature: 0.7
output:
  format: plain
---
Test prompt
"#;

    fn mock_event_stream_body() -> String {
        let events = [
            json!({
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "created": 1684,
                "model": "gpt-3.5-turbo",
                "choices": [{
                    "delta": {"content": "Hello"},
                    "index": 0,
                    "finish_reason": null
                }]
            }),
            json!({
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "created": 1684,
                "model": "gpt-3.5-turbo",
                "choices": [{
                    "delta": {"content": " world!"},
                    "index": 0,
                    "finish_reason": "stop"
                }]
            }),
        ];
        let mut body: String = events.iter().map(|e| format!("data: {e}\n\n")).collect();
        body.push_str("data: [DONE]\n\n");
        body
    }

    use crate::test_utils::create_temp_config_file;

    async fn get_test_config(server: &MockServer) -> Result<Config> {
        let config_file = create_temp_config_file(&server.uri());
        get_config(Some(config_file.path().to_path_buf()))
            .map_err(|e| anyhow!("Failed to create temp config file. Error {}", e))
    }

    #[tokio::test]
    async fn test_play_file_creation() -> Result<()> {
        let server = MockServer::start().await;
        let temp_dir = tempdir()?;
        let file_path = temp_dir.path().join("test_play.md");
        fs::write(&file_path, SAMPLE_PLAY_CONTENT)?;

        let config = get_test_config(&server).await?;

        let play_file = PlayFile::new(&file_path, &config)?;

        assert_eq!(play_file.file_path, file_path);
        assert_eq!(play_file.prompt, "Test prompt");

        match &play_file.model_settings["n_ctx"] {
            Value::Number(n) => assert_eq!(n.as_i64(), Some(4096)),
            _ => panic!("n_ctx should be a number"),
        }

        match &play_file.completion_profile["temperature"] {
            Value::Number(n) => assert!((n.as_f64().unwrap() - 0.7).abs() < f64::EPSILON),
            _ => panic!("temperature should be a number"),
        }
        assert_eq!(play_file.output_settings["format"], "plain");
        assert!(play_file.result.is_none());

        Ok(())
    }

    #[test]
    fn test_missing_file_creation() -> Result<()> {
        let temp_dir = tempdir()?;
        let file_path = temp_dir.path().join("missing_play.md");

        // File shouldn't exist initially
        assert!(!file_path.exists());

        // Create file
        let created_path = PlayFile::create_missing(Some(&file_path))?;
        assert_eq!(created_path, file_path);
        assert!(file_path.exists());

        // Verify content
        let content = fs::read_to_string(&file_path)?;
        assert!(!content.is_empty());
        assert!(content.contains("model:"));
        assert!(content.contains("profile:"));

        Ok(())
    }

    #[test]
    fn test_temp_file_creation() -> Result<()> {
        let created_path = PlayFile::create_missing(None)?;

        assert!(created_path.exists());

        // Verify filename pattern
        let file_name = created_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default();
        assert!(file_name.starts_with("arey_play"));
        assert!(file_name.ends_with(".md"));

        Ok(())
    }

    #[tokio::test]
    async fn test_ensure_session() -> Result<()> {
        let server = MockServer::start().await;
        let temp_dir = tempdir()?;
        let file_path = temp_dir.path().join("test_play.md");
        fs::write(&file_path, SAMPLE_PLAY_CONTENT)?;

        let config = get_test_config(&server).await?;
        let mut play_file = PlayFile::new(&file_path, &config)?;

        assert!(play_file.session.is_none());
        play_file.ensure_session().await?;
        assert!(play_file.session.is_some());

        // Should not fail on second call
        play_file.ensure_session().await?;

        Ok(())
    }

    #[tokio::test]
    async fn test_generate() -> Result<()> {
        let server = MockServer::start().await;
        let response = ResponseTemplate::new(200)
            .set_body_bytes(mock_event_stream_body().as_bytes())
            .insert_header("Content-Type", "text/event-stream");
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(response)
            .mount(&server)
            .await;

        let temp_dir = tempdir()?;
        let file_path = temp_dir.path().join("test_play.md");
        fs::write(&file_path, SAMPLE_PLAY_CONTENT)?;

        let config = get_test_config(&server).await?;
        let mut play_file = PlayFile::new(&file_path, &config)?;

        let mut stream = play_file.generate().await?;
        let mut response_text = String::new();
        while let Some(result) = stream.next().await {
            if let Completion::Response(response) = result? {
                response_text.push_str(&response.text);
            }
        }

        assert_eq!(response_text, "Hello world!");

        Ok(())
    }
}
