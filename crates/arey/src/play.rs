use anyhow::{Context, Result, anyhow};
use arey_core::{
    completion::{CancellationToken, Completion, CompletionMetrics, SenderType},
    config::{AreyConfigError, Config},
    model::ModelConfig,
    session::Session,
};
use chrono::Local;
use futures::{Stream, StreamExt};
use markdown::{Constructs, ParseOptions, to_mdast};
use serde_yaml::Value;
use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};
use tokio::sync::Mutex;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

/// Result of task execution
pub struct PlayResult {
    pub response: String,
    pub metrics: CompletionMetrics,
    pub finish_reason: Option<String>,
    // pub logs: Option<String>,
}

/// Play file with model settings and prompt
pub struct PlayFile {
    pub file_path: PathBuf,
    pub model_config: Option<ModelConfig>,
    pub model_settings: HashMap<String, Value>,
    pub prompt: String,
    pub completion_profile: HashMap<String, Value>,
    pub output_settings: HashMap<String, String>,
    pub session: Option<Arc<Mutex<Session>>>,
    pub result: Option<PlayResult>,
}

fn get_default_play_file() -> String {
    include_str!("../data/play.md").to_string()
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

    pub async fn ensure_session(&mut self) -> Result<()> {
        if self.session.is_none() {
            let model_config = self.model_config.as_ref().ok_or_else(|| {
                AreyConfigError::Config("Missing model configuration".to_string())
            })?;

            let session = Session::new(model_config.clone(), "").await?;

            // TODO: Apply model_settings overrides
            self.session = Some(Arc::new(Mutex::new(session)));
        }
        Ok(())
    }

    pub async fn generate(&mut self) -> Result<impl Stream<Item = Result<Completion>>> {
        self.ensure_session().await?;
        let session = self.session.clone().unwrap();
        let prompt = self.prompt.clone();
        let completion_profile = self.completion_profile.clone();

        let (tx, rx) = mpsc::channel(4);

        tokio::spawn(async move {
            let mut session_lock = session.lock().await;
            session_lock.add_message(SenderType::User, &prompt);

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

            match session_lock.generate(settings, cancel_token).await {
                Ok(mut stream) => {
                    while let Some(item) = stream.next().await {
                        if tx.send(item).await.is_err() {
                            break;
                        }
                    }
                }
                Err(e) => {
                    let _ = tx.send(Err(e)).await;
                }
            }
        });

        Ok(ReceiverStream::new(rx))
    }
}

#[cfg(test)]
mod test {
    use std::io::Write;

    use arey_core::config::get_config;

    use super::*;
    use serde_yaml::Value;
    use tempfile::{NamedTempFile, tempdir};

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

    const DUMMY_CONFIG_CONTENT: &str = r#"
models:
  dummy-7b:
    name: dummy-7b
    type: gguf
    n_ctx: 4096
    path: /path/to/dummy_model.gguf
profiles:
  default:
    temperature: 0.7
    repeat_penalty: 1.176
    top_k: 40
    top_p: 0.1
chat:
  model: dummy-7b
  profile: default
task:
  model: dummy-7b
  profile: default
"#;

    fn create_temp_config() -> Result<Config> {
        let mut config_file = NamedTempFile::new()?;
        write!(config_file, "{DUMMY_CONFIG_CONTENT}")?;

        get_config(Some(config_file.path().to_path_buf()))
            .map_err(|e| anyhow!("Failed to create temp config file. Error {}", e))
    }

    #[test]
    fn test_play_file_creation() -> Result<()> {
        let temp_dir = tempdir()?;
        let file_path = temp_dir.path().join("test_play.md");
        fs::write(&file_path, SAMPLE_PLAY_CONTENT)?;

        let config = create_temp_config()?;

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
}
