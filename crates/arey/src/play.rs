use crate::{
    console::{MessageType, format_footer_metrics, style_text},
    play::watch::watch_file,
};
use anyhow::{Context, Result, anyhow};
use arey_core::{
    completion::{
        CancellationToken, ChatMessage, Completion, CompletionMetrics, CompletionModel, SenderType,
    },
    config::{AreyConfigError, Config},
    model::{ModelConfig, ModelMetrics},
};
use chrono::Local;
use futures::StreamExt;
use markdown::{Constructs, ParseOptions, to_mdast};
use notify::Event;
use serde_yaml::Value;
use std::{
    collections::HashMap,
    fs,
    io::Write,
    path::{Path, PathBuf},
};
use tokio::sync::Mutex;

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
    pub model: Option<Mutex<Box<dyn CompletionModel + Send + Sync>>>,
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
            model: None,
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

    pub async fn load_model(&mut self) -> Result<ModelMetrics> {
        let model_config = self
            .model_config
            .as_ref()
            .ok_or_else(|| AreyConfigError::Config("Missing model configuration".to_string()))?;

        let mut config = model_config.clone();
        for (key, value) in &self.model_settings {
            config.settings.insert(key.clone(), value.clone());
        }

        let mut model = arey_core::get_completion_llm(config.clone())
            .map_err(|e| anyhow!("Model init error: {e}"))?;
        model.load("").await?;

        let metrics = model.metrics().clone();
        self.model = Some(Mutex::new(model));
        Ok(metrics)
    }

    pub async fn get_response(&mut self) -> Result<()> {
        let model_lock = self
            .model
            .as_ref()
            .ok_or_else(|| anyhow!("Model not loaded"))?;

        let messages = vec![ChatMessage {
            sender: SenderType::User,
            text: self.prompt.clone(),
        }];

        let settings: HashMap<String, String> = self
            .completion_profile
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

        let mut model_guard = model_lock.lock().await;
        let mut stream = model_guard
            .complete(&messages, None, &settings, cancel_token)
            .await;

        let mut text = String::new();
        let mut finish_reason = None;
        let mut metrics = CompletionMetrics::default();

        while let Some(chunk) = stream.next().await {
            match chunk? {
                Completion::Response(response) => {
                    text.push_str(&response.text);
                    finish_reason = response.finish_reason;

                    print!("{}", response.text);
                    std::io::stdout().flush()?;
                }
                Completion::Metrics(usage) => metrics = usage,
            }
        }

        self.result = Some(PlayResult {
            response: text,
            metrics,
            finish_reason,
            // logs: None,
        });

        Ok(())
    }
}

pub async fn run_play(play_file: &mut PlayFile, config: &Config, no_watch: bool) -> Result<()> {
    println!(
        "{}",
        style_text(
            "Welcome to arey play! Edit the play file below in your favorite editor and I'll generate a response for you. Use `Ctrl+C` to abort play session.",
            MessageType::Footer
        )
    );
    println!();

    if no_watch {
        run_once(play_file).await?;
        return Ok(());
    }

    // Watch the file for changes and rerun in loop
    let file_path = play_file.file_path.clone();

    println!(
        "{} `{}`",
        style_text("Watching", MessageType::Footer),
        file_path.display()
    );

    let (mut _watcher, mut rx) = watch_file(&file_path).await?;

    loop {
        tokio::select! {
            Some(_event) = rx.recv() => {
                println!();
                println!(
                    "{}",
                    style_text(&format!("[{}] File modified, re-generating...", Local::now().format("%Y-%m-%d %H:%M:%S")), MessageType::Footer)
                );
                println!();

                // Reload file content
                match PlayFile::new(&file_path, config) {
                    Ok(mut new_play_file) => {
                        // Reuse existing model if configuration hasn't changed
                        if play_file.model_config == new_play_file.model_config &&
                            play_file.model_settings == new_play_file.model_settings {

                            new_play_file.model = play_file.model.take();
                        }
                        *play_file = new_play_file;

                        run_once(play_file).await?;
                    }
                    Err(e) => {
                        println!("{}", style_text(&format!("Error reloading file: {e}"), MessageType::Error));
                    }
                }
                println!();
                println!(
                    "{} `{}`",
                    style_text("Watching", MessageType::Footer),
                    file_path.display()
                );
            }
            _ = tokio::signal::ctrl_c() => {
                break;
            }
        }
    }

    Ok(())
}

async fn run_once(play_file: &mut PlayFile) -> Result<()> {
    if play_file.model.is_none() {
        let metrics = play_file.load_model().await?;
        println!(
            "{} {}",
            style_text("✓ Model loaded.", MessageType::Footer),
            style_text(
                &format!("{:.2}s", metrics.init_latency_ms / 1000.0),
                MessageType::Footer
            )
        );
        println!();
    }

    play_file.get_response().await?;

    if let Some(result) = &play_file.result {
        let _ = match play_file.output_settings.get("format").map(|s| s.as_str()) {
            Some("plain") => &result.response,
            _ => &result.response,
        };

        let footer = format_footer_metrics(&result.metrics, result.finish_reason.as_deref(), false);
        println!();
        println!();
        println!("{}", style_text(&footer, MessageType::Footer));
    }

    Ok(())
}

pub mod watch {
    use super::*;
    use notify::{
        Config, EventKind, RecommendedWatcher, RecursiveMode, Watcher,
        event::{MetadataKind, ModifyKind},
    };
    use tokio::sync::mpsc;

    pub async fn watch_file(
        path: &Path,
    ) -> anyhow::Result<(
        RecommendedWatcher,
        mpsc::Receiver<anyhow::Result<notify::Event>>,
    )> {
        let (tx, rx) = mpsc::channel::<Result<Event>>(128);
        let mut watcher = RecommendedWatcher::new(
            move |res: notify::Result<Event>| {
                let tx = tx.clone();
                if let Ok(event) = res {
                    if matches!(
                        event.kind,
                        EventKind::Modify(ModifyKind::Metadata(MetadataKind::Any))
                    ) && tx.blocking_send(Ok(event)).is_err()
                    {
                        // Receiver closed
                    }
                }
            },
            Config::default(),
        )?;

        watcher.watch(path, RecursiveMode::NonRecursive)?;
        Ok((watcher, rx))
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
        assert!(play_file.model.is_none());
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
