use std::collections::HashMap;
use std::sync::Arc;
use anyhow::{Context, Result};
use tokio::sync::Mutex;
use futures::{Stream, StreamExt};

use crate::core::completion::{ChatMessage, Completion, CompletionMetrics, CompletionModel, SenderType, CancellationToken};
use crate::core::config::Config;
use crate::core::model::ModelConfig;
use crate::platform::llm::get_completion_llm;

pub struct Task {
    instruction: String,
    model_config: ModelConfig,
    overrides: Option<HashMap<String, serde_yaml::Value>>,
    model: Option<Arc<Mutex<Box<dyn CompletionModel + Send + Sync>>>>,
}

impl Task {
    pub fn new(
        instruction: String,
        model_config: ModelConfig,
        overrides: Option<HashMap<String, serde_yaml::Value>>,
    ) -> Self {
        Self {
            instruction,
            model_config,
            overrides,
            model: None,
        }
    }

    pub async fn load_model(&mut self) -> Result<CompletionMetrics> {
        let mut config = self.model_config.clone();
        
        // Apply overrides to model configuration
        if let Some(overrides) = &self.overrides {
            for (key, value) in overrides {
                config.settings.insert(key.clone(), value.clone());
            }
        }

        let model = get_completion_llm(config.clone())
            .await
            .context("Failed to initialize model")?;

        // Load empty system prompt for tasks
        model
            .lock().await
            .load("")
            .await
            .context("Failed to load model with system prompt")?;

        let metrics = model.lock().await.metrics();
        self.model = Some(model);
        Ok(metrics)
    }

    pub async fn run(&mut self) -> Result<impl Stream<Item = Result<Completion>>> {
        let model_lock = self.model
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Model not loaded"))?
            .clone();

        let message = ChatMessage {
            sender: SenderType::User,
            text: self.instruction.clone(),
        };

        let settings = HashMap::new(); // Use default settings for now
        let cancel_token = CancellationToken::new();

        let mut model = model_lock.lock().await;
        let stream = model.complete(&[message], &settings, cancel_token).await?;
        Ok(stream)
    }
}

/// Run the ask command with given instruction and overrides
pub async fn run_ask(
    instruction: &str,
    config: &Config,
    overrides_file: Option<&str>
) -> Result<()> {
    let mut task = Task::new(
        instruction.to_string(),
        config.task.model.clone(),
        parse_overrides_file(overrides_file).await?
    );

    println!("Loading model...");
    let metrics = task.load_model().await?;
    println!("Model loaded in {:.2}ms", metrics.init_latency_ms);

    println!("Generating response...");
    let mut stream = task.run().await?;
    
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        print!("{}", chunk.text);
        std::io::stdout().flush()?;
    }
    println!();

    Ok(())
}

/// Parse YAML overrides file into settings map
async fn parse_overrides_file(
    path: Option<&str>
) -> Result<Option<HashMap<String, serde_yaml::Value>>> {
    let Some(path) = path else {
        return Ok(None);
    };

    let content = tokio::fs::read_to_string(path)
        .await
        .context("Failed to read overrides file")?;

    let overrides: HashMap<String, serde_yaml::Value> = serde_yaml::from_str(&content)
        .context("Failed to parse overrides YAML")?;

    Ok(Some(overrides))
}
