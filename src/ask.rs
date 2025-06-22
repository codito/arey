use anyhow::{Context, Result};
use futures::stream::{BoxStream, StreamExt};
use std::collections::HashMap;
use std::io::Write;

use crate::core::completion::{
    CancellationToken, ChatMessage, Completion, CompletionModel, SenderType,
};
use crate::core::model::{ModelConfig, ModelMetrics};
use crate::platform::console::{style_text, MessageType};
use crate::platform::llm::get_completion_llm;

pub struct Task {
    instruction: String,
    model_config: ModelConfig,
    overrides: Option<HashMap<String, serde_yaml::Value>>,
    model: Option<Box<dyn CompletionModel + Send + Sync>>,
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

    pub async fn load_model(&mut self) -> Result<ModelMetrics> {
        let mut config = self.model_config.clone();

        // Apply overrides to model configuration
        if let Some(overrides) = &self.overrides {
            for (key, value) in overrides {
                config.settings.insert(key.clone(), value.clone());
            }
        }

        let mut model = get_completion_llm(config.clone()).context("Failed to initialize model")?;

        // Load empty system prompt for tasks
        model
            .load("")
            .await
            .context("Failed to load model with system prompt")?;

        let metrics = model.metrics();
        self.model = Some(model);
        Ok(metrics)
    }

    pub async fn run(&mut self) -> Result<BoxStream<'_, Result<Completion>>> {
        let message = ChatMessage {
            sender: SenderType::User,
            text: self.instruction.clone(),
        };

        let settings = HashMap::new(); // Use default settings for now
        let cancel_token = CancellationToken::new();

        let model = self
            .model
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("Model not loaded"))?;
        let stream = model.complete(&[message], &settings, cancel_token).await;

        Ok(stream)
    }
}

/// Run the ask command with given instruction and overrides
pub async fn run_ask(
    instruction: &str,
    model_config: ModelConfig,
    overrides_file: Option<&str>,
) -> Result<()> {
    let mut task = Task::new(
        instruction.to_string(),
        model_config,
        parse_overrides_file(overrides_file).await?,
    );

    println!("Loading model...");
    let model_metrics = task.load_model().await?;
    println!("Model loaded in {:.2}ms", model_metrics.init_latency_ms);

    println!("Generating response...");
    let mut stream = task.run().await?;
    
    // Collect the response and metrics
    let mut response = String::new();
    let mut metrics = CompletionMetrics::default();
    let mut finish_reason = None;

    while let Some(result) = stream.next().await {
        match result? {
            Completion::Response(r) => {
                if let Some(reason) = r.finish_reason {
                    finish_reason = Some(reason);
                }
                response.push_str(&r.text);
            }
            Completion::Metrics(m) => metrics = m,
        }
    }

    // Print the full response
    println!("{}", response);
    
    // Calculate tokens per second and prepare footer
    let tokens_per_sec = if metrics.completion_latency_ms > 0.0 {
        metrics.completion_tokens as f32 * 1000.0 / metrics.completion_latency_ms
    } else {
        0.0
    };

    let mut footer_complete = String::from("◼ Completed");
    if let Some(reason) = finish_reason {
        footer_complete.push_str(&format!(" ({reason})"));
    }

    println!(
        "{}",
        style_text(
            &format!("{footer_complete}. {:.2} tokens/s.", tokens_per_sec),
            MessageType::Footer,
        )
    );

    Ok(())
}

/// Parse YAML overrides file into settings map
async fn parse_overrides_file(
    path: Option<&str>,
) -> Result<Option<HashMap<String, serde_yaml::Value>>> {
    let Some(path) = path else {
        return Ok(None);
    };

    let content = tokio::fs::read_to_string(path)
        .await
        .context("Failed to read overrides file")?;

    let overrides: HashMap<String, serde_yaml::Value> =
        serde_yaml::from_str(&content).context("Failed to parse overrides YAML")?;

    Ok(Some(overrides))
}
