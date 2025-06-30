use anyhow::{Context, Result};
use futures::stream::{BoxStream, StreamExt};
use std::collections::HashMap;
use std::io::Write;

use crate::console::{MessageType, format_footer_metrics, style_text};
use arey_core::completion::{
    CancellationToken, ChatMessage, Completion, CompletionMetrics, CompletionModel, SenderType,
};
use arey_core::get_completion_llm;
use arey_core::model::{ModelConfig, ModelMetrics};

pub struct Task {
    instruction: String,
    model_config: ModelConfig,
    model: Option<Box<dyn CompletionModel + Send + Sync>>,
}

impl Task {
    pub fn new(instruction: String, model_config: ModelConfig) -> Self {
        Self {
            instruction,
            model_config,
            model: None,
        }
    }

    pub async fn load_model(&mut self) -> Result<ModelMetrics> {
        let config = self.model_config.clone();
        let mut model = get_completion_llm(config).context("Failed to initialize model")?;

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
        let stream = model
            .complete(&[message], None, &settings, cancel_token)
            .await;

        Ok(stream)
    }
}

/// Run the ask command with given instruction and overrides
pub async fn run_ask(instruction: &str, model_config: ModelConfig) -> Result<()> {
    let mut task = Task::new(instruction.to_string(), model_config);

    println!("Loading model...");
    let model_metrics = task.load_model().await?;
    println!("Model loaded in {:.2}ms", model_metrics.init_latency_ms);

    println!("Generating response...");
    let mut stream = task.run().await?;

    // Collect the response and metrics
    let mut metrics = CompletionMetrics::default();
    let mut finish_reason = None;

    while let Some(result) = stream.next().await {
        match result? {
            Completion::Response(r) => {
                if let Some(reason) = r.finish_reason {
                    finish_reason = Some(reason);
                }
                print!("{}", r.text);
                std::io::stdout().flush()?;
            }
            Completion::Metrics(m) => metrics = m,
        }
    }

    let footer = format_footer_metrics(&metrics, finish_reason.as_deref(), false);
    println!();
    println!();
    println!("{}", style_text(&footer, MessageType::Footer));

    Ok(())
}
