//! A session is a shared context between a human and the AI assistant.
//! Context includes the conversation, shared artifacts, and tools.
use crate::{
    completion::{CancellationToken, ChatMessage, Completion, CompletionModel, SenderType},
    model::{ModelConfig, ModelMetrics},
    tools::Tool,
};
use anyhow::{Context, Result};
use futures::stream::BoxStream;
use std::{collections::HashMap, sync::Arc};

/// A session with shared context between Human and AI model.
pub struct Session {
    model: Box<dyn CompletionModel + Send + Sync>,
    messages: Vec<ChatMessage>,
    tools: Vec<Arc<dyn Tool>>,
    metrics: Option<ModelMetrics>,
}

impl Session {
    /// Create a new session with the given model configuration
    pub async fn new(model_config: ModelConfig, system_prompt: &str) -> Result<Self> {
        let mut model = crate::get_completion_llm(model_config.clone())
            .context("Failed to initialize session model")?;

        model
            .load(system_prompt)
            .await
            .context("Failed to load model with system prompt")?;

        let metrics = Some(model.metrics());

        Ok(Self {
            model,
            messages: Vec::new(),
            tools: Vec::new(),
            metrics,
        })
    }

    /// Add a new message to the conversation history
    pub fn add_message(&mut self, sender: SenderType, text: &str) {
        self.messages.push(ChatMessage {
            sender,
            text: text.to_string(),
            tools: Vec::new(),
        });
    }

    /// Set the tools available for this session
    pub fn set_tools(&mut self, tools: Vec<Arc<dyn Tool>>) {
        self.tools = tools;
    }

    /// Generate a response stream for the current conversation
    pub async fn generate(
        &mut self,
        settings: HashMap<String, String>,
        cancel_token: CancellationToken,
    ) -> Result<BoxStream<'_, Result<Completion>>> {
        let tool_slice = if self.tools.is_empty() {
            None
        } else {
            Some(self.tools.as_slice())
        };

        Ok(self
            .model
            .complete(&self.messages, tool_slice, &settings, cancel_token)
            .await)
    }

    /// Clear the conversation history
    pub fn clear_history(&mut self) {
        self.messages.clear();
    }

    /// Get model metrics if available
    pub fn metrics(&self) -> Option<&ModelMetrics> {
        self.metrics.as_ref()
    }

    /// Get the last message from the assistant
    pub fn last_assistant_message(&self) -> Option<&ChatMessage> {
        self.messages
            .iter()
            .rev()
            .find(|m| m.sender == SenderType::Assistant)
    }
}
