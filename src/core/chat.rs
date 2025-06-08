use crate::core::completion::{ChatMessage, CompletionMetrics, CompletionResponse, SenderType};
use crate::core::model::{ModelConfig, ModelMetrics};
use anyhow::{Context, Result};
use futures::stream::BoxStream;
use std::collections::HashMap;
use tokio::sync::Mutex;
use std::sync::Arc;

/// Context associated with a single chat message
pub struct MessageContext {
    pub prompt: String,
    pub finish_reason: Option<String>,
    pub metrics: CompletionMetrics,
    pub logs: String,
}

/// Chat message with context
pub struct Message {
    pub text: String,
    pub sender: SenderType,
    pub timestamp: i64,
    pub context: Option<MessageContext>,
}

impl Message {
    pub fn to_chat_message(&self) -> ChatMessage {
        ChatMessage {
            text: self.text.clone(),
            sender: self.sender.clone(),
        }
    }
}

/// Context associated with a chat
pub struct ChatContext {
    pub metrics: Option<ModelMetrics>,
    pub logs: String,
}

/// Chat conversation between human and AI model
pub struct Chat {
    pub messages: Vec<Message>,
    pub context: ChatContext,
    model_config: ModelConfig,
    model: Arc<Mutex<Box<dyn crate::core::completion::CompletionModel + Send + Sync>>>,
}

impl Chat {
    pub async fn new(model_config: ModelConfig) -> Result<Self> {
        let model_arc = crate::platform::llm::get_completion_llm(model_config.clone())
            .context("Failed to initialize chat model")?;
        
        let mut model_lock = model_arc.lock().await;
        // Load system prompt - leave empty for now
        model_lock.load("").await?;
        let metrics = model_lock.metrics();

        Ok(Self {
            messages: Vec::new(),
            context: ChatContext {
                metrics: Some(metrics),
                logs: String::new(),
            },
            model_config,
            model: model_arc,
        })
    }

    pub async fn stream_response(
        &self,
        message: String,
    ) -> Result<BoxStream<'_, Result<CompletionResponse>>> {
        // Create user message
        let user_message = Message {
            text: message.clone(),
            sender: SenderType::User,
            timestamp: chrono::Utc::now().timestamp(),
            context: None,
        };

        // Add message to history (will leave storage for actual implementation)
        // In full implementation, you'd push this to self.messages

        // Convert all messages to model format
        let mut model_messages = vec![];
        for msg in &self.messages {
            model_messages.push(msg.to_chat_message());
        }
        model_messages.push(user_message.to_chat_message());

        let mut model = self.model.lock().await;
        let stream = model
            .complete(&model_messages, &HashMap::new())
            .await;
        
        Ok(stream)
    }

    // Placeholder for actual implementation
    pub fn get_completion_metrics(&self) -> Option<CompletionMetrics> {
        None
    }
}
