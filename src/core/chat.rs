use crate::core::completion::{
    ChatMessage, CompletionMetrics, CompletionModel, CompletionResponse, SenderType,
};
use crate::core::model::{ModelConfig, ModelMetrics};
use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use futures::stream::{BoxStream, StreamExt};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

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
    pub timestamp: DateTime<Utc>,
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
    pub messages: Arc<Mutex<Vec<Message>>>,
    pub context: ChatContext,
    model_config: ModelConfig,
    model: Box<dyn CompletionModel + Send + Sync>,
}

impl Chat {
    pub async fn new(model_config: ModelConfig) -> Result<Self> {
        let model = crate::platform::llm::get_completion_llm(model_config.clone())
            .context("Failed to initialize chat model")?;
        let metrics = model.metrics();
        Ok(Self {
            messages: Arc::new(Mutex::new(Vec::new())),
            context: ChatContext {
                metrics: Some(metrics),
                logs: String::new(),
            },
            model_config,
            model,
        })
    }

    pub async fn stream_response(
        &mut self,
        message: String,
    ) -> Result<BoxStream<'_, Result<CompletionResponse>>> {
        let timestamp = Utc::now();

        // Create user message and add to history
        let user_message = Message {
            text: message.clone(),
            sender: SenderType::User,
            timestamp,
            context: None,
        };

        let mut messages_lock = self.messages.lock().unwrap();
        messages_lock.push(user_message);

        // Add assistant message placeholder
        let assistant_index = messages_lock.len();
        messages_lock.push(Message {
            text: String::new(),
            sender: SenderType::Assistant,
            timestamp: Utc::now(),
            context: None,
        });
        drop(messages_lock);

        // Convert all messages to model format
        let model_messages: Vec<ChatMessage> = self
            .messages
            .lock()
            .unwrap()
            .iter()
            .map(|msg| msg.to_chat_message())
            .collect();

        // Get the stream from the model
        let mut stream = self.model.complete(&model_messages, &HashMap::new()).await;
        // This makes the returned stream borrow `self`.
        let shared_messages = self.messages.clone();

        let wrapped_stream = async_stream::stream! {
            let mut has_error = false;
            let mut assistant_response = String::new();
            while let Some(result) = stream.next().await {
                match result {
                    Ok(chunk) => {
                        // Accumulate response in chat history
                        assistant_response.push_str(&chunk.text);
                        yield Ok(chunk);
                    }
                    Err(e) => {
                        has_error = true;
                        yield Err(e);
                        break;
                    }
                }
            }

            // Clear placeholder if error occurred
            let mut messages = shared_messages.lock().unwrap();
            if has_error {
                messages.truncate(assistant_index);
            }

            if let Some(msg) = messages.get_mut(assistant_index) {
                msg.text = assistant_response;
            }
        };

        Ok(Box::pin(wrapped_stream))
    }

    // Placeholder for actual implementation
    pub fn get_completion_metrics(&self) -> Option<CompletionMetrics> {
        None
    }
}
