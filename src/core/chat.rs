use crate::core::completion::{ChatMessage, CompletionMetrics, CompletionResponse, SenderType};
use crate::core::model::{ModelConfig, ModelMetrics};
use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use futures::StreamExt; // Add at the top with other imports
use futures::stream::{BoxStream, StreamExt}; // Then update the existing use statement
use std::collections::HashMap;
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
    pub messages: Vec<Message>,
    pub context: ChatContext,
    model_config: ModelConfig,
    model: Box<dyn crate::core::completion::CompletionModel + Send + Sync>,
}

impl Chat {
    pub async fn new(model_config: ModelConfig) -> Result<Self> {
        let model = crate::platform::llm::get_completion_llm(model_config.clone())
            .context("Failed to initialize chat model")?;
        let metrics = model.metrics();
        Ok(Self {
            messages: Vec::new(),
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
        self.messages.push(user_message);

        // Add assistant message placeholder
        let assistant_index = self.messages.len();
        self.messages.push(Message {
            text: String::new(),
            sender: SenderType::Assistant,
            timestamp: Utc::now(),
            context: None,
        });

        // Convert all messages to model format
        let model_messages: Vec<ChatMessage> = self
            .messages
            .iter()
            .map(|msg| msg.to_chat_message())
            .collect();

        // Get the stream from the model
        let mut inner_stream = self.model.complete(&model_messages, &HashMap::new()).await;

        // Capture `self` mutably for the outer stream.
        // This makes the returned stream borrow `self`.
        let mut chat_ref = self;

        let wrapped_stream = async_stream::stream! {
            let mut has_error = false;
            while let Some(result) = inner_stream.next().await {
                match result {
                    Ok(chunk) => {
                        // Accumulate response in chat history
                        chat_ref.messages[assistant_index].text.push_str(&chunk.text);
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
            if has_error {
                chat_ref.messages.truncate(assistant_index);
            }
        };

        Ok(Box::pin(wrapped_stream))
    }

    // Placeholder for actual implementation
    pub fn get_completion_metrics(&self) -> Option<CompletionMetrics> {
        None
    }
}
