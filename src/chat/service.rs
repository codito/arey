use crate::core::completion::{
    CancellationToken, ChatMessage, Completion, CompletionMetrics, CompletionModel,
    CompletionResponse, SenderType,
};
use crate::core::model::{ModelConfig, ModelMetrics};
use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use futures::stream::{BoxStream, StreamExt};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Context associated with a single chat message
pub struct MessageContext {
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
        let mut model = crate::platform::llm::get_completion_llm(model_config.clone())
            .context("Failed to initialize chat model")?;

        // TODO: empty system prompt
        model
            .load("")
            .await
            .context("Failed to load model with system prompt")?;

        let metrics = model.metrics();
        Ok(Self {
            messages: Arc::new(Mutex::new(Vec::new())),
            context: ChatContext {
                metrics: Some(metrics),
                logs: String::new(), // TODO: stderr capture
            },
            model_config,
            model,
        })
    }

    pub async fn stream_response(
        &mut self,
        message: String,
        cancel_token: CancellationToken, // Added cancellation token
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

        let mut stream = self
            .model
            .complete(&model_messages, &HashMap::new(), cancel_token.clone())
            .await;
        let shared_messages = self.messages.clone();
        let mut last_finish_reason: Option<String> = None;

        let wrapped_stream = async_stream::stream! {
            let mut has_error = false;
            let mut assistant_response = String::new();
            let mut raw_logs = String::new();
            let mut metrics = CompletionMetrics::default();
            while let Some(result) = stream.next().await {
                // Check for cancellation *before* processing the chunk
                if cancel_token.is_cancelled() {
                    has_error = true; // Treat cancellation as an error for cleanup purposes
                    yield Err(anyhow::anyhow!("Cancelled by user")); // Yield a cancellation error
                    break; // Exit the loop
                }

                match result {
                    Ok(chunk) => match chunk {
                        Completion::Response(response) => {
                            // Store raw chunk if available
                            if let Some(raw) = &response.raw_chunk {
                                raw_logs.push_str(&format!("{raw}\n"));
                            }
                            // Accumulate response in chat history
                            assistant_response.push_str(&response.text);
                            last_finish_reason = response.finish_reason.clone();
                            yield Ok(response);
                        }
                        Completion::Metrics(usage) => {
                            metrics = usage;
                            break;
                        }
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

            // Update the assistant message and context, if successful
            if let Some(msg) = messages.get_mut(assistant_index) {
                let msg_context = MessageContext {
                    finish_reason: last_finish_reason,
                    metrics: metrics,
                    logs: raw_logs, // STORE ACCUMULATED LOGS
                };

                msg.text = assistant_response;
                msg.context = Some(msg_context);
            }
        };

        Ok(Box::pin(wrapped_stream))
    }

    pub fn get_last_completion_metrics(&self) -> Option<CompletionMetrics> {
        let messages_lock = self.messages.lock().unwrap();
        messages_lock
            .iter()
            .rev()
            .find(|m| m.sender == SenderType::Assistant)
            .and_then(|m| m.context.as_ref())
            .map(|ctx| ctx.metrics.clone())
    }

    pub fn get_last_assistant_logs(&self) -> Option<String> {
        let messages = self.messages.lock().unwrap();
        messages
            .iter()
            .rev()
            .find(|m| m.sender == SenderType::Assistant)
            .and_then(|m| m.context.as_ref())
            .map(|ctx| ctx.logs.clone())
    }
}
