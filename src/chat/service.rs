use crate::core::completion::combine_metrics;
use crate::core::completion::{
    CancellationToken, ChatMessage, CompletionMetrics, CompletionModel, CompletionResponse,
    SenderType,
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
    pub finish_reason: Option<String>, // 6. Finish reason tracking
    pub metrics: CompletionMetrics,    // 2. Metrics combination and storage for assistant messages
    pub logs: String,                  // 3. Assistant message logging context
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

        // 1. System prompt loading during chat initialization
        // 7. System prompt handling (currently hard-coded empty string)
        // This assumes `CompletionModel` trait has an `async fn load(&mut self, system_prompt: &str) -> Result<()>` method.
        // The `load` method was removed from the CompletionModel trait in a previous step.
        // model
        //     .load("") // For now, an empty system prompt. Proper prompt handling needs more config.
        //     .await
        //     .context("Failed to load model with system prompt")?;

        let metrics = model.metrics();
        Ok(Self {
            messages: Arc::new(Mutex::new(Vec::new())),
            context: ChatContext {
                metrics: Some(metrics),
                logs: String::new(), // Placeholder for logs from model.load, as stderr capture is complex in Rust.
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

        // Get the stream from the model
        let mut stream = self
            .model
            .complete(&model_messages, &HashMap::new(), cancel_token.clone())
            .await; // Pass cancellation token
        // This makes the returned stream borrow `self`.
        let shared_messages = self.messages.clone();

        // 4. In-stream metrics accumulation
        let mut usage_series: Vec<CompletionMetrics> = Vec::new();
        // 8. Last finish reason capture during streaming
        let mut last_finish_reason: Option<String> = None;

        let wrapped_stream = async_stream::stream! {
            let mut has_error = false;
            let mut assistant_response = String::new();
            let mut raw_logs = String::new(); // ADD THIS FOR LOG ACCUMULATION
            while let Some(result) = stream.next().await {
                // Check for cancellation *before* processing the chunk
                if cancel_token.is_cancelled() {
                    has_error = true; // Treat cancellation as an error for cleanup purposes
                    yield Err(anyhow::anyhow!("Cancelled by user")); // Yield a cancellation error
                    break; // Exit the loop
                }

                match result {
                    Ok(chunk) => {
                        // Store raw chunk if available
                        if let Some(raw) = &chunk.raw_chunk {
                            raw_logs.push_str(&format!("{raw}\n"));
                        }
                        // Accumulate response in chat history
                        assistant_response.push_str(&chunk.text);
                        usage_series.push(chunk.metrics.clone()); // Accumulate metrics
                        last_finish_reason = chunk.finish_reason.clone(); // Capture last finish reason
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

            // 2. Metrics combination and storage for assistant messages
            let combined_metrics = combine_metrics(&usage_series);
            // 3. Assistant message logging context (placeholder for now)
            // 6. Finish reason tracking
            let msg_context = MessageContext {
                prompt: String::new(), // Python has prompt, but it's not used here.
                finish_reason: last_finish_reason,
                metrics: combined_metrics,
                logs: raw_logs, // STORE ACCUMULATED LOGS
            };

            if let Some(msg) = messages.get_mut(assistant_index) {
                msg.text = assistant_response;
                msg.context = Some(msg_context); // Store context with metrics and finish reason
            }
        };

        Ok(Box::pin(wrapped_stream))
    }

    // 5. Completion metrics retrieval
    pub fn get_completion_metrics(&self) -> Option<CompletionMetrics> {
        let messages_lock = self.messages.lock().unwrap();
        messages_lock
            .iter()
            .rev() // Iterate in reverse to find the last assistant message
            .find(|m| m.sender == SenderType::Assistant)
            .and_then(|m| m.context.as_ref()) // Get its context if it exists
            .map(|ctx| ctx.metrics.clone()) // Get the metrics from the context
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
