use anyhow::{Context, Result};
use arey_core::completion::{
    CancellationToken, ChatMessage, Completion, CompletionMetrics, CompletionModel,
    CompletionResponse, SenderType,
};
use arey_core::model::{ModelConfig, ModelMetrics};
use arey_core::tools::Tool;
use chrono::{DateTime, Utc};
use futures::stream::{BoxStream, StreamExt};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

// TODO: This should load tools from config
fn get_tool_by_name(name: &str) -> Result<Arc<dyn Tool>> {
    match name {
        "search" => Err(anyhow::anyhow!("'search' tool is not implemented yet")),
        _ => Err(anyhow::anyhow!("Unknown tool: {}", name)),
    }
}

/// Context associated with a single chat message
#[derive(Clone)]
pub struct MessageContext {
    pub finish_reason: Option<String>,
    pub metrics: CompletionMetrics,
    pub logs: String,
}

/// Chat message with context
pub struct Message {
    pub text: String,
    pub sender: SenderType,
    pub _timestamp: DateTime<Utc>,
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
    pub _metrics: Option<ModelMetrics>,
    pub _logs: String,
}

/// Chat conversation between human and AI model
pub struct Chat {
    pub messages: Arc<Mutex<Vec<Message>>>,
    pub _context: ChatContext,
    _model_config: ModelConfig,
    model: Box<dyn CompletionModel + Send + Sync>,
    tools: Vec<Arc<dyn Tool>>,
}

impl Chat {
    pub async fn new(model_config: ModelConfig) -> Result<Self> {
        let mut model = arey_core::get_completion_llm(model_config.clone())
            .context("Failed to initialize chat model")?;

        model
            .load("")
            .await
            .context("Failed to load model with system prompt")?;

        Ok(Self {
            messages: Arc::new(Mutex::new(Vec::new())),
            _context: ChatContext {
                _metrics: Some(model.metrics()),
                _logs: String::new(),
            },
            _model_config: model_config,
            model,
            tools: Vec::new(),
        })
    }

    pub async fn set_tools(&mut self, tool_names: &[String]) -> Result<()> {
        let mut tools = Vec::new();
        for name in tool_names {
            let tool = get_tool_by_name(name)?;
            tools.push(tool);
        }
        self.tools = tools;
        Ok(())
    }

    pub async fn stream_response(
        &mut self,
        message: &str,
        cancel_token: CancellationToken, // Added cancellation token
    ) -> Result<BoxStream<'_, Result<CompletionResponse>>> {
        let _timestamp = Utc::now();

        // Create user message and add to history
        let user_message = Message {
            text: message.to_string(),
            sender: SenderType::User,
            _timestamp,
            context: None,
        };

        let mut messages_lock = self.messages.lock().await;
        messages_lock.push(user_message);

        // Add assistant message placeholder
        let assistant_index = messages_lock.len();
        messages_lock.push(Message {
            text: String::new(),
            sender: SenderType::Assistant,
            _timestamp: Utc::now(),
            context: None,
        });
        drop(messages_lock);

        // Convert all messages to model format
        let model_messages: Vec<ChatMessage> = self
            .messages
            .lock()
            .await
            .iter()
            .map(|msg| msg.to_chat_message())
            .collect();

        let tool_slice = if self.tools.is_empty() {
            None
        } else {
            Some(self.tools.as_slice())
        };

        let mut stream = self
            .model
            .complete(
                &model_messages,
                tool_slice,
                &HashMap::new(),
                cancel_token.clone(),
            )
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
                            if let Some(raw) = &usage.raw_chunk {
                                raw_logs.push_str(&format!("{raw}\n"));
                            }

                            metrics = usage;
                            // Do not break to ensure that stream end can be passed to the clients.
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
            let mut messages = shared_messages.lock().await;
            if has_error {
                messages.truncate(assistant_index);
            }

            // Update the assistant message and context, if successful
            if let Some(msg) = messages.get_mut(assistant_index) {
                let msg_context = MessageContext {
                    finish_reason: last_finish_reason,
                    metrics,
                    logs: raw_logs, // STORE ACCUMULATED LOGS
                };

                msg.text = assistant_response;
                msg.context = Some(msg_context);
            }
        };

        Ok(Box::pin(wrapped_stream))
    }

    pub async fn clear_messages(&self) {
        let mut messages = self.messages.lock().await;
        messages.clear();
    }

    pub async fn get_last_assistant_context(&self) -> Option<MessageContext> {
        let messages_lock = self.messages.lock().await;
        messages_lock
            .iter()
            .rev()
            .find(|m| m.sender == SenderType::Assistant)
            .and_then(|m| m.context.clone())
    }
}
