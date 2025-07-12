use anyhow::{Context, Result};
use arey_core::completion::{
    CancellationToken, ChatMessage, Completion, CompletionMetrics, CompletionModel,
    CompletionResponse, SenderType,
};
use arey_core::model::{ModelConfig, ModelMetrics};
use arey_core::tools::{Tool, ToolCall};
use chrono::{DateTime, Utc};
use futures::stream::{BoxStream, StreamExt};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Context associated with a single chat message
#[derive(Clone)]
pub struct MessageContext {
    pub finish_reason: Option<String>,
    pub metrics: CompletionMetrics,
    pub raw_api_logs: String,
    pub tool_calls: Vec<ToolCall>,
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
        let mut tools: Vec<ToolCall> = vec![];

        if let Some(ctx) = &self.context {
            tools = ctx.tool_calls.clone()
        }

        ChatMessage {
            text: self.text.clone(),
            sender: self.sender.clone(),
            tools,
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
    pub available_tools: HashMap<String, Arc<dyn Tool>>,
    _model_config: ModelConfig,
    model: Box<dyn CompletionModel + Send + Sync>,
    tools: Vec<Arc<dyn Tool>>,
}

impl Chat {
    pub async fn new(
        model_config: ModelConfig,
        available_tools: HashMap<String, Arc<dyn Tool>>,
    ) -> Result<Self> {
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
            available_tools,
            _model_config: model_config,
            model,
            tools: Vec::new(),
        })
    }

    pub fn get_available_tool_names(&self) -> Vec<String> {
        self.available_tools.keys().cloned().collect()
    }

    pub async fn set_tools(&mut self, tool_names: &[String]) -> Result<()> {
        let mut tools = Vec::new();
        for name in tool_names {
            let tool = self
                .available_tools
                .get(name.as_str())
                .ok_or_else(|| anyhow::anyhow!("Tool not found or not available: {}", name))?;
            tools.push(tool.clone());
        }
        self.tools = tools;
        Ok(())
    }

    pub async fn stream_response(
        &mut self,
        user_messages: Vec<Message>,
        tool_messages: Vec<Message>,
        cancel_token: CancellationToken, // Added cancellation token
    ) -> Result<BoxStream<'_, Result<CompletionResponse>>> {
        // Create user message and add to history
        {
            let mut messages_lock = self.messages.lock().await;
            for m in user_messages {
                messages_lock.push(m);
            }
            for m in tool_messages {
                messages_lock.push(m);
            }
        }

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

        // Add assistant message placeholder
        let assistant_index = {
            let mut messages_lock = self.messages.lock().await;
            messages_lock.push(Message {
                text: String::new(),
                sender: SenderType::Assistant,
                _timestamp: Utc::now(),
                context: None,
            });

            messages_lock.len() - 1
        };

        let wrapped_stream = async_stream::stream! {
            let mut has_error = false;
            let mut assistant_response = String::new();
            let mut tool_calls = Vec::new();
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
                            if let Some(calls) = &response.tool_calls {
                                tool_calls.extend(calls.clone());
                            }
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
                    raw_api_logs: raw_logs,
                    tool_calls,
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

#[cfg(test)]
mod tests {
    use super::*;
    use arey_core::model::ModelProvider;
    use arey_core::tools::ToolError;
    use async_trait::async_trait;
    use serde_json::json;

    // Mock Tool for testing
    #[derive(Debug)]
    struct MockTool {
        name: String,
    }

    #[async_trait]
    impl Tool for MockTool {
        fn name(&self) -> String {
            self.name.clone()
        }
        fn description(&self) -> String {
            format!("A mock tool named {}", self.name)
        }
        fn parameters(&self) -> serde_json::Value {
            json!({"type": "object"})
        }
        async fn execute(
            &self,
            _arguments: &serde_json::Value,
        ) -> Result<serde_json::Value, ToolError> {
            Ok(json!({"result": "success"}))
        }
    }

    // Mock CompletionModel for testing
    struct MockCompletionModel;

    #[async_trait]
    impl CompletionModel for MockCompletionModel {
        fn metrics(&self) -> ModelMetrics {
            ModelMetrics::default()
        }
        async fn load(&mut self, _text: &str) -> Result<()> {
            Ok(())
        }
        async fn complete(
            &mut self,
            _messages: &[ChatMessage],
            _tools: Option<&[Arc<dyn Tool>]>,
            _settings: &HashMap<String, String>,
            _cancel_token: CancellationToken,
        ) -> BoxStream<'_, Result<Completion>> {
            unimplemented!()
        }
    }

    fn create_test_chat(available_tools: HashMap<String, Arc<dyn Tool>>) -> Chat {
        Chat {
            messages: Arc::new(Mutex::new(Vec::new())),
            _context: ChatContext {
                _metrics: None,
                _logs: String::new(),
            },
            available_tools,
            _model_config: ModelConfig {
                name: "mock-model".to_string(),
                provider: ModelProvider::Gguf,
                settings: HashMap::new(),
            },
            model: Box::new(MockCompletionModel),
            tools: Vec::new(),
        }
    }

    #[test]
    fn test_get_available_tool_names() {
        let mut available_tools: HashMap<String, Arc<dyn Tool>> = HashMap::new();
        available_tools.insert(
            "tool1".to_string(),
            Arc::new(MockTool {
                name: "tool1".to_string(),
            }),
        );
        available_tools.insert(
            "tool2".to_string(),
            Arc::new(MockTool {
                name: "tool2".to_string(),
            }),
        );

        let chat = create_test_chat(available_tools);
        let mut tool_names = chat.get_available_tool_names();
        tool_names.sort();
        assert_eq!(tool_names, vec!["tool1", "tool2"]);
    }

    #[tokio::test]
    async fn test_set_tools_success() {
        let mut available_tools: HashMap<String, Arc<dyn Tool>> = HashMap::new();
        let tool1 = Arc::new(MockTool {
            name: "tool1".to_string(),
        });
        available_tools.insert("tool1".to_string(), tool1.clone());

        let mut chat = create_test_chat(available_tools);
        let result = chat.set_tools(&["tool1".to_string()]).await;

        assert!(result.is_ok());
        assert_eq!(chat.tools.len(), 1);
        assert_eq!(chat.tools[0].name(), "tool1");
    }

    #[tokio::test]
    async fn test_set_tools_not_available() {
        let available_tools = HashMap::new();
        let mut chat = create_test_chat(available_tools);
        let result = chat.set_tools(&["unknown_tool".to_string()]).await;

        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err().to_string(),
            "Tool not found or not available: unknown_tool"
        );
        assert!(chat.tools.is_empty());
    }
}
