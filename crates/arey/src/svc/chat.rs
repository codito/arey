use anyhow::{Context, Result};
use arey_core::completion::{CancellationToken, ChatMessage, Completion};
use arey_core::config::Config;
use arey_core::session::Session;
use arey_core::tools::Tool;
use futures::{StreamExt, stream::BoxStream};
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Represents an interactive chat session between a user and an AI model.
///
/// It maintains conversation history and manages tool usage.
pub struct Chat<'a> {
    session: Arc<Mutex<Session>>,
    pub available_tools: HashMap<&'a str, Arc<dyn Tool>>,
}

impl<'a> fmt::Debug for Chat<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Chat")
            .field(
                "available_tools",
                &self.available_tools.keys().collect::<Vec<_>>(),
            )
            .finish_non_exhaustive()
    }
}

impl<'a> Chat<'a> {
    /// Creates a new `Chat` session.
    ///
    /// It uses the specified model from the configuration, or the default chat model if `None`.
    pub async fn new(
        config: &Config,
        model: Option<String>,
        available_tools: HashMap<&'a str, Arc<dyn Tool>>,
    ) -> Result<Self> {
        let model_config = if let Some(model_name) = model {
            config
                .models
                .get(model_name.as_str())
                .cloned()
                .context(format!("Model '{model_name}' not found in config."))?
        } else {
            config.chat.model.clone()
        };

        let session = Session::new(model_config, "")
            .await
            .context("Failed to create chat session")?;

        Ok(Self {
            session: Arc::new(Mutex::new(session)),
            available_tools,
        })
    }

    /// Sets the tools available for the current chat session.
    pub async fn set_tools(&self, tool_names: &[String]) -> Result<()> {
        let mut tools = Vec::new();
        for name in tool_names {
            let tool = self
                .available_tools
                .get(name.as_str())
                .ok_or_else(|| anyhow::anyhow!("Tool not found or not available: {}", name))?;
            tools.push(tool.clone());
        }

        let mut session = self.session.lock().await;
        session.set_tools(tools);
        Ok(())
    }

    /// Adds messages to the conversation history.
    pub async fn add_messages(
        &self,
        user_messages: Vec<ChatMessage>,
        tool_messages: Vec<ChatMessage>,
    ) {
        let mut session = self.session.lock().await;
        for message in user_messages {
            session.add_message(message.sender, &message.text);
        }

        for message in tool_messages {
            session.add_message(message.sender, &message.text);
        }
    }

    /// Generates a streaming response from the model based on the conversation history.
    pub async fn stream_response(
        &self,
        cancel_token: CancellationToken,
    ) -> Result<BoxStream<'_, Result<Completion>>> {
        let session = self.session.clone();
        let stream = async_stream::stream! {
            let mut session_lock = session.lock().await;
            let mut inner_stream = match session_lock.generate(HashMap::new(), cancel_token.clone()).await {
                Ok(stream) => stream,
                Err(e) => {
                    yield Err(e);
                    return;
                }
            };

            while let Some(item) = inner_stream.next().await {
                yield item;
            }
        };

        Ok(Box::pin(stream))
    }

    /// Clears the conversation history of the session.
    pub async fn clear_messages(&self) {
        let mut session = self.session.lock().await;
        session.clear_history();
    }

    /// Retrieves the last message from the assistant in the conversation history.
    pub async fn get_last_assistant_message(&self) -> Option<ChatMessage> {
        let session = self.session.lock().await;
        session.last_assistant_message().cloned()
    }

    /// Get all messages from the session
    pub async fn get_all_messages(&self) -> Vec<ChatMessage> {
        let session = self.session.lock().await;
        session.all_messages()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arey_core::{
        completion::SenderType,
        config::{Config, get_config},
        tools::{Tool, ToolError},
    };
    use async_trait::async_trait;
    use serde_json::{Value, json};
    use std::sync::Arc;
    use tempfile::NamedTempFile;
    use wiremock::{
        Mock, MockServer, ResponseTemplate,
        matchers::{method, path},
    };

    fn mock_event_stream_body() -> String {
        let events = [
            json!({
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "created": 1684,
                "model": "gpt-3.5-turbo",
                "choices": [{
                    "delta": {"content": "Hello"},
                    "index": 0,
                    "finish_reason": null
                }]
            }),
            json!({
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "created": 1684,
                "model": "gpt-3.5-turbo",
                "choices": [{
                    "delta": {"content": " world!"},
                    "index": 0,
                    "finish_reason": "stop"
                }]
            }),
        ];
        let mut body: String = events.iter().map(|e| format!("data: {e}\n\n")).collect();
        body.push_str("data: [DONE]\n\n");
        body
    }

    fn create_temp_config_file(server_uri: &str) -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        let config_content = format!(
            r#"
models:
  test-model:
    provider: openai
    base_url: "{server_uri}"
    api_key: "MOCK_OPENAI_API_KEY"
profiles: {{}}
chat:
  model: test-model
task:
  model: test-model
"#,
        );
        std::io::Write::write_all(&mut file, config_content.as_bytes()).unwrap();
        file
    }

    async fn get_test_config(server: &MockServer) -> Result<Config> {
        let config_file = create_temp_config_file(&server.uri());
        get_config(Some(config_file.path().to_path_buf()))
            .map_err(|e| anyhow::anyhow!("Failed to create temp config file. Error {}", e))
    }

    #[derive(Debug, Clone)]
    struct MockTool;

    #[async_trait]
    impl Tool for MockTool {
        fn name(&self) -> String {
            "mock_tool".to_string()
        }

        fn description(&self) -> String {
            "A mock tool for testing".to_string()
        }

        fn parameters(&self) -> Value {
            json!({"type": "object", "properties": {}})
        }

        async fn execute(&self, _input: &Value) -> Result<Value, ToolError> {
            Ok(json!("Mock tool executed"))
        }
    }

    #[tokio::test]
    async fn test_chat_new() -> Result<()> {
        let server = MockServer::start().await;
        let config = get_test_config(&server).await?;

        // Test with existing model
        let chat = Chat::new(&config, Some("test-model".to_string()), HashMap::new()).await;
        assert!(chat.is_ok());

        // Test with default model
        let chat = Chat::new(&config, None, HashMap::new()).await;
        assert!(chat.is_ok());

        // Test with non-existent model
        let chat = Chat::new(&config, Some("bad-model".to_string()), HashMap::new()).await;
        assert!(chat.is_err());
        assert!(
            chat.unwrap_err()
                .to_string()
                .contains("Model 'bad-model' not found in config.")
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_set_tools() -> Result<()> {
        let server = MockServer::start().await;
        let config = get_test_config(&server).await?;

        let mock_tool: Arc<dyn Tool> = Arc::new(MockTool);
        let available_tools = HashMap::from([("mock_tool", mock_tool)]);

        let chat = Chat::new(&config, None, available_tools).await?;
        assert!(chat.set_tools(&["mock_tool".to_string()]).await.is_ok());

        // Test with a tool that is not available
        assert!(chat.set_tools(&["bad_tool".to_string()]).await.is_err());

        Ok(())
    }

    #[tokio::test]
    async fn test_add_and_get_last_message() -> Result<()> {
        let server = MockServer::start().await;
        let config = get_test_config(&server).await?;
        let chat = Chat::new(&config, None, HashMap::new()).await?;

        chat.add_messages(
            vec![ChatMessage {
                sender: SenderType::User,
                text: "Hello".to_string(),
                tools: Vec::new(),
            }],
            Vec::new(),
        )
        .await;
        assert!(chat.get_last_assistant_message().await.is_none());

        chat.add_messages(
            vec![ChatMessage {
                sender: SenderType::Assistant,
                text: "Hi there!".to_string(),
                tools: Vec::new(),
            }],
            Vec::new(),
        )
        .await;

        let last_message = chat.get_last_assistant_message().await;
        assert!(last_message.is_some());
        let last_message = last_message.unwrap();
        assert_eq!(last_message.sender, SenderType::Assistant);
        assert_eq!(last_message.text, "Hi there!");

        Ok(())
    }

    #[tokio::test]
    async fn test_clear_messages() -> Result<()> {
        let server = MockServer::start().await;
        let config = get_test_config(&server).await?;
        let chat = Chat::new(&config, None, HashMap::new()).await?;

        chat.add_messages(
            vec![
                ChatMessage {
                    sender: SenderType::User,
                    text: "Hello".to_string(),
                    tools: Vec::new(),
                },
                ChatMessage {
                    sender: SenderType::Assistant,
                    text: "Hi there!".to_string(),
                    tools: Vec::new(),
                },
            ],
            Vec::new(),
        )
        .await;

        assert!(chat.get_last_assistant_message().await.is_some());

        chat.clear_messages().await;

        assert!(chat.get_last_assistant_message().await.is_none());
        Ok(())
    }

    #[tokio::test]
    async fn test_stream_response() -> Result<()> {
        let server = MockServer::start().await;
        let response = ResponseTemplate::new(200)
            .set_body_bytes(mock_event_stream_body().as_bytes())
            .insert_header("Content-Type", "text/event-stream");
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(response)
            .mount(&server)
            .await;

        let config = get_test_config(&server).await?;
        let chat = Chat::new(&config, None, HashMap::new()).await?;

        let cancel_token = CancellationToken::new();
        let mut stream = chat.stream_response(cancel_token).await?;

        let mut response_text = String::new();
        while let Some(result) = stream.next().await {
            if let Completion::Response(response) = result? {
                response_text.push_str(&response.text);
            }
        }

        assert_eq!(response_text, "Hello world!");

        Ok(())
    }
}
