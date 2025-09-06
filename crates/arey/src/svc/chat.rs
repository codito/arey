use anyhow::{Context, Result};
use arey_core::completion::{CancellationToken, ChatMessage, Completion};
use arey_core::config::{Config, ProfileConfig};
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
    agent_name: String,
    pub available_tools: HashMap<&'a str, Arc<dyn Tool>>,
    config: &'a Config,
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
        config: &'a Config,
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

        let agent_name = config.chat.agent_name.clone();
        let agent_config = config
            .agents
            .get(&agent_name)
            .cloned()
            .context(format!("Agent '{agent_name}' not found in config."))?;

        let tools: Result<Vec<Arc<dyn Tool>>, _> = agent_config
            .tools
            .iter()
            .map(|tool_name| {
                available_tools
                    .get(tool_name.as_str())
                    .cloned()
                    .ok_or_else(|| anyhow::anyhow!("Tool '{}' not found", tool_name))
            })
            .collect();

        let mut session = Session::new(model_config, &agent_config.prompt)
            .await
            .context("Failed to create chat session")?;
        session.set_tools(tools?);

        Ok(Self {
            session: Arc::new(Mutex::new(session)),
            agent_name,
            available_tools,
            config,
        })
    }

    /// Get current agent name
    pub fn agent_name(&self) -> String {
        self.agent_name.clone()
    }

    /// Switch to a different agent
    pub async fn set_agent(&mut self, agent_name: &str) -> Result<()> {
        let agent_config = self
            .config
            .agents
            .get(agent_name)
            .cloned()
            .context(format!("Agent '{agent_name}' not found in config."))?;

        let model_key = self.model_key().await;
        let model_config = self
            .config
            .models
            .get(&model_key)
            .cloned()
            .context(format!(
                "Model '{}' associated with the current session not found in config.",
                model_key
            ))?;

        let mut session_lock = self.session.lock().await;
        let messages = session_lock.all_messages();

        let tools: Result<Vec<Arc<dyn Tool>>, _> = agent_config
            .tools
            .iter()
            .map(|name| {
                self.available_tools
                    .get(name.as_str())
                    .cloned()
                    .ok_or_else(|| anyhow::anyhow!("Tool not found: {}", name))
            })
            .collect();

        let mut new_session = Session::new(model_config, &agent_config.prompt).await?;
        new_session.set_tools(tools?);
        new_session.set_messages(messages);
        *session_lock = new_session;
        self.agent_name = agent_name.to_string();

        Ok(())
    }

    /// Get current model key identifier
    pub async fn model_key(&self) -> String {
        self.session.lock().await.model_key().to_string()
    }

    /// Switch to a different model
    pub async fn set_model(&mut self, model_name: &str) -> Result<()> {
        let model_config = self
            .config
            .models
            .get(model_name)
            .cloned()
            .context(format!("Model '{model_name}' not found in config."))?;

        let mut session_lock = self.session.lock().await;
        // Preserve existing conversation state
        let messages = session_lock.all_messages();
        let tools = session_lock.tools();
        let system_prompt = session_lock.system_prompt().to_string();

        let mut new_session = Session::new(model_config, &system_prompt)
            .await
            .context("Failed to initialize new session")?;

        new_session.set_tools(tools);
        new_session.set_messages(messages);
        *session_lock = new_session;

        Ok(())
    }

    /// Get current profile name
    pub fn profile_name(&self) -> String {
        self.agent_name.clone()
    }

    /// Get current profile name and data
    pub fn current_profile(&self) -> Option<(&String, &ProfileConfig)> {
        self.config
            .agents
            .get(&self.agent_name)
            .map(|agent| (&self.agent_name, &agent.profile))
    }

    /// Switch to a different profile
    pub fn set_profile(&mut self, _profile_name: &str) -> Result<()> {
        anyhow::bail!(
            "Profiles cannot be set directly in chat mode. Use an agent to set a profile."
        )
    }

    /// Get current system prompt
    pub async fn system_prompt(&self) -> String {
        self.session.lock().await.system_prompt().to_string()
    }

    /// Get available agent names
    pub fn available_agent_names(&self) -> Vec<&str> {
        self.config.agents.keys().map(|s| s.as_str()).collect()
    }

    /// Get available model names
    pub fn available_model_names(&self) -> Vec<&str> {
        self.config.models.keys().map(|s| s.as_str()).collect()
    }

    /// Get available profile names
    pub fn available_profile_names(&self) -> Vec<&str> {
        self.config.profiles.keys().map(|s| s.as_str()).collect()
    }

    /// Gets the tools available for the current chat session.
    pub async fn tools(&self) -> Vec<Arc<dyn Tool>> {
        self.session.lock().await.tools()
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

        let mut session_lock = self.session.lock().await;
        let model_key = session_lock.model_key().to_string();
        let model_config = self
            .config
            .models
            .get(&model_key)
            .cloned()
            .context(format!(
                "Model '{}' associated with the current session not found in config.",
                model_key
            ))?;
        let messages = session_lock.all_messages();
        let system_prompt = session_lock.system_prompt().to_string();

        let mut new_session = Session::new(model_config, &system_prompt).await?;
        new_session.set_tools(tools);
        new_session.set_messages(messages);
        *session_lock = new_session;

        Ok(())
    }

    /// Get all messages from the session
    pub async fn get_all_messages(&self) -> Vec<ChatMessage> {
        let session = self.session.lock().await;
        session.all_messages()
    }

    /// Retrieves the last message from the assistant in the conversation history.
    pub async fn get_last_assistant_message(&self) -> Option<ChatMessage> {
        let session = self.session.lock().await;
        session.last_assistant_message().cloned()
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

    /// Clears the conversation history of the session.
    pub async fn clear_messages(&self) {
        let mut session = self.session.lock().await;
        session.clear_history();
    }

    /// Generates a streaming response from the model based on the conversation history.
    pub async fn stream_response(
        &self,
        cancel_token: CancellationToken,
    ) -> Result<BoxStream<'_, Result<Completion>>> {
        let session = self.session.clone();

        let profile = self
            .config
            .agents
            .get(&self.agent_name)
            .map(|a| a.profile.clone());

        let settings = if let Some(profile) = profile {
            let mut settings = HashMap::new();
            settings.insert("temperature".to_string(), profile.temperature.to_string());
            settings.insert(
                "repeat_penalty".to_string(),
                profile.repeat_penalty.to_string(),
            );
            settings.insert("top_k".to_string(), profile.top_k.to_string());
            settings.insert("top_p".to_string(), profile.top_p.to_string());
            settings
        } else {
            HashMap::new()
        };

        let stream = async_stream::stream! {
            let mut session_lock = session.lock().await;
            let mut inner_stream = match session_lock.generate(settings, cancel_token.clone()).await {
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
chat:
  model: test-model
task:
  model: test-model
profiles:
  test-profile:
    temperature: 0.5
    top_p: 0.9
    repeat_penalty: 1.1
    top_k: 40
agents:
  test-agent:
    prompt: "You are a test agent."
    tools: [ "mock_tool" ]
    profile:
      temperature: 0.1
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
        let mut config = get_test_config(&server).await?;
        let mock_tool: Arc<dyn Tool> = Arc::new(MockTool);
        let available_tools = HashMap::from([("mock_tool", mock_tool)]);

        // Test with existing model, should use default agent from config
        let chat = Chat::new(&config, Some("test-model".to_string()), HashMap::new()).await;
        assert!(chat.is_ok());
        assert_eq!(chat.unwrap().agent_name(), "default");

        // Test with a specified agent
        config.chat.agent_name = "test-agent".to_string();
        let chat = Chat::new(&config, None, available_tools).await?;
        assert_eq!(chat.agent_name(), "test-agent");
        assert_eq!(chat.system_prompt().await, "You are a test agent.");
        assert_eq!(chat.tools().await.len(), 1);

        // Test with non-existent agent
        config.chat.agent_name = "bad-agent".to_string();
        let chat = Chat::new(&config, None, HashMap::new()).await;
        assert!(chat.is_err());
        assert!(
            chat.unwrap_err()
                .to_string()
                .contains("Agent 'bad-agent' not found in config.")
        );

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
    async fn test_get_tools() -> Result<()> {
        let server = MockServer::start().await;
        let config = get_test_config(&server).await?;

        let mock_tool: Arc<dyn Tool> = Arc::new(MockTool);
        let available_tools = HashMap::from([("mock_tool", mock_tool)]);

        let chat = Chat::new(&config, None, available_tools).await?;

        // Initially, no tools are set
        let tools = chat.tools().await;
        assert!(tools.is_empty());

        // Set a tool
        chat.set_tools(&["mock_tool".to_string()]).await?;

        // Get the tools and verify
        let tools = chat.tools().await;
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name(), "mock_tool");

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

    #[tokio::test]
    async fn test_set_model() -> Result<()> {
        let server = MockServer::start().await;
        let mut config = get_test_config(&server).await?;

        // Add second model to config
        config.models.insert(
            "test-model-2".to_string(),
            arey_core::model::ModelConfig {
                key: "test-key".to_string(),
                name: "test-model-2".to_string(),
                provider: arey_core::model::ModelProvider::Openai,
                settings: HashMap::from([
                    ("base_url".to_string(), server.uri().into()),
                    ("api_key".to_string(), "MOCK_OPENAI_API_KEY_2".into()),
                ]),
            },
        );

        let mock_tool: Arc<dyn Tool> = Arc::new(MockTool);
        let available_tools = HashMap::from([("mock_tool", mock_tool)]);

        let mut chat = Chat::new(&config, Some("test-model".to_string()), available_tools).await?;

        // Switch to second model
        chat.set_model("test-model-2").await?;

        // Verify model changed
        let models = chat.available_model_names();
        assert_eq!(models.len(), 2);
        assert!(models.contains(&"test-model"));
        assert!(models.contains(&"test-model-2"));
        Ok(())
    }

    #[tokio::test]
    async fn test_set_agent() -> Result<()> {
        let server = MockServer::start().await;
        let config = get_test_config(&server).await?;
        let mock_tool: Arc<dyn Tool> = Arc::new(MockTool);
        let available_tools = HashMap::from([("mock_tool", mock_tool)]);

        let mut chat = Chat::new(&config, None, available_tools).await?;
        assert_eq!(chat.agent_name(), "default");
        assert_eq!(chat.system_prompt().await, "You are a helpful assistant.");

        // Set a valid agent
        chat.set_agent("test-agent").await?;
        assert_eq!(chat.agent_name(), "test-agent".to_string());
        assert_eq!(chat.system_prompt().await, "You are a test agent.");
        assert_eq!(chat.tools().await.len(), 1);

        // Setting an invalid agent should fail
        let result = chat.set_agent("non-existent-agent").await;
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Agent 'non-existent-agent' not found in config.")
        );

        // Agent should remain unchanged after error
        assert_eq!(chat.agent_name(), "test-agent".to_string());

        Ok(())
    }
}
