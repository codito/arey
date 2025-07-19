use anyhow::{Context, Result};
use arey_core::completion::{CancellationToken, ChatMessage, Completion};
use arey_core::config::Config;
use arey_core::session::Session;
use arey_core::tools::Tool;
use futures::{StreamExt, stream::BoxStream};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Represents an interactive chat session between a user and an AI model.
///
/// It maintains conversation history and manages tool usage.
pub struct Chat<'a> {
    session: Arc<Mutex<Session>>,
    pub available_tools: HashMap<&'a str, Arc<dyn Tool>>,
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
}
