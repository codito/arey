use crate::cli::ux::style_chat_text;
use crate::svc::chat::Chat;
use anyhow::Result;
use clap::{Parser, Subcommand};
use std::sync::Arc;
use tokio::sync::Mutex;

// -------------
// REPL commands
// -------------
#[derive(Parser, Debug)]
#[command(multicall = true)]
pub struct CliCommand {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand, Debug, Hash, PartialEq, Eq)]
pub enum Command {
    /// Clear chat history
    Clear,
    /// Show detailed logs for the last assistant message
    Log,
    /// Manage chat models.
    ///
    /// With no arguments, shows the current model and available models.
    #[command(alias = "m", alias = "mod")]
    Model {
        /// Model name to switch to
        name: Option<String>,
    },
    /// Manage chat profiles.
    ///
    /// With no arguments, shows the current profile and available profiles.
    #[command(alias = "p")]
    Profile {
        /// Profile name to switch to
        name: Option<String>,
    },
    /// Manage chat tools.
    ///
    /// With no arguments, shows current active tools and available tools.
    /// Use "clear" to remove all active tools.
    /// Provide tool names to set active tools.
    #[command(alias = "t")]
    Tool {
        /// Tool names to set, or "clear"
        names: Vec<String>,
    },
    /// Manage chat agents.
    ///
    /// With no arguments, shows the current agent and available agents with their sources.
    #[command(alias = "a")]
    Agent {
        /// Agent name to switch to
        name: Option<String>,
    },
    /// Set or view the system prompt for the current session.
    ///
    /// With no arguments, shows the current system prompt.
    /// Provide a prompt string to set a new system prompt.
    #[command(alias = "sys")]
    System {
        /// New system prompt to set (optional)
        prompt: Option<String>,
    },
    /// Exit the chat session
    #[command(alias = "q", alias = "quit")]
    Exit,
}

impl Command {
    /// Executes a REPL command.
    ///
    /// Returns `Ok(false)` if the REPL should exit.
    pub async fn execute(self, session: Arc<Mutex<Chat<'_>>>) -> Result<bool> {
        match self {
            Command::Clear => self.execute_clear(session).await,
            Command::Log => self.execute_log(session).await,
            Command::Model { ref name } => self.execute_model(session, name).await,
            Command::Profile { ref name } => self.execute_profile(session, name).await,
            Command::Tool { ref names } => self.execute_tool(session, names).await,
            Command::Agent { ref name } => self.execute_agent(session, name).await,
            Command::System { ref prompt } => self.execute_system(session, prompt).await,
            Command::Exit => self.execute_exit(),
        }
    }

    async fn execute_clear(&self, session: Arc<Mutex<Chat<'_>>>) -> Result<bool> {
        session.lock().await.clear_messages().await;
        println!("Chat history cleared");
        Ok(true)
    }

    async fn execute_log(&self, session: Arc<Mutex<Chat<'_>>>) -> Result<bool> {
        let chat_guard = session.lock().await;
        let messages = chat_guard.get_all_messages();
        let block = format_message_block(&messages)?;
        println!("{}", block);
        Ok(true)
    }

    async fn execute_tool(&self, session: Arc<Mutex<Chat<'_>>>, names: &[String]) -> Result<bool> {
        if names.len() == 1 && names[0] == "clear" {
            // Clear all tools
            let mut chat_guard = session.lock().await;
            let current_tools = chat_guard.tools();
            if current_tools.is_empty() {
                println!("No tools are currently active.");
            } else {
                let tool_names: Vec<String> =
                    current_tools.iter().map(|t| t.name().to_string()).collect();
                chat_guard.set_tools(&[]).await?;
                println!("Tools cleared: {}", tool_names.join(", "));
            }
        } else if names.is_empty() {
            // Show current and available tools
            let chat_guard = session.lock().await;
            let current_tools = chat_guard.tools();
            let available_tools = chat_guard.available_tool_names();

            if current_tools.is_empty() {
                println!("No tools are currently active.");
            } else {
                let tool_names: Vec<String> =
                    current_tools.iter().map(|t| t.name().to_string()).collect();
                println!("Current tools: {}", tool_names.join(", "));
            }

            if !available_tools.is_empty() {
                println!("Available tools: {}", available_tools.join(", "));
            }
        } else {
            // Set new tools
            let mut chat_guard = session.lock().await;
            match chat_guard.set_tools(names).await {
                Ok(()) => {
                    println!("Tools set: {}", names.join(", "));
                }
                Err(e) => {
                    eprintln!("Error setting tools: {e}");
                }
            }
        }
        Ok(true)
    }

    async fn execute_model(
        &self,
        session: Arc<Mutex<Chat<'_>>>,
        name: &Option<String>,
    ) -> Result<bool> {
        match name {
            Some(name) => {
                let mut chat_guard = session.lock().await;
                let spinner =
                    crate::cli::ux::GenerationSpinner::new(format!("Loading model '{}'...", &name));

                match chat_guard.set_model(name).await {
                    Ok(()) => {
                        spinner.clear();
                        println!("Model switched to: {}", name);
                    }
                    Err(e) => {
                        spinner.clear();
                        let error_msg = format!("Error switching model: {}", e);
                        eprintln!(
                            "{}",
                            style_chat_text(&error_msg, crate::cli::ux::ChatMessageType::Error)
                        );
                    }
                }
            }
            None => {
                let chat_guard = session.lock().await;
                let model_key = chat_guard.model_key();
                let model_names = chat_guard.available_model_names();
                println!("Current model: {}", model_key);
                if !model_names.is_empty() {
                    println!("Available models: {}", model_names.join(", "));
                }
            }
        }
        Ok(true)
    }

    async fn execute_profile(
        &self,
        session: Arc<Mutex<Chat<'_>>>,
        name: &Option<String>,
    ) -> Result<bool> {
        match name {
            Some(name) => {
                let mut chat_guard = session.lock().await;
                match chat_guard.set_profile(name) {
                    Ok(()) => {
                        println!("Profile switched to: {}", name);
                        // Show detailed profile information
                        if let Some((_profile_name, profile_data)) = chat_guard.current_profile() {
                            match serde_yaml::to_string(&profile_data) {
                                Ok(yaml) => {
                                    // Trim to avoid printing empty "{}" for empty-but-not-null data.
                                    let trimmed = yaml.trim();
                                    if !trimmed.is_empty() && trimmed != "{}" {
                                        print!("{yaml}"); // `to_string` includes a newline
                                    }
                                }
                                Err(e) => {
                                    let error_msg = format!("Error formatting profile data: {e}");
                                    eprintln!(
                                        "{}",
                                        style_chat_text(
                                            &error_msg,
                                            crate::cli::ux::ChatMessageType::Error
                                        )
                                    );
                                }
                            }
                        }
                    }
                    Err(e) => {
                        let error_msg = format!("Error switching profile: {}", e);
                        eprintln!(
                            "{}",
                            style_chat_text(&error_msg, crate::cli::ux::ChatMessageType::Error)
                        );
                    }
                }
            }
            None => {
                let chat_guard = session.lock().await;
                let profile_names = chat_guard.available_profile_names();

                if let Some((profile_name, profile_data)) = chat_guard.current_profile() {
                    println!("Current profile: {profile_name}");
                    match serde_yaml::to_string(&profile_data) {
                        Ok(yaml) => {
                            // Trim to avoid printing empty "{}" for empty-but-not-null data.
                            let trimmed = yaml.trim();
                            if !trimmed.is_empty() && trimmed != "{}" {
                                print!("{yaml}"); // `to_string` includes a newline
                            }
                        }
                        Err(e) => {
                            let error_msg = format!("Error formatting profile data: {e}");
                            eprintln!(
                                "{}",
                                style_chat_text(&error_msg, crate::cli::ux::ChatMessageType::Error)
                            );
                        }
                    }
                } else {
                    println!("No profile is active.");
                }

                if !profile_names.is_empty() {
                    println!("Available profiles: {}", profile_names.join(", "));
                }
            }
        }
        Ok(true)
    }

    async fn execute_agent(
        &self,
        session: Arc<Mutex<Chat<'_>>>,
        name: &Option<String>,
    ) -> Result<bool> {
        match name {
            Some(name) => {
                let mut chat_guard = session.lock().await;
                let spinner =
                    crate::cli::ux::GenerationSpinner::new(format!("Loading agent '{}'...", &name));

                match chat_guard.set_agent(name).await {
                    Ok(()) => {
                        spinner.clear();
                        println!("Agent switched to: {}", name);
                    }
                    Err(e) => {
                        spinner.clear();
                        let error_msg = format!("Error switching agent: {}", e);
                        eprintln!(
                            "{}",
                            style_chat_text(&error_msg, crate::cli::ux::ChatMessageType::Error)
                        );
                    }
                }
            }
            None => {
                let chat_guard = session.lock().await;
                let agent_name = chat_guard.agent_name();
                let source = chat_guard.format_agent_source(&agent_name);
                println!("Current agent: {} ({})", agent_name, source);

                let agents = chat_guard.available_agents_with_sources();
                if !agents.is_empty() {
                    println!("Available agents:");
                    for (agent_name, source) in agents {
                        let source_str = match source {
                            arey_core::agent::AgentSource::BuiltIn => "built-in",
                            arey_core::agent::AgentSource::UserFile(_) => "user",
                        };
                        println!("  {} ({})", agent_name, source_str);
                    }
                }
            }
        }
        Ok(true)
    }

    async fn execute_system(
        &self,
        session: Arc<Mutex<Chat<'_>>>,
        prompt: &Option<String>,
    ) -> Result<bool> {
        let mut chat_guard = session.lock().await;
        match prompt {
            Some(new_prompt) => match chat_guard.set_system_prompt(new_prompt).await {
                Ok(()) => {
                    println!("System prompt updated successfully.");
                }
                Err(e) => {
                    let error_msg = format!("Error setting system prompt: {}", e);
                    eprintln!(
                        "{}",
                        style_chat_text(&error_msg, crate::cli::ux::ChatMessageType::Error)
                    );
                }
            },
            None => {
                let current_prompt = chat_guard.system_prompt();
                println!("Current system prompt:");
                println!("{}", current_prompt);
            }
        }
        Ok(true)
    }

    fn execute_exit(&self) -> Result<bool> {
        println!("Bye!");
        Ok(false)
    }
}

/// Parse command line input with robust handling for special characters
/// Returns a vector of parsed arguments suitable for clap parsing
pub fn parse_command_line(line: &str) -> Vec<String> {
    let trimmed_line = line.trim();

    let args = match shlex::split(trimmed_line) {
        Some(parsed) => parsed,
        None => {
            // Fallback for cases with unescaped quotes/apostrophes
            // Split on first space only for commands that need special handling
            if trimmed_line.starts_with("/system")
                || trimmed_line.starts_with("/sys")
                || trimmed_line.starts_with("/tool")
            {
                if let Some(space_pos) = trimmed_line.find(' ') {
                    let command = trimmed_line[..space_pos].to_string();
                    let argument = trimmed_line[space_pos + 1..].trim().to_string();
                    vec![command, argument]
                } else {
                    vec![trimmed_line.to_string()]
                }
            } else {
                // For other commands, try simple space splitting
                trimmed_line
                    .split_whitespace()
                    .map(|s| s.to_string())
                    .collect()
            }
        }
    };

    // Special handling for commands that accept multi-word arguments
    if !args.is_empty() && (args[0] == "/system" || args[0] == "/sys" || args[0] == "/tool") {
        if args.len() > 1 {
            // Join all arguments after the command
            let mut processed = vec![args[0].clone()];
            let joined_arg = args[1..].join(" ");
            processed.push(joined_arg);
            processed
        } else {
            args
        }
    } else {
        args
    }
}

/// Format the last message block (from last user message to end) into a string.
fn format_message_block(messages: &[arey_core::completion::ChatMessage]) -> Result<String> {
    use arey_core::completion::SenderType;

    // Find start of last block (last user message)
    let start_idx = messages
        .iter()
        .rposition(|msg| msg.sender == SenderType::User)
        .unwrap_or(0);

    let last_block = &messages[start_idx..];

    if last_block.is_empty() {
        return Ok("No recent messages to display".to_string());
    }

    let mut out = String::new();

    out.push_str("\n=== LAST MESSAGE BLOCK ===\n");
    for (i, msg) in last_block.iter().enumerate() {
        // Format sender with type-specific style
        let sender_tag = match msg.sender {
            SenderType::User => "USER:".to_string(),
            SenderType::Assistant => "ASSISTANT:".to_string(),
            SenderType::Tool => "TOOL:".to_string(),
            SenderType::System => "SYSTEM:".to_string(),
        };

        // Truncate long messages
        let max_length = 500;
        let mut content = msg.text.clone();
        let is_truncated = content.len() > max_length;

        if is_truncated {
            content.truncate(max_length);
            content.push_str("\n... [truncated]");
        }

        out.push_str(&format!("{} {}\n", sender_tag, content));

        // Show tool calls if any
        if !msg.tools.is_empty() {
            out.push_str("  Tools:\n");
            for tool in &msg.tools {
                out.push_str(&format!("    - {}: {}\n", tool.name, tool.arguments));
            }
        }

        if i < last_block.len() - 1 {
            out.push_str("------\n");
        }
    }
    out.push_str("========================\n");

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::chat::test_utils::{MockTool, create_test_config_with_custom_agent};
    use arey_core::completion::{ChatMessage, SenderType};
    use arey_core::tools::{Tool, ToolCall};
    use std::collections::HashMap;
    use std::sync::Arc;
    use tokio::sync::Mutex;

    #[test]
    fn test_system_command_parsing_edge_cases() {
        let test_cases = vec![
            // Test normal case with spaces
            ("/system you are an expert", Some("you are an expert")),
            // Test apostrophes (the main issue)
            ("/system you're an expert", Some("you're an expert")),
            // Test quotes (should still work with shlex)
            ("/system \"quoted text\"", Some("quoted text")),
            // Test mixed punctuation
            ("/system hello, world!", Some("hello, world!")),
            // Test no arguments (should work)
            ("/system", None),
            // Test alias
            ("/sys you're an expert", Some("you're an expert")),
        ];

        for (input, expected_prompt) in test_cases {
            // Use the shared parsing function
            let processed_args = parse_command_line(input);

            let result = CliCommand::try_parse_from(processed_args);
            assert!(result.is_ok(), "Failed to parse '{}': {:?}", input, result);

            let cli_command = result.unwrap();
            match cli_command.command {
                Command::System { prompt } => {
                    assert_eq!(
                        prompt,
                        expected_prompt.map(|s| s.to_string()),
                        "Failed for input: '{}'",
                        input
                    );
                }
                _ => panic!(
                    "Expected System command for input: '{}', got {:?}",
                    input, cli_command.command
                ),
            }
        }
    }

    #[test]
    fn test_format_message_block_empty() {
        let messages = vec![];
        let result = format_message_block(&messages).unwrap();
        assert_eq!(result, "No recent messages to display");
    }

    #[test]
    fn test_format_message_block_single_user() {
        let messages = vec![ChatMessage {
            sender: SenderType::User,
            text: "Test".to_string(),
            tools: vec![],
        }];
        let result = format_message_block(&messages).unwrap();
        let expected = r#"
=== LAST MESSAGE BLOCK ===
USER: Test
========================
"#;
        assert!(result.contains(expected.trim()));
    }

    #[test]
    fn test_format_message_block_multiple_turns() {
        let messages = vec![
            ChatMessage {
                sender: SenderType::User,
                text: "First".to_string(),
                tools: vec![],
            },
            ChatMessage {
                sender: SenderType::Assistant,
                text: "First Response".to_string(),
                tools: vec![],
            },
            ChatMessage {
                sender: SenderType::User,
                text: "Second".to_string(),
                tools: vec![],
            },
            ChatMessage {
                sender: SenderType::Assistant,
                text: "Second Response".to_string(),
                tools: vec![],
            },
        ];
        let result = format_message_block(&messages).unwrap();
        let expected = r#"
=== LAST MESSAGE BLOCK ===
USER: Second
------
ASSISTANT: Second Response
========================
"#;
        assert!(result.contains(expected.trim()));
    }

    #[test]
    fn test_format_message_block_truncation() {
        let long_text = "a".repeat(600);
        let messages = vec![ChatMessage {
            sender: SenderType::User,
            text: long_text,
            tools: vec![],
        }];
        let result = format_message_block(&messages).unwrap();
        let truncated_part = "a".repeat(500) + "\n... [truncated]";
        assert!(result.contains(&truncated_part));
        assert!(result.contains("[truncated]"));
    }

    #[test]
    fn test_format_message_block_tools() {
        let messages = vec![ChatMessage {
            sender: SenderType::User,
            text: "Run tool".to_string(),
            tools: vec![ToolCall {
                id: "id1".to_string(),
                name: "tool1".to_string(),
                arguments: "{\"arg\":1}".to_string(),
            }],
        }];
        let result = format_message_block(&messages).unwrap();
        let expected = r#"
=== LAST MESSAGE BLOCK ===
USER: Run tool
  Tools:
    - tool1: {"arg":1}
========================
"#;
        assert!(result.contains(expected.trim()));
    }

    #[tokio::test]
    async fn test_model_command_switch() -> Result<()> {
        let config = create_test_config_with_custom_agent()?;
        let chat = Chat::new(&config, Some("test-model-1".to_string()), HashMap::new()).await?;
        let chat_session = Arc::new(Mutex::new(chat));

        // Test switching to an available model
        let switch_cmd = Command::Model {
            name: Some("test-model-2".to_string()),
        };
        let result = switch_cmd.execute(chat_session.clone()).await?;
        assert!(result, "execute should return true to continue REPL");
        assert_eq!(chat_session.lock().await.model_key(), "test-model-2");

        Ok(())
    }

    #[tokio::test]
    async fn test_model_command_current() -> Result<()> {
        let config = create_test_config_with_custom_agent()?;
        let chat = Chat::new(&config, Some("test-model-1".to_string()), HashMap::new()).await?;
        let chat_session = Arc::new(Mutex::new(chat));

        // Test /model (just ensure it runs without panic)
        let current_cmd = Command::Model { name: None };
        assert!(current_cmd.execute(chat_session.clone()).await?);

        Ok(())
    }

    #[tokio::test]
    async fn test_model_command_invalid() -> Result<()> {
        let config = create_test_config_with_custom_agent()?;
        let chat = Chat::new(&config, Some("test-model-1".to_string()), HashMap::new()).await?;
        let chat_session = Arc::new(Mutex::new(chat));

        // Test switching to a non-existent model
        let switch_to_bad = Command::Model {
            name: Some("bad-model".to_string()),
        };
        let result = switch_to_bad.execute(chat_session.clone()).await?;
        assert!(result, "execute should return true even on error");
        assert_eq!(chat_session.lock().await.model_key(), "test-model-1");

        Ok(())
    }

    #[tokio::test]
    async fn test_profile_command_switch() -> Result<()> {
        let config = create_test_config_with_custom_agent()?;
        let chat = Chat::new(&config, Some("test-model-1".to_string()), HashMap::new()).await?;
        let chat_session = Arc::new(Mutex::new(chat));

        // Test switching to test-profile
        let switch_to_profile = Command::Profile {
            name: Some("test-profile".to_string()),
        };
        let result = switch_to_profile.execute(chat_session.clone()).await?;
        assert!(result, "execute should return true to continue REPL");
        assert_eq!(
            chat_session.lock().await.current_profile().unwrap().0,
            "test-profile"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_profile_command_invalid() -> Result<()> {
        let config = create_test_config_with_custom_agent()?;
        let chat = Chat::new(&config, Some("test-model-1".to_string()), HashMap::new()).await?;
        let chat_session = Arc::new(Mutex::new(chat));

        // Test switching to a non-existent profile
        let switch_to_bad = Command::Profile {
            name: Some("bad-profile".to_string()),
        };
        let result = switch_to_bad.execute(chat_session.clone()).await?;
        assert!(result, "execute should return true even on error");
        // Profile should not have changed
        assert_eq!(
            chat_session.lock().await.current_profile().unwrap().0,
            "test-agent"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_profile_command_current() -> Result<()> {
        let config = create_test_config_with_custom_agent()?;
        let chat = Chat::new(&config, Some("test-model-1".to_string()), HashMap::new()).await?;
        let chat_session = Arc::new(Mutex::new(chat));

        // Test /profile (just ensure it runs without panic)
        let current_profile = Command::Profile { name: None };
        assert!(current_profile.execute(chat_session.clone()).await?);

        Ok(())
    }

    #[tokio::test]
    async fn test_tool_command_set() -> Result<()> {
        // Create Chat instance with a mock tool
        let config = create_test_config_with_custom_agent()?;
        let mock_tool: Arc<dyn Tool> = Arc::new(MockTool);
        let available_tools: HashMap<&str, Arc<dyn Tool>> =
            HashMap::from([("mock_tool", mock_tool)]);

        let chat = Chat::new(&config, Some("test-model-1".to_string()), available_tools).await?;
        let chat_session = Arc::new(Mutex::new(chat));

        // Test setting a tool
        let set_tool_cmd = Command::Tool {
            names: vec!["mock_tool".to_string()],
        };
        let result = set_tool_cmd.execute(chat_session.clone()).await?;
        assert!(result, "execute should return true to continue REPL");

        // Check that the tool is actually set
        {
            let chat_guard = chat_session.lock().await;
            let tools = chat_guard.tools();
            assert_eq!(tools.len(), 1);
            assert_eq!(tools[0].name(), "mock_tool");
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_tool_command_clear() -> Result<()> {
        // Create Chat instance with a mock tool
        let config = create_test_config_with_custom_agent()?;
        let mock_tool: Arc<dyn Tool> = Arc::new(MockTool);
        let available_tools: HashMap<&str, Arc<dyn Tool>> =
            HashMap::from([("mock_tool", mock_tool)]);

        let chat = Chat::new(&config, Some("test-model-1".to_string()), available_tools).await?;
        let chat_session = Arc::new(Mutex::new(chat));

        // Set a tool first
        let set_tool_cmd = Command::Tool {
            names: vec!["mock_tool".to_string()],
        };
        set_tool_cmd.execute(chat_session.clone()).await?;

        // Test clearing tools
        let clear_tools_cmd = Command::Tool {
            names: vec!["clear".to_string()],
        };
        clear_tools_cmd.execute(chat_session.clone()).await?;

        let tools = chat_session.lock().await.tools();
        assert!(tools.is_empty());

        Ok(())
    }

    #[tokio::test]
    async fn test_tool_command_current() -> Result<()> {
        // Create Chat instance with a mock tool
        let config = create_test_config_with_custom_agent()?;
        let mock_tool: Arc<dyn Tool> = Arc::new(MockTool);
        let available_tools: HashMap<&str, Arc<dyn Tool>> =
            HashMap::from([("mock_tool", mock_tool)]);

        let chat = Chat::new(&config, Some("test-model-1".to_string()), available_tools).await?;
        let chat_session = Arc::new(Mutex::new(chat));

        // Test /tool (no arguments) - should show current tools
        let current_cmd = Command::Tool { names: vec![] };
        assert!(current_cmd.execute(chat_session.clone()).await?);

        Ok(())
    }

    #[tokio::test]
    async fn test_tool_command_invalid() -> Result<()> {
        let config = create_test_config_with_custom_agent()?;
        let chat = Chat::new(&config, Some("test-model-1".to_string()), HashMap::new()).await?;
        let chat_session = Arc::new(Mutex::new(chat));

        // Test setting an invalid tool
        let set_bad_tool_cmd = Command::Tool {
            names: vec!["nonexistent_tool".to_string()],
        };
        let result = set_bad_tool_cmd.execute(chat_session.clone()).await?;
        assert!(result, "execute should return true even on error");
        assert!(chat_session.lock().await.tools().is_empty());

        Ok(())
    }

    #[tokio::test]
    async fn test_clear_command() -> Result<()> {
        let config = create_test_config_with_custom_agent()?;
        let chat = Chat::new(&config, Some("test-model-1".to_string()), HashMap::new()).await?;
        let chat_session = Arc::new(Mutex::new(chat));

        // Add a message first
        chat_session
            .lock()
            .await
            .add_messages(
                vec![ChatMessage {
                    sender: SenderType::User,
                    text: "hello".to_string(),
                    tools: vec![],
                }],
                vec![],
            )
            .await;
        assert!(!chat_session.lock().await.get_all_messages().is_empty());

        // Test clear command
        let clear_cmd = Command::Clear;
        assert!(clear_cmd.execute(chat_session.clone()).await?);
        assert!(chat_session.lock().await.get_all_messages().is_empty());

        Ok(())
    }

    #[tokio::test]
    async fn test_log_command() -> Result<()> {
        let config = create_test_config_with_custom_agent()?;
        let chat = Chat::new(&config, Some("test-model-1".to_string()), HashMap::new()).await?;
        let chat_session = Arc::new(Mutex::new(chat));

        // Add a message first
        chat_session
            .lock()
            .await
            .add_messages(
                vec![ChatMessage {
                    sender: SenderType::User,
                    text: "log this".to_string(),
                    tools: vec![],
                }],
                vec![],
            )
            .await;

        // Test log command
        let log_cmd = Command::Log;
        assert!(log_cmd.execute(chat_session.clone()).await?);

        Ok(())
    }

    #[tokio::test]
    async fn test_system_command_set() -> Result<()> {
        let config = create_test_config_with_custom_agent()?;
        let chat = Chat::new(&config, Some("test-model-1".to_string()), HashMap::new()).await?;
        let chat_session = Arc::new(Mutex::new(chat));

        // Test setting system prompt
        let new_prompt = "You are a helpful coding assistant.";
        let set_prompt_cmd = Command::System {
            prompt: Some(new_prompt.to_string()),
        };
        assert!(set_prompt_cmd.execute(chat_session.clone()).await?);
        assert_eq!(chat_session.lock().await.system_prompt(), new_prompt);

        Ok(())
    }

    #[tokio::test]
    async fn test_system_command_view() -> Result<()> {
        let config = create_test_config_with_custom_agent()?;
        let chat = Chat::new(&config, Some("test-model-1".to_string()), HashMap::new()).await?;
        let chat_session = Arc::new(Mutex::new(chat));

        // Test viewing current prompt (no arguments)
        let view_prompt_cmd = Command::System { prompt: None };
        assert!(view_prompt_cmd.execute(chat_session.clone()).await?);

        Ok(())
    }

    #[tokio::test]
    async fn test_exit_command() -> Result<()> {
        let config = create_test_config_with_custom_agent()?;
        let chat = Chat::new(&config, Some("test-model-1".to_string()), HashMap::new()).await?;
        let chat_session = Arc::new(Mutex::new(chat));

        // Test exit command
        let exit_cmd = Command::Exit;
        assert!(!exit_cmd.execute(chat_session.clone()).await?);

        Ok(())
    }
}
