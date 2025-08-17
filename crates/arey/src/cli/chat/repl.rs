use crate::cli::ux::{
    ChatMessageType, GenerationSpinner, TerminalRenderer, format_footer_metrics, style_chat_text,
};
use crate::svc::chat::Chat;
use anyhow::Result;
use arey_core::completion::{
    CancellationToken, ChatMessage, Completion, CompletionMetrics, SenderType,
};
use arey_core::tools::{Tool, ToolCall, ToolResult};
use clap::{CommandFactory, Parser, Subcommand};
use futures::StreamExt;
use rustyline::completion::{Candidate, Completer};
use rustyline::error::ReadlineError;
use rustyline::hint::Hinter;
use rustyline::{CompletionType, Editor, Helper, Highlighter, Validator};
use serde_json::Value;
use serde_yaml;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::debug;

// -------------
// REPL commands
// -------------
#[derive(Parser, Debug)]
#[command(multicall = true)]
struct CliCommand {
    #[command(subcommand)]
    pub command: Command,
}

#[test]
fn test_model_command_completion() {
    use rustyline::history::DefaultHistory;

    let history = DefaultHistory::new();

    // Test command-line completion for the model command
    let repl = Repl {
        command_names: vec![],
        tool_names: vec![],
        model_names: vec!["model1".to_string(), "model2".to_string()],
        profile_names: vec![],
    };

    // Simulate user typing "/model mod"
    let line = "/model mod";
    let (start, candidates) = repl
        .complete(line, line.len(), &rustyline::Context::new(&history))
        .unwrap();

    // Expecting completion to start at the model prefix (after the space)
    assert_eq!(start, 7); // "/model ".len() is 7
    assert_eq!(candidates.len(), 2);
    assert_eq!(candidates[0].replacement(), "model1");
    assert_eq!(candidates[1].replacement(), "model2");

    // Simulate user typing "/model l"
    let line = "/model l";
    let (start, candidates) = repl
        .complete(line, line.len(), &rustyline::Context::new(&history))
        .unwrap();
    assert_eq!(start, 7);
    assert_eq!(candidates.len(), 1);
    assert_eq!(candidates[0].replacement(), "list");
}

#[derive(Subcommand, Debug, Hash, PartialEq, Eq)]
enum Command {
    /// Clear chat history
    Clear,
    /// Show detailed logs for the last assistant message
    Log,
    /// Manage chat models.
    ///
    /// With no arguments, shows the current model.
    /// Use "list" to see available models.
    #[command(alias = "m", alias = "mod")]
    Model {
        /// Model name to switch to, or "list"
        name: Option<String>,
    },
    /// Manage chat profiles.
    ///
    /// With no arguments, shows the current profile.
    /// Use "list" to see available profiles.
    #[command(alias = "p")]
    Profile {
        /// Profile name to switch to, or "list"
        name: Option<String>,
    },
    /// Set tools for the chat session. E.g. /tool search
    #[command(alias = "t")]
    Tool {
        /// Names of the tools to use
        names: Vec<String>,
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
            Command::Clear => {
                session.lock().await.clear_messages().await;
                println!("Chat history cleared");
            }
            Command::Log => {
                let chat_guard = session.lock().await;
                let messages = chat_guard.get_all_messages().await;
                let block = format_message_block(&messages)?;
                println!("{}", block);
            }
            Command::Tool { names } => {
                let chat_guard = session.lock().await;
                match chat_guard.set_tools(&names).await {
                    Ok(()) => {
                        if names.is_empty() {
                            println!("Tools cleared.");
                        } else {
                            println!("Tools set: {}", names.join(", "));
                        }
                    }
                    Err(e) => {
                        eprintln!("Error setting tools: {e}");
                    }
                }
            }
            Command::Model { name } => match name {
                Some(name) => {
                    if name == "list" {
                        let chat_guard = session.lock().await;
                        let model_names = chat_guard.available_model_names();
                        println!("Available models: {}", model_names.join(", "));
                    } else {
                        let mut chat_guard = session.lock().await;
                        match chat_guard.set_model(&name).await {
                            Ok(()) => {
                                let success_msg = format!("Model switched to: {}", name);
                                println!("{success_msg}",);
                            }
                            Err(e) => {
                                let error_msg = format!("Error switching model: {}", e);
                                eprintln!(
                                    "{}",
                                    style_chat_text(&error_msg, ChatMessageType::Error)
                                );
                            }
                        }
                    }
                }
                None => {
                    let chat_guard = session.lock().await;
                    let model_name = chat_guard.model_name().await;
                    println!("Current model: {}", model_name);
                }
            },
            Command::Profile { name } => match name {
                Some(name) => {
                    if name == "list" {
                        let chat_guard = session.lock().await;
                        let profile_names = chat_guard.available_profile_names();
                        println!("Available profiles: {}", profile_names.join(", "));
                    } else {
                        let mut chat_guard = session.lock().await;
                        match chat_guard.set_profile(&name) {
                            Ok(()) => {
                                let success_msg = format!("Profile switched to: {}", name);
                                println!("{success_msg}");
                            }
                            Err(e) => {
                                let error_msg = format!("Error switching profile: {}", e);
                                eprintln!(
                                    "{}",
                                    style_chat_text(&error_msg, ChatMessageType::Error)
                                );
                            }
                        }
                    }
                }
                None => {
                    let chat_guard = session.lock().await;
                    if let Some((profile_name, profile_data)) = chat_guard.current_profile() {
                        println!("Current profile: {profile_name}");
                        match serde_yaml::to_string(profile_data) {
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
                                    style_chat_text(&error_msg, ChatMessageType::Error)
                                );
                            }
                        }
                    } else {
                        println!("No profile is active.");
                    }
                }
            },
            Command::Exit => {
                println!("Bye!");
                return Ok(false);
            }
        }
        Ok(true)
    }
}

// Model command completion
fn model_compl(
    line: &str,
    pos: usize,
    model_names: &[String],
) -> Result<(usize, Vec<CompletionCandidate>), ReadlineError> {
    let line_to_pos = &line[..pos];
    if let Some(space_pos) = line_to_pos.rfind(' ') {
        let model_prefix_start = space_pos + 1;
        if model_prefix_start <= line_to_pos.len() {
            let model_prefix = &line_to_pos[model_prefix_start..];
            let mut candidates = model_names
                .iter()
                .filter(|name| name.starts_with(model_prefix))
                .map(|name| CompletionCandidate::new(name))
                .collect::<Vec<_>>();

            if "list".starts_with(model_prefix) && !model_names.contains(&"list".to_string()) {
                candidates.push(CompletionCandidate::new("list"));
            }
            return Ok((model_prefix_start, candidates));
        }
    }
    Ok((0, Vec::new()))
}

// Profile command completion
fn profile_compl(
    line: &str,
    pos: usize,
    profile_names: &[String],
) -> Result<(usize, Vec<CompletionCandidate>), ReadlineError> {
    let line_to_pos = &line[..pos];
    if let Some(space_pos) = line_to_pos.rfind(' ') {
        let profile_prefix_start = space_pos + 1;
        if profile_prefix_start <= line_to_pos.len() {
            let profile_prefix = &line_to_pos[profile_prefix_start..];
            let mut candidates = profile_names
                .iter()
                .filter(|name| name.starts_with(profile_prefix))
                .map(|name| CompletionCandidate::new(name))
                .collect::<Vec<_>>();

            if "list".starts_with(profile_prefix) && !profile_names.contains(&"list".to_string()) {
                candidates.push(CompletionCandidate::new("list"));
            }
            return Ok((profile_prefix_start, candidates));
        }
    }
    Ok((0, Vec::new()))
}

// -------------
// REPL completion
// -------------
#[derive(Helper, Validator, Highlighter)]
struct Repl {
    pub command_names: Vec<String>,
    pub tool_names: Vec<String>,
    pub model_names: Vec<String>,
    pub profile_names: Vec<String>,
}

#[derive(Debug)]
struct CompletionCandidate {
    text: String,
    display_string: String,
}

impl CompletionCandidate {
    pub fn new(text: &str) -> Self {
        let display_string = style_chat_text(text, ChatMessageType::Footer).to_string();
        Self {
            text: text.to_owned(),
            display_string,
        }
    }
}

impl Candidate for CompletionCandidate {
    fn display(&self) -> &str {
        &self.display_string
    }

    fn replacement(&self) -> &str {
        &self.text
    }
}

impl Completer for Repl {
    type Candidate = CompletionCandidate;

    fn complete(
        &self,
        line: &str,
        pos: usize,
        _ctx: &rustyline::Context<'_>,
    ) -> Result<(usize, Vec<Self::Candidate>), ReadlineError> {
        let is_command = line.starts_with('/') || line.starts_with('!') || line.starts_with('@');
        if !is_command {
            return Ok((0, Vec::new()));
        }

        let args = shlex::split(line).unwrap_or_default();
        if let Ok(cli_command) = CliCommand::try_parse_from(&args) {
            return match cli_command.command {
                Command::Tool { .. } => tool_compl(line, pos, &self.tool_names),
                Command::Model { .. } => model_compl(line, pos, &self.model_names),
                Command::Profile { .. } => profile_compl(line, pos, &self.profile_names),
                _ => Ok((0, Vec::new())),
            };
        }

        let candidates = self
            .command_names
            .iter()
            .filter(|name| name.starts_with(line))
            .map(|name| CompletionCandidate::new(name))
            .collect();

        Ok((0, candidates))
    }
}

impl Hinter for Repl {
    type Hint = String;

    fn hint(&self, line: &str, pos: usize, _ctx: &rustyline::Context<'_>) -> Option<Self::Hint> {
        if line.is_empty() || pos < line.len() {
            return None;
        }
        if line.starts_with('/') {
            // Suggest command completions
            self.command_names
                .iter()
                .find(|&cmd_name| cmd_name.starts_with(line))
                .map(|cmd_name| cmd_name[line.len()..].into())
        } else {
            None
        }
    }
}

fn tool_compl(
    line: &str,
    pos: usize,
    tool_names: &[String],
) -> Result<(usize, Vec<CompletionCandidate>), ReadlineError> {
    let line_to_pos = &line[..pos];
    if let Some(space_pos) = line_to_pos.rfind(' ') {
        let tool_prefix_start = space_pos + 1;
        if tool_prefix_start <= line_to_pos.len() {
            let tool_prefix = &line_to_pos[tool_prefix_start..];
            let candidates = tool_names
                .iter()
                .filter(|name| name.starts_with(tool_prefix))
                .map(|name| CompletionCandidate::new(name))
                .collect();
            return Ok((tool_prefix_start, candidates));
        }
    }

    Ok((0, Vec::new()))
}

/// Runs the interactive REPL for the chat session.
pub async fn run(chat: Arc<Mutex<Chat<'_>>>, renderer: &mut TerminalRenderer<'_>) -> Result<()> {
    println!("Welcome to arey chat! Type '/help' for commands, '/q' to exit.");

    let config = rustyline::Config::builder()
        .history_ignore_dups(true)?
        .history_ignore_space(true)
        .completion_type(CompletionType::List)
        .build();

    let command_names = CliCommand::command()
        .get_subcommands()
        .flat_map(|c| c.get_name_and_visible_aliases())
        .map(|s| format!("/{s}"))
        .collect::<Vec<_>>();
    let tool_names = {
        let chat_guard = chat.clone();
        chat_guard
            .lock()
            .await
            .available_tools
            .keys()
            .map(|s| s.to_string())
            .collect()
    };
    let model_names = {
        let chat_guard = chat.clone();
        chat_guard
            .lock()
            .await
            .available_model_names()
            .into_iter()
            .map(|s| s.to_string())
            .collect()
    };
    let profile_names = {
        let chat_guard = chat.clone();
        chat_guard
            .lock()
            .await
            .available_profile_names()
            .into_iter()
            .map(|s| s.to_string())
            .collect()
    };

    let mut rl = Editor::with_config(config)?;
    rl.set_helper(Some(Repl {
        command_names,
        tool_names,
        model_names,
        profile_names,
    }));

    loop {
        let prompt = {
            let chat_guard = chat.lock().await;
            let model_name = chat_guard.model_name().await;
            let profile_str = chat_guard
                .profile_name()
                .map(|p| format!(" | profile: {}", p))
                .unwrap_or_default();

            let tools = chat_guard.tools().await;
            let tools_str = if !tools.is_empty() {
                let tool_names: Vec<_> = tools.iter().map(|t| t.name()).collect();
                format!(" | tools: {}", tool_names.join(", "))
            } else {
                String::new()
            };

            let prompt_meta = format!("[model: {}{}{}]", model_name, profile_str, tools_str);
            format!(
                "\n{}\n{}",
                style_chat_text(&prompt_meta, ChatMessageType::Prompt),
                style_chat_text("> ", ChatMessageType::Prompt)
            )
        };
        let readline = rl.readline(&prompt);
        match readline {
            Ok(line) => {
                rl.add_history_entry(&line)?;
                let trimmed_line = line.trim();

                if trimmed_line.is_empty() {
                    continue;
                }

                let is_command = trimmed_line.starts_with('/')
                    || trimmed_line.starts_with('!')
                    || trimmed_line.starts_with('@');

                if is_command {
                    // TODO: error handling
                    let args = shlex::split(trimmed_line).unwrap_or_default();
                    match CliCommand::try_parse_from(args) {
                        Ok(cli_command) => {
                            if !cli_command.command.execute(chat.clone()).await? {
                                return Ok(()); // Exit REPL
                            }
                        }
                        Err(e) => {
                            e.print()?;
                        }
                    }
                } else {
                    let user_messages = vec![ChatMessage {
                        text: line.to_string(),
                        sender: SenderType::User,
                        tools: vec![],
                    }];
                    if !process_message(chat.clone(), renderer, user_messages, vec![]).await? {
                        return Ok(());
                    }
                }
            }
            Err(ReadlineError::Interrupted) => {
                println!("Type /quit to exit.");
                continue;
            }
            Err(ReadlineError::Eof) => {
                println!("\nBye!");
                return Ok(());
            }
            Err(err) => {
                return Err(err.into());
            }
        }
    }
}

async fn process_message(
    chat: Arc<Mutex<Chat<'_>>>,
    renderer: &mut TerminalRenderer<'_>,
    user_messages: Vec<ChatMessage>,
    tool_messages: Vec<ChatMessage>,
) -> Result<bool> {
    let mut metrics = CompletionMetrics::default();
    let mut finish_reason: Option<String> = None;

    // Clear renderer state for this new message processing cycle.
    renderer.clear();

    // Create spinner
    let spinner = GenerationSpinner::new("Generating...".to_string());
    let cancel_token = CancellationToken::new();

    // Clone for async block
    let chat_clone = chat.clone();

    // Child tool messages are created if LLM requires a set of tools to be invoked for responding
    // to a user message.
    let mut child_tool_messages: Vec<ChatMessage> = vec![];
    let mut stream_error = false;
    let was_cancelled = {
        // Get stream response
        let chat_guard = chat_clone.lock().await;
        let available_tools = chat_guard.available_tools.clone();
        let mut stream = {
            chat_guard.add_messages(user_messages, tool_messages).await;
            chat_guard.stream_response(cancel_token.clone()).await?
        };

        let mut first_token_received = false;
        let mut was_cancelled_internal = false;

        // Start listening for Ctrl-C
        let mut ctrl_c_stream = Box::pin(tokio::signal::ctrl_c());

        // Process stream with Ctrl-C and tokenization detection
        loop {
            tokio::select! {
                // Ctrl-C handling
                _ = &mut ctrl_c_stream => {
                    cancel_token.cancel();
                    was_cancelled_internal = true;
                    break;
                },

                // Process the next stream token
                next = stream.next() => {
                    match next {
                        Some(response) => {
                            if !first_token_received {
                                spinner.clear();
                                first_token_received = true;
                            }

                            if cancel_token.is_cancelled() {
                                was_cancelled_internal = true;
                                break;
                            }

                            match response {
                                Ok(Completion::Response(chunk)) => {
                                    if !&chunk.text.is_empty() {
                                        renderer.render_markdown(&chunk.text)?;
                                    }

                                    if let Some(reason) = &chunk.finish_reason {
                                        finish_reason = Some(reason.clone());
                                    }

                                    // Tool messages can come in chunks, we collate all
                                    if let Some(tools) = &chunk.tool_calls {
                                        child_tool_messages =
                                            process_tools(&available_tools, tools).await?;
                                    }
                                }
                                Ok(Completion::Metrics(m)) => {
                                    metrics = m;
                                }
                                Err(e) => {
                                    eprintln!("Error: {e}");
                                    stream_error = true;
                                    break;
                                }
                            }
                        }
                        // End of stream
                        None => break,
                    }
                }
            }
        }

        was_cancelled_internal || cancel_token.is_cancelled()
    };

    // Ensure spinner is cleared after stream processing
    spinner.clear();

    // If the stream produced an error, we're done. The error has already been printed.
    if stream_error {
        return Ok(true);
    }

    // After a successful stream, flush any remaining partial lines from the renderer.
    renderer.render_markdown("\n")?;

    // If the model produced tool calls, recursively call this function to process them.
    if !child_tool_messages.is_empty() {
        return Box::pin(process_message(
            chat_clone,
            renderer,
            vec![],
            child_tool_messages,
        ))
        .await;
    }

    // If we've reached this point, the response is complete. Print the footer.
    let (metrics, finish_reason_option) = match was_cancelled {
        true => (CompletionMetrics::default(), None),
        false => (metrics, finish_reason),
    };

    let footer = format_footer_metrics(&metrics, finish_reason_option.as_deref(), was_cancelled);

    // The `render_markdown("\n")` above ensures we start on a fresh line.
    println!();
    println!("{}", style_chat_text(&footer, ChatMessageType::Footer));

    Ok(true)
}

/// Format the last message block (from last user message to end) into a string.
fn format_message_block(messages: &[ChatMessage]) -> Result<String> {
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

/// Returns set of tool results as messages
async fn process_tools(
    available_tools: &HashMap<&str, Arc<dyn Tool>>,
    tool_calls: &Vec<ToolCall>,
) -> Result<Vec<ChatMessage>> {
    let mut tool_messages: Vec<ChatMessage> = vec![];

    for call in tool_calls {
        let tool_fmt = format!("Tool: {}({})", call.name, call.arguments);
        let tool_msg = style_chat_text(&tool_fmt, ChatMessageType::Footer);
        let spinner = GenerationSpinner::new(tool_msg.to_string());
        let tool = match available_tools.get(call.name.as_str()) {
            Some(t) => t.clone(),
            None => {
                eprintln!(
                    "{}",
                    style_chat_text(
                        &format!("Tool '{}' not available", call.name),
                        ChatMessageType::Error
                    )
                );
                continue;
            }
        };

        // Normalize tool call arguments - could be direct JSON, escaped JSON, or plain string
        debug!("Raw tool call arguments: '{}'", call.arguments);

        let args = match serde_json::from_str(&call.arguments) {
            Ok(value) => {
                debug!("Parsed as direct JSON: {}", value);
                value
            }
            Err(first_error) => {
                debug!(
                    "First parse failed: {}. Trying raw string parse",
                    first_error
                );
                match serde_json::from_str::<Value>(&call.arguments) {
                    Ok(value) => {
                        debug!("Parsed as raw JSON string: {}", value);
                        value
                    }
                    Err(second_error) => {
                        debug!(
                            "Second parse failed: {}. Falling back to input wrapper",
                            second_error
                        );
                        serde_json::json!({ "input": call.arguments })
                    }
                }
            }
        };

        // If we got a string, try to parse that string as JSON to see if it's really a structured value.
        let args = match &args {
            Value::String(s) => match serde_json::from_str(s) {
                Ok(parsed_value) => {
                    debug!("Unescaped inner JSON string successfully: {}", parsed_value);
                    parsed_value
                }
                Err(inner_error) => {
                    debug!(
                        "Failed to unescape inner JSON: {}. Keeping as string.",
                        inner_error
                    );
                    args
                }
            },
            _ => args,
        };

        debug!("Final tool arguments: {}", args);
        let output = match tool.execute(&args).await {
            Ok(out) => out,
            Err(e) => {
                eprintln!(
                    "{}",
                    style_chat_text(
                        &format!("Tool execution failed: {e}"),
                        ChatMessageType::Error
                    )
                );
                continue;
            }
        };

        spinner.clear();
        eprintln!("✓ {tool_msg}");

        // Gemini doesn't provide a tool_id, we fill it if empty
        let call_id = if call.id.is_empty() {
            call.name.to_string()
        } else {
            call.id.to_string()
        };
        let result = ToolResult {
            call: ToolCall {
                id: call_id,
                ..call.clone()
            },
            output,
        };
        tool_messages.push(ChatMessage {
            text: serde_json::to_string(&result)?,
            sender: SenderType::Tool,
            tools: vec![],
        });
    }

    eprintln!();
    Ok(tool_messages)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::ux::{TerminalRenderer, get_theme};
    use crate::svc::chat::Chat;
    use anyhow::Result;
    use arey_core::{
        completion::{ChatMessage, SenderType},
        config::{Config, get_config},
        tools::{Tool, ToolError, ToolResult},
    };
    use async_trait::async_trait;
    use rustyline::history::DefaultHistory;
    use serde_json::{Value, json};
    use std::sync::Arc;
    use std::vec;
    use tempfile::NamedTempFile;
    use tokio::sync::Mutex;

    const BASE_TEST_CONFIG: &str = r#"
models:
  test-model-1:
    provider: test
  test-model-2:
    provider: test
profiles:
  test-profile:
    temperature: 0.8
    repeat_penalty: 1.1
    top_k: 40
    top_p: 0.9
chat:
  model: test-model-1
  profile: test-profile
task:
  model: test-model-1
"#;

    fn get_test_config_from_str(content: &str) -> Result<Config> {
        let mut file = NamedTempFile::new().unwrap();
        std::io::Write::write_all(&mut file, content.as_bytes()).unwrap();
        get_config(Some(file.path().to_path_buf()))
            .map_err(|e| anyhow::anyhow!("Failed to create temp config file. Error {}", e))
    }

    #[derive(Debug)]
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
            json!({})
        }

        async fn execute(&self, _args: &Value) -> std::result::Result<Value, ToolError> {
            Ok(json!("mock tool output"))
        }
    }

    #[test]
    fn test_repl_completer_for_commands() {
        let repl = Repl {
            command_names: vec!["/help".to_string(), "/clear".to_string()],
            tool_names: vec![],
            model_names: vec![],
            profile_names: vec![],
        };
        let line = "/c";
        let history = DefaultHistory::new();
        let (start, candidates) = repl
            .complete(line, line.len(), &rustyline::Context::new(&history))
            .unwrap();
        assert_eq!(start, 0);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].replacement(), "/clear");
    }

    #[test]
    fn test_profile_command_completion() {
        use rustyline::history::DefaultHistory;
        let history = DefaultHistory::new();
        let repl = Repl {
            command_names: vec![],
            tool_names: vec![],
            model_names: vec![],
            profile_names: vec!["profile1".to_string(), "profile2".to_string()],
        };
        let line = "/profile pro";
        let (start, candidates) = repl
            .complete(line, line.len(), &rustyline::Context::new(&history))
            .unwrap();
        assert_eq!(start, 9); // "/profile ".len()
        assert_eq!(candidates.len(), 2);
    }

    #[test]
    fn test_tool_command_completion() {
        use rustyline::history::DefaultHistory;
        let history = DefaultHistory::new();
        let repl = Repl {
            command_names: vec![],
            tool_names: vec!["search".to_string(), "browse".to_string()],
            model_names: vec![],
            profile_names: vec![],
        };
        let line = "/tool se";
        let (start, candidates) = repl
            .complete(line, line.len(), &rustyline::Context::new(&history))
            .unwrap();
        assert_eq!(start, 6); // "/tool ".len()
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].replacement(), "search");
    }

    #[test]
    fn test_repl_completer_for_subcommands() {
        let repl = Repl {
            command_names: vec![],
            tool_names: vec!["search".to_string()],
            model_names: vec!["model-1".to_string()],
            profile_names: vec!["prof-1".to_string()],
        };
        let history = DefaultHistory::new();
        let ctx = rustyline::Context::new(&history);

        // Test tool completion delegation
        let line = "/tool s";
        let (start, candidates) = repl.complete(line, line.len(), &ctx).unwrap();
        assert_eq!(start, 6);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].replacement(), "search");

        // Test profile completion delegation
        let line = "/profile p";
        let (start, candidates) = repl.complete(line, line.len(), &ctx).unwrap();
        assert_eq!(start, 9);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].replacement(), "prof-1");
    }

    #[test]
    fn test_repl_hinter() {
        let repl = Repl {
            command_names: vec!["/help".to_string(), "/clear".to_string()],
            tool_names: vec![],
            model_names: vec![],
            profile_names: vec![],
        };
        let history = DefaultHistory::new();
        let ctx = rustyline::Context::new(&history);

        // Test successful hint
        let line = "/h";
        let hint = repl.hint(line, line.len(), &ctx).unwrap();
        assert_eq!(hint, "elp");

        // Test no hint for non-command
        assert!(repl.hint("abc", 3, &ctx).is_none());
        // Test no hint when cursor is not at the end
        assert!(repl.hint("/help", 3, &ctx).is_none());
        // Test no hint for empty line
        assert!(repl.hint("", 0, &ctx).is_none());
    }

    #[tokio::test]
    async fn test_process_tools() -> Result<()> {
        let mock_tool: Arc<dyn Tool> = Arc::new(MockTool {});
        let available_tools: HashMap<&str, Arc<dyn Tool>> =
            HashMap::from([("mock_tool", mock_tool.clone())]);
        let tool_calls = vec![ToolCall {
            id: "call_1".to_string(),
            name: "mock_tool".to_string(),
            arguments: "{}".to_string(),
        }];

        let messages = process_tools(&available_tools, &tool_calls).await?;

        assert_eq!(messages.len(), 1);
        let msg = &messages[0];
        assert_eq!(msg.sender, SenderType::Tool);

        let tool_result: ToolResult = serde_json::from_str(&msg.text)?;
        assert_eq!(tool_result.call.id, "call_1");
        assert_eq!(tool_result.output, json!("mock tool output"));

        Ok(())
    }

    #[tokio::test]
    async fn test_process_tools_with_stringified_json_argument() -> Result<()> {
        struct ArgRecorder;
        #[async_trait]
        impl Tool for ArgRecorder {
            fn name(&self) -> String {
                "arg_recorder".to_string()
            }

            fn description(&self) -> String {
                "Records arguments".to_string()
            }

            fn parameters(&self) -> Value {
                json!({})
            }

            async fn execute(&self, args: &Value) -> std::result::Result<Value, ToolError> {
                Ok(args.clone())
            }
        }

        let tool: Arc<dyn Tool> = Arc::new(ArgRecorder);
        let available_tools: HashMap<&str, Arc<dyn Tool>> =
            HashMap::from([("arg_recorder", tool.clone())]);
        // The arguments are a string that itself is a JSON object
        let tool_calls = vec![ToolCall {
            id: "call_1".to_string(),
            name: "arg_recorder".to_string(),
            arguments: r#""{\"arg\":42}""#.to_string(),
        }];

        let messages = process_tools(&available_tools, &tool_calls).await?;
        assert_eq!(messages.len(), 1);
        let msg = &messages[0];
        let tool_result: ToolResult = serde_json::from_str(&msg.text)?;

        // The tool should have received the parsed JSON object: {"arg":42}
        assert_eq!(tool_result.output, json!({"arg":42}));

        Ok(())
    }

    #[tokio::test]
    async fn test_process_tools_with_non_json_string() -> Result<()> {
        struct ArgRecorder;
        #[async_trait]
        impl Tool for ArgRecorder {
            fn name(&self) -> String {
                "arg_recorder".to_string()
            }

            fn description(&self) -> String {
                "Records arguments".to_string()
            }

            fn parameters(&self) -> Value {
                json!({})
            }

            async fn execute(&self, args: &Value) -> std::result::Result<Value, ToolError> {
                Ok(args.clone())
            }
        }

        let tool: Arc<dyn Tool> = Arc::new(ArgRecorder);
        let available_tools: HashMap<&str, Arc<dyn Tool>> =
            HashMap::from([("arg_recorder", tool.clone())]);
        let tool_calls = vec![ToolCall {
            id: "call_1".to_string(),
            name: "arg_recorder".to_string(),
            arguments: "plain string".to_string(),
        }];

        let messages = process_tools(&available_tools, &tool_calls).await?;

        assert_eq!(messages.len(), 1);
        let msg = &messages[0];
        let tool_result: ToolResult = serde_json::from_str(&msg.text)?;

        // We expect the tool to have received: {"input": "plain string"}
        assert_eq!(tool_result.output, json!({ "input": "plain string" }));

        Ok(())
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
    async fn test_model_command_execute() -> Result<()> {
        // 1. Setup config
        let config = get_test_config_from_str(BASE_TEST_CONFIG)?;

        // 2. Create Chat instance
        let available_tools = HashMap::new();
        let chat = Chat::new(&config, Some("test-model-1".to_string()), available_tools).await?;
        let chat_session = Arc::new(Mutex::new(chat));

        // 3. Check initial model
        assert_eq!(chat_session.lock().await.model_name().await, "test-model-1");

        // 4. Test successful model switch
        let switch_to_2 = Command::Model {
            name: Some("test-model-2".to_string()),
        };
        let result = switch_to_2.execute(chat_session.clone()).await?;
        assert!(result, "execute should return true to continue REPL");
        assert_eq!(chat_session.lock().await.model_name().await, "test-model-2");

        // 5. Test switching to a non-existent model
        let switch_to_bad = Command::Model {
            name: Some("bad-model".to_string()),
        };
        let result = switch_to_bad.execute(chat_session.clone()).await?;
        assert!(result, "execute should return true even on error");
        // Model should not have changed
        assert_eq!(chat_session.lock().await.model_name().await, "test-model-2");

        // 6. Test /model list (just ensure it runs without panic)
        let list_models = Command::Model {
            name: Some("list".to_string()),
        };
        assert!(list_models.execute(chat_session.clone()).await?);

        // 7. Test /model (just ensure it runs without panic)
        let current_model = Command::Model { name: None };
        assert!(current_model.execute(chat_session.clone()).await?);

        Ok(())
    }

    #[tokio::test]
    async fn test_profile_command_execute() -> Result<()> {
        // 1. Setup config
        let config = get_test_config_from_str(BASE_TEST_CONFIG)?;

        // 2. Create Chat instance
        let chat = Chat::new(&config, Some("test-model-2".to_string()), HashMap::new()).await?;
        assert_eq!(chat.profile_name(), Some("test-profile".to_string()));
        let chat_session = Arc::new(Mutex::new(chat));

        // 3. Test successful profile switch
        let switch_to_prof = Command::Profile {
            name: Some("test-profile".to_string()),
        };
        assert!(switch_to_prof.execute(chat_session.clone()).await?);
        let chat_guard = chat_session.lock().await;
        assert_eq!(chat_guard.profile_name(), Some("test-profile".to_string()));
        assert_eq!(
            chat_guard.model_name().await,
            "test-model-2",
            "Model should not change when setting a profile"
        );
        drop(chat_guard);

        // 4. Test switching to a non-existent profile
        let switch_to_bad = Command::Profile {
            name: Some("bad-profile".to_string()),
        };
        assert!(switch_to_bad.execute(chat_session.clone()).await?);
        let chat_guard = chat_session.lock().await;
        assert_eq!(
            chat_guard.profile_name(),
            Some("test-profile".to_string()),
            "Profile should not have changed after a failed attempt"
        );
        drop(chat_guard);

        // 5. Test /profile list (ensure it runs)
        let list_profiles = Command::Profile {
            name: Some("list".to_string()),
        };
        assert!(list_profiles.execute(chat_session.clone()).await?);

        // 6. Test /profile (ensure it shows current profile)
        let current_profile = Command::Profile { name: None };
        assert!(current_profile.execute(chat_session.clone()).await?);

        Ok(())
    }

    #[tokio::test]
    async fn test_tool_command_and_prompt() -> Result<()> {
        // 1. Setup config
        let config = get_test_config_from_str(BASE_TEST_CONFIG)?;

        // 2. Create Chat instance with a mock tool
        let mock_tool: Arc<dyn Tool> = Arc::new(MockTool);
        let available_tools: HashMap<&str, Arc<dyn Tool>> =
            HashMap::from([("mock_tool", mock_tool)]);

        let chat = Chat::new(&config, Some("test-model-1".to_string()), available_tools).await?;
        let chat_session = Arc::new(Mutex::new(chat));

        // 3. Test setting a tool
        let set_tool_cmd = Command::Tool {
            names: vec!["mock_tool".to_string()],
        };
        let result = set_tool_cmd.execute(chat_session.clone()).await?;
        assert!(result, "execute should return true to continue REPL");

        // Check that the tool is actually set
        {
            let chat_guard = chat_session.lock().await;
            let tools = chat_guard.tools().await;
            assert_eq!(tools.len(), 1);
            assert_eq!(tools[0].name(), "mock_tool");
        }

        // 4. Test prompt generation
        let prompt_meta = {
            let chat_guard = chat_session.lock().await;
            let model_name = chat_guard.model_name().await;
            let profile_str = chat_guard
                .profile_name()
                .map(|p| format!(" | profile: {}", p))
                .unwrap_or_default();
            let tools = chat_guard.tools().await;
            let tools_str = if !tools.is_empty() {
                let tool_names: Vec<_> = tools.iter().map(|t| t.name()).collect();
                format!(" | tools: {}", tool_names.join(", "))
            } else {
                String::new()
            };
            format!("[model: {}{}{}]", model_name, profile_str, tools_str)
        };

        assert_eq!(
            prompt_meta,
            "[model: test-model-1 | profile: test-profile | tools: mock_tool]"
        );

        // 5. Test clearing tools
        let clear_tools_cmd = Command::Tool { names: vec![] };
        clear_tools_cmd.execute(chat_session.clone()).await?;

        let prompt_meta_after_clear = {
            let chat_guard = chat_session.lock().await;
            let model_name = chat_guard.model_name().await;
            let profile_str = chat_guard
                .profile_name()
                .map(|p| format!(" | profile: {}", p))
                .unwrap_or_default();
            let tools = chat_guard.tools().await;
            let tools_str = if !tools.is_empty() {
                let tool_names: Vec<_> = tools.iter().map(|t| t.name()).collect();
                format!(" | tools: {}", tool_names.join(", "))
            } else {
                String::new()
            };
            format!("[model: {}{}{}]", model_name, profile_str, tools_str)
        };

        assert_eq!(
            prompt_meta_after_clear,
            "[model: test-model-1 | profile: test-profile]"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_command_execute_various() -> Result<()> {
        let config = get_test_config_from_str(BASE_TEST_CONFIG)?;
        let chat = Chat::new(&config, Some("test-model-1".to_string()), HashMap::new()).await?;
        let chat_session = Arc::new(Mutex::new(chat));

        // Test Clear command
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
        assert!(
            !chat_session
                .lock()
                .await
                .get_all_messages()
                .await
                .is_empty()
        );
        let clear_cmd = Command::Clear;
        assert!(clear_cmd.execute(chat_session.clone()).await?);
        assert!(
            chat_session
                .lock()
                .await
                .get_all_messages()
                .await
                .is_empty()
        );

        // Test Log command
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
        let log_cmd = Command::Log;
        assert!(log_cmd.execute(chat_session.clone()).await?);

        // Test Tool command error
        let set_bad_tool_cmd = Command::Tool {
            names: vec!["nonexistent_tool".to_string()],
        };
        assert!(set_bad_tool_cmd.execute(chat_session.clone()).await?);
        assert!(chat_session.lock().await.tools().await.is_empty());

        // Test Exit command
        let exit_cmd = Command::Exit;
        assert!(!exit_cmd.execute(chat_session.clone()).await?);

        Ok(())
    }

    #[tokio::test]
    async fn test_process_message_simple_response() -> Result<()> {
        // 1. Setup Chat and Renderer
        let config = get_test_config_from_str(BASE_TEST_CONFIG)?;
        let chat = Chat::new(&config, Some("test-model-1".to_string()), HashMap::new()).await?;
        let chat_session = Arc::new(Mutex::new(chat));
        let mut buffer = Vec::new();
        let theme = get_theme("ansi");
        let mut renderer = TerminalRenderer::new(&mut buffer, &theme);

        // 2. Call process_message
        let user_message = ChatMessage {
            sender: SenderType::User,
            text: "Hi".to_string(),
            tools: vec![],
        };
        process_message(chat_session, &mut renderer, vec![user_message], vec![]).await?;

        // 3. Assert rendered output
        let output = String::from_utf8(buffer).unwrap();
        assert!(output.contains("Hello world"));
        Ok(())
    }

    #[tokio::test]
    async fn test_process_message_with_tool_call() -> Result<()> {
        let tool_call_config = r#"
models:
  tool-call-model:
    provider: test
    response_mode: "tool_call"
profiles: {}
chat:
  model: tool-call-model
task:
  model: tool-call-model
"#;
        let config = get_test_config_from_str(tool_call_config)?;
        let mock_tool: Arc<dyn Tool> = Arc::new(MockTool);
        let available_tools: HashMap<&str, Arc<dyn Tool>> =
            HashMap::from([("mock_tool", mock_tool)]);
        let chat = Chat::new(
            &config,
            Some("tool-call-model".to_string()),
            available_tools,
        )
        .await?;
        let chat_session = Arc::new(Mutex::new(chat));

        let mut buffer = Vec::new();
        let theme = get_theme("ansi");
        let mut renderer = TerminalRenderer::new(&mut buffer, &theme);

        let user_message = ChatMessage {
            sender: SenderType::User,
            text: "Hi".to_string(),
            tools: vec![],
        };
        process_message(chat_session, &mut renderer, vec![user_message], vec![]).await?;

        let output = String::from_utf8(buffer).unwrap();
        assert!(output.contains("Tool output is mock tool output"));
        Ok(())
    }

    #[tokio::test]
    async fn test_process_message_stream_error() -> Result<()> {
        let error_config = r#"
models:
  error-model:
    provider: test
    response_mode: "error"
profiles: {}
chat:
  model: error-model
task:
  model: error-model
"#;
        let config = get_test_config_from_str(error_config)?;
        let chat = Chat::new(&config, Some("error-model".to_string()), HashMap::new()).await?;
        let chat_session = Arc::new(Mutex::new(chat));
        let mut buffer = Vec::new();
        let theme = get_theme("ansi");
        let mut renderer = TerminalRenderer::new(&mut buffer, &theme);

        let user_message = ChatMessage {
            sender: SenderType::User,
            text: "Hi".to_string(),
            tools: vec![],
        };
        process_message(chat_session, &mut renderer, vec![user_message], vec![]).await?;

        // Expect no output to renderer, error is printed to stderr
        let output = String::from_utf8(buffer).unwrap();
        assert!(
            output.is_empty(),
            "Output should be empty. Output: {}",
            output
        );
        Ok(())
    }
}
