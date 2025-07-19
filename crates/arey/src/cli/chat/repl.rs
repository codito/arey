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

#[derive(Subcommand, Debug, Hash, PartialEq, Eq)]
enum Command {
    /// Clear chat history
    Clear,
    /// Show detailed logs for the last assistant message
    Log,
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
    pub async fn execute(self, session: Arc<Mutex<Chat<'_>>>) -> Result<bool> {
        match self {
            Command::Clear => {
                session.lock().await.clear_messages().await;
                println!("Chat history cleared");
            }
            Command::Log => {
                let chat_guard = session.lock().await;
                match chat_guard.get_last_assistant_message().await {
                    Some(ctx) => {
                        println!(
                            "\n=== TOOL CALLS ===\n{:#?}\n====================",
                            ctx.tools
                        );
                    }
                    None => println!("No logs available"),
                }
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
            Command::Exit => {
                println!("Bye!");
                return Ok(false);
            }
        }
        Ok(true)
    }
}

// -------------
// REPL completion
// -------------
#[derive(Helper, Validator, Highlighter)]
struct Repl {
    pub command_names: Vec<String>,
    pub tool_names: Vec<String>,
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

/// Runs the interactive REPL.
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

    let mut rl = Editor::with_config(config)?;
    rl.set_helper(Some(Repl {
        command_names,
        tool_names,
    }));

    let prompt = (style_chat_text("> ", ChatMessageType::Prompt)).to_string();

    loop {
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

    // Create spinner
    let spinner = GenerationSpinner::new("Generating...".to_string());
    let cancel_token = CancellationToken::new();

    // Clone for async block
    let chat_clone = chat.clone();

    // Child tool messages are created if LLM requires a set of tools to be invoked for responding
    // to a user message.
    let mut child_tool_messages: Vec<ChatMessage> = vec![];
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
                                    was_cancelled_internal = true;
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

    // After the stream finishes, clear the markdown renderer's internal buffer
    // and reset its state for the next message. This does not clear the screen.
    renderer.clear();

    if !child_tool_messages.is_empty() {
        return Box::pin(process_message(
            chat_clone,
            renderer,
            vec![],
            child_tool_messages,
        ))
        .await;
    }

    // Print footer with metrics
    let (metrics, finish_reason_option) = match was_cancelled {
        true => (CompletionMetrics::default(), None),
        false => (metrics, finish_reason),
    };

    let footer = format_footer_metrics(&metrics, finish_reason_option.as_deref(), was_cancelled);

    // Ensure the footer starts on a newline after the markdown output.
    // The `render` function leaves the cursor at the end of the last line it drew.
    // `println!()` will handle adding a newline before printing.
    println!();
    println!();
    println!("{}", style_chat_text(&footer, ChatMessageType::Footer));
    println!();

    Ok(true)
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

        let args = serde_json::from_str(&call.arguments).unwrap_or_else(|e| {
            debug!(
                "Failed to parse tool arguments, defaulting to null. Error: {}, Args: '{}'",
                e, call.arguments
            );
            serde_json::Value::Null
        });
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
