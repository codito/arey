// Handles user interaction for chat
use crate::chat::{Chat, Message};
use crate::console::{
    GenerationSpinner, MessageType, TerminalRenderer, format_footer_metrics, style_text,
};
use anyhow::Result;
use arey_core::completion::{CancellationToken, CompletionMetrics, SenderType};
use arey_core::tools::{ToolCall, ToolResult};
use chrono::Utc;
use clap::{CommandFactory, Parser, Subcommand};
use console::Style;
use futures::StreamExt;
use rustyline::CompletionType;
use rustyline::completion::Candidate;
use rustyline::{Config, Context, Editor, Helper, Highlighter, Validator, error::ReadlineError};
use std::sync::Arc;
use tokio::sync::Mutex;

#[derive(Debug)]
struct CompletionCandidate {
    text: String,
    display_string: String,
}

impl CompletionCandidate {
    fn new(text: String) -> Self {
        let display_string = Style::new().white().apply_to(&text).to_string();
        Self {
            text,
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

#[derive(Parser, Debug)]
#[command(multicall = true)]
struct CliCommand {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
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

#[derive(Helper, Validator, Highlighter)]
struct CommandCompleter {
    command_names: Vec<String>,
    tool_names: Vec<String>,
}

impl rustyline::completion::Completer for CommandCompleter {
    type Candidate = CompletionCandidate;

    fn complete(
        &self,
        line: &str,
        pos: usize,
        _ctx: &Context,
    ) -> Result<(usize, Vec<Self::Candidate>), ReadlineError> {
        let line_to_pos = &line[..pos];

        // Autocomplete for /tool command arguments
        if line_to_pos.starts_with("/tool ") || line_to_pos.starts_with("/t ") {
            if let Some(space_pos) = line_to_pos.rfind(' ') {
                let tool_prefix_start = space_pos + 1;
                if tool_prefix_start <= line_to_pos.len() {
                    let tool_prefix = &line_to_pos[tool_prefix_start..];
                    let candidates = self
                        .tool_names
                        .iter()
                        .filter(|name| name.starts_with(tool_prefix))
                        .map(|name| CompletionCandidate::new(name.clone()))
                        .collect();
                    return Ok((tool_prefix_start, candidates));
                }
            }
        }

        // Only suggest commands at start of line
        if pos == 0 || line.starts_with('/') {
            let candidates = self
                .command_names
                .iter()
                .filter(|&cmd_name| cmd_name.starts_with(line))
                .map(|s| CompletionCandidate::new(s.clone()))
                .collect();

            Ok((0, candidates))
        } else {
            Ok((0, Vec::new()))
        }
    }
}

impl rustyline::hint::Hinter for CommandCompleter {
    type Hint = String;

    fn hint(&self, line: &str, pos: usize, _ctx: &Context) -> Option<Self::Hint> {
        if line.is_empty() || pos < line.len() {
            return None;
        }
        if line.starts_with('/') {
            // Suggest command completions
            self.command_names
                .iter()
                .find(|&cmd_name| cmd_name.starts_with(line))
                .map(|cmd_name| {
                    format!("{}", Style::new().white().apply_to(&cmd_name[line.len()..]))
                })
        } else {
            None
        }
    }
}

/// Chat UX flow
pub async fn start_chat(
    chat: Arc<Mutex<Chat>>,
    renderer: &mut TerminalRenderer<'_>,
) -> anyhow::Result<()> {
    println!("Welcome to arey chat! Type '/help' for commands, '/q' to exit.");

    // Configure rustyline
    let config = Config::builder()
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
        let chat_guard = chat.lock().await;
        chat_guard.get_available_tool_names()
    };

    let mut rl = Editor::with_config(config)?;
    rl.set_helper(Some(CommandCompleter {
        command_names,
        tool_names,
    }));

    let prompt = (style_text("> ", MessageType::Prompt)).to_string();
    loop {
        let readline = rl.readline(&prompt);
        match readline {
            Ok(line) => {
                rl.add_history_entry(&line)?;
                let user_input = line.trim();

                // Skip empty input
                if user_input.is_empty() {
                    continue;
                }

                let continue_repl = match user_input.starts_with('/') {
                    true => process_command(&chat, user_input).await?,
                    false => {
                        let user_messages = vec![Message {
                            text: line.to_string(),
                            sender: SenderType::User,
                            _timestamp: Utc::now(),
                            context: None,
                        }];
                        process_message(&chat, renderer, user_messages, vec![]).await?
                    }
                };

                if continue_repl {
                    continue;
                }

                return Ok(());
            }
            Err(ReadlineError::Interrupted) => {
                // Ctrl-C pressed, but not during generation.
                // The generation loop handles Ctrl-C during generation.
                println!("Type /quit to exit.");
                continue;
            }
            Err(ReadlineError::Eof) => {
                // Ctrl-D pressed
                println!("\nBye!");
                return Ok(());
            }
            Err(err) => {
                return Err(err.into());
            }
        }
    }
}

/// Returns false if the REPL should break.
async fn process_command(chat: &Arc<Mutex<Chat>>, user_input: &str) -> Result<bool> {
    // Handle commands
    let args = match shlex::split(user_input) {
        Some(args) => args,
        None => {
            println!("Invalid command syntax");
            return Ok(true);
        }
    };

    let continue_repl = match CliCommand::try_parse_from(args) {
        Ok(CliCommand { command }) => match command {
            Command::Clear => {
                chat.lock().await.clear_messages().await;
                println!("Chat history cleared");
                true
            }
            Command::Log => {
                let chat_guard = chat.lock().await;
                match chat_guard.get_last_assistant_context().await {
                    Some(ctx) => {
                        println!(
                            "\n=== RAW API LOGS ===\n{}\n====================",
                            ctx.raw_api_logs
                        );
                    }
                    None => println!("No logs available"),
                }
                true
            }
            Command::Tool { names } => {
                let mut chat_guard = chat.lock().await;
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
                true
            }
            Command::Exit => {
                println!("Bye!");
                false
            }
        },
        Err(e) => {
            e.print().unwrap();
            true
        }
    };

    Ok(continue_repl)
}

/// Returns false if the REPL should break.
async fn process_message(
    chat: &Arc<Mutex<Chat>>,
    renderer: &mut TerminalRenderer<'_>,
    user_messages: Vec<Message>,
    tool_messages: Vec<Message>,
) -> Result<bool> {
    // Create spinner
    let spinner = GenerationSpinner::new();
    let cancel_token = CancellationToken::new();

    // Clone for async block
    let chat_clone = chat.clone();

    // Child tool messages are created if LLM requires a set of tools to be invoked for responding
    // to a user message.
    let was_cancelled = {
        // Get stream response
        let mut chat_guard = chat_clone.lock().await;
        let mut stream = {
            chat_guard
                .stream_response(user_messages, tool_messages, cancel_token.clone())
                .await?
        };

        let mut first_token_received = false;
        let mut was_cancelled_internal = false;
        let mut tool_calls: Vec<ToolCall> = vec![];

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
                                Ok(chunk) => {
                                    if !&chunk.text.is_empty() {
                                        renderer.render_markdown(&chunk.text)?;
                                    }

                                    // Tool messages can come in chunks, we collate all
                                    for t in &chunk.tool_calls.unwrap_or_default() {
                                        tool_calls.push(t.clone());
                                    }
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

        // Process tool calls and send them to the LLM with recursion
        if !tool_calls.is_empty() {
            let tool_messages = process_tools(&chat_clone, &tool_calls).await?;
            spinner.clear();
            return Box::pin(process_message(
                &chat_clone,
                renderer,
                vec![],
                tool_messages,
            ))
            .await;
        }

        // Ensure spinner is cleared after stream processing
        spinner.clear();

        was_cancelled_internal || cancel_token.is_cancelled()
    };

    // After the stream finishes, clear the markdown renderer's internal buffer
    // and reset its state for the next message. This does not clear the screen.
    renderer.clear();

    // Print footer with metrics
    let (metrics, finish_reason_option) = match was_cancelled {
        true => (CompletionMetrics::default(), None),
        false => {
            if let Some(ctx) = chat.clone().lock().await.get_last_assistant_context().await {
                (ctx.metrics, ctx.finish_reason)
            } else {
                (CompletionMetrics::default(), None)
            }
        }
    };

    let footer = format_footer_metrics(&metrics, finish_reason_option.as_deref(), was_cancelled);

    // Ensure the footer starts on a new line after the markdown output.
    // The `render` function leaves the cursor at the end of the last line it drew.
    // `println!()` will handle adding a newline before printing.
    println!();
    println!();
    println!("{}", style_text(&footer, MessageType::Footer));
    println!();

    Ok(true)
}

/// Returns set of tool results as messages
async fn process_tools(
    chat: &Arc<Mutex<Chat>>,
    tool_calls: &Vec<ToolCall>,
) -> Result<Vec<Message>> {
    let mut tool_messages: Vec<Message> = vec![];

    let chat_guard = chat.lock().await;
    for call in tool_calls {
        println!(
            "{}",
            style_text(
                &format!("\n🛠️ Calling tool: {}({})", call.name, call.arguments),
                MessageType::Footer
            )
        );

        let tool = match chat_guard.available_tools.get(&call.name) {
            Some(t) => t.clone(),
            None => {
                eprintln!(
                    "{}",
                    style_text(
                        &format!("Tool '{}' not available", call.name),
                        MessageType::Error
                    )
                );
                continue;
            }
        };

        let output = match tool.execute(&call.arguments).await {
            Ok(out) => out,
            Err(e) => {
                eprintln!(
                    "{}",
                    style_text(&format!("Tool execution failed: {e}"), MessageType::Error)
                );
                continue;
            }
        };

        println!(
            "{}",
            style_text(&format!("✅ Tool result: {output}"), MessageType::Footer)
        );

        let result = ToolResult {
            call: call.clone(),
            output,
        };
        tool_messages.push(Message {
            text: serde_json::to_string(&result)?,
            sender: SenderType::Tool,
            _timestamp: Utc::now(),
            context: None,
        });
    }

    Ok(tool_messages)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustyline::completion::Completer;
    use rustyline::history::DefaultHistory;

    #[test]
    fn test_command_completer_commands() {
        let completer = CommandCompleter {
            command_names: vec!["/help".to_string(), "/history".to_string()],
            tool_names: vec![],
        };
        let history = DefaultHistory::new();
        let ctx = Context::new(&history);

        let (_start, candidates) = completer.complete("/h", 2, &ctx).unwrap();
        assert_eq!(candidates.len(), 2);
        assert_eq!(candidates[0].replacement(), "/help");
        assert_eq!(candidates[1].replacement(), "/history");

        let (start, candidates) = completer.complete("/hist", 5, &ctx).unwrap();
        assert_eq!(start, 0);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].replacement(), "/history");
    }

    #[test]
    fn test_command_completer_tools() {
        let completer = CommandCompleter {
            command_names: vec!["/tool".to_string()],
            tool_names: vec!["search".to_string(), "calculator".to_string()],
        };
        let history = DefaultHistory::new();
        let ctx = Context::new(&history);

        // Complete first tool
        let line = "/tool s";
        let (start, candidates) = completer.complete(line, line.len(), &ctx).unwrap();
        assert_eq!(start, 6); // after "/tool "
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].replacement(), "search");

        // Complete with empty prefix
        let line = "/tool ";
        let (start, candidates) = completer.complete(line, line.len(), &ctx).unwrap();
        assert_eq!(start, 6);
        assert_eq!(candidates.len(), 2);
        assert_eq!(candidates[0].replacement(), "search");
        assert_eq!(candidates[1].replacement(), "calculator");

        // Complete second tool
        let line = "/tool search calc";
        let (start, candidates) = completer.complete(line, line.len(), &ctx).unwrap();
        assert_eq!(start, 13); // after "/tool search "
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].replacement(), "calculator");
    }

    #[test]
    fn test_command_completer_no_match() {
        let completer = CommandCompleter {
            command_names: vec!["/help".to_string()],
            tool_names: vec!["search".to_string()],
        };
        let history = DefaultHistory::new();
        let ctx = Context::new(&history);
        let (_start, candidates) = completer.complete("foo", 3, &ctx).unwrap();
        assert_eq!(candidates.len(), 0);
    }
}
