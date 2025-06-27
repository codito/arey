// Handles user interaction for chat
use crate::chat::Chat;
use crate::console::GenerationSpinner;
use crate::console::{format_footer_metrics, style_text, MessageType};
use anyhow::Result;
use arey_core::completion::{CancellationToken, CompletionMetrics};
use clap::{Parser, Subcommand};
use console::Style;
use futures::StreamExt;
use rustyline::completion::Candidate;
use rustyline::CompletionType;
use rustyline::{error::ReadlineError, Config, Context, Editor, Helper, Highlighter, Validator};
use std::future::Future;
use std::io::{self, Write};
use std::pin::Pin;
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
#[command(multicall = true, no_binary_name = true)]
struct CliCommand {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Show detailed logs for the last assistant message
    Log,
    /// Exit the chat session
    #[command(alias = "q", alias = "quit")]
    Exit,
    /// Show help message
    Help,
}

#[derive(Helper, Validator, Highlighter)]
struct CommandCompleter {
    command_names: Arc<Vec<String>>,
}

impl rustyline::completion::Completer for CommandCompleter {
    type Candidate = CompletionCandidate;

    fn complete(
        &self,
        line: &str,
        pos: usize,
        _ctx: &Context,
    ) -> Result<(usize, Vec<Self::Candidate>), ReadlineError> {
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
pub async fn start_chat(chat: Arc<Mutex<Chat>>) -> anyhow::Result<()> {
    println!("Welcome to arey chat! Type '/help' for commands, '/q' to exit.");

    // Configure rustyline
    let config = Config::builder()
        .history_ignore_dups(true)?
        .history_ignore_space(true)
        .completion_type(CompletionType::List)
        .build();

    let command_names = Arc::new(vec![
        "/log".to_string(),
        "/exit".to_string(),
        "/q".to_string(),
        "/quit".to_string(),
        "/help".to_string(),
    ]);

    let mut rl = Editor::with_config(config)?;
    rl.set_helper(Some(CommandCompleter {
        command_names: command_names.clone(),
    }));

    loop {
        let readline = rl.readline("> ");
        match readline {
            Ok(line) => {
                rl.add_history_entry(&line)?;
                match process_message(&chat, &line).await {
                    Ok(true) => continue,
                    Ok(false) => return Ok(()),
                    Err(err) => return Err(err),
                }
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

async fn process_message(chat: &Arc<Mutex<Chat>>, line: &str) -> Result<bool> {
    let user_input = line.trim();

    // Skip empty input
    if user_input.is_empty() {
        return Ok(true);
    }

    // Handle commands
    if user_input.starts_with('/') {
        let args = match shlex::split(user_input) {
            Some(args) => args,
            None => {
                println!("Invalid command syntax");
                return Ok(true);
            }
        };

        // Parse using clap
        match CliCommand::try_parse_from(args) {
            Ok(CliCommand { command }) => match command {
                Command::Log => {
                    let chat_guard = chat.lock().await;
                    match chat_guard.get_last_assistant_context().await {
                        Some(ctx) => println!("\n=== LOGS ===\n{}\n=============", ctx.logs),
                        None => println!("No logs available"),
                    }
                    Ok(false)
                }
                Command::Exit => {
                    println!("Bye!");
                    Ok(true)
                }
                Command::Help => {
                    println!("Available commands:");
                    println!("{:<8} - {}", "/log", "Show detailed logs");
                    println!("{:<8} - {}", "/exit", "Exit the chat");
                    println!("{:<8} - {}", "/q, /quit", "Aliases for /exit");
                    println!("{:<8} - {}", "/help", "Show this help");
                    println!();
                    Ok(false)
                }
            },
            Err(e) => {
                println!("{e}");
                e.print().unwrap(); // Show help automatically
                Ok(false)
            }
        }
    } else {
        // Create spinner
        let spinner = GenerationSpinner::new();
        let cancel_token = CancellationToken::new();

        // Clone for async block
        let chat_clone = chat.clone();
        let user_input_for_future = user_input.to_string();

        let was_cancelled = {
            // Get stream response
            let mut chat_guard = chat_clone.lock().await;
            let mut stream = {
                chat_guard
                    .stream_response(user_input_for_future, cancel_token.clone())
                    .await?
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
                                    Ok(chunk) => {
                                        // Print token to console
                                        print!("{}", chunk.text);
                                        io::stdout().flush()?;
                                    }
                                    Err(e) => {
                                        eprintln!("Error: {}", e);
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

            // Ensure spinner is cleared after stream processing
            spinner.clear();

            was_cancelled_internal || cancel_token.is_cancelled()
        };

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
        println!();
        println!();
        println!("{}", style_text(&footer, MessageType::Footer));
        println!();

        Ok(true)
    }
}
