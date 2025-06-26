// Handles user interaction for chat
use crate::chat::Chat;
use crate::console::GenerationSpinner;
use crate::console::{MessageType, format_footer_metrics, style_text};
use anyhow::Result;
use arey_core::completion::{CancellationToken, CompletionMetrics};
use futures::StreamExt;
use rustyline::CompletionType;
use rustyline::{Config, Context, Editor, Helper, Highlighter, Validator, error::ReadlineError};
use std::io::{self, Write};
use std::sync::Arc;
use tokio::sync::Mutex;

static COMMANDS: [&str; 4] = ["/log", "/quit", "/q", "/help"];

#[derive(Helper, Validator, Highlighter)]
struct CommandCompleter;

impl rustyline::completion::Completer for CommandCompleter {
    type Candidate = String;

    fn complete(
        &self,
        line: &str,
        pos: usize,
        _ctx: &Context,
    ) -> Result<(usize, Vec<String>), ReadlineError> {
        // Only suggest commands at start of line
        if pos == 0 || line.starts_with('/') {
            let candidates = COMMANDS
                .iter()
                .filter(|cmd| cmd.starts_with(line))
                .map(|s| s.to_string())
                .collect();

            Ok((0, candidates))
        } else {
            Ok((0, Vec::new()))
        }
    }
}

impl rustyline::hint::Hinter for CommandCompleter {
    type Hint = String;

    fn hint(&self, line: &str, pos: usize, _ctx: &Context) -> Option<String> {
        if line.is_empty() || pos < line.len() {
            return None;
        }
        if line.starts_with('/') {
            // Suggest command completions
            COMMANDS
                .into_iter()
                .find(|cmd| cmd.starts_with(line))
                .map(|cmd| cmd[line.len()..].to_string())
        } else {
            None
        }
    }
}

/// Command handler logic
async fn handle_command(
    chat: &Chat,
    user_input: &str,
    command_list: &Vec<(&str, &str)>,
) -> Result<bool> {
    if let Some(cmd) = command_list
        .iter()
        .find(|(cmd, _)| user_input.starts_with(*cmd) || cmd.starts_with(user_input))
    {
        if cmd.0 == user_input {
            match user_input {
                "/log" => match chat.get_last_assistant_context().await {
                    Some(ctx) => println!("\n=== LOGS ===\n{}\n=============", ctx.logs),
                    None => println!("No logs available"),
                },
                "/quit" => {
                    println!("Bye!");
                    return Ok(true); // Indicate that the chat should exit
                }
                "/help" => {
                    println!("\nAvailable commands:");
                    for (cmd, desc) in command_list.iter() {
                        println!("{:<8} - {}", cmd, desc);
                    }
                    println!();
                }
                _ => {} // Should not happen with exact match
            }
        } else {
            println!("Command suggestion: {}", cmd.0);
        }
    }
    Ok(false) // Indicate that the chat should continue
}

/// Chat UX flow
pub async fn start_chat(chat: Arc<Mutex<Chat>>) -> anyhow::Result<()> {
    println!("Welcome to arey chat! Type '/help' for commands, 'q' to exit.");

    // Configure rustyline
    let config = Config::builder()
        .history_ignore_dups(true)?
        .history_ignore_space(true)
        .completion_type(CompletionType::List)
        .build();

    let mut rl = Editor::with_config(config)?;
    rl.set_helper(Some(CommandCompleter));

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
    // Add available commands with descriptions
    let command_list = vec![
        ("/log", "Show detailed logs for the last assistant message"),
        ("/q", "Alias for /quit command"),
        ("/quit", "Exit the chat session"),
        ("/help", "Show this help message"),
    ];

    let user_input = line.trim();

    // Skip empty input
    if user_input.is_empty() {
        return Ok(true);
    }

    // Handle commands
    if user_input.starts_with('/') {
        // Lock the chat for command handling
        let chat_guard = chat.lock().await;
        if handle_command(&chat_guard, user_input, &command_list).await? {
            return Ok(false); // Exit loop if command handler returns true (e.g., for /quit)
        }
        drop(chat_guard); // Explicitly drop the guard to release the lock
        return Ok(true);
    }

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

    Ok(false)
}
