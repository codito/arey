// Handles user interaction for chat
use crate::chat::Chat;
use crate::core::completion::{CancellationToken, CompletionMetrics};
use crate::platform::console::GenerationSpinner;
use crate::platform::console::{MessageType, style_text};
use anyhow::Result;
use futures::StreamExt;
use std::io::{self, Write};
use std::sync::Arc;
use tokio::sync::Mutex; // Added this import

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
                "/quit" | "/q" => {
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
    // Add available commands with descriptions
    let command_list = vec![
        ("/log", "Show detailed logs for the last assistant message"),
        ("/q", "Alias for /quit command"),
        ("/quit", "Exit the chat session"),
        ("/help", "Show this help message"),
    ];

    println!("Welcome to arey chat! Type '/help' for commands, 'q' to exit.");

    loop {
        print!("> ");
        io::stdout().flush()?;

        let mut user_input = String::new();
        let num_bytes = io::stdin().read_line(&mut user_input)?;

        // Handle Ctrl+D (EOF)
        if num_bytes == 0 {
            println!("\nBye!");
            return Ok(());
        }

        let user_input = user_input.trim();

        // Skip empty input
        if user_input.is_empty() {
            continue;
        }

        // Handle commands
        if user_input.starts_with('/') {
            // Lock the chat for command handling
            let chat_guard = chat.lock().await;
            if handle_command(&chat_guard, user_input, &command_list).await? {
                break; // Exit loop if command handler returns true (e.g., for /quit)
            }
            drop(chat_guard); // Explicitly drop the guard to release the lock
            continue;
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
        let mut finish_reason = String::new();
        let footer_details = match was_cancelled {
            true => String::new(),
            false => match chat.clone().lock().await.get_last_assistant_context().await {
                Some(ctx) => {
                    finish_reason = ctx.finish_reason.unwrap_or_default();
                    get_footer_details(ctx.metrics)
                }
                None => String::new(), // error
            },
        };
        let footer = match was_cancelled {
            true => "◼ Canceled.".to_string(),
            false => format!("◼ Completed({finish_reason})."),
        };

        println!();
        println!();
        println!(
            "{}",
            style_text(&format!("{footer} {footer_details}"), MessageType::Footer)
        );
        println!();
    }

    Ok(())
}

fn get_footer_details(combined: CompletionMetrics) -> String {
    let mut footer_details = String::new();
    if combined.prompt_eval_latency_ms > 0.0 || combined.completion_latency_ms > 0.0 {
        footer_details.push_str(&format!(
            " {:.2}s to first token.",
            combined.prompt_eval_latency_ms / 1000.0
        ));
        footer_details.push_str(&format!(
            " {:.2}s total.",
            combined.completion_latency_ms / 1000.0
        ));
    }

    if combined.completion_tokens > 0 && combined.completion_latency_ms > 0.0 {
        let tokens_per_sec =
            (combined.completion_tokens as f32 * 1000.0) / combined.completion_latency_ms;
        footer_details.push_str(&format!(" {:.2} tokens/s.", tokens_per_sec));
    }

    if combined.completion_tokens > 0 {
        footer_details.push_str(&format!(
            " {} completion tokens.",
            combined.completion_tokens
        ));
    }

    if combined.prompt_tokens > 0 {
        footer_details.push_str(&format!(" {} prompt tokens.", combined.prompt_tokens));
    }

    footer_details
}
