// Handles user interaction for chat
use crate::core::chat::Chat;
use crate::core::completion::{CompletionMetrics, combine_metrics};
use anyhow::Result;
use futures::StreamExt;
use std::io::{self, Write};
use std::sync::Arc;
use tokio::sync::Mutex; // Use tokio's Mutex for async operations
use crate::platform::console::GenerationSpinner;

/// Command handler logic
async fn handle_command(chat: &Chat, user_input: &str, command_list: &Vec<(&str, &str)>) -> Result<bool> {
    if let Some(cmd) = command_list
        .iter()
        .find(|(cmd, _)| user_input.starts_with(*cmd) || cmd.starts_with(user_input))
    {
        if cmd.0 == user_input {
            match user_input {
                "/log" => match chat.get_last_assistant_logs() {
                    Some(logs) => println!("\n=== LOGS ===\n{logs}\n============="),
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
    } else {
        println!(
            "Unknown command '{}'. Type '/help' for available commands.",
            user_input
        );
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
        io::stdin().read_line(&mut user_input)?;
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

        // Create spinner using the new utility
        let spinner = GenerationSpinner::new();
        let spinner_token = spinner.token(); // Get the token from the spinner

        // Capture user_input for the async block
        let user_input_for_future = user_input.to_string();

        // Clone the Arc for the async block
        let chat_clone_for_future = chat.clone();

        let (combined_metrics, was_cancelled) = if let Some((metrics, cancelled)) = spinner
            .handle_stream(
                async move {
                    // Acquire lock inside the async block
                    let mut chat_guard = chat_clone_for_future.lock().await;
                    let mut stream = chat_guard
                        .stream_response(user_input_for_future, spinner_token)
                        .await?;
                    drop(chat_guard); // Release the lock after getting the stream

                    let chunks_metrics = Arc::new(Mutex::new(Vec::<CompletionMetrics>::new()));
                    let mut first_token_received = false;

                    let mut current_cancel_status = false; // Track cancellation within the stream processing

                    while let Some(response) = stream.next().await {
                        // Check for cancellation *before* processing the chunk
                        if spinner_token.is_cancelled() {
                            current_cancel_status = true;
                            break; // Exit the stream processing loop
                        }

                        match response {
                            Ok(chunk) => {
                                if !first_token_received {
                                    spinner.clear_message(); // Clear spinner message on first token
                                    first_token_received = true;
                                }
                                print!("{}", chunk.text);
                                io::stdout().flush()?;
                                chunks_metrics.lock().await.push(chunk.metrics.clone());
                            }
                            Err(e) => {
                                eprintln!("Error: {}", e);
                                current_cancel_status = true; // Treat error as a reason to stop
                                break;
                            }
                        }
                    }

                    let metrics = chunks_metrics.lock().await;
                    Ok((combine_metrics(&metrics), current_cancel_status))
                }
            )
            .await
        {
            (metrics, cancelled) => (Some(metrics), cancelled),
        } else {
            // handle_stream returned None, meaning it was cancelled or an error occurred before the future completed
            (None, true) // Assume cancelled if None
        };

        // Print footer with metrics
        let footer = if was_cancelled {
            "◼ Canceled.".to_string()
        } else {
            "◼ Completed.".to_string()
        };

        let mut footer_details = String::new();
        if let Some(combined) = combined_metrics {
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
                footer_details.push_str(&format!(" {} tokens.", combined.completion_tokens));
            }

            if combined.prompt_tokens > 0 {
                footer_details.push_str(&format!(" {} prompt tokens.", combined.prompt_tokens));
            }
        }

        println!();
        println!();
        println!("{footer} {footer_details}");
        println!();
    }

    Ok(())
}
