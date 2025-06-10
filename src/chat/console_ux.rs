// Handles user interaction for chat
use crate::chat::service::Chat;
use crate::core::completion::{CancellationToken, CompletionMetrics, combine_metrics};
use anyhow::Result;
use futures::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use std::io::{self, Write};
use std::sync::Arc;
use tokio::signal;
use tokio::sync::Mutex;

/// Command handler logic
fn handle_command(chat: &Chat, user_input: &str, command_list: &Vec<(&str, &str)>) -> Result<bool> {
    if let Some(cmd) = command_list.iter().find(|(cmd, _)| {
        user_input.starts_with(*cmd) || cmd.starts_with(user_input)
    }) {
        if cmd.0 == user_input {
            match user_input {
                "/log" => {
                    match chat.get_last_assistant_logs() {
                        Some(logs) => println!("\n=== LOGS ===\n{logs}\n============="),
                        None => println!("No logs available"),
                    }
                },
                "/quit" | "/q" => {
                    println!("Bye!");
                    return Ok(true); // Indicate that the chat should exit
                },
                "/help" => {
                    println!("\nAvailable commands:");
                    for (cmd, desc) in command_list.iter() {
                        println!("{:<8} - {}", cmd, desc);
                    }
                    println!();
                },
                _ => {} // Should not happen with exact match
            }
        } else {
            println!("Command suggestion: {}", cmd.0);
        }
    } else {
        println!("Unknown command '{}'. Type '/help' for available commands.", user_input);
    }
    Ok(false) // Indicate that the chat should continue
}

/// Chat UX flow
pub async fn start_chat(mut chat: Chat) -> anyhow::Result<()> {
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
            if handle_command(&chat, user_input, &command_list)? {
                break; // Exit loop if command handler returns true (e.g., for /quit)
            }
            continue;
        }
        
        // Create per-message cancellation token
        let cancel_token = CancellationToken::new();

        // Create spinner
        let spinner = ProgressBar::new_spinner();
        spinner.set_style(
            ProgressStyle::with_template("{spinner:.blue} {msg}")
                .unwrap()
                .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]),
        );
        spinner.set_message("Generating...");
        spinner.enable_steady_tick(std::time::Duration::from_millis(100));

        let mut stream = chat
            .stream_response(user_input.to_string(), cancel_token.clone())
            .await?;
        let chunks_metrics = Arc::new(Mutex::new(Vec::<CompletionMetrics>::new()));
        let mut first_token_received = false;

        // Create and pin Ctrl-C signal future
        let ctrl_c_future = signal::ctrl_c();
        tokio::pin!(ctrl_c_future);

        'receive_loop: loop {
            tokio::select! {
                // Handle Ctrl-C signal
                _ = ctrl_c_future.as_mut(), if !cancel_token.is_cancelled() => {
                    cancel_token.cancel();
                },
                
                // Process stream response
                response = stream.next() => {
                    match response {
                        Some(Ok(chunk)) => {
                            if !first_token_received {
                                spinner.finish_and_clear();
                                first_token_received = true;
                            }
                            print!("{}", chunk.text);
                            io::stdout().flush()?;
                            chunks_metrics.lock().await.push(chunk.metrics.clone());
                        }
                        Some(Err(e)) => {
                            eprintln!("Error: {}", e);
                            break 'receive_loop;
                        }
                        None => break 'receive_loop,
                    }
                },

                // Handle cancellation by breaking the loop
                _ = tokio::task::yield_now(), if cancel_token.is_cancelled() => {
                    break 'receive_loop;
                }
            }
        }

        spinner.finish_and_clear();

        // Print footer with metrics
        let metrics_vec = chunks_metrics.lock().await;
        let combined = combine_metrics(&metrics_vec);
        let footer = if cancel_token.is_cancelled() {
            // Check the specific message's cancel token
            "◼ Canceled.".to_string()
        } else {
            "◼ Completed.".to_string()
        };

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
            footer_details.push_str(&format!(" {} tokens.", combined.completion_tokens));
        }

        if combined.prompt_tokens > 0 {
            footer_details.push_str(&format!(" {} prompt tokens.", combined.prompt_tokens));
        }

        println!();
        println!();
        println!("{footer} {footer_details}");
        println!();

        // No need to reset stop flag, as a new CancellationToken is created each loop.
    }

    Ok(())
}
