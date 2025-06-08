mod core;
mod platform;

use crate::core::chat::Chat;
use crate::core::completion::{CompletionMetrics, combine_metrics};
use crate::core::config::get_config;
use anyhow::Context;
use clap::{Parser, Subcommand};
use futures::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use std::io::{self, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::sync::Mutex;

/// Arey - a simple large language model app.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Show verbose logs.
    #[arg(short, long, global = true)]
    verbose: bool,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Run an instruction and generate response.
    Ask {
        /// Instruction to run.
        instruction: Vec<String>,
        /// Path to overrides file.
        #[arg(short, long)]
        overrides_file: Option<String>,
    },
    /// Chat with an AI model.
    Chat {
        /// Model to use for chat, must be defined in the config.
        #[arg(short, long)]
        model: Option<String>,
    },
    /// Watch FILE for model, prompt and generate response on edit.
    /// If FILE is not provided, a temporary file is created for edit.
    Play {
        /// File to watch.
        file: Option<String>,
        /// Watch the play file and regenerate response on save.
        #[arg(long)]
        no_watch: bool,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Load configuration
    let config = get_config(None).context("Failed to load configuration")?;

    match &cli.command {
        Commands::Ask {
            instruction,
            overrides_file,
        } => {
            // Placeholder for ask command logic
            println!("Running ask command with instruction: {:?}", instruction);
            if let Some(file) = overrides_file {
                println!("Overrides file: {}", file);
            }
            println!("Verbose: {}", cli.verbose);
        }
        Commands::Chat { model } => {
            println!("Welcome to arey chat! Type 'q' to exit.");
            // Start chat with the configured chat model
            let chat_model_config = if let Some(model_name) = model {
                config
                    .models
                    .get(model_name.as_str())
                    .cloned()
                    .context(format!("Model '{}' not found in config.", model_name))?
            } else {
                config.chat.model
            };

            let chat_instance = Chat::new(chat_model_config).await?;
            start_chat(chat_instance).await?;
        }
        Commands::Play { file, no_watch } => {
            // Placeholder for play command logic
            println!("Running play command.");
            if let Some(f) = file {
                println!("File: {}", f);
            }
            println!("No watch: {}", no_watch);
            println!("Verbose: {}", cli.verbose);
        }
    }

    Ok(())
}

async fn start_chat(mut chat: Chat) -> anyhow::Result<()> {
    let stop_flag = Arc::new(AtomicBool::new(false));
    let s = stop_flag.clone();
    ctrlc::set_handler(move || {
        s.store(true, Ordering::SeqCst);
    })
    .context("Error setting Ctrl-C handler")?;

    loop {
        print!("> ");
        io::stdout().flush()?;

        let mut user_input = String::new();
        io::stdin().read_line(&mut user_input)?;
        let user_input = user_input.trim();

        if user_input == "q" || user_input == "quit" {
            println!("Bye!");
            break;
        }

        // Create spinner
        let spinner = ProgressBar::new_spinner();
        spinner.set_style(
            ProgressStyle::with_template("{spinner:.blue} {msg}")
                .unwrap()
                .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]),
        );
        spinner.set_message("Generating...");
        spinner.enable_steady_tick(std::time::Duration::from_millis(100));

        let mut stream = chat.stream_response(user_input.to_string()).await?;
        let chunks_metrics = Arc::new(Mutex::new(Vec::<CompletionMetrics>::new()));
        // Add this line to clone the stop_flag for the task
        let stop_flag_clone = stop_flag.clone();
        let spinner_clone = spinner.clone();

        // Handle cancellation and streaming
        let cancel_task = tokio::spawn(async move {
            while !stop_flag_clone.load(Ordering::SeqCst) {
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            }
            spinner_clone.finish_and_clear();
        });

        let mut first_token_received = false;
        while let Some(response_result) = stream.next().await {
            if stop_flag.load(Ordering::SeqCst) {
                break;
            }

            match response_result {
                Ok(chunk) => {
                    if !first_token_received {
                        spinner.finish_and_clear();
                        first_token_received = true;
                    }
                    print!("{}", chunk.text);
                    io::stdout().flush()?;
                    chunks_metrics.lock().await.push(chunk.metrics.clone());
                }
                Err(e) => eprintln!("Error: {}", e),
            }
        }

        cancel_task.abort();
        spinner.finish_and_clear();

        // Print footer with metrics
        let metrics_vec = chunks_metrics.lock().await;
        let combined = combine_metrics(&metrics_vec);
        let footer = if stop_flag.load(Ordering::SeqCst) {
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
            let tokens_per_sec = (combined.completion_tokens as f32 * 1000.0) / combined.completion_latency_ms;
            footer_details.push_str(&format!(" {:.2} tokens/s.", tokens_per_sec));
        }

        if combined.completion_tokens > 0 {
            footer_details.push_str(&format!(" {} tokens.", combined.completion_tokens));
        }

        if combined.prompt_tokens > 0 {
            footer_details.push_str(&format!(" {} prompt tokens.", combined.prompt_tokens));
        }

        println!();
        println!("{}", footer);
        println!("{}", footer_details);
        println!();

        // Reset stop flag for next message
        stop_flag.store(false, Ordering::SeqCst);
    }

    Ok(())
}
