mod core;
mod platform;

use crate::core::chat::Chat;
use crate::core::config::get_config;
use anyhow::Context;
use clap::{Parser, Subcommand};
use futures::StreamExt;

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
                    .get(&model_name)
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
    // println!("Welcome to arey chat! Type 'q' to exit."); // Moved to main

    loop {
        print!("> ");
        use std::io::{self, Write};
        io::stdout().flush()?; // Ensure prompt is displayed immediately

        let mut user_input = String::new();
        io::stdin().read_line(&mut user_input)?;
        let user_input = user_input.trim();

        if user_input == "q" || user_input == "quit" {
            println!("Bye!");
            break;
        }

        let mut stream = chat.stream_response(user_input.to_string()).await?;

        while let Some(response_result) = stream.next().await {
            match response_result {
                Ok(chunk) => {
                    print!("{}", chunk.text);
                }
                Err(e) => eprintln!("Error: {}", e),
            }
        }
        println!(); // Newline after AI response
    }

    Ok(())
}
