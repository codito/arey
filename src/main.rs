mod ask; // Add this line
mod chat;
mod core;
mod platform;
mod play; // Added this line

use crate::chat::{Chat, start_chat};
use crate::core::config::get_config;
use ask::run_ask;
use anyhow::Context;
use clap::{Parser, Subcommand, command};
use std::path::Path;
use std::sync::Arc;
use std::io::Write; // For stdout flush
use tokio::sync::Mutex; // Added this line

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
            let instruction = instruction.join(" ");
            run_ask(&instruction, &config, overrides_file.as_deref()).await?;
        }
        Commands::Chat { model } => {
            let chat_model_config = if let Some(model_name) = model {
                config
                    .models
                    .get(model_name.as_str())
                    .cloned()
                    .context(format!("Model '{}' not found in config.", model_name))?
            } else {
                config.chat.model
            };

            let chat_instance = Arc::new(Mutex::new(Chat::new(chat_model_config).await?));
            start_chat(chat_instance).await?;
        }
        Commands::Play { file, no_watch } => {
            let file_path = play::PlayFile::create_missing(file.as_deref().map(Path::new))
                .context("Failed to create play file")?;
            let mut play_file = play::PlayFile::new(&file_path, &config)?;
            play::run_play(&mut play_file, &config, *no_watch).await?;
        }
    }

    Ok(())
}
