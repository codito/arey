use anyhow::{Context, Result};
use arey_core::{config::get_config, get_data_dir};
use clap::{Parser, Subcommand};

use crate::ext::get_tools;

pub mod ask;
pub mod chat;
pub mod play;

/// Arey - a simple large language model app.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Cli {
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
        /// Model to use for completion
        #[arg(short, long)]
        model: Option<String>,
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

pub async fn run(cli: &Cli) -> Result<()> {
    // Load configuration
    let config = get_config(None).context("Failed to load configuration")?;

    // Initialize all available tools
    let available_tools = get_tools(&config).context("Failed to get builtin tools")?;

    match &cli.command {
        Commands::Ask { instruction, model } => {
            ask::execute(instruction.clone(), model.clone(), &config).await
        }
        Commands::Chat { model } => chat::execute(model.clone(), &config, available_tools).await,
        Commands::Play { file, no_watch } => {
            play::execute(file.as_deref(), *no_watch, &config).await
        }
    }
}
