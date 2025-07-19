//! Arey app cli definition and entrypoint.
mod chat;
mod play;
mod run;
pub mod ux;

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result};
use arey_core::config::{Config, get_config};
use arey_core::tools::Tool;
use clap::{Parser, Subcommand};

use crate::ext::get_tools;
use crate::log::setup_logging;

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
    Run {
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

/// Runs the main CLI application.
pub async fn run() -> Result<()> {
    let cli = Cli::parse();

    if cli.verbose {
        setup_logging().context("Failed to set up logging")?;
    }

    // Load configuration
    let config = get_config(None).context("Failed to load configuration")?;

    // Initialize all available tools
    let available_tools = get_tools(&config).context("Failed to get builtin tools")?;

    match &cli.command {
        Commands::Run { instruction, model } => {
            run::execute(instruction.clone(), model.clone(), &config).await
        }
        Commands::Chat { model } => execute_chat(model.clone(), &config, available_tools).await,
        Commands::Play { file, no_watch } => {
            play::execute(file.as_deref(), *no_watch, &config).await
        }
    }
}

async fn execute_chat(
    model: Option<String>,
    config: &Config,
    available_tools: HashMap<&str, Arc<dyn Tool>>,
) -> Result<()> {
    crate::cli::chat::execute(model, config, available_tools).await
}

#[cfg(test)]
mod tests {
    // TODO: Add integration tests for the CLI entrypoint `run`.
    // This would involve running the binary with different arguments and
    // checking exit codes and output.
}
