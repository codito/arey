use anyhow::{Context, Result};
use arey_core::{config::get_config, get_data_dir};
use clap::{Parser, Subcommand};

use crate::ext::get_tools;

pub mod chat;
pub mod play;
pub mod run;

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

fn setup_logging() -> anyhow::Result<()> {
    let data_dir = get_data_dir().context("Failed to get data directory")?;
    let log_path = data_dir.join("arey.log");

    if log_path.exists() {
        let metadata = std::fs::metadata(&log_path)?;
        if metadata.len() > 100 * 1024 {
            // 100KB
            let backup_path = data_dir.join("arey.log.old");
            if backup_path.exists() {
                std::fs::remove_file(&backup_path)?;
            }
            std::fs::rename(&log_path, backup_path)?;
        }
    }

    let log_file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)?;

    tracing_subscriber::fmt()
        .with_env_filter("arey=debug,rustyline=info")
        .with_writer(log_file)
        .init();
    Ok(())
}

pub async fn run_app() -> Result<()> {
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
        Commands::Chat { model } => chat::execute(model.clone(), &config, available_tools).await,
        Commands::Play { file, no_watch } => {
            play::execute(file.as_deref(), *no_watch, &config).await
        }
    }
}
