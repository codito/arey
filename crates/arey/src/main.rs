mod ask;
mod chat;
mod console;
mod play;

use crate::chat::{Chat, start_chat};
use crate::console::{TerminalRenderer, get_render_theme};
use ::console::Term;
use anyhow::{Context, anyhow};
use arey_core::{config::get_config, tools::Tool};
use clap::{Parser, Subcommand, command};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
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

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Load configuration
    let config = get_config(None).context("Failed to load configuration")?;

    // Initialize all available tools
    let mut available_tools: HashMap<String, Arc<dyn Tool>> = HashMap::new();
    for (name, tool_config) in &config.tools {
        let tool: Arc<dyn Tool> = match name.as_str() {
            "search" => Arc::new(
                arey_tools_search::SearchTool::from_config(tool_config)
                    .with_context(|| format!("Failed to initialize tool: {name}"))?,
            ),
            _ => return Err(anyhow!("Unknown tool in config: {}", name)),
        };
        available_tools.insert(name.clone(), tool);
    }

    match &cli.command {
        Commands::Ask { instruction, model } => {
            let instruction = instruction.join(" ");
            let ask_model_config = if let Some(model_name) = model {
                config
                    .models
                    .get(model_name.as_str())
                    .cloned()
                    .context(format!("Model '{model_name}' not found in config."))?
            } else {
                config.task.model.clone()
            };
            ask::run_ask(&instruction, ask_model_config).await?;
        }
        Commands::Chat { model } => {
            let chat_model_config = if let Some(model_name) = model {
                config
                    .models
                    .get(model_name.as_str())
                    .cloned()
                    .context(format!("Model '{model_name}' not found in config."))?
            } else {
                config.chat.model.clone()
            };

            let mut term = Term::stdout();
            let theme = get_render_theme(&config.theme);
            let mut renderer = TerminalRenderer::new(&mut term, &theme);
            let chat_instance = Arc::new(Mutex::new(
                Chat::new(chat_model_config, available_tools).await?,
            ));

            start_chat(chat_instance, &mut renderer).await?;
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
