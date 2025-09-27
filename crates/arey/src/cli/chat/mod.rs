use crate::cli::ux::{TerminalRenderer, get_theme};
use crate::svc::chat::Chat;
use anyhow::{Context, Result};
use arey_core::config::Config;
use arey_core::tools::Tool;
use std::collections::HashMap;
use std::io::stdout;
use std::sync::Arc;
use tokio::sync::Mutex;

mod commands;
mod compl;
mod prompt;
mod repl;
mod test_utils;

/// Executes the chat command, starting an interactive REPL session.
pub async fn execute(
    model: Option<String>,
    config: &Config,
    available_tools: HashMap<&str, Arc<dyn Tool>>,
) -> Result<()> {
    let chat = Chat::new(config, model, available_tools)
        .await
        .context("Failed to initialize chat service")?;
    let theme = get_theme("ansi"); // TODO: Theme from config
    let mut stdout = stdout();
    let mut renderer = TerminalRenderer::new(&mut stdout, &theme);
    repl::run(Arc::new(Mutex::new(chat)), &mut renderer).await
}

#[cfg(test)]
mod tests {
    // TODO: Add unit tests for chat::execute. This would require significant
    // mocking of the REPL and user input.
}
